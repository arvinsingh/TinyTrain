"""Triton fused layer normalization kernel."""
import torch
import triton
import triton.language as tl


@triton.jit
def layernorm_fwd_kernel(
    X, Y, W, B, Mean, Rstd,
    stride_x, N,
    eps,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_N)
    mask = offs < N

    x = tl.load(X + row * stride_x + offs, mask=mask, other=0.0).to(tl.float32)

    mean = tl.sum(x, axis=0) / N
    xc = x - mean
    var = tl.sum(xc * xc, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + eps)

    x_hat = xc * rstd

    w = tl.load(W + offs, mask=mask, other=1.0)
    b = tl.load(B + offs, mask=mask, other=0.0)
    y = x_hat * w + b

    tl.store(Y + row * stride_x + offs, y, mask=mask)
    tl.store(Mean + row, mean)
    tl.store(Rstd + row, rstd)


@triton.jit
def layernorm_bwd_kernel(
    DY, X, W, Mean, Rstd,
    DX, DW, DB,
    stride_x, N,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_N)
    mask = offs < N

    dy = tl.load(DY + row * stride_x + offs, mask=mask, other=0.0).to(tl.float32)
    x = tl.load(X + row * stride_x + offs, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(W + offs, mask=mask, other=1.0).to(tl.float32)
    mean = tl.load(Mean + row)
    rstd = tl.load(Rstd + row)

    x_hat = (x - mean) * rstd
    wdy = w * dy

    # dx
    c1 = tl.sum(wdy, axis=0) / N
    c2 = tl.sum(wdy * x_hat, axis=0) / N
    dx = rstd * (wdy - c1 - x_hat * c2)
    tl.store(DX + row * stride_x + offs, dx, mask=mask)

    # dw, db (per-row partial; need external reduction across rows)
    tl.store(DW + row * N + offs, dy * x_hat, mask=mask)
    tl.store(DB + row * N + offs, dy, mask=mask)


def triton_layernorm_forward(x, weight, bias, eps=1e-5):
    """x: (M, N) CUDA tensor. weight, bias: (N,)."""
    assert x.ndim == 2
    M, N = x.shape
    y = torch.empty_like(x)
    mean = torch.empty(M, device=x.device, dtype=torch.float32)
    rstd = torch.empty(M, device=x.device, dtype=torch.float32)

    BLOCK_N = triton.next_power_of_2(N)
    grid = (M,)

    layernorm_fwd_kernel[grid](
        x, y, weight, bias, mean, rstd,
        x.stride(0), N, eps,
        BLOCK_N=BLOCK_N,
    )
    return y, mean, rstd


def triton_layernorm_backward(dy, x, weight, mean, rstd):
    """Backward pass. Returns dx, dw, db."""
    M, N = x.shape
    dx = torch.empty_like(x)
    dw_partial = torch.empty((M, N), device=x.device, dtype=torch.float32)
    db_partial = torch.empty((M, N), device=x.device, dtype=torch.float32)

    BLOCK_N = triton.next_power_of_2(N)
    grid = (M,)

    layernorm_bwd_kernel[grid](
        dy, x, weight, mean, rstd,
        dx, dw_partial, db_partial,
        x.stride(0), N,
        BLOCK_N=BLOCK_N,
    )
    dw = dw_partial.sum(dim=0)
    db = db_partial.sum(dim=0)
    return dx, dw, db
