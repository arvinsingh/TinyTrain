"""Triton fused activation kernels for ReLU, & GELU."""
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice
import torch


@triton.jit
def relu_kernel(X, Y, N: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * N + tl.arange(0, N)
    x = tl.load(X + offs)
    tl.store(Y + offs, tl.where(x > 0, x, 0.0))


@triton.jit
def relu_bwd_kernel(DY, X, DX, N: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * N + tl.arange(0, N)
    dy = tl.load(DY + offs)
    x = tl.load(X + offs)
    tl.store(DX + offs, tl.where(x > 0, dy, 0.0))


@triton.jit
def gelu_kernel(X, Y, N: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * N + tl.arange(0, N)
    x = tl.load(X + offs).to(tl.float32)
    k = 0.7978845608028654  # sqrt(2/pi)
    cdf = 0.5 * (1.0 + libdevice.tanh(k * (x + 0.044715 * x * x * x)))
    tl.store(Y + offs, x * cdf)


@triton.jit
def gelu_bwd_kernel(DY, X, DX, N: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * N + tl.arange(0, N)
    dy = tl.load(DY + offs).to(tl.float32)
    x = tl.load(X + offs).to(tl.float32)
    k = 0.7978845608028654
    inner = k * (x + 0.044715 * x * x * x)
    tanh_inner = libdevice.tanh(inner)
    cdf = 0.5 * (1.0 + tanh_inner)
    pdf = 0.5 * k * (1.0 + 0.134145 * x * x) * (1.0 - tanh_inner * tanh_inner)
    tl.store(DX + offs, dy * (cdf + x * pdf))


def triton_relu(x):
    M, N = x.shape
    y = torch.empty_like(x)
    BLOCK = triton.next_power_of_2(N)
    relu_kernel[(M,)](x, y, N=BLOCK)
    return y


def triton_relu_bwd(dy, x):
    M, N = x.shape
    dx = torch.empty_like(x)
    BLOCK = triton.next_power_of_2(N)
    relu_bwd_kernel[(M,)](dy, x, dx, N=BLOCK)
    return dx


def triton_gelu(x):
    M, N = x.shape
    y = torch.empty_like(x)
    BLOCK = triton.next_power_of_2(N)
    gelu_kernel[(M,)](x, y, N=BLOCK)
    return y


def triton_gelu_bwd(dy, x):
    M, N = x.shape
    dx = torch.empty_like(x)
    BLOCK = triton.next_power_of_2(N)
    gelu_bwd_kernel[(M,)](dy, x, dx, N=BLOCK)
    return dx
