"""Triton fused flash attention kernel with online softmax & optional causal mask."""
import torch
import triton
import triton.language as tl
import math


@triton.jit
def flash_attn_fwd_kernel(
    Q, K, V, O, L,
    stride_qz, stride_qm, stride_qk,
    stride_kz, stride_kn, stride_kk,
    stride_vz, stride_vn, stride_vk,
    stride_oz, stride_om, stride_ok,
    N_CTX,
    sm_scale,
    IS_CAUSAL: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, HEAD_DIM: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_z = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, HEAD_DIM)

    q_ptrs = Q + pid_z * stride_qz + offs_m[:, None] * stride_qm + offs_k[None, :] * stride_qk
    k_ptrs = K + pid_z * stride_kz + offs_n[:, None] * stride_kn + offs_k[None, :] * stride_kk
    v_ptrs = V + pid_z * stride_vz + offs_n[:, None] * stride_vn + offs_k[None, :] * stride_vk

    q_mask = (offs_m[:, None] < N_CTX) & (offs_k[None, :] < N_CTX)
    q = tl.load(q_ptrs, mask=offs_m[:, None] < N_CTX, other=0.0).to(tl.float32)

    m_i = tl.full((BLOCK_M,), float('-inf'), dtype=tl.float32)
    l_i = tl.zeros((BLOCK_M,), dtype=tl.float32)
    acc = tl.zeros((BLOCK_M, HEAD_DIM), dtype=tl.float32)

    end_n = tl.minimum((pid_m + 1) * BLOCK_M, N_CTX) if IS_CAUSAL else N_CTX

    for start_n in range(0, end_n, BLOCK_N):
        cur_offs_n = start_n + offs_n
        k = tl.load(k_ptrs, mask=cur_offs_n[:, None] < N_CTX, other=0.0).to(tl.float32)
        v = tl.load(v_ptrs, mask=cur_offs_n[:, None] < N_CTX, other=0.0).to(tl.float32)

        qk = tl.dot(q, tl.trans(k)) * sm_scale

        if IS_CAUSAL:
            causal_mask = offs_m[:, None] >= cur_offs_n[None, :]
            qk = tl.where(causal_mask, qk, float('-inf'))

        qk = tl.where(
            (offs_m[:, None] < N_CTX) & (cur_offs_n[None, :] < N_CTX),
            qk, float('-inf')
        )

        m_ij = tl.max(qk, axis=1)
        m_new = tl.maximum(m_i, m_ij)
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(qk - m_new[:, None])

        l_i = alpha * l_i + tl.sum(p, axis=1)
        acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
        m_i = m_new

        k_ptrs += BLOCK_N * stride_kn
        v_ptrs += BLOCK_N * stride_vn

    acc = acc / l_i[:, None]

    o_ptrs = O + pid_z * stride_oz + offs_m[:, None] * stride_om + offs_k[None, :] * stride_ok
    tl.store(o_ptrs, acc.to(O.dtype.element_ty), mask=offs_m[:, None] < N_CTX)

    l_ptrs = L + pid_z * N_CTX + offs_m
    tl.store(l_ptrs, m_i + tl.log(l_i), mask=offs_m < N_CTX)


def triton_flash_attention(q, k, v, causal=False):
    """Flash attention forward. q,k,v: (B*H, N, D) contiguous CUDA tensors. D must be >= 16."""
    assert q.ndim == 3
    BH, N, D = q.shape
    assert D >= 16, f"Head dim must be >= 16 for Triton tl.dot, got {D}"

    o = torch.empty_like(q)
    L = torch.empty((BH, N), device=q.device, dtype=torch.float32)

    HEAD_DIM = triton.next_power_of_2(D)
    BLOCK_M = min(64, triton.next_power_of_2(N))
    BLOCK_N = min(64, triton.next_power_of_2(N))
    BLOCK_M = max(BLOCK_M, 16)
    BLOCK_N = max(BLOCK_N, 16)

    sm_scale = 1.0 / math.sqrt(D)
    grid = (triton.cdiv(N, BLOCK_M), BH)

    if D < HEAD_DIM:
        pad = HEAD_DIM - D
        q = torch.nn.functional.pad(q, (0, pad))
        k = torch.nn.functional.pad(k, (0, pad))
        v = torch.nn.functional.pad(v, (0, pad))
        o_padded = torch.empty_like(q)
    else:
        o_padded = o

    flash_attn_fwd_kernel[grid](
        q, k, v, o_padded, L,
        q.stride(0), q.stride(1), q.stride(2),
        k.stride(0), k.stride(1), k.stride(2),
        v.stride(0), v.stride(1), v.stride(2),
        o_padded.stride(0), o_padded.stride(1), o_padded.stride(2),
        N, sm_scale,
        IS_CAUSAL=causal,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, HEAD_DIM=HEAD_DIM,
    )

    if D < HEAD_DIM:
        o = o_padded[:, :, :D]

    return o.contiguous(), L
