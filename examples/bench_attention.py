"""Benchmark CPU scaled dot-product attention vs GPU Triton flash attention."""

import time
import numpy as np
from pathlib import Path
import sys
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.tensor import Tensor
from core import functional, utils
from core.kernels.bridge import cupy_to_torch, torch_to_cupy
from core.kernels.attention import triton_flash_attention

try:
    import cupy as cp
    HAS_CUPY = True
except ImportError:
    HAS_CUPY = False


def bench(fn, warmup=3, repeats=10):
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return np.array(times)


def make_qkv(B, H, N, D, device='cpu'):
    shape = (B, H, N, D)
    q = np.random.randn(*shape).astype(np.float32)
    k = np.random.randn(*shape).astype(np.float32)
    v = np.random.randn(*shape).astype(np.float32)
    if device == 'cuda':
        q, k, v = cp.asarray(q), cp.asarray(k), cp.asarray(v)
    return Tensor(q, requires_grad=True), Tensor(k, requires_grad=True), Tensor(v, requires_grad=True)


def run_sdpa_cpu(q, k, v):
    out = functional.scaled_dot_product_attention(q, k, v)
    loss = out.sum()
    loss.backward()
    return out


def run_flash_gpu(q_cp, k_cp, v_cp):


    B, H, N, D = q_cp.shape
    q_t = cupy_to_torch(q_cp.data).reshape(B * H, N, D).contiguous()
    k_t = cupy_to_torch(k_cp.data).reshape(B * H, N, D).contiguous()
    v_t = cupy_to_torch(v_cp.data).reshape(B * H, N, D).contiguous()
    o, _ = triton_flash_attention(q_t, k_t, v_t, causal=False)
    torch.cuda.synchronize()
    return o


def fmt(times):
    return f"{times.mean()*1000:8.2f} ms  (std {times.std()*1000:.2f} ms)"


if __name__ == "__main__":
    utils.set_seed(0)

    configs = [
        # (B, H, N, D)
        (1, 4,   64,  64),
        (1, 4,  128,  64),
        (1, 8,  256,  64),
        (2, 8,  512,  64),
        (2, 8, 1024,  64),
    ]

    print("=" * 78)
    print(f"{'Config':>28s} | {'CPU SDPA':>24s} | {'GPU Flash':>24s} | Speedup")
    print("=" * 78)

    for B, H, N, D in configs:
        tag = f"B={B} H={H} N={N} D={D}"

        q, k, v = make_qkv(B, H, N, D, device='cpu')
        cpu_times = bench(lambda: run_sdpa_cpu(q, k, v), warmup=2, repeats=5)

        if HAS_CUPY:
            q_g, k_g, v_g = make_qkv(B, H, N, D, device='cuda')
            gpu_times = bench(lambda: run_flash_gpu(q_g, k_g, v_g), warmup=3, repeats=10)
            speedup = cpu_times.mean() / gpu_times.mean()
            print(f"{tag:>28s} | {fmt(cpu_times)} | {fmt(gpu_times)} | {speedup:5.1f}x")
        else:
            print(f"{tag:>28s} | {fmt(cpu_times)} | {'N/A (no CuPy)':>24s} |   -")

    if HAS_CUPY:
        print("\n--- Correctness check (B=1, H=2, N=32, D=32) ---")
        np.random.seed(123)
        q_np = np.random.randn(1, 2, 32, 32).astype(np.float32)
        k_np = np.random.randn(1, 2, 32, 32).astype(np.float32)
        v_np = np.random.randn(1, 2, 32, 32).astype(np.float32)

        q_c = Tensor(q_np.copy(), requires_grad=True)
        k_c = Tensor(k_np.copy(), requires_grad=True)
        v_c = Tensor(v_np.copy(), requires_grad=True)
        out_cpu = functional.scaled_dot_product_attention(q_c, k_c, v_c).numpy()

        B, H, N, D = 1, 2, 32, 32
        q_t = torch.tensor(q_np, device='cuda').reshape(B * H, N, D).contiguous()
        k_t = torch.tensor(k_np, device='cuda').reshape(B * H, N, D).contiguous()
        v_t = torch.tensor(v_np, device='cuda').reshape(B * H, N, D).contiguous()
        o_gpu, _ = triton_flash_attention(q_t, k_t, v_t, causal=False)
        out_gpu = o_gpu.reshape(B, H, N, D).cpu().numpy()

        max_err = np.max(np.abs(out_cpu - out_gpu))
        mean_err = np.mean(np.abs(out_cpu - out_gpu))
        print(f"  Max absolute error:  {max_err:.6e}")
        print(f"  Mean absolute error: {mean_err:.6e}")
        print(f"  [{'PASS' if max_err < 5e-3 else 'FAIL'}] Flash attention matches SDPA")

    print()
