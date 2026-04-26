"""Bridge between CuPy arrays and PyTorch CUDA tensors for Triton kernel dispatch."""
try:
    import torch
except ImportError:
    torch = None

try:
    import cupy as cp
except ImportError:
    cp = None

from core.kernels import triton_available


def cupy_to_torch(arr):
    return torch.as_tensor(arr, device='cuda')


def torch_to_cupy(t):
    return cp.from_dlpack(t)


def can_use_triton(data):
    if not triton_available():
        return False
    if cp is None:
        return False
    return isinstance(data, cp.ndarray)
