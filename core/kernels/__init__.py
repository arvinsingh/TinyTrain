"""Triton GPU kernel availability detection."""

HAS_TRITON = False
HAS_CUDA = False

try:
    import triton
    import torch
    HAS_TRITON = True
    HAS_CUDA = torch.cuda.is_available()
except ImportError:
    pass


def triton_available():
    return HAS_TRITON and HAS_CUDA
