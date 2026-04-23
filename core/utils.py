"""Device management, seeding, ... TBD"""
import numpy as np


try:
    import cupy as cp
except ImportError:
    cp = None


def set_seed(seed):
    np.random.seed(seed)
    if cp is not None:
        cp.random.seed(seed)


def param_count(module):
    return sum(p.data.size for p in module.parameters())


def get_device():
    return 'cuda' if cp is not None else 'cpu'
