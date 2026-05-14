"""Minimal data loading utilities."""
import numpy as np
from core.tensor import Tensor

try:
    import cupy as cp
except ImportError:
    cp = None


class DataLoader:
    def __init__(self, x, y, batch_size=32, shuffle=True, device='cpu'):
        self.x = x
        self.y = y
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.device = device

    def __iter__(self):
        n = len(self.x)
        idx = np.arange(n)
        if self.shuffle:
            np.random.shuffle(idx)
        for start in range(0, n, self.batch_size):
            end = min(start + self.batch_size, n)
            batch_idx = idx[start:end]
            bx = self.x[batch_idx]
            by = self.y[batch_idx]
            if self.device == 'cuda' and cp is not None:
                bx, by = cp.asarray(bx), cp.asarray(by)
            yield Tensor(bx), Tensor(by)

    def __len__(self):
        return (len(self.x) + self.batch_size - 1) // self.batch_size
