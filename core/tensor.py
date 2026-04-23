"""Tensor with autograd support, backed by CuPy / NumPy arrays."""
import numpy as np

try:
    import cupy as cp
except ImportError:
    cp = None


def _get_xp(data):
    if cp is not None and isinstance(data, cp.ndarray):
        return cp
    return np


def _ensure_array(data, xp=np):
    if isinstance(data, Tensor):
        return data.data
    if isinstance(data, (int, float)):
        return xp.array(data, dtype=xp.float32)
    if isinstance(data, np.ndarray):
        return data
    if cp is not None and isinstance(data, cp.ndarray):
        return data
    return xp.array(data, dtype=xp.float32)


class Context:
    """Saved tensors/info for backward pass."""
    def __init__(self):
        self.saved = []

    def save_for_backward(self, *tensors):
        self.saved = list(tensors)

    @property
    def saved_tensors(self):
        return self.saved


class Tensor:
    def __init__(self, data, requires_grad=False, _children=(), _op=None):
        if isinstance(data, Tensor):
            data = data.data
        if isinstance(data, (int, float, list)):
            data = np.array(data, dtype=np.float32)
        self.data = data
        self.grad = None
        self.requires_grad = requires_grad
        self._ctx = None
        self._op = _op
        self._children = tuple(_children)

    @property
    def shape(self):
        return self.data.shape

    @property
    def dtype(self):
        return self.data.dtype

    @property
    def xp(self):
        return _get_xp(self.data)

    def numpy(self):
        d = self.data
        if cp is not None and isinstance(d, cp.ndarray):
            d = d.get()
        return np.array(d)

    def cuda(self):
        if cp is None:
            raise RuntimeError("CuPy not available")
        return Tensor(cp.asarray(self.data), requires_grad=self.requires_grad)

    def cpu(self):
        return Tensor(np.array(self.numpy()), requires_grad=self.requires_grad)

    def detach(self):
        return Tensor(self.data, requires_grad=False)

    def __repr__(self):
        return f"Tensor({self.data}, requires_grad={self.requires_grad})"

    def backward(self, grad=None):
        if grad is None:
            assert self.data.size == 1, "backward requires grad for non-scalar"
            grad = self.xp.ones_like(self.data)

        self.grad = grad if self.grad is None else self.grad + grad

        topo = []
        visited = set()

        def build_topo(t):
            if id(t) not in visited:
                visited.add(id(t))
                for child in t._children:
                    build_topo(child)
                topo.append(t)

        build_topo(self)

        for t in reversed(topo):
            if t._op is not None and t._ctx is not None and t.grad is not None:
                grads = t._op.backward(t._ctx, t.grad)
                if not isinstance(grads, tuple):
                    grads = (grads,)
                for child, g in zip(t._children, grads):
                    if g is None:
                        continue
                    if child.requires_grad:
                        # handle broadcasting reduce grad to match child shape
                        g = _unbroadcast(g, child.data.shape, _get_xp(g))
                        child.grad = g if child.grad is None else child.grad + g

    def zero_grad(self):
        self.grad = None

    def __len__(self):
        return self.data.shape[0]

    # operator overloads and methods (sum, mean, exp, etc.) are wired
    # up by core.ops at import time to avoid circular imports.


class no_grad:
    """Context manager to disable gradient tracking."""
    _enabled = False # global, not thread-safe. to-do - make thread-local

    def __enter__(self):
        self._prev = no_grad._enabled
        no_grad._enabled = True
        return self

    def __exit__(self, *args):
        no_grad._enabled = self._prev


def _wrap(val, xp):
    if isinstance(val, Tensor):
        return val
    return Tensor(_ensure_array(val, xp))


def _unbroadcast(grad, target_shape, xp):
    """Reduce grad to target_shape by summing over broadcast dims."""
    if grad.shape == target_shape:
        return grad
    # pad target shape with leading 1s
    ndim_diff = len(grad.shape) - len(target_shape)
    padded = (1,) * ndim_diff + target_shape
    # sum over dims that were broadcast
    axes = []
    for i, (gs, ts) in enumerate(zip(grad.shape, padded)):
        if ts == 1 and gs != 1:
            axes.append(i)
    if ndim_diff > 0:
        axes = list(range(ndim_diff)) + [a for a in axes if a >= ndim_diff]
    axes = tuple(sorted(set(axes)))
    if axes:
        grad = grad.sum(axis=axes, keepdims=True)
    if grad.shape != target_shape:
        grad = grad.reshape(target_shape)
    return grad
