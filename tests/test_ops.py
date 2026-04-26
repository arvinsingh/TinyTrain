"""Per-op numerical gradient tests via finite differences."""
import numpy as np
import pytest
from core.tensor import Tensor
from core.utils import set_seed
from core import ops


def numerical_grad(fn, inputs, idx, eps=1e-5):
    """Compute numerical gradient for input at index idx using float64."""
    # float64 to avoid float32 cancellation errors
    orig_data = [inp.data.copy() for inp in inputs]
    for inp in inputs:
        inp.data = inp.data.astype(np.float64)
    grad = np.zeros_like(inputs[idx].data)
    it = np.nditer(grad, flags=['multi_index'])
    while not it.finished:
        mi = it.multi_index
        old = inputs[idx].data[mi].copy()
        inputs[idx].data[mi] = old + eps
        fp = float(fn(*inputs).data.sum())
        inputs[idx].data[mi] = old - eps
        fm = float(fn(*inputs).data.sum())
        inputs[idx].data[mi] = old
        grad[mi] = (fp - fm) / (2 * eps)
        it.iternext()
    for inp, d in zip(inputs, orig_data):
        inp.data = d
    return grad.astype(np.float32)


class TestOpsNumerical:
    def _check(self, fn, shapes, atol=1e-4):
        set_seed(42)
        inputs = [Tensor(np.random.randn(*s).astype(np.float32), requires_grad=True) for s in shapes]
        out = fn(*inputs)
        out.sum().backward()
        for i, inp in enumerate(inputs):
            # fresh inputs for numerical grad (no graph)
            fresh = [Tensor(inp.numpy().copy(), requires_grad=True) for inp in inputs]
            ng = numerical_grad(fn, fresh, i)
            np.testing.assert_allclose(inputs[i].grad, ng, atol=atol,
                                       err_msg=f"Grad mismatch for input {i}")

    def test_add(self):
        self._check(lambda a, b: a + b, [(3, 4), (3, 4)])

    def test_mul(self):
        self._check(lambda a, b: a * b, [(3, 4), (3, 4)])

    def test_matmul(self):
        self._check(lambda a, b: a @ b, [(3, 4), (4, 5)])

    def test_pow(self):
        self._check(lambda a: a ** 3, [(3, 4)])

    def test_div(self):
        set_seed(42)
        a = Tensor((np.random.randn(3, 4)).astype(np.float32), requires_grad=True)
        b = Tensor((np.abs(np.random.randn(3, 4)) + 0.5).astype(np.float32), requires_grad=True)
        fn = lambda a, b: a / b
        out = fn(a, b).sum()
        out.backward()
        for i, inp in enumerate([a, b]):
            fresh = [Tensor(a.numpy().copy(), requires_grad=True),
                     Tensor(b.numpy().copy(), requires_grad=True)]
            ng = numerical_grad(fn, fresh, i)
            np.testing.assert_allclose(inp.grad, ng, atol=1e-3)

    def test_relu(self):
        self._check(lambda a: ops.ReLU.apply(a), [(3, 4)])

    def test_gelu(self):
        self._check(lambda a: ops.GELU.apply(a), [(3, 4)])

    def test_tanh(self):
        self._check(lambda a: ops.Tanh.apply(a), [(3, 4)])

    def test_sigmoid(self):
        self._check(lambda a: ops.Sigmoid.apply(a), [(3, 4)])

    def test_sum_axis(self):
        self._check(lambda a: a.sum(axis=1), [(3, 4)])

    def test_mean_axis(self):
        self._check(lambda a: a.mean(axis=0), [(3, 4)])

    def test_max(self):
        self._check(lambda a: a.max(axis=1), [(3, 4)], atol=1e-3)

    def test_cat(self):
        set_seed(42)
        a = Tensor(np.random.randn(3, 4).astype(np.float32), requires_grad=True)
        b = Tensor(np.random.randn(2, 4).astype(np.float32), requires_grad=True)
        out = ops.cat([a, b], axis=0).sum()
        out.backward()
        assert a.grad.shape == (3, 4)
        assert b.grad.shape == (2, 4)
        np.testing.assert_allclose(a.grad, np.ones((3, 4)), atol=1e-6)
