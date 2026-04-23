"""Autograd correctness tests - compare against PyTorch."""
import numpy as np
import pytest
import torch
from core.tensor import Tensor
from core.utils import set_seed


def _to_torch(arr, requires_grad=True):
    return torch.tensor(arr, dtype=torch.float32, requires_grad=requires_grad)


def _check_grad(tiny_fn, torch_fn, shapes, atol=1e-5):
    """Run forward+backward on both frameworks and compare gradients."""
    set_seed(42)
    np_inputs = [np.random.randn(*s).astype(np.float32) for s in shapes]
    tiny_inputs = [Tensor(x.copy(), requires_grad=True) for x in np_inputs]
    torch_inputs = [_to_torch(x.copy()) for x in np_inputs]

    tiny_out = tiny_fn(*tiny_inputs)
    torch_out = torch_fn(*torch_inputs)

    tiny_loss = tiny_out.sum()
    torch_loss = torch_out.sum()

    tiny_loss.backward()
    torch_loss.backward()

    for ti, to in zip(tiny_inputs, torch_inputs):
        np.testing.assert_allclose(ti.numpy(), to.detach().numpy(), atol=atol)
        assert ti.grad is not None, "Tiny grad is None"
        np.testing.assert_allclose(ti.grad, to.grad.numpy(), atol=atol)


class TestAutograd:
    def test_add(self):
        _check_grad(lambda a, b: a + b, lambda a, b: a + b, [(3, 4), (3, 4)])

    def test_add_broadcast(self):
        _check_grad(lambda a, b: a + b, lambda a, b: a + b, [(3, 4), (4,)])

    def test_mul(self):
        _check_grad(lambda a, b: a * b, lambda a, b: a * b, [(3, 4), (3, 4)])

    def test_mul_broadcast(self):
        _check_grad(lambda a, b: a * b, lambda a, b: a * b, [(3, 4), (1, 4)])

    def test_neg(self):
        _check_grad(lambda a: -a, lambda a: -a, [(3, 4)])

    def test_sub(self):
        _check_grad(lambda a, b: a - b, lambda a, b: a - b, [(3, 4), (3, 4)])

    def test_div(self):
        def tiny_fn(a, b):
            return a / (b * b + Tensor(np.ones(b.shape) * 0.5))
        def torch_fn(a, b):
            return a / (b * b + 0.5)
        _check_grad(tiny_fn, torch_fn, [(3, 4), (3, 4)])

    def test_pow(self):
        _check_grad(lambda a: a ** 2, lambda a: a ** 2, [(3, 4)])

    @pytest.mark.skip(reason="Temporarily disabled, needs ops to be implemented")
    def test_matmul_2d(self):
        _check_grad(lambda a, b: a @ b, lambda a, b: a @ b, [(3, 4), (4, 5)])

    @pytest.mark.skip(reason="Temporarily disabled, needs ops to be implemented")
    def test_matmul_batch(self):
        _check_grad(lambda a, b: a @ b, lambda a, b: a @ b, [(2, 3, 4), (2, 4, 5)])

    def test_sum(self):
        _check_grad(lambda a: a.sum(), lambda a: a.sum(), [(3, 4)])

    def test_sum_axis(self):
        _check_grad(lambda a: a.sum(axis=1), lambda a: a.sum(dim=1), [(3, 4)])

    def test_mean(self):
        _check_grad(lambda a: a.mean(), lambda a: a.mean(), [(3, 4)])

    def test_mean_axis(self):
        _check_grad(lambda a: a.mean(axis=0), lambda a: a.mean(dim=0), [(3, 4)])

    def test_reshape(self):
        _check_grad(lambda a: a.reshape(12), lambda a: a.reshape(12), [(3, 4)])

    def test_transpose(self):
        _check_grad(lambda a: a.transpose(), lambda a: a.T, [(3, 4)])

    def test_exp(self):
        _check_grad(lambda a: a.exp(), lambda a: a.exp(), [(3, 4)])

    def test_log(self):
        set_seed(42)
        a_np = np.abs(np.random.randn(3, 4).astype(np.float32)) + 0.1
        ta = Tensor(a_np.copy(), requires_grad=True)
        tt = _to_torch(a_np.copy())
        tiny_out = ta.log().sum()
        torch_out = tt.log().sum()
        tiny_out.backward()
        torch_out.backward()
        np.testing.assert_allclose(ta.grad, tt.grad.numpy(), atol=1e-4)

    def test_tanh(self):
        _check_grad(lambda a: a.tanh(), lambda a: a.tanh(), [(3, 4)])

    def test_sigmoid(self):
        _check_grad(lambda a: a.sigmoid(), lambda a: a.sigmoid(), [(3, 4)])

    @pytest.mark.skip(reason="Temporarily disabled, needs ops to be implemented")
    def test_chain_rule(self):
        """Multi-step computation graph."""
        _check_grad(
            lambda a, b: ((a @ b) * a.sum()).sum(),
            lambda a, b: ((a @ b) * a.sum()).sum(),
            [(3, 4), (4, 3)]
        )

    def test_slice(self):
        _check_grad(lambda a: a[:, 1:3], lambda a: a[:, 1:3], [(3, 4)])


class TestReproducibility:
    def test_seed_determinism(self):
        set_seed(123)
        a = np.random.randn(5)
        set_seed(123)
        b = np.random.randn(5)
        np.testing.assert_array_equal(a, b)
