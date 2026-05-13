"""Module forward/backward tests."""
import numpy as np
import pytest
import torch
import torch.nn as torch_nn
from core.tensor import Tensor
from core.nn import Linear, LayerNorm, Embedding, Dropout, Sequential, GELU, Module
from core.utils import set_seed


class TestLinear:
    def test_forward_shape(self):
        set_seed(42)
        m = Linear(4, 3)
        x = Tensor(np.random.randn(2, 4).astype(np.float32), requires_grad=True)
        out = m(x)
        assert out.shape == (2, 3)

    def test_backward_vs_torch(self):
        set_seed(42)
        x_np = np.random.randn(2, 4).astype(np.float32)
        w_np = np.random.randn(4, 3).astype(np.float32)
        b_np = np.zeros(3, dtype=np.float32)

        m = Linear(4, 3)
        m.weight.data = w_np.copy()
        m.bias.data = b_np.copy()
        x = Tensor(x_np.copy(), requires_grad=True)
        out = m(x).sum()
        out.backward()

        tw = torch.tensor(w_np.T.copy(), requires_grad=True)  # torch is (out, in)
        tb = torch.tensor(b_np.copy(), requires_grad=True)
        tx = torch.tensor(x_np.copy(), requires_grad=True)
        tout = (tx @ tw.T + tb).sum()
        # Our Linear: x @ W + b where W is (in, out)
        tout2 = (torch.tensor(x_np.copy(), requires_grad=True) @ torch.tensor(w_np.copy(), requires_grad=True) + torch.tensor(b_np.copy(), requires_grad=True)).sum()

        xt = torch.tensor(x_np.copy(), requires_grad=True)
        wt = torch.tensor(w_np.copy(), requires_grad=True)
        bt = torch.tensor(b_np.copy(), requires_grad=True)
        (xt @ wt + bt).sum().backward()

        np.testing.assert_allclose(x.grad, xt.grad.numpy(), atol=1e-5)
        np.testing.assert_allclose(m.weight.grad, wt.grad.numpy(), atol=1e-5)


class TestLayerNorm:
    def test_forward_values(self):
        set_seed(42)
        x_np = np.random.randn(2, 4).astype(np.float32)

        m = LayerNorm(4)
        out = m(Tensor(x_np.copy()))

        tm = torch_nn.LayerNorm(4, dtype=torch.float32)
        tm.weight.data.fill_(1.0)
        tm.bias.data.fill_(0.0)
        tout = tm(torch.tensor(x_np.copy()))

        np.testing.assert_allclose(out.numpy(), tout.detach().numpy(), atol=1e-5)

    def test_backward(self):
        set_seed(42)
        x_np = np.random.randn(2, 4).astype(np.float32)

        m = LayerNorm(4)
        x = Tensor(x_np.copy(), requires_grad=True)
        out = m(x).sum()
        out.backward()

        tm = torch_nn.LayerNorm(4, dtype=torch.float32)
        tm.weight.data.fill_(1.0)
        tm.bias.data.fill_(0.0)
        tx = torch.tensor(x_np.copy(), requires_grad=True)
        tm(tx).sum().backward()

        np.testing.assert_allclose(x.grad, tx.grad.numpy(), atol=1e-5)


class TestEmbedding:
    def test_forward_shape(self):
        set_seed(42)
        m = Embedding(10, 8)
        idx = np.array([1, 3, 5])
        out = m(idx)
        assert out.shape == (3, 8)

    def test_lookup_values(self):
        set_seed(42)
        m = Embedding(10, 4)
        idx = np.array([2])
        out = m(idx)
        np.testing.assert_allclose(out.numpy(), m.weight.data[2:3], atol=1e-6)


class TestDropout:
    def test_eval_passthrough(self):
        m = Dropout(0.5)
        m.eval()
        x = Tensor(np.ones((3, 4), dtype=np.float32))
        out = m(x)
        np.testing.assert_array_equal(out.numpy(), x.numpy())

    def test_train_scaling(self):
        set_seed(42)
        m = Dropout(0.5)
        m.train()
        x = Tensor(np.ones((1000,), dtype=np.float32), requires_grad=True)
        out = m(x)
        # Mean should be approximately 1.0 due to scaling
        assert abs(out.numpy().mean() - 1.0) < 0.15


class TestSequential:
    def test_forward(self):
        set_seed(42)
        m = Sequential(Linear(4, 3), GELU(), Linear(3, 2))
        x = Tensor(np.random.randn(2, 4).astype(np.float32), requires_grad=True)
        out = m(x)
        assert out.shape == (2, 2)


class TestModule:
    def test_parameters(self):
        m = Linear(4, 3)
        params = m.parameters()
        assert len(params) == 2  # weight + bias

    def test_zero_grad(self):
        set_seed(42)
        m = Linear(4, 3)
        x = Tensor(np.random.randn(2, 4).astype(np.float32), requires_grad=True)
        m(x).sum().backward()
        assert m.weight.grad is not None
        m.zero_grad()
        assert m.weight.grad is None


class TestReproducibility:
    def test_linear_seed(self):
        set_seed(42)
        m1 = Linear(4, 3)
        set_seed(42)
        m2 = Linear(4, 3)
        np.testing.assert_array_equal(m1.weight.data, m2.weight.data)
