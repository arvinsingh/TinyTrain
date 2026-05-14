"""Functional ops tests."""
import numpy as np
import pytest
import torch
import torch.nn.functional as F
from core.tensor import Tensor
from core.functional import softmax, log_softmax, cross_entropy
from core.utils import set_seed


class TestSoftmax:
    def test_values(self):
        set_seed(42)
        x_np = np.random.randn(3, 5).astype(np.float32)
        out = softmax(Tensor(x_np.copy()))
        tout = F.softmax(torch.tensor(x_np.copy()), dim=-1)
        np.testing.assert_allclose(out.numpy(), tout.numpy(), atol=1e-6)

    def test_backward(self):
        set_seed(42)
        x_np = np.random.randn(3, 5).astype(np.float32)
        x = Tensor(x_np.copy(), requires_grad=True)
        softmax(x).sum().backward()
        tx = torch.tensor(x_np.copy(), requires_grad=True)
        F.softmax(tx, dim=-1).sum().backward()
        np.testing.assert_allclose(x.grad, tx.grad.numpy(), atol=1e-5)

    def test_numerical_stability(self):
        x = Tensor(np.array([1000.0, 1001.0, 1002.0], dtype=np.float32))
        out = softmax(x)
        assert np.all(np.isfinite(out.numpy()))
        np.testing.assert_allclose(out.numpy().sum(), 1.0, atol=1e-5)


class TestLogSoftmax:
    def test_values(self):
        set_seed(42)
        x_np = np.random.randn(3, 5).astype(np.float32)
        out = log_softmax(Tensor(x_np.copy()))
        tout = F.log_softmax(torch.tensor(x_np.copy()), dim=-1)
        np.testing.assert_allclose(out.numpy(), tout.numpy(), atol=1e-5)


class TestCrossEntropy:
    def test_values(self):
        set_seed(42)
        x_np = np.random.randn(4, 5).astype(np.float32)
        y_np = np.array([0, 2, 1, 4])
        out = cross_entropy(Tensor(x_np.copy()), y_np)
        tout = F.cross_entropy(torch.tensor(x_np.copy()), torch.tensor(y_np))
        np.testing.assert_allclose(out.numpy(), tout.numpy(), atol=1e-5)

    def test_backward(self):
        set_seed(42)
        x_np = np.random.randn(4, 5).astype(np.float32)
        y_np = np.array([0, 2, 1, 4])
        x = Tensor(x_np.copy(), requires_grad=True)
        loss = cross_entropy(x, y_np)
        loss.backward()
        tx = torch.tensor(x_np.copy(), requires_grad=True)
        F.cross_entropy(tx, torch.tensor(y_np)).backward()
        np.testing.assert_allclose(x.grad, tx.grad.numpy(), atol=1e-5)

    def test_numerical_stability_large_logits(self):
        x = Tensor(np.array([[1000.0, 1001.0, 1002.0]], dtype=np.float32), requires_grad=True)
        y = np.array([2])
        loss = cross_entropy(x, y)
        assert np.isfinite(loss.numpy())
        loss.backward()
        assert np.all(np.isfinite(x.grad))
