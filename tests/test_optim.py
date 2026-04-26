"""Optimizer correctness tests."""
import numpy as np
import pytest
import torch
from core.tensor import Tensor
from core.optim import SGD, Adam
from core.utils import set_seed


class TestSGD:
    def test_step(self):
        set_seed(42)
        w_np = np.random.randn(4, 3).astype(np.float32)
        x_np = np.random.randn(2, 4).astype(np.float32)

        # ours
        w = Tensor(w_np.copy(), requires_grad=True)
        x = Tensor(x_np.copy())
        opt = SGD([w], lr=0.1)
        (x @ w).sum().backward()
        opt.step()

        # torch
        wt = torch.tensor(w_np.copy(), requires_grad=True)
        xt = torch.tensor(x_np.copy())
        opt_t = torch.optim.SGD([wt], lr=0.1)
        (xt @ wt).sum().backward()
        opt_t.step()

        np.testing.assert_allclose(w.data, wt.detach().numpy(), atol=1e-6)


class TestAdam:
    def test_step(self):
        set_seed(42)
        w_np = np.random.randn(4, 3).astype(np.float32)
        x_np = np.random.randn(2, 4).astype(np.float32)

        w = Tensor(w_np.copy(), requires_grad=True)
        x = Tensor(x_np.copy())
        opt = Adam([w], lr=0.001)
        (x @ w).sum().backward()
        opt.step()

        wt = torch.tensor(w_np.copy(), requires_grad=True)
        xt = torch.tensor(x_np.copy())
        opt_t = torch.optim.Adam([wt], lr=0.001)
        (xt @ wt).sum().backward()
        opt_t.step()

        np.testing.assert_allclose(w.data, wt.detach().numpy(), atol=1e-5)

    def test_multi_step(self):
        set_seed(42)
        w_np = np.random.randn(4, 3).astype(np.float32)
        x_np = np.random.randn(2, 4).astype(np.float32)

        w = Tensor(w_np.copy(), requires_grad=True)
        opt = Adam([w], lr=0.001)
        for _ in range(5):
            opt.zero_grad()
            (Tensor(x_np) @ w).sum().backward()
            opt.step()

        wt = torch.tensor(w_np.copy(), requires_grad=True)
        opt_t = torch.optim.Adam([wt], lr=0.001)
        for _ in range(5):
            opt_t.zero_grad()
            (torch.tensor(x_np) @ wt).sum().backward()
            opt_t.step()

        np.testing.assert_allclose(w.data, wt.detach().numpy(), atol=1e-4)
