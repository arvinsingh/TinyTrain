"""e2e integration tests"""
import numpy as np
import pytest
import os
import tempfile
from core.tensor import Tensor
from core.nn import Linear, Sequential, GELU, LayerNorm, MultiHeadAttention, TransformerBlock, FeedForward
from core.functional import cross_entropy, softmax, mse_loss
from core.optim import Adam, SGD
from core.data import DataLoader
from core.utils import set_seed, save, load, clip_grad_norm, clip_grad_value, param_count
from core.utils import StepLR, CosineAnnealingLR, LinearWarmupCosineDecay


class TestMLPTraining:
    def test_xor(self):
        """Train a small MLP to solve XOR - loss must decrease."""
        set_seed(42)
        X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
        Y = np.array([0, 1, 1, 0])

        model = Sequential(Linear(2, 16), GELU(), Linear(16, 2))
        opt = Adam(model.parameters(), lr=0.01)

        losses = []
        for epoch in range(200):
            opt.zero_grad()
            out = model(Tensor(X))
            loss = cross_entropy(out, Y)
            loss.backward()
            opt.step()
            losses.append(float(loss.numpy()))

        assert losses[-1] < losses[0] * 0.1, f"Loss didn't decrease enough: {losses[0]} -> {losses[-1]}"

        # accuracy
        preds = model(Tensor(X))
        pred_classes = preds.numpy().argmax(axis=1)
        assert np.array_equal(pred_classes, Y), f"XOR not solved: {pred_classes}"


class TestTransformerBlock:
    def test_loss_decreases(self):
        """Single transformer-like block on random data - loss must decrease."""
        set_seed(42)
        B, T, D = 4, 8, 16
        V = 32

        # embed -> linear -> layernorm -> linear -> logits
        embed = Linear(D, D)
        ln = LayerNorm(D)
        head = Linear(D, V)

        params = embed.parameters() + ln.parameters() + head.parameters()
        opt = Adam(params, lr=0.001)

        X = np.random.randn(B, T, D).astype(np.float32)
        Y = np.random.randint(0, V, (B * T,))

        losses = []
        for _ in range(50):
            opt.zero_grad()
            h = embed(Tensor(X))
            h = ln(h)
            logits = head(h).reshape(B * T, V)
            loss = cross_entropy(logits, Y)
            loss.backward()
            opt.step()
            losses.append(float(loss.numpy()))

        assert losses[-1] < losses[0], f"Loss didn't decrease: {losses[0]} -> {losses[-1]}"


class TestGradientFlow:
    def test_deep_network_no_nan(self):
        """Gradients through a 5-layer network should not be NaN/Inf."""
        set_seed(42)
        layers = []
        for _ in range(5):
            layers.extend([Linear(16, 16), GELU()])
        layers.append(Linear(16, 2))
        model = Sequential(*layers)

        x = Tensor(np.random.randn(4, 16).astype(np.float32), requires_grad=True)
        out = model(x)
        loss = cross_entropy(out, np.array([0, 1, 0, 1]))
        loss.backward()

        for p in model.parameters():
            assert p.grad is not None
            assert np.all(np.isfinite(p.grad)), "NaN/Inf in gradients"


class TestMSELoss:
    def test_forward_value(self):
        pred = Tensor(np.array([1.0, 2.0, 3.0], dtype=np.float32))
        target = Tensor(np.array([1.5, 2.5, 3.5], dtype=np.float32))
        loss = mse_loss(pred, target)
        assert abs(float(loss.numpy()) - 0.25) < 1e-6

    def test_backward(self):
        set_seed(42)
        pred = Tensor(np.random.randn(4, 3).astype(np.float32), requires_grad=True)
        target = Tensor(np.random.randn(4, 3).astype(np.float32))
        loss = mse_loss(pred, target)
        loss.backward()
        assert pred.grad is not None
        assert pred.grad.shape == pred.shape


class TestSaveLoad:
    def test_linear_roundtrip(self):
        set_seed(42)
        model = Sequential(Linear(4, 8), GELU(), Linear(8, 2))
        x = Tensor(np.random.randn(2, 4).astype(np.float32))
        out1 = model(x).numpy()

        with tempfile.NamedTemporaryFile(suffix='.npz', delete=False) as f:
            path = f.name
        try:
            save(model, path)
            model2 = Sequential(Linear(4, 8), GELU(), Linear(8, 2))
            load(model2, path)
            out2 = model2(x).numpy()
            assert np.allclose(out1, out2, atol=1e-6)
        finally:
            os.remove(path)

    def test_layernorm_roundtrip(self):
        set_seed(42)
        model = LayerNorm(16)
        x = Tensor(np.random.randn(4, 16).astype(np.float32))
        out1 = model(x).numpy()
        with tempfile.NamedTemporaryFile(suffix='.npz', delete=False) as f:
            path = f.name
        try:
            save(model, path)
            model2 = LayerNorm(16)
            load(model2, path)
            assert np.allclose(out1, model2(x).numpy(), atol=1e-6)
        finally:
            os.remove(path)


class TestGradClipping:
    def test_clip_grad_norm(self):
        set_seed(42)
        model = Linear(4, 2)
        x = Tensor(np.random.randn(3, 4).astype(np.float32), requires_grad=True)
        loss = model(x).sum()
        loss.backward()
        norm = clip_grad_norm(model.parameters(), max_norm=0.1)
        assert norm > 0
        clipped_norm_sq = sum(float(np.sum(p.grad ** 2)) for p in model.parameters() if p.grad is not None)
        assert clipped_norm_sq ** 0.5 <= 0.1 + 1e-6

    def test_clip_grad_value(self):
        set_seed(42)
        model = Linear(4, 2)
        x = Tensor(np.random.randn(3, 4).astype(np.float32) * 10, requires_grad=True)
        loss = model(x).sum()
        loss.backward()
        clip_grad_value(model.parameters(), clip_value=0.5)
        for p in model.parameters():
            if p.grad is not None:
                assert np.all(p.grad <= 0.5 + 1e-8)
                assert np.all(p.grad >= -0.5 - 1e-8)


class TestLRSchedulers:
    def test_step_lr(self):
        model = Linear(4, 2)
        opt = SGD(model.parameters(), lr=0.1)
        sched = StepLR(opt, step_size=5, gamma=0.5)
        for _ in range(5):
            sched.step()
        assert abs(opt.lr - 0.05) < 1e-8

    def test_cosine_annealing(self):
        model = Linear(4, 2)
        opt = Adam(model.parameters(), lr=0.01)
        sched = CosineAnnealingLR(opt, T_max=100, eta_min=0.0)
        lrs = []
        for _ in range(100):
            sched.step()
            lrs.append(opt.lr)
        assert lrs[0] > lrs[-1]
        assert lrs[-1] < 1e-4

    def test_warmup_cosine(self):
        model = Linear(4, 2)
        opt = Adam(model.parameters(), lr=0.01)
        sched = LinearWarmupCosineDecay(opt, warmup_steps=10, total_steps=100)
        lrs = []
        for _ in range(100):
            sched.step()
            lrs.append(opt.lr)
        assert lrs[4] < lrs[9] # warmup phase - inc
        assert lrs[50] > lrs[90] # Decay phase - dec


class TestMultiHeadAttention:
    def test_forward_shape(self):
        set_seed(42)
        mha = MultiHeadAttention(d_model=32, n_heads=4)
        x = Tensor(np.random.randn(2, 8, 32).astype(np.float32))
        out = mha(x)
        assert out.shape == (2, 8, 32)

    def test_backward(self):
        set_seed(42)
        mha = MultiHeadAttention(d_model=16, n_heads=2)
        x = Tensor(np.random.randn(1, 4, 16).astype(np.float32), requires_grad=True)
        out = mha(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None
        for p in mha.parameters():
            assert p.grad is not None


class TestTransformerBlockModule:
    def test_forward_shape(self):
        set_seed(42)
        block = TransformerBlock(d_model=32, n_heads=4)
        x = Tensor(np.random.randn(2, 8, 32).astype(np.float32))
        out = block(x)
        assert out.shape == (2, 8, 32)

    def test_training_loss_decreases(self):
        set_seed(42)
        block = TransformerBlock(d_model=16, n_heads=2)
        head = Linear(16, 8)
        params = block.parameters() + head.parameters()
        opt = Adam(params, lr=1e-3)
        X = np.random.randn(2, 4, 16).astype(np.float32)
        Y = np.random.randint(0, 8, (2 * 4,))
        losses = []
        for _ in range(30):
            opt.zero_grad()
            h = block(Tensor(X))
            logits = head(h).reshape(8, 8)
            loss = cross_entropy(logits, Y)
            loss.backward()
            opt.step()
            losses.append(float(loss.numpy()))
        assert losses[-1] < losses[0]

    def test_param_count(self):
        block = TransformerBlock(d_model=32, n_heads=4)
        count = param_count(block)
        assert count > 0


class TestDataLoader:
    def test_yields_tensors(self):
        X = np.random.randn(10, 4).astype(np.float32)
        y = np.random.randn(10, 1).astype(np.float32)
        loader = DataLoader(X, y, batch_size=5, shuffle=False)
        for bx, by in loader:
            assert isinstance(bx, Tensor)
            assert isinstance(by, Tensor)
            assert bx.shape[0] == 5
            break

    def test_regression_training(self):
        set_seed(42)
        X = np.random.randn(100, 3).astype(np.float32)
        w_true = np.array([[1.0], [2.0], [3.0]], dtype=np.float32)
        y = X @ w_true
        loader = DataLoader(X, y, batch_size=20, shuffle=True)
        model = Linear(3, 1)
        opt = Adam(model.parameters(), lr=0.01)
        for epoch in range(200):
            for bx, by in loader:
                pred = model(bx)
                loss = mse_loss(pred, by)
                model.zero_grad()
                loss.backward()
                opt.step()
        test_pred = model(Tensor(X)).numpy()
        mse = float(np.mean((test_pred - y) ** 2))
        assert mse < 0.1
