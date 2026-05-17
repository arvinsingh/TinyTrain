"""stress test - numerical gradient checks, randomized shapes, broadcasting"""
import numpy as np
import pytest
import torch
from core.tensor import Tensor, no_grad
from core.utils import set_seed
from core import ops



def numerical_grad(fn, inputs, idx, eps=1e-5):
    """Finite-difference gradient for input[idx] f64."""
    orig = [inp.data.copy() for inp in inputs]
    for inp in inputs:
        inp.data = inp.data.astype(np.float64)
    grad = np.zeros_like(inputs[idx].data)
    it = np.nditer(grad, flags=["multi_index"])
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
    for inp, d in zip(inputs, orig):
        inp.data = d
    return grad.astype(np.float32)


def _check_numgrad(fn, shapes, atol=1e-3, positive=False, seed=42):
    """Compare analytic grads against numerical grads for every i/p."""
    set_seed(seed)
    if positive:
        inputs = [Tensor(np.abs(np.random.randn(*s).astype(np.float32)) + 0.1,
                         requires_grad=True) for s in shapes]
    else:
        inputs = [Tensor(np.random.randn(*s).astype(np.float32),
                         requires_grad=True) for s in shapes]
    out = fn(*inputs)

    if out.data.size > 1:
        out = out.sum()
    out.backward()
    for i, inp in enumerate(inputs):
        fresh = [Tensor(x.numpy().copy(), requires_grad=True) for x in inputs]
        ng = numerical_grad(fn, fresh, i)
        np.testing.assert_allclose(
            inp.grad, ng, atol=atol,
            err_msg=f"Grad mismatch for input {i}, shape={inp.shape}",
        )


def _to_torch(arr, requires_grad=True):
    return torch.tensor(arr, dtype=torch.float32, requires_grad=requires_grad)


def _check_vs_torch(tiny_fn, torch_fn, shapes, atol=1e-4, positive=False, seed=42):
    """Fwd + bwd comparison against PyTorch."""
    set_seed(seed)
    if positive:
        np_inputs = [np.abs(np.random.randn(*s).astype(np.float32)) + 0.1 for s in shapes]
    else:
        np_inputs = [np.random.randn(*s).astype(np.float32) for s in shapes]
    tiny_inputs = [Tensor(x.copy(), requires_grad=True) for x in np_inputs]
    torch_inputs = [_to_torch(x.copy()) for x in np_inputs]

    tiny_out = tiny_fn(*tiny_inputs).sum()
    torch_out = torch_fn(*torch_inputs).sum()
    tiny_out.backward()
    torch_out.backward()

    for ti, to_ in zip(tiny_inputs, torch_inputs):
        assert ti.grad is not None
        np.testing.assert_allclose(
            ti.grad, to_.grad.numpy(), atol=atol,
            err_msg=f"Grad vs torch mismatch, shape={ti.shape}",
        )


# 1 Numerical gradient checks - wider op coverage

class TestNumericalGradients:
    """Finite-difference grad verification fordifferentiable op."""

    def test_exp(self):
        _check_numgrad(lambda a: a.exp(), [(4, 5)])

    def test_log(self):
        _check_numgrad(lambda a: a.log(), [(4, 5)], positive=True)

    def test_tanh(self):
        _check_numgrad(lambda a: a.tanh(), [(4, 5)])

    def test_sigmoid(self):
        _check_numgrad(lambda a: a.sigmoid(), [(4, 5)])

    def test_neg(self):
        _check_numgrad(lambda a: -a, [(3, 4)])

    def test_sub(self):
        _check_numgrad(lambda a, b: a - b, [(3, 4), (3, 4)])

    def test_div(self):
        _check_numgrad(lambda a, b: a / b, [(3, 4), (3, 4)], positive=True)

    def test_pow_int(self):
        _check_numgrad(lambda a: a ** 3, [(3, 4)])

    def test_pow_float(self):
        _check_numgrad(lambda a: a ** 2.5, [(3, 4)], positive=True)

    def test_reshape(self):
        _check_numgrad(lambda a: a.reshape(12), [(3, 4)])

    def test_transpose(self):
        _check_numgrad(lambda a: a.transpose(), [(3, 5)])

    def test_slice_rows(self):
        _check_numgrad(lambda a: a[1:3], [(5, 4)])

    def test_slice_cols(self):
        _check_numgrad(lambda a: a[:, 2:], [(4, 6)])

    def test_sum_global(self):
        _check_numgrad(lambda a: a.sum(), [(3, 4)], atol=2e-3)

    def test_sum_axis0(self):
        _check_numgrad(lambda a: a.sum(axis=0), [(3, 4)])

    def test_sum_axis1(self):
        _check_numgrad(lambda a: a.sum(axis=1), [(3, 4)])

    def test_mean_global(self):
        _check_numgrad(lambda a: a.mean(), [(3, 4)])

    def test_mean_axis0(self):
        _check_numgrad(lambda a: a.mean(axis=0), [(3, 4)])

    def test_mean_axis1(self):
        _check_numgrad(lambda a: a.mean(axis=1), [(3, 4)])

    def test_max_axis(self):
        _check_numgrad(lambda a: a.max(axis=1), [(3, 4)], atol=2e-3)

    def test_matmul(self):
        _check_numgrad(lambda a, b: a @ b, [(3, 4), (4, 5)])

    def test_cat_axis0(self):
        _check_numgrad(lambda a, b: ops.cat([a, b], axis=0), [(3, 4), (2, 4)])

    def test_cat_axis1(self):
        _check_numgrad(lambda a, b: ops.cat([a, b], axis=1), [(3, 2), (3, 5)])

    def test_relu(self):
        _check_numgrad(lambda a: ops.ReLU.apply(a), [(4, 5)], atol=2e-3)

    def test_gelu(self):
        _check_numgrad(lambda a: ops.GELU.apply(a), [(4, 5)])

    def test_chain_exp_log(self):
        _check_numgrad(lambda a: (a.exp() + 1).log(), [(3, 4)])

    def test_chain_matmul_tanh(self):
        _check_numgrad(lambda a, b: (a @ b).tanh(), [(3, 4), (4, 5)])

    def test_chain_div_sigmoid(self):
        _check_numgrad(lambda a, b: (a / b).sigmoid(), [(3, 4), (3, 4)], positive=True)


# 2 Randomized shape tests

class TestRandomizedShapes:
    SEEDS = range(10)

    @staticmethod
    def _rand_shape(rng, ndim_range=(1, 4), dim_range=(1, 8)):
        ndim = rng.integers(*ndim_range, endpoint=True)
        return tuple(int(d) for d in rng.integers(*dim_range, size=ndim, endpoint=True))

    @pytest.mark.parametrize("seed", SEEDS)
    def test_add_random(self, seed):
        rng = np.random.default_rng(seed)
        shape = self._rand_shape(rng)
        _check_numgrad(lambda a, b: a + b, [shape, shape], seed=seed)

    @pytest.mark.parametrize("seed", SEEDS)
    def test_mul_random(self, seed):
        rng = np.random.default_rng(seed)
        shape = self._rand_shape(rng)
        _check_numgrad(lambda a, b: a * b, [shape, shape], seed=seed)

    @pytest.mark.parametrize("seed", SEEDS)
    def test_sum_random_axis(self, seed):
        rng = np.random.default_rng(seed)
        shape = self._rand_shape(rng, ndim_range=(2, 4))
        axis = int(rng.integers(0, len(shape)))
        _check_numgrad(lambda a: a.sum(axis=axis), [shape], seed=seed)

    @pytest.mark.parametrize("seed", SEEDS)
    def test_mean_random_axis(self, seed):
        rng = np.random.default_rng(seed)
        shape = self._rand_shape(rng, ndim_range=(2, 4))
        axis = int(rng.integers(0, len(shape)))
        _check_numgrad(lambda a: a.mean(axis=axis), [shape], seed=seed)

    @pytest.mark.parametrize("seed", SEEDS)
    def test_matmul_random(self, seed):
        rng = np.random.default_rng(seed)
        m, k, n = (int(d) for d in rng.integers(1, 8, size=3, endpoint=True))
        _check_numgrad(lambda a, b: a @ b, [(m, k), (k, n)], seed=seed)

    @pytest.mark.parametrize("seed", SEEDS)
    def test_transpose_random(self, seed):
        rng = np.random.default_rng(seed)
        shape = self._rand_shape(rng, ndim_range=(2, 2))
        _check_numgrad(lambda a: a.transpose(), [shape], seed=seed)

    @pytest.mark.parametrize("seed", SEEDS)
    def test_reshape_random(self, seed):
        rng = np.random.default_rng(seed)
        shape = self._rand_shape(rng, ndim_range=(2, 3), dim_range=(2, 6))
        flat = int(np.prod(shape))
        _check_numgrad(lambda a: a.reshape(flat), [shape], seed=seed)

    @pytest.mark.parametrize("seed", SEEDS)
    def test_unary_chain_random(self, seed):
        rng = np.random.default_rng(seed)
        shape = self._rand_shape(rng)
        _check_numgrad(lambda a: a.tanh().exp(), [shape], seed=seed)

    @pytest.mark.parametrize("seed", SEEDS)
    def test_slice_random(self, seed):
        rng = np.random.default_rng(seed)
        shape = self._rand_shape(rng, ndim_range=(2, 3), dim_range=(3, 8))
        lo = int(rng.integers(0, shape[0] - 1))
        hi = int(rng.integers(lo + 1, shape[0] + 1))
        _check_numgrad(lambda a: a[lo:hi], [shape], seed=seed)

    @pytest.mark.parametrize("seed", SEEDS)
    def test_max_random_axis(self, seed):
        rng = np.random.default_rng(seed)
        shape = self._rand_shape(rng, ndim_range=(2, 3), dim_range=(2, 6))
        axis = int(rng.integers(0, len(shape)))
        _check_numgrad(lambda a: a.max(axis=axis), [shape], seed=seed, atol=2e-3)


# 3 Broadcasting stress tests

class TestBroadcastingStress:
    """Gradient correctness under diff broadcasting scenarios."""

    # add

    def test_add_scalar_tensor(self):
        _check_vs_torch(
            lambda a: a + Tensor(np.float32(2.0)),
            lambda a: a + 2.0,
            [(3, 4)],
        )

    def test_add_row_broadcast(self):
        _check_vs_torch(
            lambda a, b: a + b,
            lambda a, b: a + b,
            [(3, 4), (1, 4)],
        )

    def test_add_col_broadcast(self):
        _check_vs_torch(
            lambda a, b: a + b,
            lambda a, b: a + b,
            [(3, 4), (3, 1)],
        )

    def test_add_leading_dim(self):
        _check_vs_torch(
            lambda a, b: a + b,
            lambda a, b: a + b,
            [(2, 3, 4), (3, 4)],
        )

    def test_add_leading_and_trailing(self):
        _check_vs_torch(
            lambda a, b: a + b,
            lambda a, b: a + b,
            [(2, 3, 4), (4,)],
        )

    # mul

    def test_mul_row_broadcast(self):
        _check_vs_torch(
            lambda a, b: a * b,
            lambda a, b: a * b,
            [(3, 4), (1, 4)],
        )

    def test_mul_col_broadcast(self):
        _check_vs_torch(
            lambda a, b: a * b,
            lambda a, b: a * b,
            [(3, 4), (3, 1)],
        )

    def test_mul_3d_vs_1d(self):
        _check_vs_torch(
            lambda a, b: a * b,
            lambda a, b: a * b,
            [(2, 3, 4), (4,)],
        )

    def test_mul_3d_vs_2d(self):
        _check_vs_torch(
            lambda a, b: a * b,
            lambda a, b: a * b,
            [(2, 3, 4), (3, 4)],
        )

    # sub / div

    def test_sub_broadcast(self):
        _check_vs_torch(
            lambda a, b: a - b,
            lambda a, b: a - b,
            [(3, 4), (1, 4)],
        )

    def test_div_broadcast(self):
        _check_vs_torch(
            lambda a, b: a / b,
            lambda a, b: a / b,
            [(3, 4), (1, 4)],
            positive=True,
        )

    def test_div_scalar_broadcast(self):
        _check_vs_torch(
            lambda a: a / Tensor(np.float32(3.0)),
            lambda a: a / 3.0,
            [(3, 4)],
        )

    # compound expressions with broadcasting

    def test_compound_add_mul_broadcast(self):
        _check_vs_torch(
            lambda a, b: (a + b) * b,
            lambda a, b: (a + b) * b,
            [(3, 4), (1, 4)],
        )

    def test_compound_matmul_add_bias(self):
        """MatMul + bias broadcast"""
        _check_vs_torch(
            lambda x, w, b: x @ w + b,
            lambda x, w, b: x @ w + b,
            [(8, 4), (4, 6), (6,)],
        )

    def test_compound_normalize(self):
        """(x - mean) / std style normalization w/ broadcasting."""
        def tiny_fn(a):
            mu = a.mean(axis=1, keepdims=True)
            centered = a - mu
            var = (centered * centered).mean(axis=1, keepdims=True)
            return centered / (var + Tensor(np.float32(1e-5))).exp().log()
        def torch_fn(a):
            mu = a.mean(dim=1, keepdim=True)
            centered = a - mu
            var = (centered * centered).mean(dim=1, keepdim=True)
            return centered / (var + 1e-5).exp().log()
        _check_vs_torch(tiny_fn, torch_fn, [(4, 8)], atol=1e-3)

    # 4D, batched attn-like

    def test_4d_add_broadcast(self):
        _check_vs_torch(
            lambda a, b: a + b,
            lambda a, b: a + b,
            [(2, 3, 4, 5), (1, 1, 1, 5)],
        )

    def test_4d_mul_broadcast(self):
        _check_vs_torch(
            lambda a, b: a * b,
            lambda a, b: a * b,
            [(2, 3, 4, 5), (4, 5)],
        )

    # randomized broadcasting pairs

    @pytest.mark.parametrize("seed", range(10))
    def test_random_broadcast_add(self, seed):
        rng = np.random.default_rng(seed)
        ndim = int(rng.integers(2, 4, endpoint=True))
        shape_a = tuple(int(d) for d in rng.integers(2, 5, size=ndim, endpoint=True))
        # For each dim, either match or set to 1
        shape_b = tuple(
            s if rng.random() > 0.5 else 1 for s in shape_a
        )
        _check_vs_torch(
            lambda a, b: a + b,
            lambda a, b: a + b,
            [shape_a, shape_b],
            seed=seed,
        )

    @pytest.mark.parametrize("seed", range(10))
    def test_random_broadcast_mul(self, seed):
        rng = np.random.default_rng(seed)
        ndim = int(rng.integers(2, 4, endpoint=True))
        shape_a = tuple(int(d) for d in rng.integers(2, 5, size=ndim, endpoint=True))
        shape_b = tuple(
            s if rng.random() > 0.5 else 1 for s in shape_a
        )
        _check_vs_torch(
            lambda a, b: a * b,
            lambda a, b: a * b,
            [shape_a, shape_b],
            seed=seed,
        )

    @pytest.mark.parametrize("seed", range(10))
    def test_random_broadcast_sub(self, seed):
        rng = np.random.default_rng(seed)
        ndim = int(rng.integers(2, 4, endpoint=True))
        shape_a = tuple(int(d) for d in rng.integers(2, 5, size=ndim, endpoint=True))
        shape_b = tuple(
            s if rng.random() > 0.5 else 1 for s in shape_a
        )
        _check_vs_torch(
            lambda a, b: a - b,
            lambda a, b: a - b,
            [shape_a, shape_b],
            seed=seed,
        )

    @pytest.mark.parametrize("seed", range(5))
    def test_random_broadcast_leading_dims(self, seed):
        """b has fewer dims than a (leading-dim broadcast)."""
        rng = np.random.default_rng(seed + 100)
        shape_a = tuple(int(d) for d in rng.integers(2, 5, size=3, endpoint=True))
        # Drop 1 or 2 leading dims
        drop = int(rng.integers(1, len(shape_a)))
        shape_b = shape_a[drop:]
        _check_vs_torch(
            lambda a, b: a * b,
            lambda a, b: a * b,
            [shape_a, shape_b],
            seed=seed,
        )


# 4 no_grad context manager tests

class TestNoGrad:

    def test_no_graph_built(self):
        a = Tensor(np.ones((3, 4), dtype=np.float32), requires_grad=True)
        with no_grad():
            b = a * 2 + 1
        assert not b.requires_grad
        assert b._op is None
        assert b._children == ()

    def test_nesting(self):
        a = Tensor(np.ones((2,), dtype=np.float32), requires_grad=True)
        with no_grad():
            with no_grad():
                b = a + 1
            c = a * 2
        d = a * 3
        assert not b.requires_grad
        assert not c.requires_grad
        assert d.requires_grad

    def test_grad_still_works_after(self):
        a = Tensor(np.array([3.0], dtype=np.float32), requires_grad=True)
        with no_grad():
            _ = a ** 2
        loss = (a ** 2).sum()
        loss.backward()
        np.testing.assert_allclose(a.grad, np.array([6.0]), atol=1e-6)
