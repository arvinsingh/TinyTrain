"""Neural network modules."""
import numpy as np
from core.tensor import Tensor, _get_xp, Context, no_grad
from core import ops as O
from core.functional import scaled_dot_product_attention
from core.kernels.bridge import can_use_triton, cupy_to_torch, torch_to_cupy
from core.kernels.layernorm import triton_layernorm_forward, triton_layernorm_backward

try:
    import cupy as cp
except ImportError:
    cp = None


class Module:
    def __init__(self):
        self._modules = {}
        self._params = {}
        self.training = True

    def __setattr__(self, name, value):
        if isinstance(value, Tensor) and value.requires_grad:
            self.__dict__.setdefault('_params', {})[name] = value
        elif isinstance(value, Module):
            self.__dict__.setdefault('_modules', {})[name] = value
        super().__setattr__(name, value)

    def parameters(self):
        params = list(self._params.values())
        for m in self._modules.values():
            params.extend(m.parameters())
        return params

    def zero_grad(self):
        for p in self.parameters():
            p.grad = None

    def train(self):
        self.training = True
        for m in self._modules.values():
            m.train()

    def eval(self):
        self.training = False
        for m in self._modules.values():
            m.eval()

    def cuda(self):
        if cp is None:
            raise RuntimeError("CuPy not available")
        for name, p in self._params.items():
            setattr(self, name, p.cuda())
        for name, m in self._modules.items():
            m.cuda()
        return self

    def cpu(self):
        for name, p in self._params.items():
            setattr(self, name, p.cpu())
        for name, m in self._modules.items():
            m.cpu()
        return self

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, *args, **kwargs):
        raise NotImplementedError


class Linear(Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        scale = np.sqrt(2.0 / in_features)
        self.weight = Tensor(
            np.random.randn(in_features, out_features).astype(np.float32) * scale,
            requires_grad=True,
        )
        self.bias = (
            Tensor(np.zeros(out_features, dtype=np.float32), requires_grad=True)
            if bias else None
        )

    def forward(self, x):
        out = x @ self.weight
        if self.bias is not None:
            out = out + self.bias
        return out


class Embedding(Module):
    def __init__(self, num_embeddings, embedding_dim):
        super().__init__()
        self.weight = Tensor(
            np.random.randn(num_embeddings, embedding_dim).astype(np.float32) * 0.02,
            requires_grad=True,
        )

    def forward(self, idx):
        return EmbeddingLookup.apply(self.weight, idx)


class EmbeddingLookup(O.Op):
    @staticmethod
    def forward(ctx, weight, idx):
        ctx.save_for_backward(weight)
        ctx.idx = idx
        return weight[idx]

    @staticmethod
    def backward(ctx, grad):
        weight, = ctx.saved_tensors
        xp = _get_xp(grad)
        grad_weight = xp.zeros_like(weight)
        np.add.at(grad_weight, ctx.idx, grad)
        return (grad_weight,)

class LayerNorm(Module):
    def __init__(self, normalized_shape, eps=1e-5):
        super().__init__()
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = normalized_shape
        self.eps = eps
        self.weight = Tensor(np.ones(normalized_shape, dtype=np.float32), requires_grad=True)
        self.bias = Tensor(np.zeros(normalized_shape, dtype=np.float32), requires_grad=True)

    def forward(self, x):
        return LayerNormOp.apply(x, self.weight, self.bias, self.eps)


class LayerNormOp:
    """
    Uses a custom apply() to implement a fast path with Triton 
    for 2D GPU tensors, and a fallback for CPU/NumPy.
    no_grad handled explicitly since some i/ps are not Tensors.
    """
    @classmethod
    def apply(cls, x, weight, bias, eps):
        ctx = Context()
        xp = _get_xp(x.data)
        axis = tuple(range(-len(weight.shape), 0))
        requires_grad = not no_grad._enabled and (
            x.requires_grad or weight.requires_grad or bias.requires_grad
        )

        # Triton fast path
        if x.data.ndim == 2 and len(weight.shape) == 1 and can_use_triton(x.data):
            x_t = cupy_to_torch(x.data)
            w_t = cupy_to_torch(weight.data)
            b_t = cupy_to_torch(bias.data)
            y_t, mean_t, rstd_t = triton_layernorm_forward(x_t, w_t, b_t, eps)
            result = torch_to_cupy(y_t)
            if requires_grad:
                mean = torch_to_cupy(mean_t).reshape(-1, 1)
                rstd = torch_to_cupy(rstd_t).reshape(-1, 1)
                ctx.save_for_backward(x.data, weight.data, mean, rstd)
                ctx.use_triton = True
                ctx.eps = eps
                ctx.axis = axis
        else:
            mean = x.data.mean(axis=axis, keepdims=True)
            var = x.data.var(axis=axis, keepdims=True)
            x_norm = (x.data - mean) / xp.sqrt(var + eps)
            result = x_norm * weight.data + bias.data
            if requires_grad:
                ctx.save_for_backward(x.data, weight.data, mean, var, x_norm)
                ctx.use_triton = False
                ctx.eps = eps
                ctx.axis = axis

        out = Tensor(
            result,
            requires_grad=requires_grad,
            _children=(x, weight, bias) if requires_grad else (),
            _op=cls if requires_grad else None,
        )
        if requires_grad:
            out._ctx = ctx
        return out

    @staticmethod
    def backward(ctx, grad):
        xp = _get_xp(grad)

        if ctx.use_triton:
            x, w, mean, rstd = ctx.saved_tensors
            dx_t, dw_t, db_t = triton_layernorm_backward(
                cupy_to_torch(grad),
                cupy_to_torch(x),
                cupy_to_torch(w),
                cupy_to_torch(mean.ravel()),
                cupy_to_torch(rstd.ravel()),
            )
            return torch_to_cupy(dx_t), torch_to_cupy(dw_t), torch_to_cupy(db_t)

        x, w, mean, var, x_norm = ctx.saved_tensors
        std_inv = 1.0 / xp.sqrt(var + ctx.eps)
        N = 1
        for a in ctx.axis:
            N *= x.shape[a]

        dx_norm = grad * w
        dvar = (dx_norm * (x - mean) * -0.5 * std_inv ** 3).sum(axis=ctx.axis, keepdims=True)
        dmean = (
            (dx_norm * -std_inv).sum(axis=ctx.axis, keepdims=True)
            + dvar * (-2.0 / N) * (x - mean).sum(axis=ctx.axis, keepdims=True)
        )
        dx = dx_norm * std_inv + dvar * 2.0 / N * (x - mean) + dmean / N

        reduce_axes = tuple(range(grad.ndim - len(ctx.axis)))
        dw = (grad * x_norm).sum(axis=reduce_axes) if reduce_axes else grad * x_norm
        db = grad.sum(axis=reduce_axes) if reduce_axes else grad.copy()

        return dx, dw, db


class Dropout(Module):
    def __init__(self, p=0.1):
        super().__init__()
        self.p = p

    def forward(self, x):
        if not self.training or self.p == 0:
            return x
        xp = x.xp
        mask = (xp.random.random(x.shape) > self.p).astype(x.dtype)
        return DropoutOp.apply(x, mask, self.p)


class DropoutOp:
    """
    With custom apply() bcuz mask and p are not Tensors
    and should not appear in _children or receive gradients.
    no_grad is handled explicitly.
    """
    @classmethod
    def apply(cls, x, mask, p):
        ctx = Context()
        requires_grad = not no_grad._enabled and x.requires_grad
        scale = 1.0 / (1.0 - p)
        result = x.data * mask * scale
        if requires_grad:
            ctx.save_for_backward(mask)
            ctx.scale = scale
        out = Tensor(
            result,
            requires_grad=requires_grad,
            _children=(x,) if requires_grad else (),
            _op=cls if requires_grad else None,
        )
        if requires_grad:
            out._ctx = ctx
        return out

    @staticmethod
    def backward(ctx, grad):
        mask, = ctx.saved_tensors
        return (grad * mask * ctx.scale,)


class Sequential(Module):
    def __init__(self, *modules):
        super().__init__()
        for i, m in enumerate(modules):
            self._modules[str(i)] = m

    def forward(self, x):
        for m in self._modules.values():
            x = m(x)
        return x


class ReLU(Module):
    def forward(self, x):
        return O.ReLU.apply(x)


class GELU(Module):
    def forward(self, x):
        return O.GELU.apply(x)


class MultiHeadAttention(Module):
    def __init__(self, d_model, n_heads, dropout=0.0):
        super().__init__()
        assert d_model % n_heads == 0, f"d_model {d_model} must be divisible by n_heads {n_heads}"
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.q_proj = Linear(d_model, d_model)
        self.k_proj = Linear(d_model, d_model)
        self.v_proj = Linear(d_model, d_model)
        self.out_proj = Linear(d_model, d_model)
        self.dropout = Dropout(dropout) if dropout > 0 else None

    def forward(self, x, mask=None):
        assert x.data.ndim == 3, f"MultiHeadAttention expects 3D input (B, T, C), got shape {x.shape}"
        B, T, C = x.shape
        q = self.q_proj(x).reshape(B, T, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = self.k_proj(x).reshape(B, T, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = self.v_proj(x).reshape(B, T, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)
        attn = scaled_dot_product_attention(q, k, v, mask=mask)
        attn = attn.transpose(0, 2, 1, 3).reshape(B, T, C)
        out = self.out_proj(attn)
        if self.dropout is not None:
            out = self.dropout(out)
        return out


class FeedForward(Module):
    def __init__(self, d_model, d_ff=None, dropout=0.0):
        super().__init__()
        d_ff = d_ff or 4 * d_model
        self.fc1 = Linear(d_model, d_ff)
        self.fc2 = Linear(d_ff, d_model)
        self.act = GELU()
        self.dropout = Dropout(dropout) if dropout > 0 else None

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        if self.dropout is not None:
            x = self.dropout(x)
        x = self.fc2(x)
        return x


class TransformerBlock(Module):
    def __init__(self, d_model, n_heads, d_ff=None, dropout=0.0):
        super().__init__()
        self.ln1 = LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ln2 = LayerNorm(d_model)
        self.ff = FeedForward(d_model, d_ff, dropout)

    def forward(self, x, mask=None):
        x = x + self.attn(self.ln1(x), mask=mask)
        x = x + self.ff(self.ln2(x))
        return x
