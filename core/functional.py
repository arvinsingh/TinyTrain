"""Functional operations (softmax, cross_entropy, mse, attention)."""
import numpy as np
from core.tensor import Tensor, Context, _get_xp, no_grad
from core.ops import Op


def mse_loss(pred, target):
    diff = pred - target
    return (diff * diff).mean()


def softmax(x, axis=-1):
    """Numerically stable softmax."""
    return Softmax.apply(x, axis=axis)


def log_softmax(x, axis=-1):
    """Numerically stable log-softmax."""
    return LogSoftmax.apply(x, axis=axis)


def cross_entropy(logits, targets):
    """Fused log-softmax + NLL loss. targets are class indices."""
    return CrossEntropy.apply(logits, targets)


def scaled_dot_product_attention(q, k, v, mask=None):
    """Standard scaled dot-product attention."""
    xp = q.xp
    d_k = q.shape[-1]
    scale = Tensor(xp.array(1.0 / np.sqrt(d_k), dtype=q.dtype))
    scores = (
        q @ k.transpose(0, 1, 3, 2) if k.data.ndim == 4 else q @ k.transpose(0, 2, 1)
    ) * scale
    if mask is not None:
        scores = MaskedFill.apply(scores, mask, -1e9)
    attn = softmax(scores, axis=-1)
    return attn @ v

class Softmax(Op):
    @staticmethod
    def forward(ctx, a, axis=-1):
        xp = _get_xp(a)
        shifted = a - a.max(axis=axis, keepdims=True)
        e = xp.exp(shifted)
        s = e / e.sum(axis=axis, keepdims=True)
        ctx.save_for_backward(s)
        ctx.axis = axis
        return s

    @staticmethod
    def backward(ctx, grad):
        s, = ctx.saved_tensors
        ds = grad * s
        return (ds - s * ds.sum(axis=ctx.axis, keepdims=True),)


class LogSoftmax(Op):
    @staticmethod
    def forward(ctx, a, axis=-1):
        xp = _get_xp(a)
        shifted = a - a.max(axis=axis, keepdims=True)
        log_sum_exp = xp.log(xp.exp(shifted).sum(axis=axis, keepdims=True))
        result = shifted - log_sum_exp
        ctx.save_for_backward(xp.exp(result))
        ctx.axis = axis
        return result

    @staticmethod
    def backward(ctx, grad):
        s, = ctx.saved_tensors
        return (grad - s * grad.sum(axis=ctx.axis, keepdims=True),)


class CrossEntropy:
    """
    Custom apply() because targets are not a Tensor and should not
    appear in _children or receive gradients.
    no_grad handled explicitly.
    """
    @classmethod
    def apply(cls, logits, targets):
        ctx = Context()
        xp = _get_xp(logits.data)
        targets_data = targets.data if isinstance(targets, Tensor) else xp.array(targets)

        # Stable log-softmax
        shifted = logits.data - logits.data.max(axis=-1, keepdims=True)
        log_sum_exp = xp.log(xp.exp(shifted).sum(axis=-1, keepdims=True))
        log_probs = shifted - log_sum_exp
        probs = xp.exp(log_probs)

        n = logits.data.shape[0]
        loss = -log_probs[xp.arange(n), targets_data].mean()

        requires_grad = not no_grad._enabled and logits.requires_grad
        if requires_grad:
            ctx.save_for_backward(probs)
            ctx.targets = targets_data
            ctx.n = n

        out = Tensor(
            loss,
            requires_grad=requires_grad,
            _children=(logits,) if requires_grad else (),
            _op=cls if requires_grad else None,
        )
        if requires_grad:
            out._ctx = ctx
        return out

    @staticmethod
    def backward(ctx, grad):
        probs, = ctx.saved_tensors
        xp = _get_xp(probs)
        dx = probs.copy()
        dx[xp.arange(ctx.n), ctx.targets] -= 1
        dx /= ctx.n
        return (dx * grad,)


class MaskedFill(Op):
    """
    Custom apply() because mask and fill_val are not Tensors.
    Inherits Op for consistency but overrides apply to handle
    the non-Tensor auxiliary arguments.
    """
    @classmethod
    def apply(cls, x, mask, fill_val):
        ctx = Context()
        xp = _get_xp(x.data)
        mask_data = mask.data if isinstance(mask, Tensor) else xp.array(mask)
        requires_grad = not no_grad._enabled and x.requires_grad
        result = cls.forward(ctx, x.data, mask_data, fill_val)
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
    def forward(ctx, a, mask, fill_val):
        xp = _get_xp(a)
        ctx.save_for_backward(mask)
        return xp.where(mask, xp.array(fill_val, dtype=a.dtype), a)

    @staticmethod
    def backward(ctx, grad):
        mask, = ctx.saved_tensors
        xp = _get_xp(grad)
        return (xp.where(mask, xp.zeros_like(grad), grad),)
