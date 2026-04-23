"""Autograd operations with forward/backward implementations."""
import numpy as np
from core.tensor import Tensor, Context, _get_xp, _wrap, no_grad


class Op:
    """
    Base class for operations.
    Subclasses implement static forward(ctx, ...) & backward(ctx, grad) methods.
    apply() handles the autograd logic.
    """
    @classmethod
    def apply(cls, *args, **kwargs):
        ctx = Context()
        tensors = [a for a in args if isinstance(a, Tensor)]
        result_data = cls.forward(ctx, *[a.data if isinstance(a, Tensor) else a for a in args], **kwargs)
        requires_grad = not no_grad._enabled and any(t.requires_grad for t in tensors)
        out = Tensor(result_data, requires_grad=requires_grad, _children=tensors if requires_grad else (), _op=cls if requires_grad else None)
        if requires_grad:
            out._ctx = ctx
        return out


class Add(Op):
    @staticmethod
    def forward(ctx, a, b):
        return a + b

    @staticmethod
    def backward(ctx, grad):
        return grad, grad


class Mul(Op):
    @staticmethod
    def forward(ctx, a, b):
        ctx.save_for_backward(a, b)
        return a * b

    @staticmethod
    def backward(ctx, grad):
        a, b = ctx.saved_tensors
        return grad * b, grad * a


class Neg(Op):
    @staticmethod
    def forward(ctx, a):
        return -a

    @staticmethod
    def backward(ctx, grad):
        return (-grad,)


class Div(Op):
    @staticmethod
    def forward(ctx, a, b):
        ctx.save_for_backward(a, b)
        return a / b

    @staticmethod
    def backward(ctx, grad):
        a, b = ctx.saved_tensors
        return grad / b, -grad * a / (b * b)


class Pow(Op):
    @staticmethod
    def forward(ctx, a, b):
        result = a ** b
        ctx.save_for_backward(a, b, result)
        return result

    @staticmethod
    def backward(ctx, grad):
        a, b, result = ctx.saved_tensors
        xp = _get_xp(a)
        grad_a = grad * b * a ** (b - 1)
        grad_b = grad * result * xp.log(xp.maximum(a, xp.array(1e-12, dtype=a.dtype)))
        return grad_a, grad_b


class MatMul(Op):
    @staticmethod
    def forward(ctx, a, b):
        pass

    @staticmethod
    def backward(ctx, grad):
        pass


class Sum(Op):
    @staticmethod
    def forward(ctx, a, axis=None, keepdims=False):
        ctx.shape = a.shape
        ctx.axis = axis
        ctx.keepdims = keepdims
        return a.sum(axis=axis, keepdims=keepdims)

    @staticmethod
    def backward(ctx, grad):
        xp = _get_xp(grad)
        if not ctx.keepdims and ctx.axis is not None:
            axes = ctx.axis if isinstance(ctx.axis, tuple) else (ctx.axis,)
            for a in sorted(axes):
                grad = xp.expand_dims(grad, axis=a)
        elif not ctx.keepdims and ctx.axis is None:
            grad = grad.reshape((1,) * len(ctx.shape))
        return (xp.broadcast_to(grad, ctx.shape).copy(),)


class Mean(Op):
    @staticmethod
    def forward(ctx, a, axis=None, keepdims=False):
        ctx.shape = a.shape
        ctx.axis = axis
        ctx.keepdims = keepdims
        if axis is None:
            ctx.n = a.size
        else:
            ctx.n = np.prod([a.shape[i] for i in np.atleast_1d(axis)])
        return a.mean(axis=axis, keepdims=keepdims)

    @staticmethod
    def backward(ctx, grad):
        xp = _get_xp(grad)
        if not ctx.keepdims and ctx.axis is not None:
            axes = ctx.axis if isinstance(ctx.axis, tuple) else (ctx.axis,)
            for a in sorted(axes):
                grad = xp.expand_dims(grad, axis=a)
        elif not ctx.keepdims and ctx.axis is None:
            grad = grad.reshape((1,) * len(ctx.shape))
        return (xp.broadcast_to(grad / ctx.n, ctx.shape).copy(),)


class Max(Op):
    @staticmethod
    def forward(ctx, a, axis=None, keepdims=False):
        xp = _get_xp(a)
        result = a.max(axis=axis, keepdims=True)
        mask = (a == xp.broadcast_to(result, a.shape)).astype(a.dtype)
        counts = mask.sum(axis=axis, keepdims=True)
        mask = mask / counts  # distribute grad equally among maxes
        ctx.save_for_backward(mask)
        ctx.shape = a.shape
        ctx.axis = axis
        ctx.keepdims = keepdims
        if not keepdims:
            result = result.squeeze(axis=axis) if axis is not None else result.squeeze()
        return result

    @staticmethod
    def backward(ctx, grad):
        xp = _get_xp(grad)
        mask, = ctx.saved_tensors
        if not ctx.keepdims and ctx.axis is not None:
            axes = ctx.axis if isinstance(ctx.axis, tuple) else (ctx.axis,)
            for a in sorted(axes):
                grad = xp.expand_dims(grad, axis=a)
        elif not ctx.keepdims and ctx.axis is None:
            grad = grad.reshape((1,) * len(ctx.shape))
        return (xp.broadcast_to(grad, ctx.shape) * mask,)


class Reshape(Op):
    @staticmethod
    def forward(ctx, a, shape=None):
        ctx.orig_shape = a.shape
        return a.reshape(shape)

    @staticmethod
    def backward(ctx, grad):
        return (grad.reshape(ctx.orig_shape),)


class Transpose(Op):
    @staticmethod
    def forward(ctx, a, axes=None):
        ctx.axes = axes
        xp = _get_xp(a)
        return xp.transpose(a, axes=axes)

    @staticmethod
    def backward(ctx, grad):
        xp = _get_xp(grad)
        if ctx.axes is None:
            return (xp.transpose(grad),)
        inv = [0] * len(ctx.axes)
        for i, a in enumerate(ctx.axes):
            inv[a] = i
        return (xp.transpose(grad, axes=inv),)


class Slice(Op):
    @staticmethod
    def forward(ctx, a, key=None):
        ctx.shape = a.shape
        ctx.key = key
        return a[key]

    @staticmethod
    def backward(ctx, grad):
        xp = _get_xp(grad)
        full_grad = xp.zeros(ctx.shape, dtype=grad.dtype)
        full_grad[ctx.key] = grad
        return (full_grad,)


class Exp(Op):
    @staticmethod
    def forward(ctx, a):
        result = _get_xp(a).exp(a)
        ctx.save_for_backward(result)
        return result

    @staticmethod
    def backward(ctx, grad):
        result, = ctx.saved_tensors
        return (grad * result,)


class Log(Op):
    @staticmethod
    def forward(ctx, a):
        xp = _get_xp(a)
        a_clamped = xp.maximum(a, xp.array(1e-12, dtype=a.dtype))
        ctx.save_for_backward(a_clamped)
        return xp.log(a_clamped)

    @staticmethod
    def backward(ctx, grad):
        a_clamped, = ctx.saved_tensors
        return (grad / a_clamped,)


class Tanh(Op):
    @staticmethod
    def forward(ctx, a):
        xp = _get_xp(a)
        result = xp.tanh(a)
        ctx.save_for_backward(result)
        return result

    @staticmethod
    def backward(ctx, grad):
        result, = ctx.saved_tensors
        return (grad * (1 - result * result),)


class Sigmoid(Op):
    @staticmethod
    def forward(ctx, a):
        xp = _get_xp(a)
        # numerically stable sigmoid
        pos = xp.where(a >= 0, 1 / (1 + xp.exp(-a)), 0)
        neg = xp.where(a < 0, xp.exp(a) / (1 + xp.exp(a)), 0)
        result = pos + neg
        ctx.save_for_backward(result)
        return result

    @staticmethod
    def backward(ctx, grad):
        result, = ctx.saved_tensors
        return (grad * result * (1 - result),)


class ReLU(Op):
    @staticmethod
    def forward(ctx, a):
        pass

    @staticmethod
    def backward(ctx, grad):
        pass


class GELU(Op):
    @staticmethod
    def forward(ctx, a):
        pass

    @staticmethod
    def backward(ctx, grad):
        pass


class Cat(Op):
    @staticmethod
    def forward(ctx, *args, axis=0):
        xp = _get_xp(args[0])
        ctx.splits = [a.shape[axis] for a in args]
        ctx.axis = axis
        return xp.concatenate(args, axis=axis)

    @staticmethod
    def backward(ctx, grad):
        xp = _get_xp(grad)
        indices = np.cumsum(ctx.splits[:-1])
        return tuple(xp.split(grad, indices.tolist(), axis=ctx.axis))


def cat(tensors, axis=0):
    """Concatenate tensors along axis."""
    return Cat.apply(*tensors, axis=axis)

# wire operator overloads onto Tensor

def _tensor_add(self, other):
    return Add.apply(self, _wrap(other, self.xp))

def _tensor_radd(self, other):
    return Add.apply(_wrap(other, self.xp), self)

def _tensor_mul(self, other):
    return Mul.apply(self, _wrap(other, self.xp))

def _tensor_rmul(self, other):
    return Mul.apply(_wrap(other, self.xp), self)

def _tensor_neg(self):
    return Neg.apply(self)

def _tensor_sub(self, other):
    return self + (-_wrap(other, self.xp))

def _tensor_rsub(self, other):
    return _wrap(other, self.xp) + (-self)

def _tensor_truediv(self, other):
    return Div.apply(self, _wrap(other, self.xp))

def _tensor_rtruediv(self, other):
    return Div.apply(_wrap(other, self.xp), self)

def _tensor_pow(self, exp):
    if isinstance(exp, Tensor):
        return Pow.apply(self, exp)
    return Pow.apply(self, _wrap(exp, self.xp))

def _tensor_matmul(self, other):
    return MatMul.apply(self, other)

def _tensor_getitem(self, key):
    return Slice.apply(self, key=key)

def _tensor_sum(self, axis=None, keepdims=False):
    return Sum.apply(self, axis=axis, keepdims=keepdims)

def _tensor_mean(self, axis=None, keepdims=False):
    return Mean.apply(self, axis=axis, keepdims=keepdims)

def _tensor_max(self, axis=None, keepdims=False):
    return Max.apply(self, axis=axis, keepdims=keepdims)

def _tensor_reshape(self, *shape):
    return Reshape.apply(self, shape=shape)

def _tensor_transpose(self, *axes):
    return Transpose.apply(self, axes=axes if axes else None)

def _tensor_exp(self):
    return Exp.apply(self)

def _tensor_log(self):
    return Log.apply(self)

def _tensor_tanh(self):
    return Tanh.apply(self)

def _tensor_sigmoid(self):
    return Sigmoid.apply(self)

Tensor.__add__ = _tensor_add
Tensor.__radd__ = _tensor_radd
Tensor.__mul__ = _tensor_mul
Tensor.__rmul__ = _tensor_rmul
Tensor.__neg__ = _tensor_neg
Tensor.__sub__ = _tensor_sub
Tensor.__rsub__ = _tensor_rsub
Tensor.__truediv__ = _tensor_truediv
Tensor.__rtruediv__ = _tensor_rtruediv
Tensor.__pow__ = _tensor_pow
Tensor.__matmul__ = _tensor_matmul
Tensor.__getitem__ = _tensor_getitem
Tensor.sum = _tensor_sum
Tensor.mean = _tensor_mean
Tensor.max = _tensor_max
Tensor.reshape = _tensor_reshape
Tensor.transpose = _tensor_transpose
Tensor.T = property(lambda self: self.transpose())
Tensor.exp = _tensor_exp
Tensor.log = _tensor_log
Tensor.tanh = _tensor_tanh
Tensor.sigmoid = _tensor_sigmoid
