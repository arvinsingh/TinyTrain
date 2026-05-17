"""Device management, seeding, save/load, grad clipping, LR schedulers."""
import numpy as np
import os
import math

from core.tensor import _get_xp

try:
    import cupy as cp
except ImportError:
    cp = None


def set_seed(seed):
    np.random.seed(seed)
    if cp is not None:
        cp.random.seed(seed)


def param_count(module):
    return sum(p.data.size for p in module.parameters())


def get_device():
    return 'cuda' if cp is not None else 'cpu'


# save/load

def save(module, path):
    state = {}
    _collect_state(module, '', state)
    np.savez(path, **state)


def load(module, path):
    data = np.load(path, allow_pickle=False)
    _load_state(module, '', data)


def _collect_state(module, prefix, state):
    for name, param in module._params.items():
        key = f"{prefix}{name}" if prefix else name
        d = param.data
        if cp is not None and isinstance(d, cp.ndarray):
            d = d.get()
        state[key] = np.array(d)
    for name, child in module._modules.items():
        child_prefix = f"{prefix}{name}." if prefix else f"{name}."
        _collect_state(child, child_prefix, state)


def _load_state(module, prefix, data):
    for name, param in module._params.items():
        key = f"{prefix}{name}" if prefix else name
        if key in data:
            loaded = data[key]
            xp = _get_xp(param.data)
            if xp is not np:
                loaded = xp.asarray(loaded)
            param.data = loaded.astype(param.data.dtype)
    for name, child in module._modules.items():
        child_prefix = f"{prefix}{name}." if prefix else f"{name}."
        _load_state(child, child_prefix, data)


# gradient clipping

def clip_grad_norm(params, max_norm):
    total_norm_sq = 0.0
    grads = []
    for p in params:
        if p.grad is not None:
            g = p.grad
            if cp is not None and isinstance(g, cp.ndarray):
                total_norm_sq += float(cp.sum(g * g).get())
            else:
                total_norm_sq += float(np.sum(g * g))
            grads.append(p)
    total_norm = math.sqrt(total_norm_sq)
    clip_coef = max_norm / (total_norm + 1e-6)
    if clip_coef < 1.0:
        for p in grads:
            p.grad = p.grad * clip_coef
    return total_norm


def clip_grad_value(params, clip_value):
    for p in params:
        if p.grad is not None:
            xp = np if not (cp is not None and isinstance(p.grad, cp.ndarray)) else cp
            p.grad = xp.clip(p.grad, -clip_value, clip_value)


# lr schedulers

class _Scheduler:
    def __init__(self, optimizer):
        self.optimizer = optimizer
        self.base_lr = optimizer.lr
        self._step_count = 0

    def step(self):
        self._step_count += 1
        self.optimizer.lr = self.get_lr()

    def get_lr(self):
        raise NotImplementedError


class StepLR(_Scheduler):
    def __init__(self, optimizer, step_size, gamma=0.1):
        super().__init__(optimizer)
        self.step_size = step_size
        self.gamma = gamma

    def get_lr(self):
        return self.base_lr * (self.gamma ** (self._step_count // self.step_size))


class CosineAnnealingLR(_Scheduler):
    def __init__(self, optimizer, T_max, eta_min=0.0):
        super().__init__(optimizer)
        self.T_max = T_max
        self.eta_min = eta_min

    def get_lr(self):
        t = min(self._step_count, self.T_max)
        return self.eta_min + (self.base_lr - self.eta_min) * (1 + math.cos(math.pi * t / self.T_max)) / 2


class LinearWarmupCosineDecay(_Scheduler):
    def __init__(self, optimizer, warmup_steps, total_steps, eta_min=0.0):
        super().__init__(optimizer)
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.eta_min = eta_min

    def get_lr(self):
        if self._step_count <= self.warmup_steps:
            return self.base_lr * self._step_count / max(1, self.warmup_steps)
        progress = (self._step_count - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps)
        progress = min(progress, 1.0)
        return self.eta_min + (self.base_lr - self.eta_min) * (1 + math.cos(math.pi * progress)) / 2
