"""Optimizers SGD and AdamW."""
from core.tensor import _get_xp


class SGD:
    def __init__(self, params, lr=0.01, momentum=0.0, weight_decay=0.0):
        self.params = list(params)
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.velocities = [None] * len(self.params) if momentum > 0 else None

    def step(self):
        for i, p in enumerate(self.params):
            if p.grad is None:
                continue
            xp = _get_xp(p.data)
            g = p.grad
            if self.weight_decay > 0:
                g = g + self.weight_decay * p.data
            if self.momentum > 0:
                if self.velocities[i] is None:
                    self.velocities[i] = xp.zeros_like(p.data)
                self.velocities[i] = self.momentum * self.velocities[i] + g
                p.data -= self.lr * self.velocities[i]
            else:
                p.data -= self.lr * g

    def zero_grad(self):
        for p in self.params:
            p.grad = None


class Adam:
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0):
        self.params = list(params)
        self.lr = lr
        self.b1, self.b2 = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.t = 0
        self.m = [None] * len(self.params)
        self.v = [None] * len(self.params)

    def step(self):
        self.t += 1
        for i, p in enumerate(self.params):
            if p.grad is None:
                continue
            xp = _get_xp(p.data)
            if self.m[i] is None:
                self.m[i] = xp.zeros_like(p.data)
                self.v[i] = xp.zeros_like(p.data)

            g = p.grad
            # decoupled weight decay
            if self.weight_decay > 0:
                p.data -= self.lr * self.weight_decay * p.data

            self.m[i] = self.b1 * self.m[i] + (1 - self.b1) * g
            self.v[i] = self.b2 * self.v[i] + (1 - self.b2) * g * g

            m_hat = self.m[i] / (1 - self.b1 ** self.t)
            v_hat = self.v[i] / (1 - self.b2 ** self.t)

            p.data -= self.lr * m_hat / (xp.sqrt(v_hat) + self.eps)

    def zero_grad(self):
        for p in self.params:
            p.grad = None
