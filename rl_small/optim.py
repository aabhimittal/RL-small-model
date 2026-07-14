"""Adam optimizer operating on the engine's Tensor parameters."""

from __future__ import annotations

import numpy as np


class Adam:
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0.0, grad_clip=1.0):
        self.params = list(params)
        self.lr = lr
        self.b1, self.b2 = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.grad_clip = grad_clip
        self.m = [np.zeros_like(p.data) for p in self.params]
        self.v = [np.zeros_like(p.data) for p in self.params]
        self.t = 0

    def _global_norm(self):
        total = 0.0
        for p in self.params:
            total += float(np.sum(p.grad * p.grad))
        return np.sqrt(total)

    def step(self):
        self.t += 1
        scale = 1.0
        if self.grad_clip is not None:
            norm = self._global_norm()
            if norm > self.grad_clip:
                scale = self.grad_clip / (norm + 1e-12)
        for i, p in enumerate(self.params):
            g = p.grad * scale
            if self.weight_decay:
                g = g + self.weight_decay * p.data
            self.m[i] = self.b1 * self.m[i] + (1 - self.b1) * g
            self.v[i] = self.b2 * self.v[i] + (1 - self.b2) * (g * g)
            mhat = self.m[i] / (1 - self.b1 ** self.t)
            vhat = self.v[i] / (1 - self.b2 ** self.t)
            p.data -= self.lr * mhat / (np.sqrt(vhat) + self.eps)

    def zero_grad(self):
        for p in self.params:
            p.zero_grad()
