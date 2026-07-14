"""Neural-network building blocks built on the tiny autograd engine.

These mirror the familiar PyTorch layers (``Linear``, ``Embedding``,
``LayerNorm``) but are a few lines each so the whole forward/backward path is
inspectable. Parameters are just :class:`~rl_small.autograd.Tensor` objects
collected via :meth:`Module.parameters`.
"""

from __future__ import annotations

import numpy as np

from .autograd import Tensor, layer_norm


class Module:
    """Base class: knows how to enumerate its parameters and sub-modules."""

    def parameters(self):
        params = []
        for name, value in vars(self).items():
            if isinstance(value, Tensor) and value.requires_grad:
                params.append(value)
            elif isinstance(value, Module):
                params.extend(value.parameters())
            elif isinstance(value, (list, tuple)):
                for v in value:
                    if isinstance(v, Module):
                        params.extend(v.parameters())
                    elif isinstance(v, Tensor) and v.requires_grad:
                        params.append(v)
        return params

    def zero_grad(self):
        for p in self.parameters():
            p.zero_grad()

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


def _init(shape, scale, rng):
    return Tensor(rng.standard_normal(shape) * scale)


class Linear(Module):
    def __init__(self, d_in, d_out, rng, bias=True):
        self.weight = _init((d_in, d_out), 1.0 / np.sqrt(d_in), rng)
        self.bias = Tensor(np.zeros(d_out)) if bias else None

    def forward(self, x: Tensor) -> Tensor:
        out = x @ self.weight
        if self.bias is not None:
            out = out + self.bias
        return out


class Embedding(Module):
    def __init__(self, num, dim, rng):
        self.weight = _init((num, dim), 0.02, rng)

    def forward(self, idx: np.ndarray) -> Tensor:
        return self.weight.take_rows(np.asarray(idx))


class LayerNorm(Module):
    def __init__(self, dim, eps=1e-5):
        self.gamma = Tensor(np.ones(dim))
        self.beta = Tensor(np.zeros(dim))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        return layer_norm(x, self.gamma, self.beta, self.eps)


def gelu(x: Tensor) -> Tensor:
    """Tanh approximation of GELU, written with autograd primitives."""
    c = np.sqrt(2.0 / np.pi)
    inner = (x + x ** 3 * 0.044715) * c
    return x * (inner.tanh() + 1.0) * 0.5
