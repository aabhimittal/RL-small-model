"""A tiny reverse-mode automatic differentiation engine over NumPy arrays.

This is intentionally small and readable. It exists so that the rest of the
project (the TinyGPT policy, GRPO, reward shaping) can be expressed with plain
math and a single ``loss.backward()`` call -- exactly like PyTorch, but with
nothing hidden.

The core object is :class:`Tensor`. Every operation on a ``Tensor`` records how
to propagate gradients backwards. Calling :meth:`Tensor.backward` walks the
recorded graph in reverse topological order and fills in ``.grad`` on every
tensor that requires gradients.

We use ``float64`` everywhere. The models here are tiny, so speed is a
non-issue, and ``float64`` makes the finite-difference gradient checks in
``tests/test_autograd.py`` clean and trustworthy.
"""

from __future__ import annotations

import numpy as np

Array = np.ndarray


def _unbroadcast(grad: Array, shape: tuple) -> Array:
    """Reduce ``grad`` back to ``shape`` by summing over broadcasted axes.

    NumPy broadcasting lets ``a + b`` produce a result larger than either
    operand. During backprop we must sum the incoming gradient over exactly the
    axes that were broadcast so it matches the operand's original shape.
    """
    # Sum away leading dimensions that did not exist in the original tensor.
    while grad.ndim > len(shape):
        grad = grad.sum(axis=0)
    # Sum away dimensions that were size-1 in the original and expanded.
    for i, dim in enumerate(shape):
        if dim == 1 and grad.shape[i] != 1:
            grad = grad.sum(axis=i, keepdims=True)
    return grad.reshape(shape)


class Tensor:
    """A node in the autodiff graph wrapping a NumPy array."""

    __slots__ = ("data", "grad", "requires_grad", "_backward", "_prev")

    def __init__(self, data, _children=(), requires_grad: bool = True):
        self.data = np.asarray(data, dtype=np.float64)
        self.requires_grad = requires_grad
        self.grad = np.zeros_like(self.data) if requires_grad else None
        # Function that pushes this node's gradient to its parents.
        self._backward = lambda: None
        self._prev = tuple(_children)

    # -- construction helpers -------------------------------------------------
    @property
    def shape(self):
        return self.data.shape

    @property
    def ndim(self):
        return self.data.ndim

    def __repr__(self):
        return f"Tensor(shape={self.data.shape}, requires_grad={self.requires_grad})"

    @staticmethod
    def _ensure(x) -> "Tensor":
        return x if isinstance(x, Tensor) else Tensor(x, requires_grad=False)

    def _new(self, data, children, backward):
        out = Tensor(data, _children=children)
        out._backward = backward
        return out

    # -- elementwise ops ------------------------------------------------------
    def __add__(self, other):
        other = Tensor._ensure(other)
        out = Tensor(self.data + other.data, _children=(self, other))

        def _backward():
            if self.requires_grad:
                self.grad += _unbroadcast(out.grad, self.shape)
            if other.requires_grad:
                other.grad += _unbroadcast(out.grad, other.shape)

        out._backward = _backward
        return out

    def __mul__(self, other):
        other = Tensor._ensure(other)
        out = Tensor(self.data * other.data, _children=(self, other))

        def _backward():
            if self.requires_grad:
                self.grad += _unbroadcast(out.grad * other.data, self.shape)
            if other.requires_grad:
                other.grad += _unbroadcast(out.grad * self.data, other.shape)

        out._backward = _backward
        return out

    def __pow__(self, p):
        assert isinstance(p, (int, float)), "only scalar powers supported"
        out = Tensor(self.data ** p, _children=(self,))

        def _backward():
            if self.requires_grad:
                self.grad += _unbroadcast(out.grad * p * (self.data ** (p - 1)), self.shape)

        out._backward = _backward
        return out

    def __neg__(self):
        return self * -1.0

    def __sub__(self, other):
        return self + (-Tensor._ensure(other))

    def __rsub__(self, other):
        return Tensor._ensure(other) + (-self)

    def __truediv__(self, other):
        other = Tensor._ensure(other)
        return self * (other ** -1)

    def __rtruediv__(self, other):
        return Tensor._ensure(other) * (self ** -1)

    __radd__ = __add__
    __rmul__ = __mul__

    # -- matmul ---------------------------------------------------------------
    def matmul(self, other):
        other = Tensor._ensure(other)
        out = Tensor(self.data @ other.data, _children=(self, other))

        def _backward():
            g = out.grad
            if self.requires_grad:
                ga = g @ np.swapaxes(other.data, -1, -2)
                self.grad += _unbroadcast(ga, self.shape)
            if other.requires_grad:
                gb = np.swapaxes(self.data, -1, -2) @ g
                other.grad += _unbroadcast(gb, other.shape)

        out._backward = _backward
        return out

    def __matmul__(self, other):
        return self.matmul(other)

    # -- reductions -----------------------------------------------------------
    def sum(self, axis=None, keepdims=False):
        out = Tensor(self.data.sum(axis=axis, keepdims=keepdims), _children=(self,))

        def _backward():
            if not self.requires_grad:
                return
            g = out.grad
            if axis is not None and not keepdims:
                g = np.expand_dims(g, axis)
            self.grad += np.broadcast_to(g, self.shape).copy()

        out._backward = _backward
        return out

    def mean(self, axis=None, keepdims=False):
        if axis is None:
            n = self.data.size
        else:
            n = self.data.shape[axis]
        return self.sum(axis=axis, keepdims=keepdims) * (1.0 / n)

    # -- shape ops ------------------------------------------------------------
    def reshape(self, *shape):
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = tuple(shape[0])
        out = Tensor(self.data.reshape(shape), _children=(self,))

        def _backward():
            if self.requires_grad:
                self.grad += out.grad.reshape(self.shape)

        out._backward = _backward
        return out

    def transpose(self, axes):
        out = Tensor(np.transpose(self.data, axes), _children=(self,))
        inv = np.argsort(axes)

        def _backward():
            if self.requires_grad:
                self.grad += np.transpose(out.grad, inv)

        out._backward = _backward
        return out

    def swapaxes(self, a, b):
        axes = list(range(self.ndim))
        axes[a], axes[b] = axes[b], axes[a]
        return self.transpose(tuple(axes))

    # -- unary math -----------------------------------------------------------
    def exp(self):
        e = np.exp(self.data)
        out = Tensor(e, _children=(self,))

        def _backward():
            if self.requires_grad:
                self.grad += out.grad * e

        out._backward = _backward
        return out

    def log(self):
        out = Tensor(np.log(self.data), _children=(self,))

        def _backward():
            if self.requires_grad:
                self.grad += out.grad / self.data

        out._backward = _backward
        return out

    def tanh(self):
        t = np.tanh(self.data)
        out = Tensor(t, _children=(self,))

        def _backward():
            if self.requires_grad:
                self.grad += out.grad * (1 - t * t)

        out._backward = _backward
        return out

    def relu(self):
        out = Tensor(np.maximum(self.data, 0.0), _children=(self,))

        def _backward():
            if self.requires_grad:
                self.grad += out.grad * (self.data > 0)

        out._backward = _backward
        return out

    # -- indexing / gather ----------------------------------------------------
    def take_rows(self, idx: Array):
        """Row lookup used by embedding layers. ``idx`` is an integer array.

        Result shape is ``idx.shape + (self.shape[1],)``. Gradient is scattered
        back with accumulation (repeated indices add up).
        """
        idx = np.asarray(idx)
        out = Tensor(self.data[idx], _children=(self,))

        def _backward():
            if not self.requires_grad:
                return
            np.add.at(self.grad, idx, out.grad)

        out._backward = _backward
        return out

    def gather_last(self, idx: Array):
        """Select one entry along the last axis per row.

        ``self`` has shape ``(..., V)`` and ``idx`` has shape ``(...)``. Returns
        shape ``(...)``. Used to pick the log-prob of a chosen token.
        """
        idx = np.asarray(idx)
        picked = np.take_along_axis(self.data, idx[..., None], axis=-1)[..., 0]
        out = Tensor(picked, _children=(self,))

        def _backward():
            if not self.requires_grad:
                return
            g = np.zeros_like(self.data)
            np.put_along_axis(g, idx[..., None], out.grad[..., None], axis=-1)
            self.grad += g

        out._backward = _backward
        return out

    # -- backprop -------------------------------------------------------------
    def backward(self):
        """Populate ``.grad`` for every ancestor of this (scalar) tensor."""
        topo = []
        visited = set()

        def build(v):
            if id(v) in visited:
                return
            visited.add(id(v))
            for child in v._prev:
                build(child)
            topo.append(v)

        build(self)
        # Seed the output gradient. Typically ``self`` is a scalar loss.
        self.grad = np.ones_like(self.data)
        for node in reversed(topo):
            node._backward()

    def zero_grad(self):
        if self.grad is not None:
            self.grad = np.zeros_like(self.data)


# -- functional helpers -------------------------------------------------------
def log_softmax(x: Tensor, axis: int = -1) -> Tensor:
    """Numerically stable ``log(softmax(x))`` built from primitives.

    The max-shift is detached (treated as a constant), which is valid because
    log-softmax is invariant to a constant shift along ``axis``.
    """
    m = x.data.max(axis=axis, keepdims=True)
    shifted = x + Tensor(-m, requires_grad=False)
    return shifted - shifted.exp().sum(axis=axis, keepdims=True).log()


def softmax_np(logits: Array, axis: int = -1) -> Array:
    """Plain-NumPy softmax for sampling paths that don't need gradients."""
    m = logits.max(axis=axis, keepdims=True)
    e = np.exp(logits - m)
    return e / e.sum(axis=axis, keepdims=True)


def layer_norm(x: Tensor, gamma: Tensor, beta: Tensor, eps: float = 1e-5) -> Tensor:
    """Layer normalization over the last axis, expressed with primitives."""
    mu = x.mean(axis=-1, keepdims=True)
    centered = x - mu
    var = (centered * centered).mean(axis=-1, keepdims=True)
    std = (var + eps) ** 0.5
    normed = centered / std
    return normed * gamma + beta
