"""Finite-difference gradient checks for the autograd engine.

If these pass, we can trust every downstream module (TinyGPT, GRPO) because
they are built entirely out of the primitives exercised here.
"""

import numpy as np

from rl_small.autograd import Tensor, log_softmax, layer_norm


def numeric_grad(f, x: Tensor, eps: float = 1e-6) -> np.ndarray:
    """Central-difference gradient of scalar ``f`` w.r.t. tensor ``x.data``."""
    grad = np.zeros_like(x.data)
    it = np.nditer(x.data, flags=["multi_index"])
    while not it.finished:
        i = it.multi_index
        orig = x.data[i]
        x.data[i] = orig + eps
        fpos = f().data
        x.data[i] = orig - eps
        fneg = f().data
        x.data[i] = orig
        grad[i] = (fpos - fneg) / (2 * eps)
        it.iternext()
    return grad


def check(f, inputs, tol=1e-5):
    for x in inputs:
        x.zero_grad()
    loss = f()
    loss.backward()
    for x in inputs:
        ng = numeric_grad(f, x)
        assert np.allclose(x.grad, ng, atol=tol), (
            f"grad mismatch\nanalytic=\n{x.grad}\nnumeric=\n{ng}"
        )


def test_add_mul_sub_div():
    rng = np.random.default_rng(0)
    a = Tensor(rng.standard_normal((3, 4)))
    b = Tensor(rng.standard_normal((3, 4)))
    check(lambda: ((a * b + a - b) / (b * b + 3.0)).sum(), [a, b])


def test_broadcasting():
    rng = np.random.default_rng(1)
    a = Tensor(rng.standard_normal((3, 4)))
    bias = Tensor(rng.standard_normal((4,)))
    check(lambda: (a + bias).sum(), [a, bias])
    check(lambda: (a * bias).sum(), [a, bias])


def test_matmul_batched():
    rng = np.random.default_rng(2)
    a = Tensor(rng.standard_normal((2, 3, 4)))
    w = Tensor(rng.standard_normal((4, 5)))
    check(lambda: (a @ w).sum(), [a, w])
    b = Tensor(rng.standard_normal((2, 4, 5)))
    check(lambda: (a @ b).sum(), [a, b])


def test_pow_exp_log_tanh_relu():
    rng = np.random.default_rng(3)
    x = Tensor(np.abs(rng.standard_normal((4, 5))) + 0.5)
    check(lambda: (x ** 3).sum(), [x])
    check(lambda: x.exp().sum(), [x])
    check(lambda: x.log().sum(), [x])
    check(lambda: x.tanh().sum(), [x])
    y = Tensor(rng.standard_normal((4, 5)))
    check(lambda: y.relu().sum(), [y])


def test_reshape_transpose():
    rng = np.random.default_rng(4)
    x = Tensor(rng.standard_normal((2, 3, 4)))
    check(lambda: x.reshape(6, 4).sum(), [x])
    check(lambda: x.transpose((0, 2, 1)).sum(), [x])


def test_log_softmax_and_gather():
    rng = np.random.default_rng(5)
    logits = Tensor(rng.standard_normal((3, 6)))
    idx = np.array([1, 4, 0])
    check(lambda: log_softmax(logits).gather_last(idx).sum(), [logits])


def test_take_rows_embedding():
    rng = np.random.default_rng(6)
    table = Tensor(rng.standard_normal((5, 3)))
    idx = np.array([[0, 2, 2], [4, 1, 0]])
    check(lambda: table.take_rows(idx).sum(), [table])


def test_layer_norm():
    rng = np.random.default_rng(7)
    x = Tensor(rng.standard_normal((3, 8)))
    gamma = Tensor(np.ones(8))
    beta = Tensor(np.zeros(8))
    check(lambda: (layer_norm(x, gamma, beta) ** 2).sum(), [x, gamma, beta])
