"""Sanity + gradient checks for TinyGPT."""

import numpy as np

from rl_small.model import TinyGPT
from rl_small.autograd import log_softmax


def test_forward_shape():
    model = TinyGPT(vocab_size=11, block_size=8, d_model=16, n_head=2, n_layer=2, seed=0)
    idx = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])
    logits = model(idx)
    assert logits.shape == (2, 4, 11)
    assert model.num_params() > 0


def test_backward_runs_and_matches_numeric():
    # Small enough to finite-difference one parameter slice.
    model = TinyGPT(vocab_size=7, block_size=6, d_model=8, n_head=2, n_layer=1, seed=1)
    idx = np.array([[1, 2, 3]])
    targets = np.array([[2, 3, 4]])

    def loss_fn():
        logits = model(idx)
        lp = log_softmax(logits).gather_last(targets)
        return -(lp.sum())

    model.zero_grad()
    loss = loss_fn()
    loss.backward()

    # Check one parameter (the token embedding) against finite differences.
    p = model.tok_emb.weight
    eps = 1e-5
    idxs = [(0, 0), (3, 2), (5, 1)]
    for i in idxs:
        orig = p.data[i]
        p.data[i] = orig + eps
        fpos = loss_fn().data
        p.data[i] = orig - eps
        fneg = loss_fn().data
        p.data[i] = orig
        num = (fpos - fneg) / (2 * eps)
        assert np.isclose(p.grad[i], num, atol=1e-4), (p.grad[i], num)
