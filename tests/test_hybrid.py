"""Tests for the dynamic hybrid reasoning controller."""

import numpy as np

from rl_small.tokenizer import Tokenizer
from rl_small.env import ArithmeticEnv
from rl_small.model import TinyGPT
from rl_small.hybrid import (HybridConfig, answer_confidence,
                             estimated_think_budget, dynamic_decode)


def test_confidence_in_unit_interval():
    tok = Tokenizer()
    env = ArithmeticEnv(tok, seed=0)
    model = TinyGPT(tok.vocab_size, block_size=48, d_model=16, n_head=2, n_layer=1)
    p = env.sample()
    c = answer_confidence(model, tok, tok.encode(p.prompt_tokens))
    assert 0.0 <= c <= 1.0


def test_budget_scales_inversely_with_confidence():
    cfg = HybridConfig(base_think_budget=10, max_think_budget=40)
    assert estimated_think_budget(1.0, cfg) == 10
    assert estimated_think_budget(0.0, cfg) == 40
    assert estimated_think_budget(0.5, cfg) == 25


def test_dynamic_decode_returns_valid_mode():
    tok = Tokenizer()
    env = ArithmeticEnv(tok, seed=0)
    model = TinyGPT(tok.vocab_size, block_size=48, d_model=16, n_head=2, n_layer=1)
    rng = np.random.default_rng(0)
    p = env.sample()
    d = dynamic_decode(model, tok, p, HybridConfig(), rng)
    assert d.mode in ("fast", "think")
    assert len(d.gen_ids) >= 1
    if d.mode == "think":
        assert d.gen_ids[0] == tok.think
    else:
        assert d.gen_ids[0] == tok.answer
