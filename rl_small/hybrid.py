"""Dynamic Hybrid Reasoning -- choosing *when* to think, per input.

"Hybrid reasoning" means a single model can operate in two modes:

* **Fast (System 1):** answer directly, no visible chain of thought.
* **Slow (System 2):** emit an explicit ``<think>`` trace, then answer.

"Dynamic" means the choice is made *per problem*, at inference time, rather than
fixed globally. This project realizes hybrid reasoning two complementary ways:

1. **Learned gating (emergent).** After GRPO training the policy itself decides,
   via its first generated token, whether to open ``<think>`` or go straight to
   ``<answer>``. The reward shaping in :mod:`rl_small.rewards` makes this gate
   line up with difficulty. Just decode with ``mode="auto"``.

2. **Confidence-gated control (explicit).** A wrapper that *measures* the
   model's confidence in a direct answer and only spends reasoning tokens when
   confidence is low -- and scales the reasoning-token *budget* to how unsure it
   is. This gives an interpretable dial for the compute/accuracy trade-off even
   on top of a weak policy.

Both are "dynamic hybrid reasoning"; (2) is the classic test-time controller and
is what this module implements.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np

from .autograd import softmax_np
from .env import Problem
from .model import TinyGPT
from .sampling import Rollout, generate
from .tokenizer import Tokenizer


@dataclass
class HybridConfig:
    confidence_threshold: float = 0.5   # answer directly if confidence >= this
    fast_budget: int = 8                # max new tokens for a direct answer
    base_think_budget: int = 12         # reasoning budget floor
    max_think_budget: int = 40          # reasoning budget ceiling
    temperature: float = 0.7


def answer_confidence(model: TinyGPT, tok: Tokenizer, prompt_ids: List[int]) -> float:
    """How sure is the model of a *direct* answer?

    We append the ``<answer>`` token to the prompt and read the model's
    distribution over the first answer digit. The probability mass on its most
    likely digit is a cheap, label-free proxy for confidence: high on easy
    problems it "knows", low on ones it should reason through.
    """
    ctx = list(prompt_ids) + [tok.answer]
    logits = model(np.array([ctx])).data[0, -1]
    probs = softmax_np(logits)
    return float(probs.max())


def estimated_think_budget(confidence: float, cfg: HybridConfig) -> int:
    """Spend more reasoning tokens the less confident the model is."""
    span = cfg.max_think_budget - cfg.base_think_budget
    return int(cfg.base_think_budget + span * (1.0 - confidence))


@dataclass
class HybridDecision:
    mode: str            # "fast" or "think"
    confidence: float
    budget: int
    gen_ids: List[int]


def dynamic_decode(model: TinyGPT, tok: Tokenizer, problem: Problem,
                   cfg: HybridConfig, rng: np.random.Generator) -> HybridDecision:
    """Decide fast-vs-think from confidence, then decode with a matched budget."""
    prompt_ids = tok.encode(problem.prompt_tokens)
    conf = answer_confidence(model, tok, prompt_ids)

    if conf >= cfg.confidence_threshold:
        gen = generate(model, tok, prompt_ids, cfg.fast_budget, rng,
                       temperature=cfg.temperature, force_first=tok.answer)
        return HybridDecision("fast", conf, cfg.fast_budget, gen)

    budget = estimated_think_budget(conf, cfg)
    gen = generate(model, tok, prompt_ids, budget, rng,
                   temperature=cfg.temperature, force_first=tok.think,
                   force_after_think=True)
    return HybridDecision("think", conf, budget, gen)


def auto_decode(model: TinyGPT, tok: Tokenizer, problem: Problem,
                max_new_tokens: int, rng: np.random.Generator,
                temperature: float = 0.7) -> Rollout:
    """Let the trained policy pick the mode itself (learned gating)."""
    prompt_ids = tok.encode(problem.prompt_tokens)
    gen = generate(model, tok, prompt_ids, max_new_tokens, rng,
                   temperature=temperature)
    return Rollout(prompt_ids=prompt_ids, gen_ids=gen, problem=problem)
