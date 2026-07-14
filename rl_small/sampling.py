"""Autoregressive rollout generation for the policy.

During RL we repeatedly *sample* completions from the current policy, score
them, and push the policy toward the high-reward ones. This module handles the
sampling half: given a prompt, generate tokens one at a time until ``<eos>`` or
a length cap.

Generation is done one sequence at a time. That is a little slower than batched
decoding but keeps the code obviously correct -- there is no padding inside the
context, so the causal attention never attends to filler tokens.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from .autograd import softmax_np
from .model import TinyGPT
from .tokenizer import Tokenizer


@dataclass
class Rollout:
    prompt_ids: List[int]
    gen_ids: List[int]                 # generated tokens (may include <eos>)
    problem: object = None             # the Problem it was sampled for
    reward: float = 0.0
    reward_parts: dict = field(default_factory=dict)
    advantage: float = 0.0

    @property
    def full_ids(self) -> List[int]:
        return self.prompt_ids + self.gen_ids

    @property
    def prompt_len(self) -> int:
        return len(self.prompt_ids)


def generate(model: TinyGPT, tok: Tokenizer, prompt_ids: List[int],
             max_new_tokens: int, rng: np.random.Generator,
             temperature: float = 1.0,
             force_first: Optional[int] = None,
             force_after_think: bool = False) -> List[int]:
    """Sample a completion for a single prompt.

    ``force_first`` optionally pins the first generated token (used by the
    hybrid controller to *force* a fast answer or a reasoning trace). If
    ``force_after_think`` is set, once a ``</think>`` is emitted the decoder is
    nudged to open an ``<answer>`` span so a truncated trace still yields a
    parseable answer.
    """
    context = list(prompt_ids)
    generated: List[int] = []
    forced_answer_next = False
    for step in range(max_new_tokens):
        if len(context) >= model.block_size:
            break
        logits = model(np.array([context])).data[0, -1]  # (vocab,)
        if step == 0 and force_first is not None:
            nxt = force_first
        elif forced_answer_next:
            nxt = tok.answer
            forced_answer_next = False
        else:
            probs = softmax_np(logits / max(temperature, 1e-6))
            nxt = int(rng.choice(len(probs), p=probs))
        generated.append(nxt)
        context.append(nxt)
        if nxt == tok.eos:
            break
        if force_after_think and nxt == tok.end_think:
            forced_answer_next = True
    return generated


def sample_group(model: TinyGPT, tok: Tokenizer, problem, group_size: int,
                 max_new_tokens: int, rng: np.random.Generator,
                 temperature: float = 1.0, **gen_kwargs) -> List[Rollout]:
    """Sample ``group_size`` completions for one problem (a GRPO group)."""
    prompt_ids = tok.encode(problem.prompt_tokens)
    rollouts = []
    for _ in range(group_size):
        gen = generate(model, tok, prompt_ids, max_new_tokens, rng,
                       temperature=temperature, **gen_kwargs)
        rollouts.append(Rollout(prompt_ids=prompt_ids, gen_ids=gen, problem=problem))
    return rollouts
