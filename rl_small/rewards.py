"""Reward design -- the lever that *tightens* small-model behavior.

In pure RL there is no supervised target; the only teaching signal is the
reward. Its shape therefore determines everything the model becomes. We combine
four ingredients:

1. **Correctness** (the dominant term): is the extracted answer right? This is
   the verifiable signal that pure RL ultimately optimizes.
2. **Graded format shaping**: partial credit for structural progress -- opening
   an ``<answer>`` span, closing it, stopping with ``<eos>``. This is what makes
   pure RL *from scratch* work. A pure correct/incorrect reward is far too
   sparse for a randomly-initialized model: it essentially never stumbles onto a
   fully correct sequence, every rollout scores the same, and GRPO sees zero
   variance so there is nothing to learn. Graded shaping creates a smooth ladder
   the policy can climb -- first learn to emit a clean answer span, *then* learn
   to make it correct.
3. **Full-format bonus**: an extra reward once the whole structure is clean.
4. **Length / efficiency** (a small penalty): shorter is better, *all else
   equal*. This is the "tightening" term -- it trims needless verbosity.

The magic is in the *interaction*. Because correctness dwarfs the length
penalty, the length term only ever breaks ties between equally-correct answers.
On easy problems a short direct answer is already correct, so brevity wins and
the model learns to answer fast. On hard problems a direct guess is usually
wrong, so the large correctness bonus makes step-by-step reasoning worth its
extra length. Nobody hand-labels "reason here, don't reason there" -- the
*dynamic hybrid reasoning* behavior emerges from this single reward.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

from .env import ArithmeticEnv
from .sampling import Rollout


@dataclass
class RewardConfig:
    correct_bonus: float = 1.0        # reward for a correct answer
    wrong_penalty: float = 0.0        # reward for a parseable but wrong answer
    # Graded shaping (bootstraps exploration from random init). Every rung
    # requires real progress toward an *answer* -- we deliberately do NOT reward
    # bare structural tokens like a lone <eos>, because a from-scratch policy
    # will happily collapse onto any cheap, contentless reward it can reach.
    answer_number_bonus: float = 0.15 # produced a non-empty, parseable number
    format_bonus: float = 0.10        # extra once the whole structure is clean
    no_answer_penalty: float = -0.05  # emitted no answer at all (breaks the
                                      #   "just stop immediately" collapse)
    length_coef: float = 0.004        # per-token brevity penalty (small!)
    max_gen_tokens: int = 40
    # Optional *dense* reward: partial credit for being numerically close to the
    # answer. 0.0 = purely sparse (correct-or-not). A small positive value turns
    # the sparse "did you nail it" signal into a smooth slope the policy can
    # follow, which dramatically speeds up correctness discovery for a tiny model
    # (see docs/04_reward_design.md, "sparse vs dense").
    proximity_coef: float = 0.0
    proximity_scale: float = 10.0


def analyze_format(env: ArithmeticEnv, gen_ids) -> Dict[str, bool]:
    """Structural checks on a generation (independent of correctness)."""
    tok = env.tok
    opened = tok.answer in gen_ids
    closed = opened and tok.end_answer in gen_ids and (
        gen_ids.index(tok.end_answer) > gen_ids.index(tok.answer)
    )
    ends_eos = len(gen_ids) > 0 and gen_ids[-1] == tok.eos
    used_reasoning = tok.think in gen_ids

    well_formed = closed and ends_eos
    if used_reasoning:
        # <think> must be opened and closed before the answer starts.
        if tok.end_think not in gen_ids or not opened:
            well_formed = False
        else:
            well_formed = well_formed and (
                gen_ids.index(tok.think)
                < gen_ids.index(tok.end_think)
                < gen_ids.index(tok.answer)
            )
    return {
        "opened": opened,
        "closed": bool(closed),
        "ends_eos": ends_eos,
        "used_reasoning": used_reasoning,
        "well_formed": bool(well_formed),
    }


def compute_reward(env: ArithmeticEnv, rollout: Rollout,
                   cfg: RewardConfig) -> Tuple[float, Dict[str, float]]:
    """Return ``(total_reward, parts)`` for a single rollout."""
    gen = rollout.gen_ids
    fmt = analyze_format(env, gen)
    correct = env.is_correct(rollout.problem, gen)
    parseable = env.extract_answer(gen) is not None

    # A structure counts as fully well-formed only if it carries a real number.
    # (Without this, the policy discovers the degenerate "<answer></answer>"
    # reward-hack: perfectly structured, perfectly empty, and a dead end.)
    well_formed = fmt["well_formed"] and parseable

    # Graded structural shaping -- partial credit that gives the from-scratch
    # policy a gradient to climb before it can ever produce a correct answer.
    shaping = 0.0
    if parseable:
        shaping += cfg.answer_number_bonus
    else:
        shaping += cfg.no_answer_penalty
    if well_formed:
        shaping += cfg.format_bonus

    if correct:
        acc_term = cfg.correct_bonus
    else:
        acc_term = cfg.wrong_penalty
        if cfg.proximity_coef > 0.0 and parseable:
            pred = env.extract_answer(gen)
            closeness = max(0.0, 1.0 - abs(pred - rollout.problem.answer) / cfg.proximity_scale)
            acc_term += cfg.proximity_coef * closeness

    # The brevity penalty applies ONLY once the model actually produced an
    # answer. Penalizing length on answer-less rollouts is a trap: when nothing
    # yet emits a number, "shorter" becomes the only reward signal and the
    # policy collapses to emitting <eos> immediately (len 1) instead of learning
    # to answer. Conciseness pressure belongs on real answers, not on silence.
    length_term = 0.0
    if parseable:
        length_term = -cfg.length_coef * min(len(gen), cfg.max_gen_tokens)

    total = acc_term + shaping + length_term
    parts = {
        "accuracy": acc_term,
        "shaping": shaping,
        "format": cfg.format_bonus if well_formed else 0.0,
        "length": length_term,
        "correct": float(correct),
        "parseable": float(parseable),
        "used_reasoning": float(fmt["used_reasoning"]),
        "well_formed": float(well_formed),
        "gen_len": float(len(gen)),
    }
    return total, parts
