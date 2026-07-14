"""Evaluation helpers: measure accuracy, reasoning rate, and length by mode.

These functions quantify exactly the behaviors the project cares about:
* Did pure RL make the model *correct*?
* Did *dynamic hybrid reasoning* emerge -- reasoning more on hard problems?
* Did the confidence-gated controller trade compute for accuracy sensibly?
"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict

import numpy as np

from .env import ArithmeticEnv
from .hybrid import HybridConfig, auto_decode, dynamic_decode
from .model import TinyGPT
from .sampling import Rollout, generate
from .tokenizer import Tokenizer


def _stats_from(records):
    if not records:
        return {"n": 0, "accuracy": 0.0, "reasoning_rate": 0.0, "gen_len": 0.0}
    correct = np.mean([r["correct"] for r in records])
    reasoning = np.mean([r["reasoning"] for r in records])
    length = np.mean([r["len"] for r in records])
    return {
        "n": len(records),
        "accuracy": float(correct),
        "reasoning_rate": float(reasoning),
        "gen_len": float(length),
    }


def evaluate(model: TinyGPT, tok: Tokenizer, env: ArithmeticEnv,
             n_per_difficulty: int = 64, max_new_tokens: int = 40,
             temperature: float = 0.7, seed: int = 12345,
             mode: str = "auto", hybrid_cfg: HybridConfig = None) -> Dict:
    """Evaluate the policy, bucketed by difficulty.

    ``mode`` is one of:
      * ``"auto"``  -- let the trained policy gate itself (learned hybrid).
      * ``"fast"``  -- force a direct answer (no reasoning).
      * ``"think"`` -- force a reasoning trace.
      * ``"hybrid"``-- the confidence-gated controller from :mod:`rl_small.hybrid`.
    """
    rng = np.random.default_rng(seed)
    hybrid_cfg = hybrid_cfg or HybridConfig()
    per_diff = defaultdict(list)

    for difficulty in env.difficulties:
        for _ in range(n_per_difficulty):
            prob = env.sample(difficulty)
            if mode == "hybrid":
                d = dynamic_decode(model, tok, prob, hybrid_cfg, rng)
                gen = d.gen_ids
            else:
                force = None
                if mode == "fast":
                    force = tok.answer
                elif mode == "think":
                    force = tok.think
                gen = generate(model, tok, tok.encode(prob.prompt_tokens),
                               max_new_tokens, rng, temperature=temperature,
                               force_first=force,
                               force_after_think=(mode == "think"))
            per_diff[difficulty].append({
                "correct": float(env.is_correct(prob, gen)),
                "reasoning": float(tok.think in gen),
                "len": float(len(gen)),
            })

    result = {"by_difficulty": {d: _stats_from(v) for d, v in per_diff.items()}}
    allrec = [r for v in per_diff.values() for r in v]
    result["overall"] = _stats_from(allrec)
    return result


def format_eval(result: Dict) -> str:
    lines = []
    ov = result["overall"]
    lines.append(f"overall: acc={ov['accuracy']:.3f} "
                 f"reasoning={ov['reasoning_rate']:.3f} len={ov['gen_len']:.1f}")
    for d in sorted(result["by_difficulty"]):
        s = result["by_difficulty"][d]
        lines.append(f"  {d}-operand: acc={s['accuracy']:.3f} "
                     f"reasoning={s['reasoning_rate']:.3f} len={s['gen_len']:.1f}")
    return "\n".join(lines)
