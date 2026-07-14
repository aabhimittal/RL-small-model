#!/usr/bin/env python3
"""Evaluate a trained policy under every decoding mode and compare them.

    python scripts/evaluate.py --ckpt runs/demo/policy.pkl

Prints accuracy / reasoning-rate / length by difficulty for the learned
auto-gating policy, forced-fast, forced-think, and the confidence-gated hybrid
controller -- so you can see the compute/accuracy trade-off directly.
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rl_small.tokenizer import Tokenizer
from rl_small.env import ArithmeticEnv
from rl_small.model import TinyGPT
from rl_small.hybrid import HybridConfig
from rl_small.evaluate import evaluate, format_eval
from rl_small.utils import load_params_into


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ckpt", type=str, default="runs/demo/policy.pkl")
    p.add_argument("--n", type=int, default=96)
    p.add_argument("--block-size", type=int, default=64)
    p.add_argument("--d-model", type=int, default=64)
    p.add_argument("--n-head", type=int, default=4)
    p.add_argument("--n-layer", type=int, default=3)
    p.add_argument("--max-operand", type=int, default=20)
    p.add_argument("--difficulties", type=int, nargs="+", default=[2, 3, 4])
    p.add_argument("--confidence-threshold", type=float, default=0.5)
    return p.parse_args()


def main():
    args = parse_args()
    tok = Tokenizer()
    env = ArithmeticEnv(tok, max_operand=args.max_operand,
                        difficulties=tuple(args.difficulties), seed=999)
    model = TinyGPT(tok.vocab_size, block_size=args.block_size,
                    d_model=args.d_model, n_head=args.n_head,
                    n_layer=args.n_layer, seed=0)
    load_params_into(model, args.ckpt)

    hcfg = HybridConfig(confidence_threshold=args.confidence_threshold)
    for mode in ["auto", "fast", "think", "hybrid"]:
        res = evaluate(model, tok, env, n_per_difficulty=args.n, mode=mode,
                       hybrid_cfg=hcfg)
        print(f"\n=== mode={mode} ===")
        print(format_eval(res))


if __name__ == "__main__":
    main()
