#!/usr/bin/env python3
"""Show a trained policy solving individual problems, with its reasoning.

    python scripts/demo.py --ckpt runs/demo/policy.pkl

For each sampled problem it prints the prompt, the model's raw generation
(including any ``<think>`` trace), the confidence-gated controller's decision,
and whether the answer is correct.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rl_small.tokenizer import Tokenizer
from rl_small.env import ArithmeticEnv
from rl_small.model import TinyGPT
from rl_small.hybrid import HybridConfig, dynamic_decode, answer_confidence
from rl_small.sampling import generate
from rl_small.utils import load_params_into


def pretty(tok, ids):
    return tok.decode(ids)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ckpt", type=str, default="runs/demo/policy.pkl")
    p.add_argument("--n", type=int, default=8)
    p.add_argument("--block-size", type=int, default=64)
    p.add_argument("--d-model", type=int, default=64)
    p.add_argument("--n-head", type=int, default=4)
    p.add_argument("--n-layer", type=int, default=3)
    p.add_argument("--max-operand", type=int, default=20)
    p.add_argument("--difficulties", type=int, nargs="+", default=[2, 3, 4])
    p.add_argument("--seed", type=int, default=7)
    return p.parse_args()


def main():
    args = parse_args()
    tok = Tokenizer()
    env = ArithmeticEnv(tok, max_operand=args.max_operand,
                        difficulties=tuple(args.difficulties), seed=args.seed)
    model = TinyGPT(tok.vocab_size, block_size=args.block_size,
                    d_model=args.d_model, n_head=args.n_head,
                    n_layer=args.n_layer, seed=0)
    load_params_into(model, args.ckpt)
    rng = np.random.default_rng(args.seed)
    hcfg = HybridConfig()

    for _ in range(args.n):
        prob = env.sample()
        prompt_ids = tok.encode(prob.prompt_tokens)
        problem_str = pretty(tok, prompt_ids).replace("<bos>", "")

        auto = generate(model, tok, prompt_ids, 40, rng, temperature=0.7)
        conf = answer_confidence(model, tok, prompt_ids)
        dec = dynamic_decode(model, tok, prob, hcfg, rng)

        print("=" * 60)
        print(f"problem: {problem_str}{prob.answer}   (difficulty={prob.difficulty})")
        print(f"  auto  : {pretty(tok, auto)}")
        print(f"          correct={env.is_correct(prob, auto)} "
              f"reasoning={tok.think in auto}")
        print(f"  hybrid: conf={conf:.2f} -> mode={dec.mode} (budget={dec.budget})")
        print(f"          {pretty(tok, dec.gen_ids)}")
        print(f"          correct={env.is_correct(prob, dec.gen_ids)}")


if __name__ == "__main__":
    main()
