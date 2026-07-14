#!/usr/bin/env python3
"""Train a TinyGPT policy with pure RL (GRPO) on the arithmetic task.

Example
-------
    python scripts/train.py --steps 300 --out runs/demo

This runs entirely on CPU in a couple of minutes. Watch ``accuracy`` climb and
``reasoning_rate`` self-organize by difficulty as training proceeds.
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rl_small.tokenizer import Tokenizer
from rl_small.env import ArithmeticEnv
from rl_small.model import TinyGPT
from rl_small.rewards import RewardConfig
from rl_small.grpo import GRPOTrainer, GRPOConfig
from rl_small.evaluate import evaluate, format_eval
from rl_small.utils import set_seed, save_checkpoint, format_stats, append_jsonl


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=str, default="runs/demo")
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--eval-every", type=int, default=100)
    # model
    p.add_argument("--d-model", type=int, default=64)
    p.add_argument("--n-head", type=int, default=4)
    p.add_argument("--n-layer", type=int, default=3)
    p.add_argument("--block-size", type=int, default=64)
    # task
    p.add_argument("--max-operand", type=int, default=20)
    p.add_argument("--difficulties", type=int, nargs="+", default=[2, 3, 4])
    # grpo
    p.add_argument("--group-size", type=int, default=8)
    p.add_argument("--prompts-per-step", type=int, default=8)
    p.add_argument("--max-new-tokens", type=int, default=40)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--ppo-epochs", type=int, default=2)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--entropy-coef", type=float, default=0.03)
    # Dense proximity reward: partial credit for being close to the answer.
    # On by default because it makes from-scratch correctness reliable; set to
    # 0 for the purely sparse (correct-or-not) setting -- see docs/04.
    p.add_argument("--proximity-coef", type=float, default=0.3)
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    os.makedirs(args.out, exist_ok=True)
    log_path = os.path.join(args.out, "log.jsonl")

    tok = Tokenizer()
    env = ArithmeticEnv(tok, max_operand=args.max_operand,
                        difficulties=tuple(args.difficulties), seed=args.seed)
    model = TinyGPT(tok.vocab_size, block_size=args.block_size,
                    d_model=args.d_model, n_head=args.n_head,
                    n_layer=args.n_layer, seed=args.seed)
    print(f"TinyGPT: {model.num_params():,} parameters, vocab={tok.vocab_size}")

    cfg = GRPOConfig(
        group_size=args.group_size, prompts_per_step=args.prompts_per_step,
        max_new_tokens=args.max_new_tokens, temperature=args.temperature,
        lr=args.lr, ppo_epochs=args.ppo_epochs, entropy_coef=args.entropy_coef,
    )
    reward_cfg = RewardConfig(proximity_coef=args.proximity_coef)
    trainer = GRPOTrainer(model, tok, env, cfg, reward_cfg, seed=args.seed)

    for step in range(1, args.steps + 1):
        stats = trainer.step()
        if step % args.log_every == 0 or step == 1:
            print(format_stats(step, stats))
            append_jsonl(log_path, {"step": step, **stats})
        if step % args.eval_every == 0:
            res = evaluate(model, tok, env, n_per_difficulty=48, mode="auto")
            print("  [eval/auto]\n    " + format_eval(res).replace("\n", "\n    "))

    ckpt = os.path.join(args.out, "policy.pkl")
    save_checkpoint(model, ckpt, meta={"args": vars(args)})
    print(f"\nSaved checkpoint -> {ckpt}")

    print("\nFinal evaluation (learned auto-gating):")
    print(format_eval(evaluate(model, tok, env, n_per_difficulty=96, mode="auto")))


if __name__ == "__main__":
    main()
