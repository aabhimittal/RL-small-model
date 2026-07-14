#!/usr/bin/env python3
"""Minimal end-to-end pure-RL run in ~40 lines. Trains, then shows samples.

    python examples/quickstart.py

Deliberately tiny (2-3 operand problems, small model, ~120 steps) so it finishes
in about a minute on a laptop CPU and clearly demonstrates the full loop:
sample -> reward -> group-relative advantage -> policy update.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rl_small import (Tokenizer, ArithmeticEnv, TinyGPT, RewardConfig,
                      GRPOTrainer, GRPOConfig)
from rl_small.evaluate import evaluate, format_eval
from rl_small.sampling import generate

tok = Tokenizer()
env = ArithmeticEnv(tok, max_operand=9, difficulties=(2,), seed=0)
model = TinyGPT(tok.vocab_size, block_size=40, d_model=48, n_head=4,
                n_layer=2, seed=0)
print(f"TinyGPT with {model.num_params():,} parameters")

# proximity_coef>0 turns on the dense reward that makes from-scratch correctness
# reliable and fast (see docs/04_reward_design.md).
reward_cfg = RewardConfig(proximity_coef=0.3, proximity_scale=8.0)
cfg = GRPOConfig(group_size=8, prompts_per_step=8, max_new_tokens=18,
                 lr=5e-3, ppo_epochs=2, entropy_coef=0.025)
trainer = GRPOTrainer(model, tok, env, cfg, reward_cfg, seed=0)

print("\nBefore training:")
print(format_eval(evaluate(model, tok, env, n_per_difficulty=48, mode="auto")))

print("\nTraining (pure RL, no labels)...")
for step in range(1, 161):
    s = trainer.step()
    if step % 20 == 0:
        print(f"  step {step:3d} | reward={s['reward']:.3f} "
              f"acc={s['accuracy']:.3f} reasoning={s['reasoning_rate']:.3f} "
              f"len={s['gen_len']:.1f}")

print("\nAfter training:")
print(format_eval(evaluate(model, tok, env, n_per_difficulty=48, mode="auto")))

print("\nSample generations:")
rng = np.random.default_rng(1)
for _ in range(5):
    p = env.sample()
    gen = generate(model, tok, tok.encode(p.prompt_tokens), 32, rng, temperature=0.6)
    prompt = tok.decode(tok.encode(p.prompt_tokens)).replace("<bos>", "")
    print(f"  {prompt}{p.answer}  ->  {tok.decode(gen)}  "
          f"[{'OK' if env.is_correct(p, gen) else 'x'}]")
