"""Tests for GRPO packing and that a short run improves reward.

This is the integration test: it runs a handful of real GRPO steps on a tiny
model and asserts the policy actually learns (reward goes up). It is the
end-to-end proof that the autograd engine, model, rewards, and trainer compose
correctly.
"""

import numpy as np

from rl_small.tokenizer import Tokenizer
from rl_small.env import ArithmeticEnv
from rl_small.model import TinyGPT
from rl_small.rewards import RewardConfig
from rl_small.grpo import GRPOTrainer, GRPOConfig, _pack
from rl_small.sampling import Rollout


def test_pack_mask_selects_generated_positions():
    tok = Tokenizer()
    r = Rollout(prompt_ids=[tok.bos, 7, 8, tok.stoi["="]],
                gen_ids=[tok.answer, 9, tok.end_answer, tok.eos])
    inp, tgt, mask = _pack([r], tok)
    # Generated tokens start at prompt_len=4; mask marks positions 3..6.
    assert mask.shape == inp.shape
    assert mask[0].sum() == 4
    assert mask[0, 3] == 1.0 and mask[0, 2] == 0.0


def test_grpo_improves_reward_on_easy_task():
    # End-to-end proof that autograd + model + rewards + trainer compose and the
    # policy actually learns. We use the dense proximity reward (the recommended
    # from-scratch setting) and a small answer range so it converges quickly.
    tok = Tokenizer()
    env = ArithmeticEnv(tok, max_operand=6, difficulties=(2,), seed=0)
    model = TinyGPT(tok.vocab_size, block_size=40, d_model=32, n_head=4,
                    n_layer=2, seed=0)
    cfg = GRPOConfig(group_size=8, prompts_per_step=6, max_new_tokens=16,
                     lr=5e-3, ppo_epochs=2, entropy_coef=0.025, difficulty=2)
    reward_cfg = RewardConfig(proximity_coef=0.3, proximity_scale=6.0)
    trainer = GRPOTrainer(model, tok, env, cfg, reward_cfg, seed=0)

    first = np.mean([trainer.step()["reward"] for _ in range(3)])
    for _ in range(40):
        trainer.step()
    last = np.mean([trainer.step()["reward"] for _ in range(3)])
    assert last > first + 0.15, f"reward did not improve: {first:.3f} -> {last:.3f}"
