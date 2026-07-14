"""GRPO -- Group Relative Policy Optimization (the pure-RL trainer).

GRPO is the algorithm behind DeepSeek-R1-Zero's "pure RL" recipe. It is a
policy-gradient method with one elegant simplification: **it has no value
network**. Ordinary PPO trains a separate critic to estimate a baseline for the
advantage. GRPO instead samples a *group* of ``G`` completions for the same
prompt and uses the group's own mean reward as the baseline:

    advantage_i = (reward_i - mean(rewards)) / (std(rewards) + eps)

Completions that beat their group's average get a positive advantage (made more
likely); below-average ones get pushed down. That's it -- no critic, no reward
model, just a verifiable reward and relative comparison within a group.

The update is the clipped PPO surrogate applied to that group-normalized
advantage, optionally with a KL penalty to a frozen reference policy:

    ratio  = exp(logp_new - logp_old)
    L_clip = min(ratio * A, clip(ratio, 1-eps, 1+eps) * A)
    loss   = -mean(L_clip) - beta_ent * entropy + beta_kl * KL(pi || pi_ref)

With a single inner epoch the ratio is 1 and this reduces to REINFORCE with a
group baseline -- the simplest faithful form of "pure RL". Multiple inner epochs
make the clip active and improve sample efficiency.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from .autograd import Tensor, log_softmax
from .env import ArithmeticEnv
from .model import TinyGPT
from .optim import Adam
from .rewards import RewardConfig, compute_reward
from .sampling import Rollout, sample_group
from .tokenizer import Tokenizer


@dataclass
class GRPOConfig:
    group_size: int = 8               # completions sampled per prompt (G)
    prompts_per_step: int = 8         # distinct prompts per update
    max_new_tokens: int = 40
    temperature: float = 1.0
    lr: float = 3e-3
    ppo_epochs: int = 2               # inner optimization passes over the batch
    clip_eps: float = 0.2
    entropy_coef: float = 0.03
    kl_coef: float = 0.0              # KL to frozen reference (0 = pure R1-Zero style)
    grad_clip: float = 1.0
    difficulty: Optional[int] = None  # None => mix of difficulties


def _pack(rollouts: List[Rollout], tok: Tokenizer):
    """Right-pad rollouts into batched arrays for a teacher-forced forward pass.

    Right-padding is safe: causal attention means a real token never attends to
    a later pad token, so the padded columns cannot corrupt earlier logits. We
    return the input ids (all but last token), the targets (all but first), and
    a mask selecting the positions that predict a *generated* token.
    """
    fulls = [r.full_ids for r in rollouts]
    L = max(len(f) for f in fulls)
    B = len(fulls)
    inp = np.full((B, L - 1), tok.pad, dtype=np.int64)
    tgt = np.full((B, L - 1), tok.pad, dtype=np.int64)
    mask = np.zeros((B, L - 1), dtype=np.float64)
    for b, r in enumerate(rollouts):
        f = r.full_ids
        inp[b, : len(f) - 1] = f[:-1]
        tgt[b, : len(f) - 1] = f[1:]
        p, g = r.prompt_len, len(r.gen_ids)
        # positions t (into the length L-1 arrays) that predict generated tokens
        mask[b, p - 1 : p - 1 + g] = 1.0
    return inp, tgt, mask


def _token_logprobs(model: TinyGPT, inp, tgt):
    """Per-token log-probabilities of ``tgt`` under the policy: shape (B, L-1)."""
    logits = model(inp)                      # (B, L-1, V)
    logp = log_softmax(logits)               # (B, L-1, V)
    return logp.gather_last(tgt), logits     # (B, L-1)


def _entropy(logits: Tensor) -> Tensor:
    """Mean token entropy of the policy distribution (for the entropy bonus)."""
    logp = log_softmax(logits)
    p = logp.exp()
    return -(p * logp).sum(axis=-1)          # (B, L-1)


class GRPOTrainer:
    def __init__(self, model: TinyGPT, tok: Tokenizer, env: ArithmeticEnv,
                 cfg: GRPOConfig, reward_cfg: RewardConfig, seed: int = 0,
                 reference: Optional[TinyGPT] = None):
        self.model = model
        self.tok = tok
        self.env = env
        self.cfg = cfg
        self.reward_cfg = reward_cfg
        self.reference = reference
        self.rng = np.random.default_rng(seed)
        self.opt = Adam(model.parameters(), lr=cfg.lr, grad_clip=cfg.grad_clip)

    # -- one full GRPO step ---------------------------------------------------
    def step(self) -> dict:
        cfg = self.cfg
        problems = self.env.sample_batch(cfg.prompts_per_step, cfg.difficulty)

        all_rollouts: List[Rollout] = []
        advantages: List[float] = []
        for prob in problems:
            group = sample_group(self.model, self.tok, prob, cfg.group_size,
                                 cfg.max_new_tokens, self.rng,
                                 temperature=cfg.temperature)
            rewards = []
            for r in group:
                total, parts = compute_reward(self.env, r, self.reward_cfg)
                r.reward = total
                r.reward_parts = parts
                rewards.append(total)
            rewards = np.asarray(rewards)
            # Group-relative advantage: the defining trick of GRPO.
            baseline = rewards.mean()
            std = rewards.std()
            adv = (rewards - baseline) / (std + 1e-4)
            for r, a in zip(group, adv):
                r.advantage = float(a)
            all_rollouts.extend(group)
            advantages.extend(adv.tolist())

        adv_arr = np.asarray(advantages)
        stats = self._log_stats(all_rollouts)
        # Note: even if every group tied (advantages ~0), we still run the update
        # so the *entropy bonus* keeps injecting exploration. Skipping here was a
        # bug -- it let a collapsed policy (e.g. "always emit <eos>") get stuck
        # forever, because the one force that could revive exploration was gated
        # behind a nonzero advantage that a collapsed policy can never produce.
        stats["degenerate_batch"] = float(np.allclose(adv_arr, 0.0))

        inp, tgt, mask = _pack(all_rollouts, self.tok)
        adv_col = adv_arr[:, None]                       # (B, 1) broadcast over tokens
        mask_t = Tensor(mask, requires_grad=False)
        tok_count = float(mask.sum())

        # Old (behavior) log-probs: fixed snapshot for the PPO ratio.
        old_logp, _ = _token_logprobs(self.model, inp, tgt)
        old_logp_data = old_logp.data.copy()

        ref_logp_data = None
        if self.reference is not None and cfg.kl_coef > 0:
            ref_logp, _ = _token_logprobs(self.reference, inp, tgt)
            ref_logp_data = ref_logp.data.copy()

        last_loss = 0.0
        for _ in range(cfg.ppo_epochs):
            self.opt.zero_grad()
            logp, logits = _token_logprobs(self.model, inp, tgt)
            ratio = (logp - Tensor(old_logp_data, requires_grad=False)).exp()
            unclipped = ratio * adv_col
            clipped = _clip(ratio, 1 - cfg.clip_eps, 1 + cfg.clip_eps) * adv_col
            surrogate = _elementwise_min(unclipped, clipped)
            pg_loss = -(surrogate * mask_t).sum() * (1.0 / tok_count)

            ent = (_entropy(logits) * mask_t).sum() * (1.0 / tok_count)
            loss = pg_loss - ent * cfg.entropy_coef

            if ref_logp_data is not None:
                # Unbiased k3 KL estimator: KL ~= exp(dref) - dref - 1, dref = ref - new.
                dref = Tensor(ref_logp_data, requires_grad=False) - logp
                kl = ((dref.exp() - dref - 1.0) * mask_t).sum() * (1.0 / tok_count)
                loss = loss + kl * cfg.kl_coef

            loss.backward()
            self.opt.step()
            last_loss = float(loss.data)

        stats["loss"] = last_loss
        return stats

    # -- metrics --------------------------------------------------------------
    def _log_stats(self, rollouts: List[Rollout]) -> dict:
        correct = np.mean([r.reward_parts["correct"] for r in rollouts])
        used = np.mean([r.reward_parts["used_reasoning"] for r in rollouts])
        length = np.mean([r.reward_parts["gen_len"] for r in rollouts])
        reward = np.mean([r.reward for r in rollouts])
        return {
            "reward": float(reward),
            "accuracy": float(correct),
            "reasoning_rate": float(used),
            "gen_len": float(length),
        }


# -- small autograd helpers used only by GRPO --------------------------------
def _clip(x: Tensor, lo: float, hi: float) -> Tensor:
    """Clamp with straight-through gradient in the active region."""
    out = Tensor(np.clip(x.data, lo, hi), _children=(x,))
    passthrough = (x.data >= lo) & (x.data <= hi)

    def _backward():
        if x.requires_grad:
            x.grad += out.grad * passthrough

    out._backward = _backward
    return out


def _elementwise_min(a: Tensor, b: Tensor) -> Tensor:
    out = Tensor(np.minimum(a.data, b.data), _children=(a, b))
    a_smaller = a.data <= b.data

    def _backward():
        if a.requires_grad:
            a.grad += out.grad * a_smaller
        if b.requires_grad:
            b.grad += out.grad * (~a_smaller)

    out._backward = _backward
    return out
