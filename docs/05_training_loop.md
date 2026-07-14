# 5. The training loop, line by line

This walks through one call to [`GRPOTrainer.step`](../rl_small/grpo.py) — a
single pure-RL update. Keep the file open beside this.

## Step 0 — sample a batch of prompts

```python
problems = self.env.sample_batch(cfg.prompts_per_step, cfg.difficulty)
```

Each problem is an arithmetic expression plus its verified answer
([`ArithmeticEnv`](../rl_small/env.py)). These are the "states" we roll out from.

## Step 1 — roll out a *group* per prompt

```python
for prob in problems:
    group = sample_group(model, tok, prob, cfg.group_size, ...)
```

For each prompt we sample `group_size` completions from the current policy
([`sampling.py`](../rl_small/sampling.py)), decoding token by token until `<eos>`
or the length cap. These are the actions/trajectories.

## Step 2 — score every completion

```python
total, parts = compute_reward(self.env, r, self.reward_cfg)
```

The verifier checks correctness; the reward adds structure shaping and a length
penalty ([doc 4](04_reward_design.md)). Each completion gets one scalar reward.

## Step 3 — group-relative advantages (the GRPO heart)

```python
adv = (rewards - rewards.mean()) / (rewards.std() + 1e-4)
```

Within each prompt's group, center by the mean and scale by the std. Above-average
completions get positive advantage; below-average, negative. **No value network.**
If a group is a tie (all equal), its advantages are ~0 and it contributes nothing
— which is why we skip the update when *all* advantages vanish.

## Step 4 — pack sequences and snapshot old log-probs

```python
inp, tgt, mask = _pack(all_rollouts, self.tok)
old_logp_data = _token_logprobs(self.model, inp, tgt)[0].data.copy()
```

`_pack` right-pads completions into arrays and builds a mask over the *generated*
tokens. `old_logp` is the behavior policy's log-probs — the fixed denominator of
the PPO ratio.

## Step 5 — inner PPO epochs

```python
for _ in range(cfg.ppo_epochs):
    logp, logits = _token_logprobs(model, inp, tgt)
    ratio     = (logp - old_logp).exp()
    surrogate = min(ratio * adv, clip(ratio, 1-ε, 1+ε) * adv)
    loss = -(surrogate * mask).mean_over_tokens
           - entropy_coef * entropy
           + kl_coef * KL_to_reference
    loss.backward();  optimizer.step()
```

- `ratio` is 1 on the first epoch (new == old); the clip bites only as the policy
  moves within the batch.
- The **mask** ensures only generated tokens get gradient; prompt tokens don't.
- `loss.backward()` is our tiny autograd engine ([doc appendix](07_autograd_appendix.md));
  `optimizer.step()` is Adam with global-norm gradient clipping.

## Step 6 — log and repeat

`stats` reports `reward`, `accuracy`, `reasoning_rate`, and `gen_len`. Over many
steps you'll see the [doc 4](04_reward_design.md) ladder: structure first (length
grows into well-formed answers), then accuracy lifts off, then — if the task has a
difficulty spread — `reasoning_rate` self-sorts by difficulty.

## The whole thing in ten lines

```python
for step in range(num_steps):
    for prompt in sample_prompts():
        group   = [policy.sample(prompt) for _ in range(G)]
        rewards = [verify_and_shape(c) for c in group]
        adv     = (rewards - mean(rewards)) / (std(rewards) + eps)
    loss = -(clipped_surrogate(adv) ).mean() - ent*H + kl*KL
    loss.backward(); adam.step()
```

That is pure RL for a language model. Everything else in this repo is making those
ten lines correct, observable, and teachable.

Next: [6. Scaling to real models](06_scaling_to_real_models.md).
