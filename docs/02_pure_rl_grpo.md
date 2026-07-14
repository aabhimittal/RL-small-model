# 2. Pure RL with GRPO

GRPO (**Group Relative Policy Optimization**) is the policy-gradient algorithm we
use for pure RL. This doc derives it from scratch. Code:
[`rl_small/grpo.py`](../rl_small/grpo.py).

## 2.1 The policy-gradient starting point

We want to maximize expected reward `J(θ) = E[R(τ)]` over completions `τ` sampled
from the policy `πθ`. The REINFORCE gradient is

```
∇θ J = E[ R(τ) · ∇θ log πθ(τ) ]
```

Intuition: nudge the log-probability of each completion up in proportion to its
reward. High-reward completions become more likely; low-reward ones less.

Raw REINFORCE has enormous variance because `R(τ)` can be large and is always
positive-ish. The standard fix is a **baseline** `b`: subtracting it doesn't bias
the gradient but slashes variance.

```
∇θ J = E[ (R(τ) − b) · ∇θ log πθ(τ) ]
```

The quantity `A = R(τ) − b` is the **advantage**: "how much better than baseline
was this completion?"

## 2.2 GRPO's idea: the group *is* the baseline

PPO learns a separate **value network** to predict `b`. That is another model to
build, tune, and store. GRPO's insight: for a fixed prompt, sample a **group** of
`G` completions and use the group's own mean reward as the baseline. Normalizing
by the group's standard deviation gives a clean, scale-free advantage:

```
        r_i − mean(r_1..r_G)
A_i  =  --------------------          (same value applied to every token of completion i)
         std(r_1..r_G) + eps
```

No critic. No reward model. Just relative comparison inside a group. A completion
that beats its peers on the *same* prompt gets a positive advantage and is
reinforced; a below-average one is pushed down. This is exactly
[`GRPOTrainer.step`](../rl_small/grpo.py):

```python
baseline = rewards.mean()
adv = (rewards - baseline) / (rewards.std() + 1e-4)
```

**Consequence to remember:** if every completion in a group gets the *same*
reward, the advantages are all zero and that group teaches nothing. Variance
within the group is the fuel. This is why reward shaping and exploration
(temperature, entropy bonus, group size) matter so much — see
[doc 4](04_reward_design.md).

## 2.3 The clipped surrogate (PPO objective)

To reuse each batch of rollouts for several gradient steps without the policy
running away, GRPO borrows PPO's clipped objective. Let
`ratio = πθ_new(a|s) / πθ_old(a|s)`:

```
L_clip = min( ratio · A,  clip(ratio, 1−ε, 1+ε) · A )
```

The `min` + `clip` prevents any single update from moving a token's probability
too far from the behavior policy that generated the data. With a **single** inner
epoch, `ratio ≡ 1` and this reduces to REINFORCE-with-group-baseline — the
simplest honest form of pure RL. With several inner epochs the clip becomes
active and improves sample efficiency. See `ppo_epochs` and `clip_eps` in
[`GRPOConfig`](../rl_small/grpo.py).

## 2.4 The full loss

```
loss = −mean(L_clip)                # maximize advantage-weighted log-probs
       − β_ent · entropy            # keep exploring; fight premature collapse
       + β_kl  · KL(πθ ‖ π_ref)     # optional: stay near a reference policy
```

- The **entropy bonus** keeps the policy stochastic so the group keeps producing
  variety (without it, small models collapse to one output and learning stalls).
- The **KL term** is optional here (`kl_coef=0` by default = R1-Zero style). When
  you start from a capable base model you add it to avoid drifting into gibberish;
  from random init there is nothing to stay near, so we lean on the entropy bonus
  instead. We use the low-variance *k3* estimator `exp(Δ) − Δ − 1`.

## 2.5 The token-level bookkeeping

A subtle but important detail is *which* tokens get the gradient. We only score
the **generated** tokens, not the prompt. [`_pack`](../rl_small/grpo.py) builds a
mask that is 1 exactly on positions predicting a generated token, and the loss is
averaged over masked tokens only. Right-padding shorter completions is safe
because causal attention never lets a real token attend to a later pad token.

## 2.6 Why GRPO suits *small* models on a *CPU*

- No value network ⇒ half the parameters to train and no critic-tuning headaches.
- The signal is a single scalar per completion ⇒ works with any verifier.
- Group sampling parallelizes trivially and needs no replay buffer.

Next: [3. Dynamic hybrid reasoning](03_dynamic_hybrid_reasoning.md).
