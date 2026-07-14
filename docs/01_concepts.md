# 1. Core concepts

This project is built around four ideas. Read this once and the rest of the code
will read like prose.

## 1.1 A language model *is* a policy

A decoder-only Transformer maps a sequence of tokens to a probability
distribution over the next token. In reinforcement-learning language:

| RL term        | Language-model equivalent                                  |
|----------------|------------------------------------------------------------|
| policy πθ      | the model — it defines P(next token \| context)            |
| state          | the tokens generated so far (prompt + partial completion)  |
| action         | the next token emitted                                     |
| trajectory     | a full completion, sampled token by token                  |
| reward         | a score for the *finished* completion (e.g. "is it right?")|
| return         | here, just the final reward (no discounting mid-sequence)  |

So "training a model with RL" means: sample completions, score them, and shift
the policy's probabilities toward the high-scoring ones. That is literally what
[`GRPOTrainer.step`](../rl_small/grpo.py) does.

## 1.2 Pure RL (vs. SFT-then-RL)

Most instruction-tuned models are made in two stages: **supervised fine-tuning
(SFT)** on human demonstrations, then RL on top. **Pure RL** skips the
demonstrations entirely — the model learns *only* from a reward signal. This is
the DeepSeek-R1-Zero recipe: no answer keys to imitate, just a verifier that says
"correct / not correct."

Pure RL is attractive because it needs no labeled reasoning traces (which are
expensive) and it lets the model discover its *own* problem-solving style rather
than copying ours. The cost is a harder optimization problem — the reward is
sparse, so exploration matters a lot (see [doc 4](04_reward_design.md)).

In this repo the TinyGPT policy starts from **random weights** and is trained
purely by GRPO. There is no SFT step anywhere in the training path.

## 1.3 Verifiable rewards

RL needs a reward. The cleanest reward is one you can *compute*: for arithmetic,
just evaluate the expression and check the model's answer. No human raters, no
learned reward model, no reward hacking of a fuzzy proxy (mostly — see the
cautionary tale in [doc 4](04_reward_design.md)). Verifiable-reward RL is the
setting where pure RL is most reliable, which is why math and code are the
canonical domains for it.

## 1.4 Dynamic hybrid reasoning

Small models have a characteristic failure mode: they either **overthink** easy
questions (rambling, wasting tokens, sometimes talking themselves out of a
correct answer) or **underthink** hard ones (blurting a wrong answer with no
working). "Hybrid reasoning" gives one model two gears:

- **fast / System 1** — answer directly.
- **slow / System 2** — write an explicit `<think>` trace, then answer.

"**Dynamic**" means picking the gear *per problem*. We get this two ways:

1. **Learned gating** — GRPO + the reward in [doc 4](04_reward_design.md) make
   the policy choose the gear itself (via its first token). No rule tells it
   when to reason; the behavior emerges because reasoning pays off on hard
   problems and costs a little on easy ones.
2. **Confidence-gated control** — a decode-time wrapper
   ([`rl_small/hybrid.py`](../rl_small/hybrid.py)) that measures the model's
   confidence in a direct answer and only spends reasoning tokens when it's
   unsure, scaling the reasoning budget to the uncertainty.

"Tightening behavior" is the umbrella goal: make a small model correct,
well-structured, appropriately concise, and reliable — using RL pressure plus
the hybrid gear instead of a bigger model.

Next: [2. Pure RL with GRPO](02_pure_rl_grpo.md).
