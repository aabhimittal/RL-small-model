# 6. Scaling the ideas to real small models

The NumPy code here is for *understanding*. The exact same concepts scale to real
small language models (0.5B–3B params) with a GPU and a few mature libraries. This
doc maps our from-scratch pieces onto the production stack — nothing here is
required to run the repo.

## 6.1 The one-to-one mapping

| This repo (NumPy)                    | Real stack                                        |
|--------------------------------------|---------------------------------------------------|
| `TinyGPT`                            | `Qwen2.5-0.5B/1.5B`, `Llama-3.2-1B`, `SmolLM2`…    |
| `rl_small.autograd`                  | PyTorch autograd                                   |
| `GRPOTrainer`                        | TRL `GRPOTrainer`, `verl`, or OpenRLHF            |
| `ArithmeticEnv` verifier             | a reward function over GSM8K/MATH answers          |
| `compute_reward`                     | reward funcs (correctness + format + length)       |
| `<think>` / `<answer>` tokens        | the same tags, e.g. R1's `<think></think>` format  |
| `answer_confidence` / hybrid decode  | logit-based routing, budget forcing, self-consistency |

## 6.2 A minimal TRL GRPO sketch

```python
from datasets import load_dataset
from trl import GRPOConfig, GRPOTrainer

ds = load_dataset("openai/gsm8k", "main", split="train")

def reward_correct(completions, answer, **kwargs):
    # verify each completion against the gold answer -> 1.0 / 0.0
    return [1.0 if extract(c) == a else 0.0 for c, a in zip(completions, answer)]

def reward_format(completions, **kwargs):
    return [0.2 if well_formed(c) else 0.0 for c in completions]

trainer = GRPOTrainer(
    model="Qwen/Qwen2.5-1.5B-Instruct",
    reward_funcs=[reward_correct, reward_format],   # list == summed, like ours
    args=GRPOConfig(
        num_generations=8,          # == our group_size G
        max_completion_length=512,
        num_iterations=2,           # == our ppo_epochs
        beta=0.0,                   # KL coeff; 0 == R1-Zero pure RL
        temperature=1.0,
    ),
    train_dataset=ds,
)
trainer.train()
```

Every knob has a twin in [`GRPOConfig`](../rl_small/grpo.py) /
[`RewardConfig`](../rl_small/rewards.py). If you understand the small version you
can drive the big one.

## 6.3 What changes at scale (and what doesn't)

**Doesn't change:** the algorithm (group-relative advantage, clipped surrogate),
the reward philosophy (verifiable + shaping + length), and the emergence of
dynamic reasoning from reward pressure.

**Does change:**
- **Cold start.** Real models start from a pretrained base, not random weights, so
  they already speak fluent language and format. That makes the sparse
  correctness reward workable *without* our heavy structural shaping — DeepSeek-
  R1-Zero used almost none. Our graded shaping and optional dense proximity
  reward exist precisely because we start from *scratch* on a CPU.
- **KL to a reference.** From a good base you usually keep a small KL penalty
  (`beta>0`) to a frozen copy so the policy doesn't drift into degenerate text.
  From random init there is nothing worth staying near, so we default `kl_coef=0`
  and rely on the entropy bonus.
- **Throughput.** Real rollouts use batched KV-cached generation (vLLM/SGLang);
  we generate one sequence at a time for clarity.
- **Length control.** At scale you fight *overthinking* explicitly — length
  penalties, budget forcing, "think/no-think" system prompts. That is exactly the
  "tightening + dynamic hybrid reasoning" theme of this repo, just with more
  tokens at stake.

## 6.4 Further reading

- **DeepSeek-R1 / R1-Zero** — pure RL producing emergent reasoning.
- **GRPO** (DeepSeekMath) — the algorithm, derived in [doc 2](02_pure_rl_grpo.md).
- **TRL GRPOTrainer docs** — the production API mirrored above.
- Adaptive / hybrid reasoning and test-time compute — the theme of
  [doc 3](03_dynamic_hybrid_reasoning.md).

Back to the [README](../README.md).
