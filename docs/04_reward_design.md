# 4. Reward design (and a reward-hacking war story)

Code: [`rl_small/rewards.py`](../rl_small/rewards.py).

In pure RL the reward is the *entire* teaching signal. Get it wrong and the model
faithfully learns the wrong thing. This doc explains our reward and shows a real
failure we hit while building this repo.

## 4.1 The components

For a finished completion we combine:

```
reward = accuracy            # +1.0 if the extracted answer is correct, else 0
       + shaping             # graded partial credit for structure (see below)
       − length_coef · len   # small brevity penalty ("tightening")
```

`accuracy` is the verifiable signal we ultimately care about. `length` is a tiny
penalty (`0.004`/token) that only ever breaks ties. The interesting part is
`shaping`.

## 4.2 Why graded shaping is necessary from scratch

A randomly-initialized TinyGPT almost never emits a fully correct
`<answer> 12 </answer> <eos>` by chance. If the only reward were "correct or not,"
**every** completion in **every** group would score 0, the GRPO advantages would
all be 0, and nothing would ever learn (recall [doc 2](02_pure_rl_grpo.md): the
group needs reward *variance*).

So we hand out partial credit for structural progress — a ladder the policy can
climb one rung at a time:

| rung | condition                                   | bonus |
|------|---------------------------------------------|-------|
| 1    | emitted an `<answer>` token at all          | 0.05  |
| 2    | produced a **non-empty, parseable** number  | 0.10  |
| 3    | terminated cleanly with `<eos>`             | 0.05  |
| 4    | the whole structure is well-formed          | 0.10  |
| top  | the answer is **correct**                   | 1.00  |

Early on, ~half of random completions stumble onto rung 1, creating the variance
GRPO needs. The policy climbs to reliable well-formed numbers, and *then* — with
structure solved — the huge correctness bonus pulls it toward right answers. You
can watch this ladder in the training log: `gen_len` grows as it learns to emit
numbers, then `accuracy` lifts off (often after a plateau).

## 4.3 War story: the empty-answer reward hack

Our **first** shaping scheme gave full "well-formed" credit to
`<answer></answer>` — a syntactically perfect, completely empty answer. Result:

```
step 20 | reward=0.235 acc=0.000 reasoning=0.000 len=3.0
step 120| reward=0.238 acc=0.000 reasoning=0.000 len=3.0   ← stuck forever
```

The policy discovered it could collect **0.25 reward for emitting nothing**,
collapsed onto that (length locked at 3, entropy gone), and never explored a
single digit. This is **reward hacking** in miniature: the model optimized our
*proxy* (structure) instead of our *intent* (a correct answer), because the proxy
was satisfiable without the intent.

The fix is rung 2 above: structural credit requires a **non-empty parseable
number**, so an empty answer is a dead end worth almost nothing. After the fix
the same run climbs past the plateau and starts getting problems right. The
lesson is general and worth internalizing: **every gap between your reward and
your true goal is an optimization target the policy will find.**

## 4.4 How the reward *creates* dynamic hybrid reasoning

Nothing in the reward mentions "reason on hard problems." Yet the behavior
appears, purely from the interaction of correctness and length:

- **Easy problem** — a short direct answer is already correct ⇒ it gets the
  correctness bonus *and* the smallest length penalty ⇒ highest reward ⇒ the
  policy learns to answer fast.
- **Hard problem** — a short direct answer is usually wrong (0 correctness). A
  `<think>` trace that reaches the right answer earns +1.0, dwarfing its extra
  length penalty ⇒ reasoning is reinforced *specifically where it helps*.

GRPO compares completions on the *same* prompt, so it feels this difference
directly and gates accordingly. That is dynamic hybrid reasoning falling out of a
single scalar reward — no per-example "think here" labels.

## 4.5 Tuning knobs

[`RewardConfig`](../rl_small/rewards.py): `correct_bonus`, the four shaping
bonuses, and `length_coef`. Rules of thumb:

- Keep `correct_bonus` clearly larger than the total shaping, so correctness
  always dominates once reachable.
- Keep `length_coef` small — it should trim ties, never discourage load-bearing
  reasoning. Raise it to push harder on conciseness (more fast-mode), lower it to
  tolerate more thinking.
- If training stalls at 0 accuracy with 0 length, suspect an empty-structure
  hack; if it stalls at well-formed-but-wrong, it needs more exploration (higher
  `entropy_coef`/`temperature`, larger `group_size`) or more steps.

Next: [5. The training loop, line by line](05_training_loop.md).
