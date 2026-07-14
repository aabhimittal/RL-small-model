# 3. Dynamic hybrid reasoning

Code: [`rl_small/hybrid.py`](../rl_small/hybrid.py).

"Hybrid reasoning" = one model, two modes:

- **fast (System 1):** `<answer> 12 </answer>` — answer directly.
- **slow (System 2):** `<think> 3+4=7; 7+5=12 </think> <answer> 12 </answer>` —
  reason, then answer.

"**Dynamic**" = the mode is chosen *per problem*. We implement it two ways that
stack.

## 3.1 Learned gating (it emerges)

The policy's **first generated token** is the gate: `<think>` opens a reasoning
trace, `<answer>` commits to a direct answer. Nothing in the code forces the
right choice. Instead the reward ([doc 4](04_reward_design.md)) makes reasoning
*worth it* on hard problems (where a direct guess is usually wrong) and *not
worth it* on easy ones (where the length penalty favors a short direct answer).

Because GRPO compares completions on the *same* prompt, it directly experiences
"on this hard problem, the `<think>` completions were the ones that got the
reward" and reinforces the gate accordingly. The fast/slow split therefore
**emerges** and lines up with difficulty. Evaluate it with `mode="auto"` in
[`rl_small/evaluate.py`](../rl_small/evaluate.py) and watch `reasoning_rate` rise
with the operand count.

## 3.2 Confidence-gated control (you can steer it)

Learned gating depends on the policy being good. On top of it — or on a weak
policy — we add an explicit, interpretable controller:

```
confidence = P(model's top first-answer-digit | prompt + "<answer>")
if confidence >= threshold:   answer directly           (cheap)
else:                         reason, with a token budget ∝ (1 − confidence)
```

- `answer_confidence` peeks at the model's distribution over the *first answer
  digit* if it were forced to answer now. High mass on one digit ⇒ it "knows" ⇒
  don't waste tokens. Low/flat ⇒ it's guessing ⇒ let it think.
- `estimated_think_budget` scales the reasoning budget with uncertainty: the less
  sure it is, the more tokens it may spend. This is the **dynamic compute** knob —
  test-time compute allocated where it helps.

This is the classic "adaptive computation / test-time scaling" idea in miniature:
spend more thinking on harder inputs, decided from the model's own signal, with
no label peeking.

## 3.3 Why "tightening"?

Three forces converge on tighter behavior:

1. **Correctness pressure** (RL) makes answers right.
2. **Format shaping** removes rambling / non-termination.
3. **Length penalty + hybrid gating** cut needless reasoning on easy inputs while
   *preserving* it where it's load-bearing.

The result is a small model that is right more often, structured, and no more
verbose than the problem demands — without scaling parameters.

## 3.4 Knobs

See [`HybridConfig`](../rl_small/hybrid.py): `confidence_threshold`,
`fast_budget`, `base_think_budget`, `max_think_budget`, `temperature`. Raising the
threshold makes the controller reason more (safer, slower); lowering it answers
faster (cheaper, riskier). `scripts/evaluate.py` prints the accuracy/length
trade-off across `fast`, `think`, `auto`, and `hybrid` so you can pick a point.

Next: [4. Reward design](04_reward_design.md).
