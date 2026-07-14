# 7. Appendix: the autograd engine

Code: [`rl_small/autograd.py`](../rl_small/autograd.py) (~200 lines).

You don't need this to *use* the repo, but reading it once demystifies "what a
deep-learning framework actually does." Everything else — TinyGPT, GRPO, the loss
— is just arithmetic on top of it.

## 7.1 The idea: a tape of operations

Every [`Tensor`](../rl_small/autograd.py) wraps a NumPy array and remembers:

- `data` — the forward value.
- `_prev` — the tensors it was computed from.
- `_backward` — a closure that, given this tensor's gradient, adds the correct
  contribution to its parents' `.grad`.

As you compute, you build a **graph**. Calling `.backward()` on a scalar loss
does a reverse topological sort and runs each `_backward` once, from output back
to inputs. This is textbook reverse-mode automatic differentiation.

```python
z = (a * b + c).sum()
z.backward()          # fills a.grad, b.grad, c.grad
```

## 7.2 One operation, in full

Multiplication shows the whole pattern — forward value, plus a closure applying
the chain rule to each operand:

```python
def __mul__(self, other):
    out = Tensor(self.data * other.data, _children=(self, other))
    def _backward():
        self.grad  += _unbroadcast(out.grad * other.data, self.shape)
        other.grad += _unbroadcast(out.grad * self.data,  self.shape)
    out._backward = _backward
    return out
```

`d(a*b)/da = b`, so `a`'s gradient is `out.grad * b`. The `_unbroadcast` handles
NumPy broadcasting: if `a` was broadcast up to a bigger shape, we sum the incoming
gradient back down over the broadcast axes (the one genuinely fiddly detail).

## 7.3 What's implemented

- **Elementwise:** `+ - * /`, scalar `**`, `exp`, `log`, `tanh`, `relu`.
- **Linear algebra:** batched `matmul`, `sum`, `mean`, `reshape`, `transpose`.
- **Indexing:** `take_rows` (embedding lookup) and `gather_last` (pick a chosen
  token's log-prob) — both scatter gradients back with `np.add.at`.
- **Composites** built from the above: `log_softmax`, `layer_norm`, and (in
  `model.py`) attention softmax and GELU.

That set is enough for a full GPT and the entire GRPO loss.

## 7.4 Why you can trust it

`tests/test_autograd.py` **gradient-checks every primitive** against central
finite differences, and `tests/test_model.py` checks TinyGPT's gradients the same
way. If analytic and numerical gradients agree to `1e-5`, the backward pass is
correct — so every result the trainer produces rests on a verified foundation.

```bash
pytest tests/test_autograd.py -q     # 8 gradient checks
```

Back to the [README](../README.md).
