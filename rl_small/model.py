"""TinyGPT: a minimal decoder-only Transformer used as the RL policy.

The architecture is the standard GPT block (causal self-attention + MLP with
pre-LayerNorm and residual connections), just scaled down to a handful of
dimensions so it trains in seconds on a CPU. It is the *policy* network: given a
token sequence it outputs a distribution over the next token, and RL adjusts
those distributions to maximize reward.
"""

from __future__ import annotations

import numpy as np

from .autograd import Tensor
from .nn import Module, Linear, Embedding, LayerNorm, gelu


class CausalSelfAttention(Module):
    def __init__(self, d_model, n_head, rng):
        assert d_model % n_head == 0
        self.n_head = n_head
        self.d_head = d_model // n_head
        self.qkv = Linear(d_model, 3 * d_model, rng)
        self.proj = Linear(d_model, d_model, rng)

    def forward(self, x: Tensor) -> Tensor:
        B, T, C = x.shape
        qkv = self.qkv(x)  # (B, T, 3C)
        # Reshape to separate the q/k/v groups and the attention heads, then
        # move the q/k/v axis to the front so we can pick each one out.
        qkv_r = qkv.reshape(B, T, 3, self.n_head, self.d_head)
        # Move to (3, B, n_head, T, d_head) so we can pick q/k/v.
        qkv_t = qkv_r.transpose((2, 0, 3, 1, 4))
        q = _index0(qkv_t, 0)
        k = _index0(qkv_t, 1)
        v = _index0(qkv_t, 2)  # each (B, n_head, T, d_head)

        scale = 1.0 / np.sqrt(self.d_head)
        att = (q @ k.swapaxes(-1, -2)) * scale  # (B, n_head, T, T)
        att = att + _causal_mask(T)
        att = _softmax_lastdim(att)
        out = att @ v  # (B, n_head, T, d_head)
        out = out.transpose((0, 2, 1, 3)).reshape(B, T, C)
        return self.proj(out)


def _index0(x: Tensor, i: int) -> Tensor:
    """Select index ``i`` along axis 0 of a Tensor, keeping gradients."""
    out = Tensor(x.data[i], _children=(x,))

    def _backward():
        if x.requires_grad:
            g = np.zeros_like(x.data)
            g[i] = out.grad
            x.grad += g

    out._backward = _backward
    return out


def _causal_mask(T: int) -> Tensor:
    mask = np.triu(np.full((T, T), -1e9), k=1)
    return Tensor(mask, requires_grad=False)


def _softmax_lastdim(x: Tensor) -> Tensor:
    m = x.data.max(axis=-1, keepdims=True)
    e = (x + Tensor(-m, requires_grad=False)).exp()
    return e / e.sum(axis=-1, keepdims=True)


class MLP(Module):
    def __init__(self, d_model, mult, rng):
        self.fc = Linear(d_model, mult * d_model, rng)
        self.proj = Linear(mult * d_model, d_model, rng)

    def forward(self, x):
        return self.proj(gelu(self.fc(x)))


class Block(Module):
    def __init__(self, d_model, n_head, mult, rng):
        self.ln1 = LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_head, rng)
        self.ln2 = LayerNorm(d_model)
        self.mlp = MLP(d_model, mult, rng)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class TinyGPT(Module):
    """A small GPT. ``forward`` returns logits of shape (B, T, vocab)."""

    def __init__(self, vocab_size, block_size, d_model=64, n_head=4,
                 n_layer=2, mult=4, seed=0):
        rng = np.random.default_rng(seed)
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.tok_emb = Embedding(vocab_size, d_model, rng)
        self.pos_emb = Embedding(block_size, d_model, rng)
        self.blocks = [Block(d_model, n_head, mult, rng) for _ in range(n_layer)]
        self.ln_f = LayerNorm(d_model)
        self.head = Linear(d_model, vocab_size, rng, bias=False)

    def forward(self, idx: np.ndarray) -> Tensor:
        idx = np.asarray(idx)
        B, T = idx.shape
        assert T <= self.block_size, f"sequence length {T} exceeds block size"
        pos = np.arange(T)
        x = self.tok_emb(idx) + self.pos_emb(pos)  # (B, T, d_model) via broadcast
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        return self.head(x)

    def num_params(self) -> int:
        return int(sum(p.data.size for p in self.parameters()))
