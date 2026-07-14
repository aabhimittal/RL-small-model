"""A tiny token-level vocabulary for the arithmetic-reasoning task.

The "language" the policy speaks is deliberately small: digits, the ``+``/``-``
operators, an equals sign, a step separator, and the structural tokens that
delimit reasoning (``<think>``) and the final answer (``<answer>``). Keeping the
vocabulary tiny is what makes pure-RL-from-scratch tractable on a CPU.
"""

from __future__ import annotations

from typing import List

SPECIALS = ["<pad>", "<bos>", "<eos>", "<think>", "</think>", "<answer>", "</answer>"]
SYMBOLS = [str(d) for d in range(10)] + ["+", "-", "=", ";"]
VOCAB = SPECIALS + SYMBOLS


class Tokenizer:
    def __init__(self):
        self.itos = list(VOCAB)
        self.stoi = {tok: i for i, tok in enumerate(self.itos)}
        # Convenience id lookups used all over the codebase.
        self.pad = self.stoi["<pad>"]
        self.bos = self.stoi["<bos>"]
        self.eos = self.stoi["<eos>"]
        self.think = self.stoi["<think>"]
        self.end_think = self.stoi["</think>"]
        self.answer = self.stoi["<answer>"]
        self.end_answer = self.stoi["</answer>"]

    @property
    def vocab_size(self) -> int:
        return len(self.itos)

    def encode(self, tokens: List[str]) -> List[int]:
        return [self.stoi[t] for t in tokens]

    def decode(self, ids: List[int]) -> str:
        return "".join(self.itos[i] for i in ids)

    def encode_number(self, n: int) -> List[str]:
        """Turn an integer into a list of digit tokens (with a leading ``-``)."""
        if n < 0:
            return ["-"] + list(str(-n))
        return list(str(n))
