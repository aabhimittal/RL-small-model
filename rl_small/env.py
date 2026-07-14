"""The arithmetic reasoning task -- a small *verifiable* environment.

Why arithmetic? Reinforcement learning needs a reward signal, and the cleanest
reward is one you can *check* programmatically. "Is this arithmetic answer
correct?" is trivially verifiable, which is exactly the setting where pure RL
shines (no human labels, no learned reward model -- just a verifier).

Crucially, the task has a *difficulty knob* (the number of operands). Easy
2-operand problems can be answered directly; harder 3-4 operand problems are far
more reliable when the model works step by step. That gap is what makes
*dynamic hybrid reasoning* meaningful: the policy should learn to answer easy
problems fast and reason through hard ones.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from .tokenizer import Tokenizer


@dataclass
class Problem:
    operands: List[int]
    ops: List[str]          # length len(operands) - 1, each '+' or '-'
    answer: int
    prompt_tokens: List[str]

    @property
    def difficulty(self) -> int:
        return len(self.operands)


def _evaluate(operands: List[int], ops: List[str]) -> int:
    """Left-to-right evaluation (only + and -, so precedence is irrelevant)."""
    total = operands[0]
    for op, val in zip(ops, operands[1:]):
        total = total + val if op == "+" else total - val
    return total


class ArithmeticEnv:
    """Generates problems and verifies answers extracted from generations."""

    def __init__(self, tokenizer: Tokenizer, max_operand: int = 20,
                 difficulties=(2, 3, 4), seed: int = 0):
        self.tok = tokenizer
        self.max_operand = max_operand
        self.difficulties = tuple(difficulties)
        self.rng = np.random.default_rng(seed)

    def sample(self, difficulty: Optional[int] = None) -> Problem:
        if difficulty is None:
            difficulty = int(self.rng.choice(self.difficulties))
        operands = [int(self.rng.integers(0, self.max_operand + 1)) for _ in range(difficulty)]
        ops = [("+" if self.rng.random() < 0.5 else "-") for _ in range(difficulty - 1)]
        answer = _evaluate(operands, ops)

        tokens: List[str] = ["<bos>"]
        tokens += list(str(operands[0]))
        for op, val in zip(ops, operands[1:]):
            tokens.append(op)
            tokens += list(str(val))
        tokens.append("=")
        return Problem(operands, ops, answer, tokens)

    def sample_batch(self, n: int, difficulty: Optional[int] = None) -> List[Problem]:
        return [self.sample(difficulty) for _ in range(n)]

    # -- verification ---------------------------------------------------------
    def extract_answer(self, gen_ids: List[int]) -> Optional[int]:
        """Parse the integer inside the first ``<answer>...</answer>`` span.

        Returns ``None`` if the answer span is missing or malformed -- which the
        reward function treats as a (mildly penalized) failure.
        """
        tok = self.tok
        try:
            start = gen_ids.index(tok.answer)
        except ValueError:
            return None
        digits = []
        i = start + 1
        while i < len(gen_ids) and gen_ids[i] != tok.end_answer:
            digits.append(gen_ids[i])
            i += 1
        if i >= len(gen_ids):          # never closed the answer span
            return None
        s = self.tok.decode(digits)
        return _parse_int(s)

    def is_correct(self, problem: Problem, gen_ids: List[int]) -> bool:
        pred = self.extract_answer(gen_ids)
        return pred is not None and pred == problem.answer

    # -- reference completions (used only for tests / optional format warmup) --
    def reference_completion(self, problem: Problem, reason: bool) -> List[int]:
        """A correct completion, either direct or with explicit reasoning.

        This is *not* used for pure-RL training (there are no supervised
        targets there). It exists so tests can exercise the verifier and so the
        optional format-warmup in the docs has something to imitate.
        """
        tok = self.tok
        out: List[str] = []
        if reason:
            out.append("<think>")
            running = problem.operands[0]
            out += list(str(running))
            for op, val in zip(problem.ops, problem.operands[1:]):
                nxt = running + val if op == "+" else running - val
                out += [op] + list(str(val)) + ["="] + tok.encode_number(nxt) + [";"]
                running = nxt
            out.append("</think>")
        out += ["<answer>"] + tok.encode_number(problem.answer) + ["</answer>", "<eos>"]
        return tok.encode(out)


def _parse_int(s: str) -> Optional[int]:
    if s == "" or s == "-":
        return None
    body = s[1:] if s.startswith("-") else s
    if not body.isdigit():
        return None
    return int(s)


def prompt_matrix(problems: List[Problem], tokenizer: Tokenizer) -> Tuple[np.ndarray, int]:
    """Left-pad a batch of prompts into a single array for batched decoding.

    Returns the padded id matrix and the common (right-aligned) length. We
    left-pad so that the *last* column is always the real final prompt token
    (the ``=``), which keeps autoregressive generation simple.
    """
    ids = [tokenizer.encode(p.prompt_tokens) for p in problems]
    maxlen = max(len(x) for x in ids)
    mat = np.full((len(ids), maxlen), tokenizer.pad, dtype=np.int64)
    for r, seq in enumerate(ids):
        mat[r, maxlen - len(seq):] = seq
    return mat, maxlen
