"""Tests for the arithmetic environment and verifier."""

from rl_small.tokenizer import Tokenizer
from rl_small.env import ArithmeticEnv


def test_problem_generation_and_eval():
    tok = Tokenizer()
    env = ArithmeticEnv(tok, seed=0)
    for _ in range(50):
        p = env.sample()
        assert 2 <= p.difficulty <= 4
        assert len(p.ops) == p.difficulty - 1
        # Prompt decodes to "<bos>...=".
        s = tok.decode(tok.encode(p.prompt_tokens))
        assert s.startswith("<bos>") and s.endswith("=")


def test_reference_completion_verifies():
    tok = Tokenizer()
    env = ArithmeticEnv(tok, seed=1)
    for reason in (False, True):
        for _ in range(50):
            p = env.sample()
            gen = env.reference_completion(p, reason=reason)
            assert env.is_correct(p, gen), (p.operands, p.ops, p.answer, tok.decode(gen))
            assert env.extract_answer(gen) == p.answer


def test_malformed_answer_returns_none():
    tok = Tokenizer()
    env = ArithmeticEnv(tok, seed=2)
    p = env.sample()
    # No answer span at all.
    assert env.extract_answer(tok.encode(["<think>", "1", "+", "1"])) is None
    # Unterminated answer span.
    assert env.extract_answer(tok.encode(["<answer>", "1", "2"])) is None
    # Empty answer.
    assert env.extract_answer(tok.encode(["<answer>", "</answer>"])) is None


def test_negative_answers():
    tok = Tokenizer()
    env = ArithmeticEnv(tok, seed=3)
    # Force a known negative result via the reference completion path.
    from rl_small.env import Problem
    p = Problem(operands=[2, 5], ops=["-"], answer=-3,
                prompt_tokens=["<bos>", "2", "-", "5", "="])
    gen = env.reference_completion(p, reason=True)
    assert env.extract_answer(gen) == -3
