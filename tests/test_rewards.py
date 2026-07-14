"""Tests for reward shaping and format analysis."""

from rl_small.tokenizer import Tokenizer
from rl_small.env import ArithmeticEnv, Problem
from rl_small.rewards import RewardConfig, compute_reward, analyze_format
from rl_small.sampling import Rollout


def _rollout(env, tok, problem, tokens):
    return Rollout(prompt_ids=tok.encode(problem.prompt_tokens),
                   gen_ids=tok.encode(tokens), problem=problem)


def test_correct_beats_wrong_and_malformed():
    tok = Tokenizer()
    env = ArithmeticEnv(tok, seed=0)
    p = Problem([3, 4], ["+"], 7, ["<bos>", "3", "+", "4", "="])
    cfg = RewardConfig()

    correct = _rollout(env, tok, p, ["<answer>", "7", "</answer>", "<eos>"])
    wrong = _rollout(env, tok, p, ["<answer>", "9", "</answer>", "<eos>"])
    malformed = _rollout(env, tok, p, ["<answer>", "7"])

    rc, _ = compute_reward(env, correct, cfg)
    rw, _ = compute_reward(env, wrong, cfg)
    rm, _ = compute_reward(env, malformed, cfg)
    assert rc > rw > rm


def test_brevity_breaks_ties_between_correct_answers():
    tok = Tokenizer()
    env = ArithmeticEnv(tok, seed=0)
    p = Problem([3, 4], ["+"], 7, ["<bos>", "3", "+", "4", "="])
    cfg = RewardConfig()

    short = _rollout(env, tok, p, ["<answer>", "7", "</answer>", "<eos>"])
    long = _rollout(env, tok, p, ["<think>", "3", "+", "4", "=", "7", ";",
                                  "</think>", "<answer>", "7", "</answer>", "<eos>"])
    rs, _ = compute_reward(env, short, cfg)
    rl, _ = compute_reward(env, long, cfg)
    # Both correct, so the shorter one should score higher (tightening).
    assert rs > rl


def test_format_analysis_flags():
    tok = Tokenizer()
    env = ArithmeticEnv(tok, seed=0)
    good = tok.encode(["<answer>", "7", "</answer>", "<eos>"])
    f = analyze_format(env, good)
    assert f["well_formed"] and f["ends_eos"] and not f["used_reasoning"]

    reasoned = tok.encode(["<think>", "1", "</think>", "<answer>", "7",
                           "</answer>", "<eos>"])
    f2 = analyze_format(env, reasoned)
    assert f2["well_formed"] and f2["used_reasoning"]

    bad = tok.encode(["<answer>", "7"])  # never terminated
    assert not analyze_format(env, bad)["well_formed"]
