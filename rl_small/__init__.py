"""rl_small -- pure-RL training and dynamic hybrid reasoning for small models.

A from-scratch, NumPy-only implementation you can read end to end:

* :mod:`rl_small.autograd`  -- reverse-mode autodiff engine
* :mod:`rl_small.model`     -- TinyGPT policy network
* :mod:`rl_small.env`       -- verifiable arithmetic reasoning task
* :mod:`rl_small.rewards`   -- reward shaping that tightens behavior
* :mod:`rl_small.grpo`      -- Group Relative Policy Optimization (pure RL)
* :mod:`rl_small.hybrid`    -- dynamic fast-vs-think decoding
"""

from .tokenizer import Tokenizer
from .env import ArithmeticEnv, Problem
from .model import TinyGPT
from .rewards import RewardConfig, compute_reward
from .grpo import GRPOTrainer, GRPOConfig
from .hybrid import HybridConfig, dynamic_decode, auto_decode

__version__ = "0.1.0"

__all__ = [
    "Tokenizer",
    "ArithmeticEnv",
    "Problem",
    "TinyGPT",
    "RewardConfig",
    "compute_reward",
    "GRPOTrainer",
    "GRPOConfig",
    "HybridConfig",
    "dynamic_decode",
    "auto_decode",
]
