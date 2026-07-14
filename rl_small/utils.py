"""Small shared utilities: seeding, checkpoint save/load, pretty logging."""

from __future__ import annotations

import json
import os
import pickle
from typing import Dict

import numpy as np


def set_seed(seed: int) -> np.random.Generator:
    np.random.seed(seed)
    return np.random.default_rng(seed)


def snapshot_params(model):
    """Deep-copy the model's parameters (for best-so-far / early-stopping)."""
    return [p.data.copy() for p in model.parameters()]


def restore_params(model, snapshot):
    """Load a snapshot from :func:`snapshot_params` back into the model."""
    for p, saved in zip(model.parameters(), snapshot):
        p.data[...] = saved


def save_checkpoint(model, path: str, meta: Dict = None):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    state = {
        "params": [p.data for p in model.parameters()],
        "config": {
            "vocab_size": model.vocab_size,
            "block_size": model.block_size,
        },
        "meta": meta or {},
    }
    with open(path, "wb") as f:
        pickle.dump(state, f)


def load_params_into(model, path: str):
    with open(path, "rb") as f:
        state = pickle.load(f)
    params = model.parameters()
    assert len(params) == len(state["params"]), "checkpoint / model mismatch"
    for p, saved in zip(params, state["params"]):
        assert p.data.shape == saved.shape, "parameter shape mismatch"
        p.data[...] = saved
    return state.get("meta", {})


def format_stats(step: int, stats: Dict[str, float]) -> str:
    keys = ["reward", "accuracy", "reasoning_rate", "gen_len", "loss"]
    parts = [f"step {step:4d}"]
    for k in keys:
        if k in stats:
            parts.append(f"{k}={stats[k]:.3f}")
    return " | ".join(parts)


def append_jsonl(path: str, record: Dict):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(record) + "\n")
