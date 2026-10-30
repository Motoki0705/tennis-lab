"""Process RNG state saved alongside Lightning's optimizer and loop state."""

from __future__ import annotations

import random
from typing import Any

import numpy as np
import torch


def capture_random_state() -> dict[str, Any]:
    numpy = np.random.get_state()
    return {
        "python": random.getstate(),
        "numpy": {
            "algorithm": numpy[0],
            "keys": numpy[1].tolist(),
            "position": numpy[2],
            "has_gauss": numpy[3],
            "cached_gaussian": numpy[4],
        },
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else [],
    }


def restore_random_state(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    numpy = state["numpy"]
    np.random.set_state(
        (
            numpy["algorithm"],
            np.asarray(numpy["keys"], dtype=np.uint32),
            numpy["position"],
            numpy["has_gauss"],
            numpy["cached_gaussian"],
        )
    )
    torch.set_rng_state(state["torch"])
    if state["cuda"]:
        if not torch.cuda.is_available():
            raise ValueError("CUDA training RNG requires a CUDA runtime")
        torch.cuda.set_rng_state_all(state["cuda"])
