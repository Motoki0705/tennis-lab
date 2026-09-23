from __future__ import annotations

import random
from typing import Any

import numpy as np
import torch

from src.tasks.plcs.generate_dataset.utils import parallel_runner


def test_cold_worker_initialization_cannot_change_scene_seed(monkeypatch):
    cold = True

    class Source:
        def generate_scene(self, *, scene_id: str) -> Any:
            return scene_id, random.random(), np.random.random(), torch.rand(1).item()

    def build(config: Any, device: str) -> Any:
        nonlocal cold
        if cold:
            random.getrandbits(64)
            np.random.random(20)
            torch.rand(20)
            cold = False
        return Source()

    monkeypatch.setattr(parallel_runner, "_get_worker_scene_generator", build)
    first = parallel_runner._generate_scene_task(7, {"run": {"seed": 42}}, "cpu")
    warm = parallel_runner._generate_scene_task(7, {"run": {"seed": 42}}, "cpu")
    assert first == warm
    different = parallel_runner._generate_scene_task(7, {"run": {"seed": 43}}, "cpu")
    assert different != first
