"""Independent, checkpointed loader RNG streams for epoch-boundary resumes."""

from __future__ import annotations

from typing import Any

import pytorch_lightning as pl
import torch

from src.tasks.base.data.rng import derive_seed


class LoaderRNGDataModule(pl.LightningDataModule):
    """Keep loader initialization and shuffle draws out of the model RNG stream.

    Lightning owns the completed epoch. Each new iterator continues its stage's
    generator, restored before the iterator is created on resume. Dataset-owned
    random streams must be persisted separately by a task that uses them.
    """

    def __init__(self, *, loader_seed: int) -> None:
        super().__init__()
        self._loader_rng = {
            stage: torch.Generator().manual_seed(
                derive_seed(0, loader_seed, "loader", stage)
            )
            for stage in ("train", "val", "test")
        }

    def loader_generator(self, stage: str) -> torch.Generator:
        return self._loader_rng[stage]

    def state_dict(self) -> dict[str, Any]:
        return {
            "loader_rng": {
                stage: generator.get_state()
                for stage, generator in self._loader_rng.items()
            }
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        states = state_dict["loader_rng"]
        if states.keys() != self._loader_rng.keys():
            raise ValueError("Loader RNG stages differ from the saved data contract")
        for stage, state in states.items():
            self._loader_rng[stage].set_state(state)
