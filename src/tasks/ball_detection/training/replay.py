"""Replay membership and exact FPS/group budgets for cumulative training."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator
from fractions import Fraction
from pathlib import Path
from typing import Any

import torch

from src.tasks.ball_detection.data.windows.dataset import FrozenBallWindowDataset
from src.tasks.ball_detection.training.sampling import FPSMixSampler
from src.utils.checksum import dual_sha256


def replay_recipe(manifest: Path | None, untrained_fraction: float | None) -> dict[str, Any] | None:
    if (manifest is None) != (untrained_fraction is None):
        raise ValueError("Replay requires both the previously trained manifest and untrained fraction")
    if manifest is None:
        return None
    if untrained_fraction is None or not 0 < untrained_fraction < 1:
        raise ValueError("Both untrained and replay groups must have positive weight")
    return dict(previously_trained_manifest_sha256=dual_sha256(manifest),
                untrained_fraction=untrained_fraction, unit="window", fps_weight="equal")


class ReplayFPSMixSampler(FPSMixSampler):
    """All eligible windows remain candidates; draw frequency is explicit.

    Within each FPS/group bucket, exhaust a random permutation before repeating
    a window. Every epoch has exact integer quotas, including small test runs.
    """

    def __init__(self, dataset: FrozenBallWindowDataset, *, windows_per_epoch: int,
                 seed: int, previously_trained_manifest: Path, untrained_fraction: float) -> None:
        super().__init__(dataset, windows_per_epoch=windows_per_epoch, seed=seed)
        self.recipe = replay_recipe(previously_trained_manifest, untrained_fraction)
        previous = json.loads(previously_trained_manifest.read_text())
        previous_train = {r["clip"]["clip_id"]: r for r in previous["clips"]
                          if r["clip"]["split"] == "train" and r["windows"]}
        if not previous_train:
            raise ValueError("Previously trained manifest has no eligible train clips")
        current = {r["clip"]["clip_id"]: r for r in dataset.records}
        if not previous_train.keys() <= current.keys():
            raise ValueError("Cumulative training removed previously trained clips")
        for clip_id, old in previous_train.items():
            new = current[clip_id]
            for field in ("semantic_sha256", "rgb_shard_sha256"):
                if not old.get(field) or old[field] != new.get(field):
                    raise ValueError("Replay clip identity changed from the previously trained manifest")
        holdouts = {r["clip"]["clip_id"] for r in previous["clips"] if r["clip"]["split"] != "train"}
        if holdouts & current.keys():
            raise ValueError("Previously held-out clips cannot become untrained training examples")
        self.membership = {name: "replay" if name in previous_train else "untrained" for name in current}
        self.buckets: dict[tuple[int, str], list[int]] = {}
        self.quotas: dict[tuple[int, str], int] = {}
        fraction = Fraction(str(untrained_fraction))
        per_fps = Fraction(windows_per_epoch, len(self.steps))
        for step in self.steps:
            for group, weight in (("untrained", fraction), ("replay", 1 - fraction)):
                quota = per_fps * weight
                if quota.denominator != 1 or quota < 1:
                    raise ValueError("Window budget must give every FPS/replay group an exact positive integer quota")
                key = (step, group)
                self.quotas[key] = int(quota)
                self.buckets[key] = [i for i in self.groups[step]
                                    if self.membership[dataset.records[dataset.windows[i][0]]["clip"]["clip_id"]] == group]
                if not self.buckets[key]:
                    raise ValueError(f"No eligible windows in FPS/replay group {key}")
        self.last_report: dict[str, Any] | None = None

    def __iter__(self) -> Iterator[int]:
        rng = torch.Generator().manual_seed(self.seed + self.epoch)
        assignments: list[tuple[int, str]] = []
        draws: dict[tuple[int, str], Iterator[int]] = {}
        report = []
        for key, available in self.buckets.items():
            selected: list[int] = []
            quota = self.quotas[key]
            while len(selected) < quota:
                selected.extend(available[i] for i in torch.randperm(len(available), generator=rng).tolist()
                                [:quota - len(selected)])
            assignments.extend([key] * quota)
            draws[key] = iter(selected)
            report.append(dict(frame_step=key[0], group=key[1], candidates=len(available),
                               draws=len(selected), unique_windows=len(set(selected))))
        order = torch.randperm(len(assignments), generator=rng).tolist()
        membership = json.dumps(self.membership, sort_keys=True, separators=(",", ":")).encode()
        self.last_report = dict(epoch=self.epoch, membership_sha256=hashlib.sha256(membership).hexdigest(),
                                groups=report, **(self.recipe or {}))
        yield from (next(draws[assignments[i]]) for i in order)
