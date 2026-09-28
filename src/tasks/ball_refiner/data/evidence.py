"""CPU detector evidence on one source camera's exact, unpadded timeline."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from numpy.typing import NDArray

from src.tasks.ball_detection.model_io.contracts import (
    BallCandidateConfig,
    BallCandidates,
)


@dataclass(frozen=True)
class ClipEvidence:
    """Local features retain native cells; UV uses source (W-1,H-1)."""

    frame_index: NDArray[np.int32]
    pts: NDArray[np.int64]
    timestamps_seconds: NDArray[np.float32]
    window_start: NDArray[np.int64]
    time_index: NDArray[np.int64]
    argmax_uv: NDArray[np.float32]
    argmax_score: NDArray[np.float32]
    candidates: BallCandidates  # singleton batch axis; every tensor on CPU
    heatmap_size_hw: tuple[int, int]
    window_length: int

    def __post_init__(self) -> None:
        n = len(self.frame_index)
        if n < self.window_length or self.window_length < 1:
            raise ValueError("Evidence must contain real, unpadded detector windows")
        for name, value, dtype, shape in (
            ("frame_index", self.frame_index, np.int32, (n,)),
            ("pts", self.pts, np.int64, (n,)),
            ("timestamps_seconds", self.timestamps_seconds, np.float32, (n,)),
            ("window_start", self.window_start, np.int64, (n,)),
            ("time_index", self.time_index, np.int64, (n,)),
            ("argmax_uv", self.argmax_uv, np.float32, (n, 2)),
            ("argmax_score", self.argmax_score, np.float32, (n,)),
        ):
            if value.dtype != dtype or value.shape != shape or not np.isfinite(value).all():
                raise ValueError(f"Invalid evidence array: {name}")
        if not np.array_equal(self.frame_index, np.arange(n)):
            raise ValueError("Evidence frame index must cover the dense source timeline")
        if (np.diff(self.pts) <= 0).any() or (np.diff(self.timestamps_seconds) <= 0).any():
            raise ValueError("Evidence PTS and seconds must be strictly increasing")
        if (
            (self.window_start < 0).any() or (self.window_start + self.window_length > n).any()
            or (self.time_index < 0).any() or (self.time_index >= self.window_length).any()
            or not np.array_equal(self.window_start + self.time_index, self.frame_index)
        ):
            raise ValueError("Evidence window provenance does not address the source frame")
        for value in (self.argmax_uv, self.argmax_score):
            if ((value < 0) | (value > 1)).any():
                raise ValueError("Evidence coordinates/scores must lie in [0,1]")
        if len(self.heatmap_size_hw) != 2 or min(self.heatmap_size_hw) < 1:
            raise ValueError("Native heatmap size must be positive")
        c = self.candidates
        k, p = c.config.max_candidates, c.config.patch_size
        for name, value, dtype, tensor_shape in (
            ("coords", c.coords, torch.float32, (1, n, k, 2)),
            ("scores", c.scores, torch.float32, (1, n, k)),
            ("valid", c.valid, torch.bool, (1, n, k)),
            ("cells", c.cells, torch.int64, (1, n, k, 2)),
            ("patches", c.patches, torch.float32, (1, n, k, p, p)),
            ("patch_valid", c.patch_valid, torch.bool, (1, n, k, p, p)),
        ):
            if value.device.type != "cpu" or value.dtype != dtype or tuple(value.shape) != tensor_shape:
                raise ValueError(f"Invalid candidate evidence tensor: {name}")
            if dtype == torch.float32 and not bool(torch.isfinite(value).all()):
                raise ValueError(f"Nonfinite candidate evidence: {name}")
        for value in (c.coords, c.scores, c.patches):
            if bool(((value < 0) | (value > 1)).any()):
                raise ValueError("Candidate coordinates/scores/patches must lie in [0,1]")
        if bool((c.patch_valid & ~c.valid[..., None, None]).any()):
            raise ValueError("Invalid candidates cannot have valid patch cells")
        if (
            bool((c.coords[~c.valid] != 0).any()) or bool((c.scores[~c.valid] != 0).any())
            or bool((c.cells[~c.valid] != 0).any()) or bool((c.patches[~c.patch_valid] != 0).any())
        ):
            raise ValueError("Invalid evidence slots must contain zero")
        if not torch.equal(c.patch_valid[..., p // 2, p // 2], c.valid) or not torch.equal(
            c.patches[..., p // 2, p // 2], c.scores,
        ):
            raise ValueError("Patch centres must match native scores and candidate validity")
        height, width = self.heatmap_size_hw
        if bool(((c.cells < 0) | (c.cells >= torch.tensor([width, height])))[c.valid].any()):
            raise ValueError("Candidate cells must lie on the native heatmap")
        offsets = torch.arange(p) - p // 2
        x = c.cells[..., 0, None, None] + offsets[None, :]
        y = c.cells[..., 1, None, None] + offsets[:, None]
        expected = c.valid[..., None, None] & (x >= 0) & (x < width) & (y >= 0) & (y < height)
        if not torch.equal(c.patch_valid, expected):
            raise ValueError("Patch boundary mask disagrees with native lattice")

    def arrays(self) -> dict[str, NDArray[np.generic]]:
        """No object arrays or labels; safe to load with allow_pickle=False."""
        return {
            **{name: getattr(self, name) for name in (
                "frame_index", "pts", "timestamps_seconds", "window_start",
                "time_index", "argmax_uv", "argmax_score",
            )},
            **{f"candidate_{name}": getattr(self.candidates, name)[0].numpy() for name in (
                "coords", "scores", "valid", "cells", "patches", "patch_valid",
            )},
        }

    @classmethod
    def from_arrays(
        cls, arrays: dict[str, NDArray[np.generic]], *, config: BallCandidateConfig,
        heatmap_size_hw: tuple[int, int], window_length: int,
    ) -> ClipEvidence:
        """Validate every persisted value before returning a usable cache."""
        timeline = (
            "frame_index", "pts", "timestamps_seconds", "window_start",
            "time_index", "argmax_uv", "argmax_score",
        )
        candidate_names = ("coords", "scores", "valid", "cells", "patches", "patch_valid")
        if set(arrays) != {*timeline, *(f"candidate_{name}" for name in candidate_names)}:
            raise ValueError("Unexpected or missing evidence arrays")
        candidates = BallCandidates(
            **{name: torch.from_numpy(arrays[f"candidate_{name}"])[None] for name in candidate_names},
            config=config,
        )
        return cls(
            **{name: arrays[name] for name in timeline}, candidates=candidates,
            heatmap_size_hw=heatmap_size_hw, window_length=window_length,
        )
