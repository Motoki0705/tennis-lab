"""Trajectory completion inference utilities.

This module provides inference-time wrappers around trajectory completion models:
- Rule-based interpolation (`PhysicsInterpolator`)
- Learned model wrappers (`BiLSTMCompleter`, `TransformerCompleter`, `IterativeRefinementCompleter`)
- Composition utilities (`HybridCompleter`, `build_completer`)

Architectures live under `src/wasb/models/trajectory_completion/` and are kept pure
`torch.nn.Module` implementations.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray


@dataclass
class CompletionResult:
    """Result of trajectory completion."""

    xy: NDArray[np.float32]
    visibility: NDArray[np.int32]
    confidence: NDArray[np.float32]
    gaps_filled: int = 0
    outliers_removed: int = 0


class TrajectoryCompleter(ABC):
    @abstractmethod
    def complete(
        self,
        xy: NDArray[np.float32],
        visibility: NDArray[np.bool_],
        score: NDArray[np.float32],
    ) -> CompletionResult: ...


class PhysicsInterpolator(TrajectoryCompleter):
    """Physics-based trajectory interpolation using quadratic fitting."""

    def __init__(
        self,
        max_gap: int = 15,
        min_anchor_points: int = 2,
        velocity_threshold: float = 100.0,
        acceleration_threshold: float = 50.0,
        score_threshold: float = 0.5,
    ) -> None:
        self.max_gap = max_gap
        self.min_anchor_points = min_anchor_points
        self.velocity_threshold = velocity_threshold
        self.acceleration_threshold = acceleration_threshold
        self.score_threshold = score_threshold

    def complete(
        self,
        xy: NDArray[np.float32],
        visibility: NDArray[np.bool_],
        score: NDArray[np.float32],
    ) -> CompletionResult:
        T = len(xy)
        if T == 0:
            return CompletionResult(
                xy=np.zeros((0, 2), dtype=np.float32),
                visibility=np.zeros(0, dtype=np.int32),
                confidence=np.zeros(0, dtype=np.float32),
            )

        completed_xy = xy.copy()
        new_visibility = np.where(
            visibility & (score >= self.score_threshold), 1, 0
        ).astype(np.int32)
        confidence = np.where(visibility, score, 0.0).astype(np.float32)

        outliers_removed = self._remove_outliers(
            completed_xy, new_visibility, confidence, score
        )
        gaps_filled = self._fill_gaps(completed_xy, new_visibility, confidence)

        return CompletionResult(
            xy=completed_xy,
            visibility=new_visibility,
            confidence=confidence,
            gaps_filled=gaps_filled,
            outliers_removed=outliers_removed,
        )

    def _remove_outliers(
        self,
        xy: NDArray[np.float32],
        visibility: NDArray[np.int32],
        confidence: NDArray[np.float32],
        score: NDArray[np.float32],
    ) -> int:
        T = len(xy)
        if T < 3:
            return 0

        valid_indices = np.where(visibility == 1)[0]
        if len(valid_indices) < 3:
            return 0

        outliers_removed = 0
        for i in range(1, len(valid_indices) - 1):
            prev_idx = valid_indices[i - 1]
            curr_idx = valid_indices[i]
            next_idx = valid_indices[i + 1]

            if curr_idx - prev_idx > 5 or next_idx - curr_idx > 5:
                continue

            dt1 = curr_idx - prev_idx
            dt2 = next_idx - curr_idx

            v1 = (xy[curr_idx] - xy[prev_idx]) / dt1
            v2 = (xy[next_idx] - xy[curr_idx]) / dt2

            speed1 = np.linalg.norm(v1)
            speed2 = np.linalg.norm(v2)
            acc = np.linalg.norm(v2 - v1) / ((dt1 + dt2) / 2)

            is_outlier = (
                (speed1 > self.velocity_threshold and speed2 > self.velocity_threshold)
                or acc > self.acceleration_threshold
            )

            if is_outlier and score[curr_idx] < 0.8:
                visibility[curr_idx] = 0
                confidence[curr_idx] = 0.0
                outliers_removed += 1

        return outliers_removed

    def _fill_gaps(
        self,
        xy: NDArray[np.float32],
        visibility: NDArray[np.int32],
        confidence: NDArray[np.float32],
    ) -> int:
        gaps = self._find_gaps(visibility)
        gaps_filled = 0

        for gap_start, gap_end in gaps:
            gap_length = gap_end - gap_start
            if gap_length > self.max_gap:
                continue

            before_anchors = self._get_anchor_points(
                xy, visibility, gap_start, direction=-1
            )
            after_anchors = self._get_anchor_points(
                xy, visibility, gap_end - 1, direction=1
            )

            if (
                len(before_anchors) < self.min_anchor_points
                or len(after_anchors) < self.min_anchor_points
            ):
                continue

            filled = self._quadratic_interpolate(
                before_anchors, after_anchors, gap_start, gap_end
            )
            if filled is None:
                continue

            for t, pos in filled.items():
                xy[t] = pos
                visibility[t] = 2
                confidence[t] = max(0.3, 1.0 - gap_length / self.max_gap * 0.5)
                gaps_filled += 1

        return gaps_filled

    def _find_gaps(
        self, visibility: NDArray[np.int32]
    ) -> list[tuple[int, int]]:
        gaps: list[tuple[int, int]] = []
        T = len(visibility)
        i = 0
        while i < T:
            if visibility[i] != 1:
                gap_start = i
                while i < T and visibility[i] != 1:
                    i += 1
                gaps.append((gap_start, i))
            else:
                i += 1
        return gaps

    def _get_anchor_points(
        self,
        xy: NDArray[np.float32],
        visibility: NDArray[np.int32],
        start_idx: int,
        direction: int,
        max_search: int = 10,
    ) -> list[tuple[int, NDArray[np.float32]]]:
        anchors: list[tuple[int, NDArray[np.float32]]] = []
        T = len(xy)
        idx = start_idx + direction

        while 0 <= idx < T and len(anchors) < max_search:
            if visibility[idx] == 1:
                anchors.append((idx, xy[idx].copy()))
                if len(anchors) >= self.min_anchor_points:
                    break
            idx += direction

        return anchors

    def _quadratic_interpolate(
        self,
        before_anchors: list[tuple[int, NDArray[np.float32]]],
        after_anchors: list[tuple[int, NDArray[np.float32]]],
        gap_start: int,
        gap_end: int,
    ) -> dict[int, NDArray[np.float32]] | None:
        all_anchors = before_anchors + after_anchors
        if len(all_anchors) < 3:
            return self._linear_interpolate(before_anchors, after_anchors, gap_start, gap_end)

        times = np.array([a[0] for a in all_anchors], dtype=np.float32)
        positions = np.array([a[1] for a in all_anchors], dtype=np.float32)

        try:
            coeffs_x = np.polyfit(times, positions[:, 0], deg=min(2, len(times) - 1))
            coeffs_y = np.polyfit(times, positions[:, 1], deg=min(2, len(times) - 1))
        except (np.linalg.LinAlgError, ValueError):
            return self._linear_interpolate(before_anchors, after_anchors, gap_start, gap_end)

        result: dict[int, NDArray[np.float32]] = {}
        for t in range(gap_start, gap_end):
            x = np.polyval(coeffs_x, t)
            y = np.polyval(coeffs_y, t)
            result[t] = np.array([x, y], dtype=np.float32)
        return result

    def _linear_interpolate(
        self,
        before_anchors: list[tuple[int, NDArray[np.float32]]],
        after_anchors: list[tuple[int, NDArray[np.float32]]],
        gap_start: int,
        gap_end: int,
    ) -> dict[int, NDArray[np.float32]] | None:
        if not before_anchors or not after_anchors:
            return None
        t1, p1 = before_anchors[0]
        t2, p2 = after_anchors[0]
        if t2 == t1:
            return None

        result: dict[int, NDArray[np.float32]] = {}
        for t in range(gap_start, gap_end):
            alpha = (t - t1) / (t2 - t1)
            result[t] = ((1 - alpha) * p1 + alpha * p2).astype(np.float32)
        return result


def _ensure_torch_device(device: str):
    import torch

    torch_device = torch.device(device)
    if torch_device.type == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch_device


class _TorchModelCompleter(TrajectoryCompleter):
    def __init__(
        self,
        model,
        score_threshold: float = 0.5,
        device: str = "cpu",
        *,
        is_trained: bool = False,
    ) -> None:
        import torch

        self.score_threshold = float(score_threshold)
        self.device = _ensure_torch_device(device)
        self.is_trained = bool(is_trained)
        self.model = model.to(self.device) if isinstance(model, torch.nn.Module) else model

    @staticmethod
    def _normalize_xy(xy: NDArray[np.float32]) -> NDArray[np.float32]:
        data = xy.astype(np.float32).copy()
        data[:, 0] /= 1920.0
        data[:, 1] /= 1080.0
        return data

    @staticmethod
    def _denormalize_xy(xy_norm: NDArray[np.float32]) -> NDArray[np.float32]:
        data = xy_norm.astype(np.float32).copy()
        data[:, 0] *= 1920.0
        data[:, 1] *= 1080.0
        return data


class TransformerCompleter(_TorchModelCompleter):
    @classmethod
    def load_from_checkpoint(
        cls,
        checkpoint_path: str | Path,
        device: str = "cpu",
        score_threshold: float = 0.5,
    ) -> TransformerCompleter:
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        from src.wasb.training import TrajectoryLightningModule

        module = TrajectoryLightningModule.load_from_checkpoint(
            str(checkpoint_path), map_location=_ensure_torch_device(device)
        )
        return cls(
            model=module.model,
            score_threshold=score_threshold,
            device=device,
            is_trained=True,
        )

    def complete(
        self,
        xy: NDArray[np.float32],
        visibility: NDArray[np.bool_],
        score: NDArray[np.float32],
    ) -> CompletionResult:
        import torch

        T = len(xy)
        if T == 0:
            return CompletionResult(
                xy=np.zeros((0, 2), dtype=np.float32),
                visibility=np.zeros(0, dtype=np.int32),
                confidence=np.zeros(0, dtype=np.float32),
            )

        x = torch.from_numpy(self._normalize_xy(xy)).float().unsqueeze(0).to(self.device)
        self.model.eval()
        with torch.no_grad():
            pred = self.model(x)[0].detach().cpu().numpy()

        pred_xy = self._denormalize_xy(pred)
        completed_xy = xy.copy()
        new_visibility = np.where(
            visibility & (score >= self.score_threshold), 1, 0
        ).astype(np.int32)
        confidence = score.copy().astype(np.float32)

        gaps_filled = 0
        for t in range(T):
            if new_visibility[t] == 0:
                completed_xy[t] = pred_xy[t]
                new_visibility[t] = 2
                confidence[t] = 0.5
                gaps_filled += 1

        return CompletionResult(
            xy=completed_xy.astype(np.float32),
            visibility=new_visibility,
            confidence=confidence,
            gaps_filled=gaps_filled,
        )


class BiLSTMCompleter(TransformerCompleter):
    @classmethod
    def load_from_checkpoint(
        cls,
        checkpoint_path: str | Path,
        device: str = "cpu",
        score_threshold: float = 0.5,
    ) -> BiLSTMCompleter:
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        from src.wasb.training import TrajectoryLightningModule

        module = TrajectoryLightningModule.load_from_checkpoint(
            str(checkpoint_path), map_location=_ensure_torch_device(device)
        )
        return cls(
            model=module.model,
            score_threshold=score_threshold,
            device=device,
            is_trained=True,
        )


class IterativeRefinementCompleter(_TorchModelCompleter):
    def __init__(
        self,
        model,
        num_steps: int = 3,
        score_threshold: float = 0.5,
        device: str = "cpu",
        *,
        is_trained: bool = False,
    ) -> None:
        super().__init__(
            model=model,
            score_threshold=score_threshold,
            device=device,
            is_trained=is_trained,
        )
        self.num_steps = max(int(num_steps), 1)

    @classmethod
    def load_from_checkpoint(
        cls,
        checkpoint_path: str | Path,
        device: str = "cpu",
        score_threshold: float = 0.5,
        num_steps: int | None = None,
    ) -> IterativeRefinementCompleter:
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        from src.wasb.training import TrajectoryLightningModule

        module = TrajectoryLightningModule.load_from_checkpoint(
            str(checkpoint_path), map_location=_ensure_torch_device(device)
        )
        steps = int(module.num_steps) if num_steps is None else int(num_steps)
        return cls(
            model=module.model,
            num_steps=steps,
            score_threshold=score_threshold,
            device=device,
            is_trained=True,
        )

    def complete(
        self,
        xy: NDArray[np.float32],
        visibility: NDArray[np.bool_],
        score: NDArray[np.float32],
    ) -> CompletionResult:
        import torch

        T = len(xy)
        if T == 0:
            return CompletionResult(
                xy=np.zeros((0, 2), dtype=np.float32),
                visibility=np.zeros(0, dtype=np.int32),
                confidence=np.zeros(0, dtype=np.float32),
            )

        xk = (
            torch.from_numpy(self._normalize_xy(xy))
            .float()
            .unsqueeze(0)
            .to(self.device)
        )
        self.model.eval()
        with torch.no_grad():
            for _ in range(self.num_steps):
                delta = self.model(xk)
                xk = xk + delta

        pred_xy = self._denormalize_xy(xk[0].detach().cpu().numpy())
        new_visibility = np.where(
            visibility & (score >= self.score_threshold), 1, 0
        ).astype(np.int32)
        confidence = score.copy().astype(np.float32)

        gaps_filled = 0
        for t in range(T):
            if new_visibility[t] != 1:
                new_visibility[t] = 2
                confidence[t] = 0.5
                gaps_filled += 1

        return CompletionResult(
            xy=pred_xy.astype(np.float32),
            visibility=new_visibility,
            confidence=confidence,
            gaps_filled=gaps_filled,
        )


class HybridCompleter(TrajectoryCompleter):
    """Combine physics interpolation with a learned coarse model and optional refiner."""

    def __init__(
        self,
        score_threshold: float = 0.5,
        coarse_model: TrajectoryCompleter | None = None,
        refiner: IterativeRefinementCompleter | None = None,
    ) -> None:
        self.score_threshold = float(score_threshold)
        self.coarse = coarse_model
        self.refiner = refiner

    def complete(
        self,
        xy: NDArray[np.float32],
        visibility: NDArray[np.bool_],
        score: NDArray[np.float32],
    ) -> CompletionResult:
        if self.coarse is None:
            coarse = build_completer(
                method="transformer",
                checkpoint_path=None,
                score_threshold=self.score_threshold,
                device="cpu",
            )
        else:
            coarse = self.coarse

        result = coarse.complete(xy, visibility, score)
        if self.refiner is None or not getattr(self.refiner, "is_trained", False):
            return result

        refined = self.refiner.complete(
            result.xy, result.visibility == 1, result.confidence
        )
        result.xy = refined.xy
        return result


def build_completer(
    method: Literal["physics", "bilstm", "transformer", "refiner", "hybrid"] = "hybrid",
    checkpoint_path: str | Path | None = None,
    **kwargs,
) -> TrajectoryCompleter:
    max_gap = int(kwargs.pop("max_gap", 15))
    score_threshold = float(kwargs.pop("score_threshold", 0.5))
    physics_gap_threshold = int(kwargs.pop("physics_gap_threshold", 5))
    device = str(kwargs.pop("device", "cpu"))

    min_anchor_points = int(kwargs.pop("min_anchor_points", 2))
    velocity_threshold = float(kwargs.pop("velocity_threshold", 100.0))
    acceleration_threshold = float(kwargs.pop("acceleration_threshold", 50.0))

    if method == "physics":
        return PhysicsInterpolator(
            max_gap=max_gap,
            min_anchor_points=min_anchor_points,
            velocity_threshold=velocity_threshold,
            acceleration_threshold=acceleration_threshold,
            score_threshold=score_threshold,
        )

    if method == "bilstm":
        if checkpoint_path is None:
            from src.wasb.models.trajectory_completion import TrajectoryBiLSTM

            model = TrajectoryBiLSTM(
                hidden_dim=int(kwargs.pop("hidden_dim", 64)),
                num_layers=int(kwargs.pop("num_layers", 2)),
                dropout=float(kwargs.pop("dropout", 0.1)),
            )
            return BiLSTMCompleter(
                model=model, score_threshold=score_threshold, device=device
            )
        return BiLSTMCompleter.load_from_checkpoint(
            checkpoint_path, device=device, score_threshold=score_threshold
        )

    if method == "transformer":
        if checkpoint_path is None:
            from src.wasb.models.trajectory_completion import TrajectoryTransformer

            model = TrajectoryTransformer(
                d_model=int(kwargs.pop("d_model", 128)),
                num_layers=int(kwargs.pop("num_layers", 2)),
                num_heads=int(kwargs.pop("num_heads", 4)),
                dim_feedforward=int(kwargs.pop("dim_feedforward", 256)),
                dropout=float(kwargs.pop("dropout", 0.1)),
            )
            return TransformerCompleter(
                model=model, score_threshold=score_threshold, device=device
            )
        return TransformerCompleter.load_from_checkpoint(
            checkpoint_path, device=device, score_threshold=score_threshold
        )

    if method == "refiner":
        steps = int(kwargs.pop("num_steps", 3))
        if checkpoint_path is None:
            from src.wasb.models.trajectory_completion import TrajectoryDeltaTransformer

            model = TrajectoryDeltaTransformer(
                d_model=int(kwargs.pop("d_model", 128)),
                num_layers=int(kwargs.pop("num_layers", 2)),
                num_heads=int(kwargs.pop("num_heads", 4)),
                dim_feedforward=int(kwargs.pop("dim_feedforward", 256)),
                dropout=float(kwargs.pop("dropout", 0.1)),
            )
            return IterativeRefinementCompleter(
                model=model,
                num_steps=steps,
                score_threshold=score_threshold,
                device=device,
            )
        return IterativeRefinementCompleter.load_from_checkpoint(
            checkpoint_path,
            device=device,
            score_threshold=score_threshold,
            num_steps=steps,
        )

    if method == "hybrid":
        coarse_method = str(kwargs.pop("coarse_method", "transformer")).lower()
        coarse_checkpoint = kwargs.pop("coarse_checkpoint_path", checkpoint_path)
        refiner_checkpoint = kwargs.pop("refiner_checkpoint_path", None)
        refiner_steps = int(kwargs.pop("num_steps", 3))

        coarse: TrajectoryCompleter | None = None
        if coarse_method == "bilstm":
            coarse = build_completer(
                method="bilstm",
                checkpoint_path=coarse_checkpoint,
                score_threshold=score_threshold,
                device=device,
            )
        elif coarse_method == "transformer":
            coarse = build_completer(
                method="transformer",
                checkpoint_path=coarse_checkpoint,
                score_threshold=score_threshold,
                device=device,
            )
        elif coarse_method == "physics":
            coarse = PhysicsInterpolator(
                max_gap=max_gap,
                min_anchor_points=min_anchor_points,
                velocity_threshold=velocity_threshold,
                acceleration_threshold=acceleration_threshold,
                score_threshold=score_threshold,
            )
        else:
            raise ValueError(f"Unknown coarse_method for hybrid: {coarse_method}")

        refiner = build_completer(
            method="refiner",
            checkpoint_path=refiner_checkpoint,
            score_threshold=score_threshold,
            device=device,
            num_steps=refiner_steps,
        )

        _ = physics_gap_threshold
        return HybridCompleter(
            coarse_model=coarse,
            refiner=refiner if isinstance(refiner, IterativeRefinementCompleter) else None,
            score_threshold=score_threshold,
        )

    raise ValueError(f"Unknown completion method: {method}")


__all__ = [
    "BiLSTMCompleter",
    "CompletionResult",
    "HybridCompleter",
    "IterativeRefinementCompleter",
    "PhysicsInterpolator",
    "TrajectoryCompleter",
    "TransformerCompleter",
    "build_completer",
]
