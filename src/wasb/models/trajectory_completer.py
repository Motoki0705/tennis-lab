"""Trajectory completion models for noisy/missing ball trajectories.

This module provides models to complete and refine partially observed
2D ball trajectories. Supports both rule-based interpolation and learned
models (Bi-LSTM, Transformer).

Key features:
- Physics-aware interpolation (parabolic motion)
- Confidence-weighted completion
- Gap bridging for missing segments
- Outlier detection and filtering
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
    """Result of trajectory completion.

    Attributes:
        xy: Completed trajectory [T, 2].
        visibility: Updated visibility flags [T].
            - 1: Original detection (kept as-is).
            - 2: Completed by model.
            - 0: Could not be completed (unreliable).
        confidence: Completion confidence per frame [T].
            Higher values indicate more reliable completions.
        gaps_filled: Number of gaps successfully filled.
        outliers_removed: Number of outlier detections replaced.

    """

    xy: NDArray[np.float32]
    visibility: NDArray[np.int32]
    confidence: NDArray[np.float32]
    gaps_filled: int = 0
    outliers_removed: int = 0


class TrajectoryCompleter(ABC):
    """Abstract base class for trajectory completion models."""

    @abstractmethod
    def complete(
        self,
        xy: NDArray[np.float32],
        visibility: NDArray[np.bool_],
        score: NDArray[np.float32],
    ) -> CompletionResult:
        """Complete a partially observed trajectory.

        Args:
            xy: Ball positions [T, 2]. May contain invalid values
                (NaN, zeros) where visibility is False.
            visibility: Boolean mask [T] indicating valid observations.
            score: Detection confidence scores [T].

        Returns:
            CompletionResult with filled trajectory.

        """
        ...


class PhysicsInterpolator(TrajectoryCompleter):
    """Physics-based trajectory interpolation using parabolic motion.

    Uses quadratic interpolation to fill gaps, which approximates
    the parabolic motion of a tennis ball under gravity.

    Attributes:
        max_gap: Maximum gap length to interpolate.
        min_anchor_points: Minimum points needed on each side of gap.
        velocity_threshold: Max velocity (px/frame) to consider valid.
        acceleration_threshold: Max acceleration for outlier detection.

    """

    def __init__(
        self,
        max_gap: int = 15,
        min_anchor_points: int = 2,
        velocity_threshold: float = 100.0,
        acceleration_threshold: float = 50.0,
        score_threshold: float = 0.5,
    ) -> None:
        """Initialize the physics interpolator.

        Args:
            max_gap: Maximum gap size to interpolate.
            min_anchor_points: Minimum anchor points on each side.
            velocity_threshold: Maximum valid velocity in pixels/frame.
            acceleration_threshold: Maximum valid acceleration.
            score_threshold: Minimum score for reliable detection.

        """
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
        """Complete trajectory using physics-based interpolation."""
        T = len(xy)
        if T == 0:
            return CompletionResult(
                xy=np.zeros((0, 2), dtype=np.float32),
                visibility=np.zeros(0, dtype=np.int32),
                confidence=np.zeros(0, dtype=np.float32),
            )

        # Initialize outputs
        completed_xy = xy.copy()
        new_visibility = np.where(
            visibility & (score >= self.score_threshold), 1, 0
        ).astype(np.int32)
        confidence = np.where(visibility, score, 0.0).astype(np.float32)

        # Step 1: Remove outliers from valid detections
        outliers_removed = self._remove_outliers(
            completed_xy, new_visibility, confidence, score
        )

        # Step 2: Find and fill gaps
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
        """Detect and remove outlier detections based on motion consistency."""
        T = len(xy)
        outliers_removed = 0

        if T < 3:
            return 0

        # Find valid indices
        valid_mask = visibility == 1
        valid_indices = np.where(valid_mask)[0]

        if len(valid_indices) < 3:
            return 0

        # Compute velocities between consecutive valid points
        for i in range(1, len(valid_indices) - 1):
            prev_idx = valid_indices[i - 1]
            curr_idx = valid_indices[i]
            next_idx = valid_indices[i + 1]

            # Skip if frames are not close enough
            if curr_idx - prev_idx > 5 or next_idx - curr_idx > 5:
                continue

            # Compute velocities
            dt1 = curr_idx - prev_idx
            dt2 = next_idx - curr_idx

            v1 = (xy[curr_idx] - xy[prev_idx]) / dt1
            v2 = (xy[next_idx] - xy[curr_idx]) / dt2

            # Check velocity magnitude
            speed1 = np.linalg.norm(v1)
            speed2 = np.linalg.norm(v2)

            # Check acceleration (velocity change)
            acc = np.linalg.norm(v2 - v1) / ((dt1 + dt2) / 2)

            # Mark as outlier if motion is inconsistent
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
        """Fill gaps using quadratic interpolation."""
        T = len(xy)
        gaps_filled = 0

        # Find gaps (consecutive frames with visibility != 1)
        gaps = self._find_gaps(visibility)

        for gap_start, gap_end in gaps:
            gap_length = gap_end - gap_start

            # Skip gaps that are too long
            if gap_length > self.max_gap:
                continue

            # Find anchor points before and after gap
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

            # Interpolate gap using quadratic fitting
            filled = self._quadratic_interpolate(
                before_anchors, after_anchors, gap_start, gap_end
            )

            if filled is not None:
                for t, pos in filled.items():
                    xy[t] = pos
                    visibility[t] = 2  # Mark as completed
                    # Confidence based on gap size and anchor quality
                    confidence[t] = max(0.3, 1.0 - gap_length / self.max_gap * 0.5)
                    gaps_filled += 1

        return gaps_filled

    def _find_gaps(
        self, visibility: NDArray[np.int32]
    ) -> list[tuple[int, int]]:
        """Find contiguous gaps in visibility."""
        gaps = []
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
        """Get anchor points in specified direction."""
        anchors = []
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
        """Perform quadratic interpolation across a gap.

        Fits a quadratic curve through anchor points and evaluates
        at gap positions.
        """
        # Combine anchors
        all_anchors = before_anchors + after_anchors
        if len(all_anchors) < 3:
            # Fall back to linear interpolation
            return self._linear_interpolate(before_anchors, after_anchors, gap_start, gap_end)

        # Extract times and positions
        times = np.array([a[0] for a in all_anchors], dtype=np.float32)
        positions = np.array([a[1] for a in all_anchors], dtype=np.float32)

        # Fit quadratic for x and y separately
        try:
            coeffs_x = np.polyfit(times, positions[:, 0], deg=min(2, len(times) - 1))
            coeffs_y = np.polyfit(times, positions[:, 1], deg=min(2, len(times) - 1))
        except (np.linalg.LinAlgError, ValueError):
            return self._linear_interpolate(before_anchors, after_anchors, gap_start, gap_end)

        # Evaluate at gap positions
        result = {}
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
        """Simple linear interpolation fallback."""
        if not before_anchors or not after_anchors:
            return None

        # Use closest anchors
        t1, p1 = before_anchors[0]
        t2, p2 = after_anchors[0]

        if t2 == t1:
            return None

        result = {}
        for t in range(gap_start, gap_end):
            alpha = (t - t1) / (t2 - t1)
            result[t] = ((1 - alpha) * p1 + alpha * p2).astype(np.float32)

        return result


class BiLSTMCompleter(TrajectoryCompleter):
    """Bi-directional LSTM model for trajectory completion.

    This model learns to predict missing ball positions from
    partially observed trajectories using bidirectional context.

    Architecture:
    - Input projection (2D coords + visibility + score -> hidden_dim)
    - Bidirectional LSTM layers
    - Output projection (hidden_dim -> 2D coords)
    """

    def __init__(
        self,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.1,
        score_threshold: float = 0.5,
        device: str = "cuda",
    ) -> None:
        """Initialize Bi-LSTM completer.

        Args:
            hidden_dim: LSTM hidden dimension.
            num_layers: Number of LSTM layers.
            dropout: Dropout probability.
            score_threshold: Minimum score for reliable detection.
            device: Device for computation.

        """
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.score_threshold = score_threshold
        self.device = device
        self._model = None
        self._is_trained = False

    def _build_model(self) -> None:
        """Build the PyTorch model (lazy initialization)."""
        try:
            import torch
            import torch.nn as nn
        except ImportError as e:
            raise ImportError(
                "PyTorch is required for BiLSTMCompleter. "
                "Install with: pip install torch"
            ) from e

        class BiLSTMModel(nn.Module):
            def __init__(
                self,
                input_dim: int,
                hidden_dim: int,
                num_layers: int,
                output_dim: int,
                dropout: float,
            ):
                super().__init__()
                self.input_proj = nn.Linear(input_dim, hidden_dim)
                self.lstm = nn.LSTM(
                    hidden_dim,
                    hidden_dim,
                    num_layers=num_layers,
                    batch_first=True,
                    bidirectional=True,
                    dropout=dropout if num_layers > 1 else 0,
                )
                self.output_proj = nn.Sequential(
                    nn.Linear(hidden_dim * 2, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim, output_dim),
                )

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                # x: [B, T, input_dim]
                h = self.input_proj(x)
                h, _ = self.lstm(h)
                return self.output_proj(h)

        self._model = BiLSTMModel(
            input_dim=4,  # x, y, visibility, score
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
            output_dim=2,  # x, y
            dropout=self.dropout,
        )
        self._model = self._model.to(self.device)

    def complete(
        self,
        xy: NDArray[np.float32],
        visibility: NDArray[np.bool_],
        score: NDArray[np.float32],
    ) -> CompletionResult:
        """Complete trajectory using Bi-LSTM model."""
        import torch

        if self._model is None:
            self._build_model()

        T = len(xy)
        if T == 0:
            return CompletionResult(
                xy=np.zeros((0, 2), dtype=np.float32),
                visibility=np.zeros(0, dtype=np.int32),
                confidence=np.zeros(0, dtype=np.float32),
            )

        # Prepare input
        vis_float = visibility.astype(np.float32)
        input_data = np.stack([xy[:, 0], xy[:, 1], vis_float, score], axis=-1)

        # Normalize coordinates (assuming 1920x1080)
        input_data[:, 0] /= 1920.0
        input_data[:, 1] /= 1080.0

        # Convert to tensor
        x = torch.from_numpy(input_data).float().unsqueeze(0).to(self.device)

        # Run model
        self._model.eval()
        with torch.no_grad():
            pred = self._model(x)

        # Get predictions
        pred_np = pred[0].cpu().numpy()
        pred_np[:, 0] *= 1920.0
        pred_np[:, 1] *= 1080.0

        # Build output
        completed_xy = xy.copy()
        new_visibility = np.where(
            visibility & (score >= self.score_threshold), 1, 0
        ).astype(np.int32)
        confidence = score.copy()

        # Fill missing positions with predictions
        gaps_filled = 0
        for t in range(T):
            if new_visibility[t] == 0:
                completed_xy[t] = pred_np[t]
                new_visibility[t] = 2
                confidence[t] = 0.5  # Model confidence
                gaps_filled += 1

        return CompletionResult(
            xy=completed_xy,
            visibility=new_visibility,
            confidence=confidence,
            gaps_filled=gaps_filled,
        )
    def load_from_checkpoint(self, checkpoint_path: str | Path) -> None:
        """Load model weights from a Lightning-style checkpoint."""
        import torch

        if self._model is None:
            self._build_model()

        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        checkpoint = torch.load(
            checkpoint_path,
            map_location=self.device,
            weights_only=False,
        )
        if not isinstance(checkpoint, dict):
            raise TypeError(f"Checkpoint must be a dict, got {type(checkpoint)}")

        state_dict = None
        if "state_dict" in checkpoint and isinstance(checkpoint["state_dict"], dict):
            state_dict = checkpoint["state_dict"]
        elif "model_state_dict" in checkpoint and isinstance(
            checkpoint["model_state_dict"], dict
        ):
            # Backwards-compatibility with older custom checkpoints.
            state_dict = checkpoint["model_state_dict"]
        else:
            raise ValueError(
                f"Checkpoint must contain a 'state_dict' or 'model_state_dict': {checkpoint_path}"
            )

        if self._model is None:
            self._build_model()
        model_state = self._model.state_dict()
        filtered_state = {}
        for key, value in state_dict.items():
            trimmed_key = key
            if key.startswith("model."):
                trimmed_key = key.removeprefix("model.")
            if trimmed_key in model_state:
                filtered_state[trimmed_key] = value

        if not filtered_state:
            raise ValueError(
                f"No matching model keys were found in checkpoint: {checkpoint_path}"
            )

        model_state.update(filtered_state)
        self._model.load_state_dict(model_state, strict=False)
        self._is_trained = True


class HybridCompleter(TrajectoryCompleter):
    """Hybrid completer combining physics-based and learned approaches.

    Uses physics interpolation for short gaps and learned model
    for longer or more complex gaps.

    Strategy:
    1. First apply physics-based outlier removal
    2. Fill short gaps (< threshold) with physics interpolation
    3. Fill remaining gaps with learned model (if available)
    4. Fall back to physics for everything if no learned model
    """

    def __init__(
        self,
        physics_gap_threshold: int = 5,
        max_gap: int = 30,
        score_threshold: float = 0.5,
        learned_model: BiLSTMCompleter | None = None,
    ) -> None:
        """Initialize hybrid completer.

        Args:
            physics_gap_threshold: Max gap size for physics interpolation.
            max_gap: Max gap size for fallback physics interpolation.
            score_threshold: Minimum score for reliable detection.
            learned_model: Optional learned model for complex gaps.

        """
        self.physics_gap_threshold = physics_gap_threshold
        self.max_gap = max_gap
        self.score_threshold = score_threshold

        self.physics = PhysicsInterpolator(
            max_gap=physics_gap_threshold,
            score_threshold=score_threshold,
        )
        self.learned = learned_model

        # Fallback physics for longer gaps
        self.physics_fallback = PhysicsInterpolator(
            max_gap=max_gap,
            score_threshold=score_threshold,
        )

    def complete(
        self,
        xy: NDArray[np.float32],
        visibility: NDArray[np.bool_],
        score: NDArray[np.float32],
    ) -> CompletionResult:
        """Complete trajectory using hybrid approach."""
        # Step 1: Physics-based completion for short gaps
        result = self.physics.complete(xy, visibility, score)

        # Check if there are remaining gaps
        remaining_gaps = np.sum(result.visibility == 0)

        if remaining_gaps == 0:
            return result

        # Step 2: Try learned model for remaining gaps
        if self.learned is not None and self.learned._is_trained:
            # Convert visibility back to bool for learned model
            vis_bool = result.visibility == 1
            learned_result = self.learned.complete(
                result.xy, vis_bool, result.confidence
            )

            # Merge results
            for t in range(len(result.xy)):
                if result.visibility[t] == 0 and learned_result.visibility[t] == 2:
                    result.xy[t] = learned_result.xy[t]
                    result.visibility[t] = 2
                    result.confidence[t] = learned_result.confidence[t]
                    result.gaps_filled += 1

        else:
            # Fallback to extended physics interpolation
            vis_bool = result.visibility == 1
            fallback_result = self.physics_fallback.complete(
                result.xy, vis_bool, result.confidence
            )

            # Merge results
            for t in range(len(result.xy)):
                if result.visibility[t] == 0 and fallback_result.visibility[t] == 2:
                    result.xy[t] = fallback_result.xy[t]
                    result.visibility[t] = 2
                    result.confidence[t] = fallback_result.confidence[t] * 0.8  # Lower confidence
                    result.gaps_filled += 1

        return result


def create_completer(
    method: Literal["physics", "bilstm", "hybrid"] = "hybrid",
    checkpoint_path: str | Path | None = None,
    **kwargs,
) -> TrajectoryCompleter:
    """Factory function to create trajectory completer.

    Args:
        method: Completion method ("physics", "bilstm", "hybrid").
        checkpoint_path: Path to model checkpoint (for bilstm/hybrid).
        **kwargs: Additional arguments for the completer.
            - max_gap: Maximum gap to interpolate (physics/hybrid).
            - score_threshold: Min detection score (all methods).
            - physics_gap_threshold: Max gap for physics in hybrid mode.

    Returns:
        Configured TrajectoryCompleter instance.

    """
    # Extract common parameters
    max_gap = kwargs.pop("max_gap", 15)
    score_threshold = kwargs.pop("score_threshold", 0.5)
    physics_gap_threshold = kwargs.pop("physics_gap_threshold", 5)

    # Physics-specific parameters
    min_anchor_points = kwargs.pop("min_anchor_points", 2)
    velocity_threshold = kwargs.pop("velocity_threshold", 100.0)
    acceleration_threshold = kwargs.pop("acceleration_threshold", 50.0)

    if method == "physics":
        return PhysicsInterpolator(
            max_gap=max_gap,
            min_anchor_points=min_anchor_points,
            velocity_threshold=velocity_threshold,
            acceleration_threshold=acceleration_threshold,
            score_threshold=score_threshold,
        )

    elif method == "bilstm":
        completer = BiLSTMCompleter(score_threshold=score_threshold)
        if checkpoint_path is not None:
            completer.load_from_checkpoint(checkpoint_path)
        return completer

    elif method == "hybrid":
        learned = None
        if checkpoint_path is not None:
            learned = BiLSTMCompleter(score_threshold=score_threshold)
            learned.load_from_checkpoint(checkpoint_path)
        return HybridCompleter(
            learned_model=learned,
            physics_gap_threshold=physics_gap_threshold,
            max_gap=max_gap,
            score_threshold=score_threshold,
        )

    else:
        raise ValueError(f"Unknown completion method: {method}")
