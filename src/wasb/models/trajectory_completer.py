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

import math
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
            input_dim=2,  # x, y
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
        input_data = xy.astype(np.float32).copy()

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
                f"No matching model keys were found in checkpoint: {checkpoint_path}\n"
                f"Expected model keys (sample): {list(model_state.keys())[:5]}\n"
                f"Checkpoint keys (sample): {list(state_dict.keys())[:5]}"
            )

        model_state.update(filtered_state)
        self._model.load_state_dict(model_state, strict=False)
        self._is_trained = True


class IterativeRefinementCompleter(TrajectoryCompleter):
    def __init__(
        self,
        d_model: int = 128,
        num_layers: int = 2,
        num_heads: int = 4,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        num_steps: int = 3,
        score_threshold: float = 0.5,
        device: str = "cuda",
    ) -> None:
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout
        self.num_steps = num_steps
        self.score_threshold = score_threshold
        self.device = device
        self._model = None
        self._is_trained = False

    def _build_model(self) -> None:
        try:
            import torch
            import torch.nn as nn
        except ImportError as e:
            raise ImportError(
                "PyTorch is required for IterativeRefinementCompleter. "
                "Install with: pip install torch"
            ) from e

        class PositionalEncoding(nn.Module):
            def __init__(
                self,
                d_model: int,
                dropout: float = 0.1,
                max_len: int = 5000,
            ) -> None:
                super().__init__()
                self.dropout = nn.Dropout(p=dropout)

                position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
                div_term = torch.exp(
                    torch.arange(0, d_model, 2, dtype=torch.float32)
                    * (-math.log(10000.0) / d_model)
                )
                pe = torch.zeros(max_len, d_model, dtype=torch.float32)
                pe[:, 0::2] = torch.sin(position * div_term)
                pe[:, 1::2] = torch.cos(position * div_term)
                pe = pe.unsqueeze(0)
                self.register_buffer("pe", pe)

            def forward(self, x: "torch.Tensor") -> "torch.Tensor":
                if x.size(1) > self.pe.size(1):
                    msg = (
                        "Sequence length exceeds maximum positional encoding length: "
                        f"{x.size(1)} > {self.pe.size(1)}"
                    )
                    raise ValueError(msg)
                x = x + self.pe[:, : x.size(1)]
                return self.dropout(x)

        class DeltaTransformerModel(nn.Module):
            def __init__(
                self,
                d_model: int,
                num_layers: int,
                num_heads: int,
                dim_feedforward: int,
                dropout: float,
            ) -> None:
                super().__init__()
                self.input_proj = nn.Linear(2, d_model)
                encoder_layer = nn.TransformerEncoderLayer(
                    d_model=d_model,
                    nhead=num_heads,
                    dim_feedforward=dim_feedforward,
                    dropout=dropout,
                    batch_first=True,
                )
                self.pos_encoder = PositionalEncoding(d_model=d_model, dropout=dropout)
                self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
                self.output_proj = nn.Sequential(
                    nn.Linear(d_model, d_model),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(d_model, 2),
                )

            def forward(self, x: "torch.Tensor") -> "torch.Tensor":
                h = self.input_proj(x)
                h = self.pos_encoder(h)
                h = self.encoder(h)
                return self.output_proj(h)

        if self.device.startswith("cuda") and not torch.cuda.is_available():
            self.device = "cpu"

        self._model = DeltaTransformerModel(
            d_model=self.d_model,
            num_layers=self.num_layers,
            num_heads=self.num_heads,
            dim_feedforward=self.dim_feedforward,
            dropout=self.dropout,
        )
        self._model = self._model.to(self.device)

    def complete(
        self,
        xy: NDArray[np.float32],
        visibility: NDArray[np.bool_],
        score: NDArray[np.float32],
    ) -> CompletionResult:
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

        input_data = xy.astype(np.float32).copy()
        input_data[:, 0] /= 1920.0
        input_data[:, 1] /= 1080.0
        xk = torch.from_numpy(input_data).float().unsqueeze(0).to(self.device)

        self._model.eval()
        with torch.no_grad():
            for _ in range(max(int(self.num_steps), 1)):
                delta = self._model(xk)
                xk = xk + delta

        pred_np = xk[0].cpu().numpy()
        pred_np[:, 0] *= 1920.0
        pred_np[:, 1] *= 1080.0

        new_visibility = np.where(
            visibility & (score >= self.score_threshold), 1, 0
        ).astype(np.int32)
        confidence = score.copy()
        gaps_filled = 0
        for t in range(T):
            if new_visibility[t] != 1:
                new_visibility[t] = 2
                confidence[t] = 0.5
                gaps_filled += 1

        return CompletionResult(
            xy=pred_np.astype(np.float32),
            visibility=new_visibility,
            confidence=confidence.astype(np.float32),
            gaps_filled=gaps_filled,
        )

    def load_from_checkpoint(self, checkpoint_path: str | Path) -> None:
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


class TransformerCompleter(TrajectoryCompleter):
    """Transformer-based model for trajectory completion.

    Uses a Transformer encoder over normalized 2D coordinates [x_norm, y_norm]
    to predict completed trajectories.
    """

    def __init__(
        self,
        d_model: int = 128,
        num_layers: int = 2,
        num_heads: int = 4,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        score_threshold: float = 0.5,
        device: str = "cuda",
    ) -> None:
        """Initialize Transformer completer.

        Args:
            d_model: Transformer embedding dimension.
            num_layers: Number of encoder layers.
            num_heads: Number of attention heads.
            dim_feedforward: Feedforward layer dimension.
            dropout: Dropout probability.
            score_threshold: Minimum score for reliable detection.
            device: Device for computation.

        """
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout
        self.score_threshold = score_threshold
        self.device = device
        self._model = None
        self._is_trained = False

    def _build_model(self) -> None:
        """Build the PyTorch Transformer model (lazy initialization)."""
        try:
            import torch
            import torch.nn as nn
        except ImportError as e:
            raise ImportError(
                "PyTorch is required for TransformerCompleter. "
                "Install with: pip install torch"
            ) from e

        class PositionalEncoding(nn.Module):
            def __init__(
                self,
                d_model: int,
                dropout: float = 0.1,
                max_len: int = 5000,
            ) -> None:
                super().__init__()
                self.dropout = nn.Dropout(p=dropout)

                position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
                div_term = torch.exp(
                    torch.arange(0, d_model, 2, dtype=torch.float32)
                    * (-math.log(10000.0) / d_model)
                )
                pe = torch.zeros(max_len, d_model, dtype=torch.float32)
                pe[:, 0::2] = torch.sin(position * div_term)
                pe[:, 1::2] = torch.cos(position * div_term)
                pe = pe.unsqueeze(0)
                self.register_buffer("pe", pe)

            def forward(self, x: "torch.Tensor") -> "torch.Tensor":
                # x: [B, T, D]
                if x.size(1) > self.pe.size(1):
                    msg = (
                        "Sequence length exceeds maximum positional encoding length: "
                        f"{x.size(1)} > {self.pe.size(1)}"
                    )
                    raise ValueError(msg)
                x = x + self.pe[:, : x.size(1)]
                return self.dropout(x)

        class TransformerModel(nn.Module):
            def __init__(
                self,
                d_model: int,
                num_layers: int,
                num_heads: int,
                dim_feedforward: int,
                dropout: float,
            ) -> None:
                super().__init__()
                self.input_proj = nn.Linear(2, d_model)
                encoder_layer = nn.TransformerEncoderLayer(
                    d_model=d_model,
                    nhead=num_heads,
                    dim_feedforward=dim_feedforward,
                    dropout=dropout,
                    batch_first=True,
                )
                self.pos_encoder = PositionalEncoding(d_model=d_model, dropout=dropout)
                self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
                self.output_proj = nn.Sequential(
                    nn.Linear(d_model, d_model),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(d_model, 2),
                )

            def forward(self, x: "torch.Tensor") -> "torch.Tensor":
                # x: [B, T, 2]
                h = self.input_proj(x)
                h = self.pos_encoder(h)
                h = self.encoder(h)
                return self.output_proj(h)

        self._model = TransformerModel(
            d_model=self.d_model,
            num_layers=self.num_layers,
            num_heads=self.num_heads,
            dim_feedforward=self.dim_feedforward,
            dropout=self.dropout,
        )
        self._model = self._model.to(self.device)

    def complete(
        self,
        xy: NDArray[np.float32],
        visibility: NDArray[np.bool_],
        score: NDArray[np.float32],
    ) -> CompletionResult:
        """Complete trajectory using Transformer model."""
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
        input_data = xy.astype(np.float32).copy()

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
                "Checkpoint must contain a 'state_dict' or 'model_state_dict': "
                f"{checkpoint_path}"
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
        learned_model: TrajectoryCompleter | None = None,
        coarse_model: TrajectoryCompleter | None = None,
        refiner: IterativeRefinementCompleter | None = None,
    ) -> None:
        """Initialize hybrid completer.

        Args:
            physics_gap_threshold: Max gap size for physics interpolation.
            max_gap: Max gap size for fallback physics interpolation.
            score_threshold: Minimum score for reliable detection.
            learned_model: Optional learned model for complex gaps.

        """
        self.score_threshold = score_threshold
        self.physics_gap_threshold = physics_gap_threshold
        self.max_gap = max_gap
        self.coarse = coarse_model or learned_model
        self.refiner = refiner

    def complete(
        self,
        xy: NDArray[np.float32],
        visibility: NDArray[np.bool_],
        score: NDArray[np.float32],
    ) -> CompletionResult:
        """Complete trajectory using hybrid approach."""
        if self.coarse is None:
            coarse = TransformerCompleter(score_threshold=self.score_threshold, device="cpu")
        else:
            coarse = self.coarse

        result = coarse.complete(xy, visibility, score)
        if self.refiner is None or not self.refiner._is_trained:
            return result

        refined = self.refiner.complete(result.xy, result.visibility == 1, result.confidence)
        result.xy = refined.xy
        return result


def create_completer(
    method: Literal[
        "physics", "bilstm", "transformer", "refiner", "hybrid"
    ] = "hybrid",
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

    elif method == "refiner":
        d_model = int(kwargs.pop("d_model", 128))
        num_layers = int(kwargs.pop("num_layers", 2))
        num_heads = int(kwargs.pop("num_heads", 4))
        dim_ff = int(kwargs.pop("dim_feedforward", 256))
        dropout = float(kwargs.pop("dropout", 0.1))
        num_steps = int(kwargs.pop("num_steps", 3))

        completer = IterativeRefinementCompleter(
            d_model=d_model,
            num_layers=num_layers,
            num_heads=num_heads,
            dim_feedforward=dim_ff,
            dropout=dropout,
            num_steps=num_steps,
            score_threshold=score_threshold,
        )
        if checkpoint_path is not None:
            completer.load_from_checkpoint(checkpoint_path)
        return completer

    elif method == "transformer":
        d_model = int(kwargs.pop("d_model", 128))
        num_layers = int(kwargs.pop("num_layers", 2))
        num_heads = int(kwargs.pop("num_heads", 4))
        dim_ff = int(kwargs.pop("dim_feedforward", 256))
        dropout = float(kwargs.pop("dropout", 0.1))

        completer = TransformerCompleter(
            d_model=d_model,
            num_layers=num_layers,
            num_heads=num_heads,
            dim_feedforward=dim_ff,
            dropout=dropout,
            score_threshold=score_threshold,
        )
        if checkpoint_path is not None:
            completer.load_from_checkpoint(checkpoint_path)
        return completer

    elif method == "hybrid":
        coarse_method = str(kwargs.pop("coarse_method", "transformer")).lower()
        coarse_checkpoint = kwargs.pop("coarse_checkpoint_path", checkpoint_path)
        refiner_checkpoint = kwargs.pop("refiner_checkpoint_path", None)
        refiner_steps = int(kwargs.pop("num_steps", 3))

        coarse: TrajectoryCompleter | None = None
        if coarse_method == "bilstm":
            coarse = BiLSTMCompleter(score_threshold=score_threshold)
            if coarse_checkpoint is not None:
                coarse.load_from_checkpoint(coarse_checkpoint)
        elif coarse_method == "transformer":
            coarse = TransformerCompleter(score_threshold=score_threshold)
            if coarse_checkpoint is not None:
                coarse.load_from_checkpoint(coarse_checkpoint)
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

        refiner = IterativeRefinementCompleter(
            num_steps=refiner_steps,
            score_threshold=score_threshold,
        )
        if refiner_checkpoint is not None:
            refiner.load_from_checkpoint(refiner_checkpoint)

        _ = physics_gap_threshold
        return HybridCompleter(
            coarse_model=coarse,
            refiner=refiner,
            score_threshold=score_threshold,
        )

    else:
        raise ValueError(f"Unknown completion method: {method}")
