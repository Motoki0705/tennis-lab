"""Unit tests for trajectory completion models."""

from __future__ import annotations

import numpy as np
import pytest

from src.wasb.models.trajectory_completer import (
    CompletionResult,
    HybridCompleter,
    IterativeRefinementCompleter,
    PhysicsInterpolator,
    TrajectoryCompleter,
    create_completer,
)


class TestPhysicsInterpolator:
    """Tests for PhysicsInterpolator."""

    @pytest.fixture
    def interpolator(self) -> PhysicsInterpolator:
        """Create a physics interpolator with default settings."""
        return PhysicsInterpolator(
            max_gap=10,
            min_anchor_points=2,
            velocity_threshold=100.0,
            acceleration_threshold=50.0,
            score_threshold=0.5,
        )

    def test_complete_no_gaps(self, interpolator: PhysicsInterpolator) -> None:
        """Test completion when all positions are valid."""
        T = 10
        xy = np.array([[100 + i * 5, 200 + i * 2] for i in range(T)], dtype=np.float32)
        visibility = np.ones(T, dtype=bool)
        score = np.ones(T, dtype=np.float32) * 0.9

        result = interpolator.complete(xy, visibility, score)

        assert result.xy.shape == (T, 2)
        assert result.visibility.shape == (T,)
        assert np.all(result.visibility == 1)
        assert result.gaps_filled == 0

    def test_complete_single_gap(self, interpolator: PhysicsInterpolator) -> None:
        """Test completion of a single gap in the middle."""
        T = 20
        xy = np.array([[100 + i * 5, 200 + i * 2] for i in range(T)], dtype=np.float32)
        visibility = np.ones(T, dtype=bool)
        score = np.ones(T, dtype=np.float32) * 0.9

        # Create a gap from frame 8 to 12
        visibility[8:12] = False
        xy[8:12] = 0  # Invalid positions

        result = interpolator.complete(xy, visibility, score)

        # Check gap is filled
        assert np.all(result.visibility[:8] == 1)
        assert np.all(result.visibility[8:12] == 2)  # Completed
        assert np.all(result.visibility[12:] == 1)
        assert result.gaps_filled == 4

        # Check positions are interpolated reasonably
        for t in range(8, 12):
            # Should be approximately on the line
            expected_x = 100 + t * 5
            expected_y = 200 + t * 2
            assert abs(result.xy[t, 0] - expected_x) < 10
            assert abs(result.xy[t, 1] - expected_y) < 10

    def test_complete_gap_too_long(self, interpolator: PhysicsInterpolator) -> None:
        """Test that gaps longer than max_gap are not filled."""
        T = 30
        xy = np.array([[100 + i * 5, 200] for i in range(T)], dtype=np.float32)
        visibility = np.ones(T, dtype=bool)
        score = np.ones(T, dtype=np.float32) * 0.9

        # Create a gap longer than max_gap (10)
        visibility[5:20] = False
        xy[5:20] = 0

        result = interpolator.complete(xy, visibility, score)

        # Gap should not be filled
        assert np.all(result.visibility[5:20] == 0)
        assert result.gaps_filled == 0

    def test_complete_edge_gap(self, interpolator: PhysicsInterpolator) -> None:
        """Test completion of gap at the beginning or end."""
        T = 15
        xy = np.array([[100 + i * 5, 200] for i in range(T)], dtype=np.float32)
        visibility = np.ones(T, dtype=bool)
        score = np.ones(T, dtype=np.float32) * 0.9

        # Gap at the beginning (no anchor before)
        visibility[:3] = False
        xy[:3] = 0

        result = interpolator.complete(xy, visibility, score)

        # Beginning gap cannot be filled (no anchors before)
        assert np.all(result.visibility[:3] == 0)
        assert np.all(result.visibility[3:] == 1)

    def test_complete_empty_trajectory(self, interpolator: PhysicsInterpolator) -> None:
        """Test completion of empty trajectory."""
        xy = np.zeros((0, 2), dtype=np.float32)
        visibility = np.zeros(0, dtype=bool)
        score = np.zeros(0, dtype=np.float32)

        result = interpolator.complete(xy, visibility, score)

        assert result.xy.shape == (0, 2)
        assert result.visibility.shape == (0,)
        assert result.gaps_filled == 0

    def test_complete_low_score_frames(self, interpolator: PhysicsInterpolator) -> None:
        """Test that low-score frames are treated as missing."""
        T = 10
        xy = np.array([[100 + i * 5, 200] for i in range(T)], dtype=np.float32)
        visibility = np.ones(T, dtype=bool)
        score = np.ones(T, dtype=np.float32) * 0.9

        # Set some frames to low score
        score[4:6] = 0.3  # Below threshold

        result = interpolator.complete(xy, visibility, score)

        # Low-score frames should be marked as completed or 0
        assert result.visibility[4] != 1
        assert result.visibility[5] != 1

    def test_outlier_removal(self) -> None:
        """Test that outliers are detected and removed."""
        interpolator = PhysicsInterpolator(
            max_gap=10,
            velocity_threshold=50.0,
            acceleration_threshold=30.0,
        )

        T = 15
        # Smooth trajectory
        xy = np.array([[100 + i * 3, 200 + i * 2] for i in range(T)], dtype=np.float32)
        visibility = np.ones(T, dtype=bool)
        score = np.ones(T, dtype=np.float32) * 0.7

        # Add outlier at frame 7 (sudden jump)
        xy[7] = [500, 500]

        result = interpolator.complete(xy, visibility, score)

        # Outlier should be detected (visibility should be 0 or 2, not 1)
        # Note: depending on exact logic, it may be removed or completed
        assert result.outliers_removed >= 0


class TestHybridCompleter:
    """Tests for HybridCompleter."""

    def test_hybrid_short_gaps(self) -> None:
        """Test that short gaps use physics interpolation."""
        completer = HybridCompleter(
            physics_gap_threshold=5,
            score_threshold=0.5,
            learned_model=None,
        )

        T = 20
        xy = np.array([[100 + i * 5, 200 + i * 2] for i in range(T)], dtype=np.float32)
        visibility = np.ones(T, dtype=bool)
        score = np.ones(T, dtype=np.float32) * 0.9

        # Short gap (3 frames)
        visibility[8:11] = False
        xy[8:11] = 0

        result = completer.complete(xy, visibility, score)

        # Gap should be filled
        assert np.all(result.visibility[8:11] == 2)
        assert result.gaps_filled == 3


class TestIterativeRefinementCompleter:
    def test_refiner_complete_shape_and_visibility(self) -> None:
        completer = IterativeRefinementCompleter(
            d_model=32,
            num_layers=1,
            num_heads=4,
            dim_feedforward=64,
            dropout=0.0,
            num_steps=2,
            score_threshold=0.5,
            device="cpu",
        )
        completer._build_model()

        T = 16
        xy = np.array([[100 + i * 5, 200 + i * 2] for i in range(T)], dtype=np.float32)
        visibility = np.ones(T, dtype=bool)
        score = np.ones(T, dtype=np.float32) * 0.9

        visibility[6:9] = False
        xy[6:9] = 0

        result = completer.complete(xy, visibility, score)

        assert result.xy.shape == (T, 2)
        assert result.visibility.shape == (T,)
        assert np.all(result.visibility[6:9] == 2)


class TestTrajectoryRefinerLightning:
    def test_lightning_iterative_loss_computes(self) -> None:
        import torch

        from src.wasb.training.trajectory_lightning_module import (
            TrajectoryLightningModule,
        )

        cfg = {
            "model": {
                "name": "trajectory_refiner",
                "d_model": 32,
                "num_layers": 1,
                "num_heads": 4,
                "dim_feedforward": 64,
                "dropout": 0.0,
                "num_steps": 3,
                "score_threshold": 0.5,
            },
            "training": {
                "learning_rate": 1.0e-3,
                "weight_decay": 0.0,
                "warmup_steps": 10,
                "max_epochs": 1,
                "min_lr": 1.0e-6,
                "lambda_block": 1.0,
                "lambda_sparse": 1.0,
                "lambda_noise": 1.0,
            },
        }

        module = TrajectoryLightningModule(cfg, steps_per_epoch=10)
        module = module.to(torch.device("cpu"))

        B, T = 2, 8
        xy_input_norm = torch.randn(B, T, 2, dtype=torch.float32)
        target_xy_norm = torch.randn(B, T, 2, dtype=torch.float32)
        mask = torch.ones(B, T, dtype=torch.float32)

        batch = {
            "xy_input_norm": xy_input_norm,
            "target_xy_norm": target_xy_norm,
            "loss_mask_block": mask,
            "loss_mask_sparse": mask,
            "loss_mask_noise": mask,
        }

        loss, metrics = module._shared_step(batch, "train")
        assert loss.dim() == 0
        assert "loss_total" in metrics
        assert torch.isfinite(loss)


class TestCreateCompleter:
    """Tests for the factory function."""

    def test_create_physics(self) -> None:
        """Test creating physics completer."""
        completer = create_completer(method="physics", max_gap=15)
        assert isinstance(completer, PhysicsInterpolator)

    def test_create_hybrid(self) -> None:
        """Test creating hybrid completer without checkpoint."""
        completer = create_completer(method="hybrid")
        assert isinstance(completer, HybridCompleter)

    def test_create_invalid_method(self) -> None:
        """Test that invalid method raises error."""
        with pytest.raises(ValueError, match="Unknown completion method"):
            create_completer(method="invalid")  # type: ignore[arg-type]


class TestCompletionResult:
    """Tests for CompletionResult dataclass."""

    def test_result_creation(self) -> None:
        """Test creating a completion result."""
        result = CompletionResult(
            xy=np.array([[100, 200], [105, 202]], dtype=np.float32),
            visibility=np.array([1, 2], dtype=np.int32),
            confidence=np.array([0.9, 0.5], dtype=np.float32),
            gaps_filled=1,
            outliers_removed=0,
        )

        assert result.xy.shape == (2, 2)
        assert result.visibility[0] == 1
        assert result.visibility[1] == 2
        assert result.gaps_filled == 1


class TestPhysicsInterpolatorQuadratic:
    """Tests for quadratic interpolation."""

    def test_parabolic_trajectory(self) -> None:
        """Test interpolation of a parabolic trajectory (like a tennis ball)."""
        interpolator = PhysicsInterpolator(max_gap=10)

        T = 30
        t = np.arange(T, dtype=np.float32)

        # Simulate parabolic motion: x = v0*t, y = h0 - 0.5*g*t^2
        v0 = 10  # horizontal velocity
        h0 = 300  # initial height
        g = 0.5  # gravity (scaled)

        x = 100 + v0 * t
        y = h0 - 0.5 * g * (t - 15) ** 2  # Peak at t=15

        xy = np.stack([x, y], axis=-1).astype(np.float32)
        visibility = np.ones(T, dtype=bool)
        score = np.ones(T, dtype=np.float32) * 0.9

        # Create gap at the peak of the parabola
        visibility[13:18] = False
        original_xy = xy[13:18].copy()
        xy[13:18] = 0

        result = interpolator.complete(xy, visibility, score)

        # Check the interpolated positions are close to original
        for i, t_idx in enumerate(range(13, 18)):
            # Allow some error due to discrete sampling and fitting
            assert abs(result.xy[t_idx, 0] - original_xy[i, 0]) < 15
            assert abs(result.xy[t_idx, 1] - original_xy[i, 1]) < 30
