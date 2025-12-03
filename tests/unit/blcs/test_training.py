"""Tests for BLCS training components."""

import torch
from omegaconf import OmegaConf

from src.blcs.training.lightning_module import BLCSLightningModule
from src.blcs.training.losses import (
    BLCSLoss,
    position_error_meters,
    smoothness_loss,
    trajectory_position_loss,
    velocity_loss,
)
from src.blcs.training.metrics import BLCSMetrics, compute_trajectory_metrics


class TestLosses:
    """Tests for loss functions."""

    def test_trajectory_position_loss_zero(self) -> None:
        """Test position loss is zero for identical inputs."""
        pred = torch.tensor([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]])
        target = torch.tensor([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]])

        loss = trajectory_position_loss(pred, target)
        assert loss.item() == 0.0

    def test_trajectory_position_loss_nonzero(self) -> None:
        """Test position loss with difference."""
        pred = torch.tensor([[[1.0, 2.0, 3.0]]])
        target = torch.tensor([[[1.5, 2.0, 3.0]]])

        loss = trajectory_position_loss(pred, target)
        assert loss.item() > 0.0

    def test_trajectory_position_loss_with_mask(self) -> None:
        """Test position loss with mask."""
        pred = torch.randn(4, 30, 3)
        target = torch.randn(4, 30, 3)
        mask = torch.ones(4, 30)
        mask[:, 20:] = 0  # Ignore last 10 frames

        loss = trajectory_position_loss(pred, target, mask)
        assert loss.item() >= 0.0

    def test_velocity_loss(self) -> None:
        """Test velocity loss computation."""
        # Create linear trajectory (constant velocity)
        pred = torch.linspace(0, 1, 30).unsqueeze(0).unsqueeze(-1).expand(2, 30, 3)
        target = pred.clone()

        loss = velocity_loss(pred, target)
        assert torch.isclose(loss, torch.tensor(0.0), atol=1e-5)

    def test_velocity_loss_with_mask(self) -> None:
        """Test velocity loss with mask."""
        pred = torch.randn(4, 30, 3)
        target = torch.randn(4, 30, 3)
        mask = torch.ones(4, 30)

        loss = velocity_loss(pred, target, mask)
        assert loss.item() >= 0.0

    def test_smoothness_loss(self) -> None:
        """Test smoothness loss (penalizes high acceleration)."""
        # Linear trajectory should have zero acceleration
        t = torch.linspace(0, 1, 30).unsqueeze(0).unsqueeze(-1).expand(2, 30, 3)
        pred = t.clone()

        loss = smoothness_loss(pred)
        assert torch.isclose(loss, torch.tensor(0.0), atol=1e-5)

    def test_smoothness_loss_nonlinear(self) -> None:
        """Test smoothness loss with non-linear trajectory."""
        # Quadratic trajectory has constant acceleration
        t = torch.linspace(0, 1, 30).unsqueeze(0).unsqueeze(-1).expand(2, 30, 3)
        pred = t**2

        loss = smoothness_loss(pred)
        assert loss.item() > 0.0

    def test_position_error_meters(self) -> None:
        """Test position error in meters."""
        pred = torch.zeros(2, 10, 3)
        target = torch.zeros(2, 10, 3)

        error = position_error_meters(pred, target)
        # Allow small epsilon due to sqrt(1e-8) term
        assert error.item() < 1e-3

    def test_blcs_loss(self) -> None:
        """Test combined BLCS loss."""
        loss_fn = BLCSLoss(
            position_weight=1.0,
            velocity_weight=0.1,
            smoothness_weight=0.05,
        )

        pred = torch.randn(4, 30, 3)
        target = torch.randn(4, 30, 3)
        mask = torch.ones(4, 30)

        losses = loss_fn(pred, target, mask)

        assert "total" in losses
        assert "position" in losses
        assert "velocity" in losses
        assert "smoothness" in losses
        assert losses["total"] >= 0


class TestMetrics:
    """Tests for BLCSMetrics."""

    def test_update_and_compute(self) -> None:
        """Test metrics update and compute."""
        metrics = BLCSMetrics()

        pred = torch.randn(10, 30, 3)
        target = pred + 0.1 * torch.randn(10, 30, 3)
        mask = torch.ones(10, 30)

        batch_metrics = metrics.update(pred, target, mask)

        assert "position_error_m" in batch_metrics
        assert "x_error_m" in batch_metrics
        assert "y_error_m" in batch_metrics
        assert "z_error_m" in batch_metrics

        final_metrics = metrics.compute()
        assert "mean_position_error_m" in final_metrics
        assert "position_accuracy_0_3m" in final_metrics
        assert "position_accuracy_0_6m" in final_metrics
        assert "position_accuracy_1_2m" in final_metrics
        assert "endpoint_accuracy_0_5m" in final_metrics
        assert "endpoint_accuracy_1m" in final_metrics

    def test_reset(self) -> None:
        """Test metrics reset."""
        metrics = BLCSMetrics()

        pred = torch.randn(5, 30, 3)
        target = pred.clone()
        mask = torch.ones(5, 30)
        metrics.update(pred, target, mask)

        metrics.reset()
        result = metrics.compute()

        assert result["mean_position_error_m"] == 0.0

    def test_perfect_prediction(self) -> None:
        """Test metrics for perfect prediction."""
        metrics = BLCSMetrics(position_threshold_m=0.3)

        pred = torch.zeros(5, 30, 3)
        target = torch.zeros(5, 30, 3)
        mask = torch.ones(5, 30)

        metrics.update(pred, target, mask)
        result = metrics.compute()

        # Allow small epsilon due to sqrt(1e-8) term
        assert result["mean_position_error_m"] < 1e-3
        assert result["position_accuracy_0_3m"] > 0.99

    def test_compute_trajectory_metrics(self) -> None:
        """Test compute_trajectory_metrics function."""
        pred = torch.randn(4, 30, 3)
        target = torch.randn(4, 30, 3)

        result = compute_trajectory_metrics(pred, target)

        assert "position_error_m" in result
        assert "x_error_m" in result
        assert "y_error_m" in result
        assert "z_error_m" in result


class TestLightningModule:
    """Tests for BLCSLightningModule."""

    def test_init(self) -> None:
        """Test module initialization."""
        config = OmegaConf.create(
            {
                "model": {"hidden_dim": 64, "num_layers": 2},
                "training": {"learning_rate": 1e-4},
                "data": {"batch_size": 32},
            }
        )

        module = BLCSLightningModule(config)
        assert module.model is not None
        assert module.learning_rate == 1e-4

    def test_forward(self) -> None:
        """Test forward pass."""
        config = OmegaConf.create(
            {
                "model": {"hidden_dim": 64, "num_layers": 2},
            }
        )

        module = BLCSLightningModule(config)

        ball_uv = torch.randn(4, 30, 2)
        court_kp = torch.randn(4, 20, 2)

        outputs = module(ball_uv, court_kp)

        assert outputs["position"].shape == (4, 30, 3)

    def test_training_step(self) -> None:
        """Test training step."""
        config = OmegaConf.create(
            {
                "model": {"hidden_dim": 64, "num_layers": 2},
            }
        )

        module = BLCSLightningModule(config)

        batch = {
            "ball_uv": torch.randn(4, 30, 2),
            "court_kp": torch.randn(4, 20, 2),
            "ball_mask": torch.ones(4, 30),
            "court_vis": torch.ones(4, 20),
            "position_3d": torch.randn(4, 30, 3),
        }

        loss = module.training_step(batch, 0)
        assert loss.shape == ()
        assert loss >= 0

    def test_configure_optimizers(self) -> None:
        """Test optimizer configuration."""
        config = OmegaConf.create(
            {
                "model": {"hidden_dim": 64},
                "training": {
                    "learning_rate": 1e-4,
                    "weight_decay": 1e-5,
                    "warmup_steps": 100,
                    "max_epochs": 10,
                },
                "data": {"batch_size": 32},
                "simulation": {"num_train_scenes": 1000},
            }
        )

        module = BLCSLightningModule(config)
        opt_config = module.configure_optimizers()

        assert "optimizer" in opt_config
        assert "lr_scheduler" in opt_config
