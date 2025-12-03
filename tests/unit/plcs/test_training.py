"""Tests for PLCS training components."""

import torch
from omegaconf import OmegaConf

from src.plcs.training.lightning_module import PLCSLightningModule
from src.plcs.training.losses import (
    PLCSLoss,
    angular_error,
    position_loss,
    rotation_loss,
)
from src.plcs.training.metrics import PLCSMetrics


class TestLosses:
    """Tests for loss functions."""

    def test_position_loss(self) -> None:
        """Test position loss computation."""
        pred = torch.tensor([[1.0, 2.0, 3.0]])
        target = torch.tensor([[1.0, 2.0, 3.0]])

        loss = position_loss(pred, target)
        assert loss.item() == 0.0

    def test_position_loss_nonzero(self) -> None:
        """Test position loss with difference."""
        pred = torch.tensor([[1.0, 2.0, 3.0]])
        target = torch.tensor([[1.5, 2.0, 3.0]])

        loss = position_loss(pred, target)
        assert loss.item() > 0.0

    def test_rotation_loss_aligned(self) -> None:
        """Test rotation loss when aligned."""
        pred = torch.tensor([[0.0, 1.0]])  # 0 degrees
        target = torch.tensor([[0.0, 1.0]])

        loss = rotation_loss(pred, target)
        assert torch.isclose(loss, torch.tensor(0.0), atol=1e-5)

    def test_rotation_loss_opposite(self) -> None:
        """Test rotation loss when opposite."""
        pred = torch.tensor([[0.0, 1.0]])  # 0 degrees
        target = torch.tensor([[0.0, -1.0]])  # 180 degrees

        loss = rotation_loss(pred, target)
        assert torch.isclose(loss, torch.tensor(2.0), atol=1e-5)

    def test_angular_error(self) -> None:
        """Test angular error computation."""
        pred = torch.tensor([[0.0, 1.0]])  # 0 degrees
        target = torch.tensor([[1.0, 0.0]])  # 90 degrees

        error = angular_error(pred, target)
        # Should be ~pi/2 radians
        assert torch.isclose(error[0], torch.tensor(1.5708), atol=0.01)

    def test_plcs_loss(self) -> None:
        """Test combined PLCS loss."""
        loss_fn = PLCSLoss(position_weight=1.0, rotation_weight=1.0)

        pred_pos = torch.randn(4, 3)
        pred_rot = torch.randn(4, 2)
        pred_rot = torch.nn.functional.normalize(pred_rot, dim=-1)
        target_pos = torch.randn(4, 3)
        target_rot = torch.randn(4, 2)
        target_rot = torch.nn.functional.normalize(target_rot, dim=-1)

        losses = loss_fn(pred_pos, pred_rot, target_pos, target_rot)

        assert "total" in losses
        assert "position" in losses
        assert "rotation" in losses
        assert losses["total"] >= 0


class TestMetrics:
    """Tests for PLCSMetrics."""

    def test_update_and_compute(self) -> None:
        """Test metrics update and compute."""
        metrics = PLCSMetrics()

        pred_pos = torch.randn(10, 3)
        pred_rot = torch.randn(10, 2)
        pred_rot = torch.nn.functional.normalize(pred_rot, dim=-1)
        target_pos = pred_pos + 0.1 * torch.randn(10, 3)
        target_rot = pred_rot

        batch_metrics = metrics.update(pred_pos, pred_rot, target_pos, target_rot)

        assert "position_error_m" in batch_metrics
        assert "angular_error_deg" in batch_metrics

        final_metrics = metrics.compute()
        assert "position_error_m" in final_metrics
        assert "position_error_std_m" in final_metrics

    def test_reset(self) -> None:
        """Test metrics reset."""
        metrics = PLCSMetrics()

        pred = torch.randn(5, 3)
        rot = torch.nn.functional.normalize(torch.randn(5, 2), dim=-1)
        metrics.update(pred, rot, pred, rot)

        metrics.reset()
        result = metrics.compute()

        assert result["position_error_m"] == 0.0


class TestLightningModule:
    """Tests for PLCSLightningModule."""

    def test_init(self) -> None:
        """Test module initialization."""
        config = OmegaConf.create(
            {
                "model": {"hidden_dim": 64, "num_layers": 2},
                "training": {"learning_rate": 1e-4},
                "data": {"batch_size": 32},
            }
        )

        module = PLCSLightningModule(config)
        assert module.model is not None
        assert module.learning_rate == 1e-4

    def test_forward(self) -> None:
        """Test forward pass."""
        config = OmegaConf.create(
            {
                "model": {"hidden_dim": 64, "num_layers": 2},
            }
        )

        module = PLCSLightningModule(config)

        human_kp = torch.randn(4, 34)
        court_kp = torch.randn(4, 40)

        outputs = module(human_kp, court_kp)

        assert outputs["position"].shape == (4, 3)
        assert outputs["rotation"].shape == (4, 2)

    def test_training_step(self) -> None:
        """Test training step."""
        config = OmegaConf.create(
            {
                "model": {"hidden_dim": 64, "num_layers": 2},
            }
        )

        module = PLCSLightningModule(config)

        batch = {
            "human_kp": torch.randn(4, 34),
            "court_kp": torch.randn(4, 40),
            "human_vis": torch.ones(4, 17),
            "court_vis": torch.ones(4, 20),
            "position": torch.randn(4, 3),
            "rotation": torch.nn.functional.normalize(torch.randn(4, 2), dim=-1),
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
                "data": {"num_scenes_per_epoch": 1000, "batch_size": 32},
            }
        )

        module = PLCSLightningModule(config)
        opt_config = module.configure_optimizers()

        assert "optimizer" in opt_config
        assert "lr_scheduler" in opt_config
