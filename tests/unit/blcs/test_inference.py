"""Tests for BLCS inference components."""

import torch

from src.blcs.inference.predictor import BLCSPredictor
from src.blcs.models.blcs_model import BLCSModel


class TestBLCSPredictor:
    """Tests for BLCSPredictor."""

    def test_init_with_model(self) -> None:
        """Test initialization with pre-built model."""
        model = BLCSModel(hidden_dim=64, num_layers=2)
        predictor = BLCSPredictor(model=model, device="cpu")

        assert predictor.model is not None

    def test_init_with_config(self) -> None:
        """Test initialization with config."""
        from src.blcs.utils.config import load_config

        config = load_config()
        predictor = BLCSPredictor(config=config, device="cpu")

        assert predictor.model is not None

    def test_predict_single_sample(self) -> None:
        """Test prediction on single sample."""
        model = BLCSModel(hidden_dim=64, num_layers=2)
        predictor = BLCSPredictor(model=model, device="cpu")

        ball_uv = torch.randn(30, 2)
        court_kp = torch.randn(20, 2)

        outputs = predictor.predict(ball_uv, court_kp, denormalize=False)

        assert "position" in outputs
        assert outputs["position"].shape == (1, 30, 3)

    def test_predict_batch(self) -> None:
        """Test prediction on batch."""
        model = BLCSModel(hidden_dim=64, num_layers=2)
        predictor = BLCSPredictor(model=model, device="cpu")

        ball_uv = torch.randn(4, 30, 2)
        court_kp = torch.randn(4, 20, 2)

        outputs = predictor.predict(ball_uv, court_kp, denormalize=False)

        assert outputs["position"].shape == (4, 30, 3)

    def test_predict_with_mask(self) -> None:
        """Test prediction with visibility mask."""
        model = BLCSModel(hidden_dim=64, num_layers=2)
        predictor = BLCSPredictor(model=model, device="cpu")

        ball_uv = torch.randn(30, 2)
        court_kp = torch.randn(20, 2)
        ball_mask = torch.ones(30)
        ball_mask[25:] = 0

        outputs = predictor.predict(
            ball_uv, court_kp, ball_mask=ball_mask, denormalize=False
        )

        assert outputs["position"].shape == (1, 30, 3)

    def test_predict_denormalize(self) -> None:
        """Test prediction with denormalization."""
        from src.blcs.utils.constants import NORM_SCALE_X, NORM_SCALE_Y, NORM_SCALE_Z

        model = BLCSModel(hidden_dim=64, num_layers=2)
        predictor = BLCSPredictor(model=model, device="cpu")

        ball_uv = torch.randn(30, 2)
        court_kp = torch.randn(20, 2)

        outputs_norm = predictor.predict(ball_uv, court_kp, denormalize=False)
        outputs_meters = predictor.predict(ball_uv, court_kp, denormalize=True)

        # Denormalized should be scaled
        scale = torch.tensor([NORM_SCALE_X, NORM_SCALE_Y, NORM_SCALE_Z])
        expected = outputs_norm["position"] * scale

        assert torch.allclose(outputs_meters["position"], expected, atol=1e-5)


class TestPredictorEdgeCases:
    """Tests for predictor edge cases."""

    def test_single_frame(self) -> None:
        """Test prediction with single frame."""
        model = BLCSModel(hidden_dim=64, num_layers=2)
        predictor = BLCSPredictor(model=model, device="cpu")

        ball_uv = torch.randn(1, 2)
        court_kp = torch.randn(20, 2)

        outputs = predictor.predict(ball_uv, court_kp, denormalize=False)
        assert outputs["position"].shape == (1, 1, 3)

    def test_long_sequence(self) -> None:
        """Test prediction with long sequence."""
        model = BLCSModel(hidden_dim=64, num_layers=2)
        predictor = BLCSPredictor(model=model, device="cpu")

        ball_uv = torch.randn(120, 2)
        court_kp = torch.randn(20, 2)

        outputs = predictor.predict(ball_uv, court_kp, denormalize=False)
        assert outputs["position"].shape == (1, 120, 3)

    def test_partial_masked(self) -> None:
        """Test prediction with some frames masked."""
        model = BLCSModel(hidden_dim=64, num_layers=2)
        predictor = BLCSPredictor(model=model, device="cpu")

        ball_uv = torch.randn(30, 2)
        court_kp = torch.randn(20, 2)
        ball_mask = torch.ones(30)
        ball_mask[20:] = 0  # Mask last 10 frames

        outputs = predictor.predict(
            ball_uv, court_kp, ball_mask=ball_mask, denormalize=False
        )
        assert outputs["position"].shape == (1, 30, 3)
