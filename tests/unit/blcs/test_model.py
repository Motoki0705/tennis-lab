"""Tests for BLCS model."""

import torch
from omegaconf import OmegaConf

from src.blcs.models.blcs_model import BLCSModel
from src.blcs.models.components.encoders import (
    BallTrajectoryEncoder,
    CourtBallCrossAttention,
    CourtContextEncoder,
    TemporalPositionalEncoding,
)
from src.blcs.models.components.heads import Trajectory3DHead, VelocityHead


class TestTemporalPositionalEncoding:
    """Tests for TemporalPositionalEncoding."""

    def test_forward(self) -> None:
        """Test forward pass."""
        pe = TemporalPositionalEncoding(d_model=128, max_len=200)

        x = torch.randn(4, 50, 128)
        output = pe(x)

        assert output.shape == (4, 50, 128)

    def test_different_seq_lengths(self) -> None:
        """Test with different sequence lengths."""
        pe = TemporalPositionalEncoding(d_model=64, max_len=200)

        for seq_len in [10, 50, 100, 150]:
            x = torch.randn(2, seq_len, 64)
            output = pe(x)
            assert output.shape == (2, seq_len, 64)


class TestCourtContextEncoder:
    """Tests for CourtContextEncoder."""

    def test_forward(self) -> None:
        """Test forward pass."""
        encoder = CourtContextEncoder(
            num_court_kp=20,
            hidden_dim=128,
            num_heads=4,
            num_layers=2,
        )

        court_kp = torch.randn(4, 20, 2)
        output = encoder(court_kp)

        assert output.shape == (4, 128)

    def test_with_visibility(self) -> None:
        """Test with visibility mask."""
        encoder = CourtContextEncoder(num_court_kp=20, hidden_dim=128)

        court_kp = torch.randn(4, 20, 2)
        court_vis = torch.ones(4, 20)
        court_vis[:, 15:] = 0  # Mark some keypoints as invisible

        output = encoder(court_kp, court_vis)
        assert output.shape == (4, 128)


class TestBallTrajectoryEncoder:
    """Tests for BallTrajectoryEncoder."""

    def test_forward(self) -> None:
        """Test forward pass."""
        encoder = BallTrajectoryEncoder(
            hidden_dim=128,
            num_heads=4,
            num_layers=2,
        )

        ball_uv = torch.randn(4, 30, 2)
        output = encoder(ball_uv)

        assert output.shape == (4, 30, 128)

    def test_with_mask(self) -> None:
        """Test with attention mask."""
        encoder = BallTrajectoryEncoder(hidden_dim=64, num_layers=2)

        ball_uv = torch.randn(4, 30, 2)
        mask = torch.ones(4, 30)
        mask[:, 20:] = 0  # Mask last 10 frames

        output = encoder(ball_uv, mask)
        assert output.shape == (4, 30, 64)


class TestCourtBallCrossAttention:
    """Tests for CourtBallCrossAttention."""

    def test_forward(self) -> None:
        """Test forward pass."""
        cross_attn = CourtBallCrossAttention(
            hidden_dim=128,
            num_heads=4,
        )

        ball_features = torch.randn(4, 30, 128)
        court_context = torch.randn(4, 128)

        output = cross_attn(ball_features, court_context)
        assert output.shape == (4, 30, 128)


class TestTrajectory3DHead:
    """Tests for Trajectory3DHead."""

    def test_forward(self) -> None:
        """Test forward pass."""
        head = Trajectory3DHead(input_dim=128, hidden_dim=64)

        x = torch.randn(4, 30, 128)
        output = head(x)

        assert output.shape == (4, 30, 3)


class TestVelocityHead:
    """Tests for VelocityHead."""

    def test_forward(self) -> None:
        """Test forward pass."""
        head = VelocityHead(input_dim=128, hidden_dim=64)

        x = torch.randn(4, 30, 128)
        output = head(x)

        assert output.shape == (4, 30, 3)


class TestBLCSModel:
    """Tests for BLCSModel."""

    def test_forward(self) -> None:
        """Test forward pass."""
        model = BLCSModel(hidden_dim=128, num_layers=2, num_heads=4)

        ball_uv = torch.randn(4, 30, 2)
        court_kp = torch.randn(4, 20, 2)

        outputs = model(ball_uv, court_kp)

        assert "position" in outputs
        assert outputs["position"].shape == (4, 30, 3)

    def test_forward_with_masks(self) -> None:
        """Test forward pass with visibility masks."""
        model = BLCSModel(hidden_dim=64, num_layers=2)

        ball_uv = torch.randn(4, 30, 2)
        court_kp = torch.randn(4, 20, 2)
        ball_mask = torch.ones(4, 30)
        court_vis = torch.ones(4, 20)

        outputs = model(ball_uv, court_kp, ball_mask, court_vis)
        assert outputs["position"].shape == (4, 30, 3)

    def test_from_config(self) -> None:
        """Test model creation from config."""
        config = OmegaConf.create(
            {
                "model": {
                    "hidden_dim": 64,
                    "num_layers": 2,
                    "num_heads": 4,
                    "dropout": 0.1,
                }
            }
        )

        model = BLCSModel.from_config(config)

        assert model.hidden_dim == 64

    def test_predict(self) -> None:
        """Test predict method."""
        model = BLCSModel(hidden_dim=64, num_layers=2)

        ball_uv = torch.randn(4, 30, 2)
        court_kp = torch.randn(4, 20, 2)

        outputs = model.predict(ball_uv, court_kp)

        assert outputs["position"].shape == (4, 30, 3)

    def test_get_num_params(self) -> None:
        """Test parameter counting."""
        model = BLCSModel(hidden_dim=64, num_layers=2)
        num_params = model.get_num_params()

        assert num_params > 0
        assert isinstance(num_params, int)

    def test_variable_sequence_length(self) -> None:
        """Test with different sequence lengths."""
        model = BLCSModel(hidden_dim=64, num_layers=2)

        for seq_len in [15, 30, 60, 100]:
            ball_uv = torch.randn(2, seq_len, 2)
            court_kp = torch.randn(2, 20, 2)
            outputs = model(ball_uv, court_kp)
            assert outputs["position"].shape == (2, seq_len, 3)
