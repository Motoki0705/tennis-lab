"""Tests for PLCS model."""

import torch
from omegaconf import OmegaConf

from src.plcs.models.components.encoders import (
    KeypointEncoder,
    TransformerKeypointEncoder,
)
from src.plcs.models.components.heads import CombinedHead, PositionHead, RotationHead
from src.plcs.models.plcs_model import PLCSModel


class TestKeypointEncoder:
    """Tests for KeypointEncoder."""

    def test_forward(self) -> None:
        """Test forward pass."""
        encoder = KeypointEncoder(
            human_kp_dim=34,
            court_kp_dim=40,
            hidden_dim=128,
        )

        human_kp = torch.randn(4, 34)
        court_kp = torch.randn(4, 40)

        output = encoder(human_kp, court_kp)
        assert output.shape == (4, 128)


class TestTransformerKeypointEncoder:
    """Tests for TransformerKeypointEncoder."""

    def test_forward(self) -> None:
        """Test forward pass."""
        encoder = TransformerKeypointEncoder(
            num_human_kp=17,
            num_court_kp=20,
            hidden_dim=128,
            num_heads=4,
            num_layers=2,
        )

        human_kp = torch.randn(4, 34)
        court_kp = torch.randn(4, 40)

        output = encoder(human_kp, court_kp)
        assert output.shape == (4, 128)

    def test_forward_with_visibility(self) -> None:
        """Test forward pass with visibility masks."""
        encoder = TransformerKeypointEncoder(
            num_human_kp=17,
            num_court_kp=20,
            hidden_dim=128,
            num_heads=4,
            num_layers=2,
        )

        human_kp = torch.randn(4, 34)
        court_kp = torch.randn(4, 40)
        human_vis = torch.ones(4, 17)
        court_vis = torch.ones(4, 20)

        output = encoder(human_kp, court_kp, human_vis, court_vis)
        assert output.shape == (4, 128)


class TestPositionHead:
    """Tests for PositionHead."""

    def test_forward(self) -> None:
        """Test forward pass."""
        head = PositionHead(input_dim=128, hidden_dim=64, output_dim=3)

        x = torch.randn(4, 128)
        output = head(x)

        assert output.shape == (4, 3)


class TestRotationHead:
    """Tests for RotationHead."""

    def test_forward(self) -> None:
        """Test forward pass."""
        head = RotationHead(input_dim=128, hidden_dim=64)

        x = torch.randn(4, 128)
        output = head(x)

        assert output.shape == (4, 2)

    def test_output_is_normalized(self) -> None:
        """Test that output is normalized to unit circle."""
        head = RotationHead(input_dim=128)

        x = torch.randn(4, 128)
        output = head(x)

        norms = output.norm(dim=-1)
        assert torch.allclose(norms, torch.ones(4), atol=1e-5)


class TestCombinedHead:
    """Tests for CombinedHead."""

    def test_forward(self) -> None:
        """Test forward pass."""
        head = CombinedHead(input_dim=128)

        x = torch.randn(4, 128)
        position, rotation = head(x)

        assert position.shape == (4, 3)
        assert rotation.shape == (4, 2)


class TestPLCSModel:
    """Tests for PLCSModel."""

    def test_forward_mlp_encoder(self) -> None:
        """Test forward pass with MLP encoder."""
        model = PLCSModel(
            hidden_dim=128,
            num_layers=2,
            use_transformer=False,
        )

        human_kp = torch.randn(4, 34)
        court_kp = torch.randn(4, 40)

        outputs = model(human_kp, court_kp)

        assert "position" in outputs
        assert "rotation" in outputs
        assert outputs["position"].shape == (4, 3)
        assert outputs["rotation"].shape == (4, 2)

    def test_forward_transformer_encoder(self) -> None:
        """Test forward pass with transformer encoder."""
        model = PLCSModel(
            hidden_dim=128,
            num_layers=2,
            num_heads=4,
            use_transformer=True,
        )

        human_kp = torch.randn(4, 34)
        court_kp = torch.randn(4, 40)

        outputs = model(human_kp, court_kp)

        assert outputs["position"].shape == (4, 3)
        assert outputs["rotation"].shape == (4, 2)

    def test_from_config(self) -> None:
        """Test model creation from config."""
        config = OmegaConf.create(
            {
                "model": {
                    "hidden_dim": 64,
                    "num_layers": 2,
                    "num_heads": 4,
                    "dropout": 0.1,
                    "use_transformer": True,
                }
            }
        )

        model = PLCSModel.from_config(config)

        assert model.hidden_dim == 64
        assert model.use_transformer is True
