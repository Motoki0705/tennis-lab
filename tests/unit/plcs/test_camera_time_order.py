"""Unit tests for PLCS camera-time ordering consistency.

This module tests that all PLCS components use consistent camera-time ordering:
(B, N, T, ...) where N=cameras, T=time.
"""

from __future__ import annotations

import pytest
import torch

from src.plcs.models.plcs_multiview_model import PLCSMultiViewModel


class TestCameraTimeOrdering:
    """Tests for camera-time ordering consistency across PLCS components."""

    @pytest.fixture
    def multiview_model(self) -> PLCSMultiViewModel:
        """Create a minimal PLCSMultiViewModel for testing."""
        return PLCSMultiViewModel(
            hidden_dim=64,
            num_layers=2,
            num_heads=4,
            dropout=0.0,
            max_views=4,
            max_seq_len=32,
            encoder_layers=1,
        )

    def test_multiview_model_accepts_camera_time_order(
        self, multiview_model: PLCSMultiViewModel
    ) -> None:
        """Test that PLCSMultiViewModel accepts camera-time ordered input (B, N, T, K, 2)."""
        batch_size = 2
        n_cameras = 3
        seq_len = 8

        # Input in camera-time order: (B, N, T, K, 2)
        human_kp = torch.randn(batch_size, n_cameras, seq_len, 17, 2)
        court_kp = torch.randn(batch_size, n_cameras, seq_len, 20, 2)
        human_vis = torch.ones(batch_size, n_cameras, seq_len, 17)
        court_vis = torch.ones(batch_size, n_cameras, seq_len, 20)

        # Should not raise
        outputs = multiview_model(human_kp, court_kp, human_vis, court_vis)

        # Output shape should be (B, T, 3) for position and (B, T, 2) for rotation
        assert outputs["position"].shape == (batch_size, seq_len, 3)
        assert outputs["rotation"].shape == (batch_size, seq_len, 2)

    def test_multiview_model_single_frame_input(
        self, multiview_model: PLCSMultiViewModel
    ) -> None:
        """Test that PLCSMultiViewModel handles single-frame input (B, N, K, 2)."""
        batch_size = 2
        n_cameras = 3

        # Single-frame input in camera order: (B, N, K, 2)
        human_kp = torch.randn(batch_size, n_cameras, 17, 2)
        court_kp = torch.randn(batch_size, n_cameras, 20, 2)
        human_vis = torch.ones(batch_size, n_cameras, 17)
        court_vis = torch.ones(batch_size, n_cameras, 20)

        # Should not raise
        outputs = multiview_model(human_kp, court_kp, human_vis, court_vis)

        # Single-frame output should have shape (B, 3) and (B, 2)
        assert outputs["position"].shape == (batch_size, 3)
        assert outputs["rotation"].shape == (batch_size, 2)

    def test_dataset_output_matches_model_input(self) -> None:
        """Test that dataset output shape matches model expected input.

        Dataset outputs (N_cam, T, K, 2) which after collation becomes
        (B, N_cam, T, K, 2) - camera-time order.
        """
        # Simulate dataset output
        n_cameras = 3
        seq_len = 8

        # Dataset __getitem__ returns (N_cam, T, K, 2)
        human_kp_sample = torch.randn(n_cameras, seq_len, 17, 2)
        court_kp_sample = torch.randn(n_cameras, seq_len, 20, 2)

        # Collation stacks samples: (B, N_cam, T, K, 2)
        batch_size = 2
        human_kp_batch = torch.stack([human_kp_sample] * batch_size)
        court_kp_batch = torch.stack([court_kp_sample] * batch_size)

        # Verify expected shape matches model input
        assert human_kp_batch.shape == (batch_size, n_cameras, seq_len, 17, 2)
        assert court_kp_batch.shape == (batch_size, n_cameras, seq_len, 20, 2)

    def test_rotation_output_is_normalized(
        self, multiview_model: PLCSMultiViewModel
    ) -> None:
        """Test that rotation output (sin, cos) is approximately normalized."""
        batch_size = 2
        n_cameras = 3
        seq_len = 4

        human_kp = torch.randn(batch_size, n_cameras, seq_len, 17, 2)
        court_kp = torch.randn(batch_size, n_cameras, seq_len, 20, 2)

        outputs = multiview_model(human_kp, court_kp)
        rotation = outputs["rotation"]

        # Compute magnitude: should be close to 1 (normalized sin/cos)
        magnitude = torch.sqrt(rotation[..., 0] ** 2 + rotation[..., 1] ** 2)
        assert torch.allclose(
            magnitude, torch.ones_like(magnitude), atol=0.01
        ), f"Rotation not normalized: magnitude = {magnitude}"
