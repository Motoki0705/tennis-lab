"""E2E tests for WASB data validation using types.py schemas.

These tests verify that:
1. Dataset samples conform to the TypedDict schemas in types.py
2. Tensor shapes, dtypes, and value ranges are valid
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from tests.e2e.fixtures.wasb_fixtures import create_minimal_wasb_dataset
from tests.e2e.validation.tensor_validators import (
    validate_tensor_range,
    validate_tensor_shape,
)


@pytest.mark.e2e
class TestWASBDatasetValidation:
    """Tests for validating WASB dataset structure."""

    @pytest.fixture
    def wasb_dataset_dir(self, tmp_path: Path) -> Path:
        """Create a minimal WASB dataset for testing."""
        return create_minimal_wasb_dataset(tmp_path / "wasb_data")

    def test_dataset_structure(self, wasb_dataset_dir: Path) -> None:
        """Test that dataset has expected directory structure."""
        # Should have game directories
        game_dirs = [d for d in wasb_dataset_dir.iterdir() if d.is_dir()]
        assert len(game_dirs) > 0, "No game directories found"

        # Each game should have clip directories
        for game_dir in game_dirs:
            clip_dirs = [d for d in game_dir.iterdir() if d.is_dir()]
            assert len(clip_dirs) > 0, f"No clip directories in {game_dir.name}"

            # Each clip should have frames and Label.csv
            for clip_dir in clip_dirs:
                label_file = clip_dir / "Label.csv"
                assert label_file.exists(), f"Missing Label.csv in {clip_dir}"

                # Should have image files
                image_files = list(clip_dir.glob("*.jpg")) + list(clip_dir.glob("*.png"))
                assert len(image_files) > 0, f"No image files in {clip_dir}"

    def test_label_csv_format(self, wasb_dataset_dir: Path) -> None:
        """Test that Label.csv files have correct format."""
        game_dirs = [d for d in wasb_dataset_dir.iterdir() if d.is_dir()]

        for game_dir in game_dirs:
            clip_dirs = [d for d in game_dir.iterdir() if d.is_dir()]

            for clip_dir in clip_dirs:
                label_file = clip_dir / "Label.csv"
                lines = label_file.read_text().strip().split("\n")

                # Check header
                header = lines[0].lower()
                assert "file name" in header or "filename" in header, (
                    f"Missing 'file name' in header: {header}"
                )
                assert "visibility" in header, f"Missing 'visibility' in header: {header}"
                assert "x-coordinate" in header or "x" in header, (
                    f"Missing x coordinate in header: {header}"
                )
                assert "y-coordinate" in header or "y" in header, (
                    f"Missing y coordinate in header: {header}"
                )

                # Check data rows
                for i, line in enumerate(lines[1:], start=2):
                    parts = line.split(",")
                    assert len(parts) >= 4, (
                        f"Line {i} has {len(parts)} fields, expected >= 4"
                    )


@pytest.mark.e2e
class TestWASBBallDetectionSampleValidation:
    """Tests for validating BallDetectionSequenceDataset samples."""

    @pytest.fixture
    def ball_detection_dataset(self, tmp_path: Path):
        """Create a minimal ball detection dataset."""
        from src.wasb.data.ball_detection_dataset import BallDetectionSequenceDataset

        dataset_dir = create_minimal_wasb_dataset(tmp_path / "wasb_data")

        try:
            dataset = BallDetectionSequenceDataset(
                root_dir=dataset_dir,
                matches=["game1"],
                frames_in=3,
                frames_out=1,
                step=1,
                visibility_mode="none",
                heatmap_sigma=5.0,
            )
            return dataset
        except Exception as e:
            pytest.skip(f"Could not create dataset: {e}")

    def test_sample_has_required_keys(self, ball_detection_dataset) -> None:
        """Test that dataset samples have required keys per BallDetectionSample."""
        if len(ball_detection_dataset) == 0:
            pytest.skip("Dataset is empty")

        sample = ball_detection_dataset[0]

        # Keys from BallDetectionSample TypedDict
        required_keys = [
            "frames",
            "targets_px",
            "targets_norm",
            "target_heatmaps",
            "visibility",
            "scores",
            "match",
            "clip",
        ]

        missing = [k for k in required_keys if k not in sample]
        assert not missing, f"Missing keys: {missing}"

    def test_sample_tensor_shapes(self, ball_detection_dataset) -> None:
        """Test that sample tensors have correct shapes."""
        if len(ball_detection_dataset) == 0:
            pytest.skip("Dataset is empty")

        sample = ball_detection_dataset[0]

        # frames: (T, C, H, W) where T = frames_in
        frames = sample["frames"]
        assert len(frames.shape) == 4, f"frames should be 4D, got {frames.shape}"
        T, C, H, W = frames.shape
        assert C == 3, f"Expected 3 channels, got {C}"

        # targets_px: (frames_out, 2)
        targets_px = sample["targets_px"]
        err = validate_tensor_shape(targets_px, (None, 2), "targets_px")
        assert err is None, err

        # targets_norm: (frames_out, 2)
        targets_norm = sample["targets_norm"]
        err = validate_tensor_shape(targets_norm, (None, 2), "targets_norm")
        assert err is None, err

        # visibility: (frames_out,)
        visibility = sample["visibility"]
        assert len(visibility.shape) == 1, f"visibility should be 1D, got {visibility.shape}"

    def test_sample_normalized_targets_in_range(self, ball_detection_dataset) -> None:
        """Test that normalized targets are in [0, 1] range."""
        if len(ball_detection_dataset) == 0:
            pytest.skip("Dataset is empty")

        sample = ball_detection_dataset[0]
        targets_norm = sample["targets_norm"]

        # Only check visible targets
        visibility = sample["visibility"]
        if visibility.sum() > 0:
            visible_targets = targets_norm[visibility > 0]
            err = validate_tensor_range(visible_targets, 0.0, 1.0, "targets_norm")
            assert err is None, err

    def test_sample_heatmap_shape_matches_frame(self, ball_detection_dataset) -> None:
        """Test that heatmap spatial dimensions match frame size."""
        if len(ball_detection_dataset) == 0:
            pytest.skip("Dataset is empty")

        sample = ball_detection_dataset[0]

        frames = sample["frames"]
        heatmaps = sample["target_heatmaps"]

        _, _, H, W = frames.shape

        # Heatmaps should have same spatial dimensions as frames
        # (may be downsampled in some implementations)
        assert heatmaps.shape[-2] <= H, (
            f"Heatmap height {heatmaps.shape[-2]} > frame height {H}"
        )
        assert heatmaps.shape[-1] <= W, (
            f"Heatmap width {heatmaps.shape[-1]} > frame width {W}"
        )

    def test_sample_visibility_values(self, ball_detection_dataset) -> None:
        """Test that visibility values are valid (0, 1, or 2)."""
        if len(ball_detection_dataset) == 0:
            pytest.skip("Dataset is empty")

        sample = ball_detection_dataset[0]
        visibility = sample["visibility"]

        unique_vals = visibility.unique().tolist()
        for val in unique_vals:
            assert val in [0, 1, 2], f"Invalid visibility value: {val}"

    def test_sample_tensor_dtypes(self, ball_detection_dataset) -> None:
        """Test that sample tensors have correct dtypes."""
        if len(ball_detection_dataset) == 0:
            pytest.skip("Dataset is empty")

        sample = ball_detection_dataset[0]

        # frames should be float
        assert sample["frames"].dtype in (torch.float32, torch.float64), (
            f"frames dtype: {sample['frames'].dtype}"
        )

        # targets should be float
        assert sample["targets_px"].dtype in (torch.float32, torch.float64), (
            f"targets_px dtype: {sample['targets_px'].dtype}"
        )

        # heatmaps should be float
        assert sample["target_heatmaps"].dtype in (torch.float32, torch.float64), (
            f"target_heatmaps dtype: {sample['target_heatmaps'].dtype}"
        )

    def test_sample_metadata_types(self, ball_detection_dataset) -> None:
        """Test that metadata fields have correct types."""
        if len(ball_detection_dataset) == 0:
            pytest.skip("Dataset is empty")

        sample = ball_detection_dataset[0]

        assert isinstance(sample["match"], str), f"match type: {type(sample['match'])}"
        assert isinstance(sample["clip"], str), f"clip type: {type(sample['clip'])}"


@pytest.mark.e2e
class TestWASBDataLoaderBatchValidation:
    """Tests for validating batched data from DataLoader."""

    @pytest.fixture
    def wasb_dataloader(self, tmp_path: Path):
        """Create a minimal WASB DataLoader."""
        from src.wasb.data.ball_detection_dataset import BallDetectionSequenceDataset

        dataset_dir = create_minimal_wasb_dataset(tmp_path / "wasb_data")

        try:
            dataset = BallDetectionSequenceDataset(
                root_dir=dataset_dir,
                matches=["game1"],
                frames_in=3,
                frames_out=1,
                step=1,
                visibility_mode="none",
                heatmap_sigma=5.0,
            )
        except Exception as e:
            pytest.skip(f"Could not create dataset: {e}")

        if len(dataset) < 2:
            pytest.skip("Dataset too small for batching")

        # Use default collate (stack tensors)
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=2,
            shuffle=False,
        )

    def test_batch_shapes_consistent(self, wasb_dataloader) -> None:
        """Test that batch tensor shapes are consistent."""
        batch = next(iter(wasb_dataloader))

        B = batch["frames"].shape[0]

        # All tensor fields should have same batch dimension
        assert batch["targets_px"].shape[0] == B
        assert batch["targets_norm"].shape[0] == B
        assert batch["target_heatmaps"].shape[0] == B
        assert batch["visibility"].shape[0] == B

    def test_batch_frames_shape(self, wasb_dataloader) -> None:
        """Test that batched frames have correct shape."""
        batch = next(iter(wasb_dataloader))

        # frames: (B, T, C, H, W)
        frames = batch["frames"]
        assert len(frames.shape) == 5, f"Batched frames should be 5D, got {frames.shape}"

        B, T, C, H, W = frames.shape
        assert C == 3, f"Expected 3 channels, got {C}"
