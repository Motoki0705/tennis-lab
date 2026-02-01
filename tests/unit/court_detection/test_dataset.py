"""Unit tests for court keypoint dataset."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from src.court_detection.data.dataset import CourtKeypointDataset


def test_court_keypoint_dataset_numpy_import() -> None:
    """Test that numpy is properly imported and dataset can load samples."""
    # Create a temporary directory with test data
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)

        # Create a dummy image
        img_path = tmpdir_path / "test_image.jpg"
        test_image = Image.new("RGB", (640, 480), color=(128, 128, 128))
        test_image.save(img_path)

        # Create a dummy keypoints JSON file
        keypoints_data = {
            "image_path": str(img_path),
            "keypoints": [
                {"x": 100, "y": 100, "visibility": 1},
                {"x": 200, "y": 150, "visibility": 1},
                {"x": 300, "y": 200, "visibility": 0},
            ]
            + [{"x": 0, "y": 0, "visibility": 0} for _ in range(17)],  # Fill to 20
        }

        json_path = tmpdir_path / "test_keypoints.json"
        with open(json_path, "w") as f:
            json.dump(keypoints_data, f)

        # Create dataset
        dataset = CourtKeypointDataset(
            data_dir=tmpdir_path,
            split="train",
            input_size=(256, 256),
            heatmap_size=(64, 64),
        )

        # Verify dataset has samples
        assert len(dataset) == 1

        # Test __getitem__ - this would fail with NameError if numpy is not imported
        sample = dataset[0]

        # Verify sample structure
        assert "image" in sample
        assert "keypoints" in sample
        assert "visibility" in sample
        assert "heatmaps" in sample

        # Verify shapes
        assert sample["image"].shape == (3, 256, 256)
        assert sample["keypoints"].shape == (20, 2)
        assert sample["visibility"].shape == (20,)
        assert sample["heatmaps"].shape == (20, 64, 64)

        # Verify image values are normalized [0, 1]
        assert sample["image"].min() >= 0.0
        assert sample["image"].max() <= 1.0

        # Verify keypoints are normalized [0, 1]
        assert sample["keypoints"].min() >= 0.0
        assert sample["keypoints"].max() <= 1.0


def test_court_keypoint_dataset_empty_directory() -> None:
    """Test dataset with no keypoint files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        dataset = CourtKeypointDataset(
            data_dir=tmpdir,
            split="train",
        )
        assert len(dataset) == 0


def test_court_keypoint_dataset_split_distribution() -> None:
    """Test that dataset splits samples correctly."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)

        # Create dummy image
        img_path = tmpdir_path / "test_image.jpg"
        test_image = Image.new("RGB", (640, 480), color=(128, 128, 128))
        test_image.save(img_path)

        # Create multiple keypoint files (10 files)
        for i in range(10):
            keypoints_data = {
                "image_path": str(img_path),
                "keypoints": [{"x": 0, "y": 0, "visibility": 0} for _ in range(20)],
            }
            json_path = tmpdir_path / f"sample_{i:02d}_keypoints.json"
            with open(json_path, "w") as f:
                json.dump(keypoints_data, f)

        # Test splits
        train_dataset = CourtKeypointDataset(data_dir=tmpdir_path, split="train")
        val_dataset = CourtKeypointDataset(data_dir=tmpdir_path, split="val")
        test_dataset = CourtKeypointDataset(data_dir=tmpdir_path, split="test")

        # Verify split sizes (80% train, 10% val, 10% test)
        assert len(train_dataset) == 8  # 80% of 10
        assert len(val_dataset) == 1  # 10% of 10
        assert len(test_dataset) == 1  # 10% of 10
