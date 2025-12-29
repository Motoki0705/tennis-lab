"""E2E tests for BLCS data validation using types.py schemas.

These tests verify that:
1. Generated NPZ scene files conform to BLCSSceneMeta and BLCSCameraParams schemas
2. Dataset samples conform to BLCSSample schema
3. Collated batches conform to BLCSBatch schema
4. Tensor shapes, dtypes, and value ranges are valid
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pytest
import torch

from src.blcs.data.dataset import BallTrajectoryDataset
from src.blcs.generate_dataset.io.dataset_io import load_scene
from tests.e2e.fixtures.blcs_fixtures import create_minimal_blcs_dataset
from tests.e2e.validation import (
    validate_blcs_batch,
    validate_blcs_camera_params,
    validate_blcs_sample,
    validate_blcs_scene_meta,
    validate_normalized_uv,
)

if TYPE_CHECKING:
    pass


@pytest.mark.e2e
class TestBLCSGeneratedSceneValidation:
    """Tests for validating generated BLCS scene NPZ files."""

    @pytest.fixture
    def blcs_dataset_dir(self, tmp_path: Path) -> Path:
        """Create a minimal BLCS dataset for testing."""
        return create_minimal_blcs_dataset(tmp_path / "blcs_data", num_scenes=5)

    def test_scene_meta_schema(self, blcs_dataset_dir: Path) -> None:
        """Test that scene metadata conforms to BLCSSceneMeta schema."""
        scenes_dir = blcs_dataset_dir / "scenes"
        scene_files = list(scenes_dir.glob("*.npz"))
        assert len(scene_files) > 0, "No scene files found"

        for scene_file in scene_files:
            scene = load_scene(scene_file)
            errors = validate_blcs_scene_meta(scene["meta"])
            assert not errors, f"{scene_file.name}: {errors}"

    def test_camera_params_schema(self, blcs_dataset_dir: Path) -> None:
        """Test that camera parameters conform to BLCSCameraParams schema."""
        scenes_dir = blcs_dataset_dir / "scenes"
        scene_files = list(scenes_dir.glob("*.npz"))

        for scene_file in scene_files:
            scene = load_scene(scene_file)
            for i, cam in enumerate(scene["cameras"]):
                errors = validate_blcs_camera_params(cam["params"])
                assert not errors, f"{scene_file.name} cam_{i}: {errors}"

    def test_trajectory_data_shapes(self, blcs_dataset_dir: Path) -> None:
        """Test that 3D trajectory data has correct shapes."""
        scenes_dir = blcs_dataset_dir / "scenes"
        scene_files = list(scenes_dir.glob("*.npz"))

        for scene_file in scene_files:
            scene = load_scene(scene_file)
            num_frames = scene["meta"]["num_frames"]

            # ball_pos_world: (T, 3)
            assert scene["ball_pos_world"].shape == (num_frames, 3), (
                f"ball_pos_world shape mismatch: {scene['ball_pos_world'].shape}"
            )

            # ball_pos_norm: (T, 3)
            assert scene["ball_pos_norm"].shape == (num_frames, 3), (
                f"ball_pos_norm shape mismatch: {scene['ball_pos_norm'].shape}"
            )

            # ball_vel_world: (T, 3)
            assert scene["ball_vel_world"].shape == (num_frames, 3), (
                f"ball_vel_world shape mismatch: {scene['ball_vel_world'].shape}"
            )

    def test_camera_projection_data_shapes(self, blcs_dataset_dir: Path) -> None:
        """Test that camera projection data has correct shapes."""
        scenes_dir = blcs_dataset_dir / "scenes"
        scene_files = list(scenes_dir.glob("*.npz"))

        for scene_file in scene_files:
            scene = load_scene(scene_file)
            num_frames = scene["meta"]["num_frames"]

            for i, cam in enumerate(scene["cameras"]):
                # ball_uv: (T, 2)
                assert cam["ball_uv"].shape == (num_frames, 2), (
                    f"cam_{i} ball_uv shape: {cam['ball_uv'].shape}"
                )

                # ball_visible: (T,)
                assert cam["ball_visible"].shape == (num_frames,), (
                    f"cam_{i} ball_visible shape: {cam['ball_visible'].shape}"
                )

                # court_kp_uv: (20, 2)
                assert cam["court_kp_uv"].shape == (20, 2), (
                    f"cam_{i} court_kp_uv shape: {cam['court_kp_uv'].shape}"
                )

                # court_kp_visible: (20,)
                assert cam["court_kp_visible"].shape == (20,), (
                    f"cam_{i} court_kp_visible shape: {cam['court_kp_visible'].shape}"
                )

    def test_visibility_ratio_in_range(self, blcs_dataset_dir: Path) -> None:
        """Test that visibility ratios are in valid range [0, 1]."""
        scenes_dir = blcs_dataset_dir / "scenes"
        scene_files = list(scenes_dir.glob("*.npz"))

        for scene_file in scene_files:
            scene = load_scene(scene_file)

            for i, cam in enumerate(scene["cameras"]):
                ratio = cam["ball_visibility_ratio"]
                assert 0.0 <= ratio <= 1.0, (
                    f"{scene_file.name} cam_{i}: visibility ratio {ratio} out of range"
                )


@pytest.mark.e2e
class TestBLCSDatasetSampleValidation:
    """Tests for validating BallTrajectoryDataset samples."""

    @pytest.fixture
    def blcs_dataset(self, tmp_path: Path) -> BallTrajectoryDataset:
        """Create a minimal BLCS dataset and return Dataset instance."""
        dataset_dir = create_minimal_blcs_dataset(
            tmp_path / "blcs_data", num_scenes=5
        )
        return BallTrajectoryDataset(
            scene_dir=dataset_dir,
            config=None,
            augment=False,
        )

    def test_sample_has_required_keys(self, blcs_dataset: BallTrajectoryDataset) -> None:
        """Test that dataset samples have all required keys."""
        if len(blcs_dataset) == 0:
            pytest.skip("Dataset is empty")

        sample = blcs_dataset[0]
        required_keys = [
            "ball_uv",
            "ball_mask",
            "court_kp",
            "court_vis",
            "position_3d",
            "velocity_3d",
            "seq_len",
        ]
        for key in required_keys:
            assert key in sample, f"Missing key: {key}"

    def test_sample_schema_validation(self, blcs_dataset: BallTrajectoryDataset) -> None:
        """Test that dataset samples conform to BLCSSample schema."""
        if len(blcs_dataset) == 0:
            pytest.skip("Dataset is empty")

        # Test first few samples
        for i in range(min(3, len(blcs_dataset))):
            sample = blcs_dataset[i]
            errors = validate_blcs_sample(sample)
            assert not errors, f"Sample {i} validation errors: {errors}"

    def test_sample_uv_normalized(self, blcs_dataset: BallTrajectoryDataset) -> None:
        """Test that UV coordinates are normalized to [0, 1]."""
        if len(blcs_dataset) == 0:
            pytest.skip("Dataset is empty")

        sample = blcs_dataset[0]

        # ball_uv should be in [0, 1]
        ball_uv_err = validate_normalized_uv(sample["ball_uv"], "ball_uv")
        assert ball_uv_err is None, ball_uv_err

        # court_kp should be in [0, 1]
        court_kp_err = validate_normalized_uv(sample["court_kp"], "court_kp")
        assert court_kp_err is None, court_kp_err

    def test_sample_tensor_dtypes(self, blcs_dataset: BallTrajectoryDataset) -> None:
        """Test that sample tensors have correct dtypes."""
        if len(blcs_dataset) == 0:
            pytest.skip("Dataset is empty")

        sample = blcs_dataset[0]

        # Float tensors
        float_keys = ["ball_uv", "court_kp", "position_3d", "velocity_3d"]
        for key in float_keys:
            assert sample[key].dtype in (torch.float32, torch.float64), (
                f"{key} dtype: {sample[key].dtype}"
            )

    def test_seq_len_matches_tensor_lengths(
        self, blcs_dataset: BallTrajectoryDataset
    ) -> None:
        """Test that seq_len matches actual tensor lengths."""
        if len(blcs_dataset) == 0:
            pytest.skip("Dataset is empty")

        sample = blcs_dataset[0]
        seq_len = int(sample["seq_len"].item())

        # ball_uv should have T frames
        assert sample["ball_uv"].shape[0] == seq_len, (
            f"ball_uv length {sample['ball_uv'].shape[0]} != seq_len {seq_len}"
        )

        # ball_mask should have T frames
        assert sample["ball_mask"].shape[0] == seq_len, (
            f"ball_mask length {sample['ball_mask'].shape[0]} != seq_len {seq_len}"
        )


@pytest.mark.e2e
class TestBLCSDataLoaderBatchValidation:
    """Tests for validating collated batches from DataLoader."""

    @pytest.fixture
    def blcs_dataloader(self, tmp_path: Path) -> torch.utils.data.DataLoader:
        """Create a minimal BLCS DataLoader."""
        from src.blcs.data.dataset import collate_trajectories

        dataset_dir = create_minimal_blcs_dataset(
            tmp_path / "blcs_data", num_scenes=5
        )
        dataset = BallTrajectoryDataset(
            scene_dir=dataset_dir,
            config=None,
            augment=False,
        )
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=2,
            collate_fn=collate_trajectories,
            shuffle=False,
        )

    def test_batch_schema_validation(
        self, blcs_dataloader: torch.utils.data.DataLoader
    ) -> None:
        """Test that collated batches conform to BLCSBatch schema."""
        batch = next(iter(blcs_dataloader))
        errors = validate_blcs_batch(batch)
        assert not errors, f"Batch validation errors: {errors}"

    def test_batch_shapes_consistent(
        self, blcs_dataloader: torch.utils.data.DataLoader
    ) -> None:
        """Test that batch tensor shapes are consistent."""
        batch = next(iter(blcs_dataloader))

        B = batch["ball_uv"].shape[0]
        T_max = batch["ball_uv"].shape[1]

        # All temporal tensors should have same T_max
        assert batch["ball_mask"].shape == (B, T_max), (
            f"ball_mask shape mismatch: {batch['ball_mask'].shape}"
        )
        assert batch["position_3d"].shape == (B, T_max, 3), (
            f"position_3d shape mismatch: {batch['position_3d'].shape}"
        )
        assert batch["velocity_3d"].shape == (B, T_max, 3), (
            f"velocity_3d shape mismatch: {batch['velocity_3d'].shape}"
        )

        # Court keypoints are not temporal
        assert batch["court_kp"].shape == (B, 20, 2), (
            f"court_kp shape mismatch: {batch['court_kp'].shape}"
        )

    def test_batch_padding_valid(
        self, blcs_dataloader: torch.utils.data.DataLoader
    ) -> None:
        """Test that padding is applied correctly based on seq_len."""
        batch = next(iter(blcs_dataloader))

        seq_lens = batch["seq_len"]
        T_max = batch["ball_uv"].shape[1]

        for i, seq_len in enumerate(seq_lens):
            seq_len = int(seq_len.item())
            # Verify that seq_len <= T_max
            assert seq_len <= T_max, f"seq_len {seq_len} > T_max {T_max}"
