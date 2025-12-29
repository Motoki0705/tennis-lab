"""E2E tests for PLCS data validation using types.py schemas.

These tests verify that:
1. Generated NPZ scene files conform to PLCSSceneMeta and PLCSCameraParams schemas
2. Dataset samples conform to PLCSFrameBatch and PLCSSequenceBatch schemas
3. Tensor shapes, dtypes, and value ranges are valid
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from src.plcs.data.dataset import SceneDataset
from src.plcs.generate_dataset.io.dataset_io import load_scene
from tests.e2e.fixtures.plcs_fixtures import create_minimal_plcs_dataset
from tests.e2e.validation import (
    validate_plcs_camera_params,
    validate_plcs_frame_batch,
    validate_plcs_scene_meta,
    validate_tensor_range,
    validate_tensor_shape,
    validate_visibility_mask,
)


@pytest.mark.e2e
class TestPLCSGeneratedSceneValidation:
    """Tests for validating generated PLCS scene NPZ files."""

    @pytest.fixture
    def plcs_dataset_dir(self, tmp_path: Path) -> Path:
        """Create a minimal PLCS dataset for testing."""
        return create_minimal_plcs_dataset(tmp_path / "plcs_data", num_scenes=5)

    def test_scene_meta_schema(self, plcs_dataset_dir: Path) -> None:
        """Test that scene metadata conforms to PLCSSceneMeta schema."""
        scenes_dir = plcs_dataset_dir / "scenes"
        scene_files = list(scenes_dir.glob("*.npz"))
        assert len(scene_files) > 0, "No scene files found"

        for scene_file in scene_files:
            scene = load_scene(scene_file)
            errors = validate_plcs_scene_meta(scene["meta"])
            assert not errors, f"{scene_file.name}: {errors}"

    def test_camera_params_schema(self, plcs_dataset_dir: Path) -> None:
        """Test that camera parameters conform to PLCSCameraParams schema."""
        scenes_dir = plcs_dataset_dir / "scenes"
        scene_files = list(scenes_dir.glob("*.npz"))

        for scene_file in scene_files:
            scene = load_scene(scene_file)
            for i, cam in enumerate(scene["cameras"]):
                errors = validate_plcs_camera_params(cam["params"])
                assert not errors, f"{scene_file.name} cam_{i}: {errors}"

    def test_position_data_shapes(self, plcs_dataset_dir: Path) -> None:
        """Test that position data has correct shapes."""
        scenes_dir = plcs_dataset_dir / "scenes"
        scene_files = list(scenes_dir.glob("*.npz"))

        for scene_file in scene_files:
            scene = load_scene(scene_file)
            num_frames = scene["meta"]["num_frames"]

            # position: (T, 3)
            assert scene["position"].shape == (num_frames, 3), (
                f"position shape mismatch: {scene['position'].shape}"
            )

            # rotation: (T, 2) for [sin(yaw), cos(yaw)]
            assert scene["rotation"].shape == (num_frames, 2), (
                f"rotation shape mismatch: {scene['rotation'].shape}"
            )

    def test_rotation_normalized(self, plcs_dataset_dir: Path) -> None:
        """Test that rotation values are in valid range [-1, 1]."""
        scenes_dir = plcs_dataset_dir / "scenes"
        scene_files = list(scenes_dir.glob("*.npz"))

        for scene_file in scene_files:
            scene = load_scene(scene_file)
            rotation = torch.from_numpy(scene["rotation"])

            err = validate_tensor_range(rotation, -1.0, 1.0, "rotation")
            assert err is None, f"{scene_file.name}: {err}"

    def test_camera_keypoint_data_shapes(self, plcs_dataset_dir: Path) -> None:
        """Test that camera keypoint data has correct shapes."""
        scenes_dir = plcs_dataset_dir / "scenes"
        scene_files = list(scenes_dir.glob("*.npz"))

        for scene_file in scene_files:
            scene = load_scene(scene_file)
            num_frames = scene["meta"]["num_frames"]

            for i, cam in enumerate(scene["cameras"]):
                # human_kp_uv: (T, 17, 2)
                assert cam["human_kp_uv"].shape == (num_frames, 17, 2), (
                    f"cam_{i} human_kp_uv shape: {cam['human_kp_uv'].shape}"
                )

                # human_kp_visible: (T, 17)
                assert cam["human_kp_visible"].shape == (num_frames, 17), (
                    f"cam_{i} human_kp_visible shape: {cam['human_kp_visible'].shape}"
                )

                # court_kp_uv: (T, 20, 2)
                assert cam["court_kp_uv"].shape == (num_frames, 20, 2), (
                    f"cam_{i} court_kp_uv shape: {cam['court_kp_uv'].shape}"
                )

                # court_kp_visible: (T, 20)
                assert cam["court_kp_visible"].shape == (num_frames, 20), (
                    f"cam_{i} court_kp_visible shape: {cam['court_kp_visible'].shape}"
                )

    def test_visibility_ratio_in_range(self, plcs_dataset_dir: Path) -> None:
        """Test that visibility ratios are in valid range [0, 1]."""
        scenes_dir = plcs_dataset_dir / "scenes"
        scene_files = list(scenes_dir.glob("*.npz"))

        for scene_file in scene_files:
            scene = load_scene(scene_file)

            for i, cam in enumerate(scene["cameras"]):
                ratio = cam["human_visibility_ratio"]
                assert 0.0 <= ratio <= 1.0, (
                    f"{scene_file.name} cam_{i}: human visibility ratio {ratio} out of range"
                )


@pytest.mark.e2e
class TestPLCSDatasetSampleValidation:
    """Tests for validating SceneDataset samples."""

    @pytest.fixture
    def plcs_dataset(self, tmp_path: Path) -> SceneDataset:
        """Create a minimal PLCS dataset and return Dataset instance."""
        dataset_dir = create_minimal_plcs_dataset(
            tmp_path / "plcs_data", num_scenes=5
        )
        return SceneDataset(
            scene_dir=dataset_dir,
            config=None,
            augment=False,
        )

    def test_sample_has_required_keys(self, plcs_dataset: SceneDataset) -> None:
        """Test that dataset samples have all required keys."""
        if len(plcs_dataset) == 0:
            pytest.skip("Dataset is empty")

        sample = plcs_dataset[0]
        required_keys = [
            "human_kp",
            "court_kp",
            "human_vis",
            "court_vis",
            "position",
            "rotation",
        ]
        for key in required_keys:
            assert key in sample, f"Missing key: {key}"

    def test_sample_schema_validation(self, plcs_dataset: SceneDataset) -> None:
        """Test that dataset samples conform to PLCSFrameBatch schema."""
        if len(plcs_dataset) == 0:
            pytest.skip("Dataset is empty")

        # Test first few samples
        for i in range(min(3, len(plcs_dataset))):
            sample = plcs_dataset[i]
            errors = validate_plcs_frame_batch(sample)
            assert not errors, f"Sample {i} validation errors: {errors}"

    def test_sample_tensor_shapes(self, plcs_dataset: SceneDataset) -> None:
        """Test that sample tensors have correct shapes."""
        if len(plcs_dataset) == 0:
            pytest.skip("Dataset is empty")

        sample = plcs_dataset[0]

        # human_kp: (34,) = 17 keypoints * 2 coords
        err = validate_tensor_shape(sample["human_kp"], (34,), "human_kp")
        assert err is None, err

        # court_kp: (40,) = 20 keypoints * 2 coords
        err = validate_tensor_shape(sample["court_kp"], (40,), "court_kp")
        assert err is None, err

        # position: (3,)
        err = validate_tensor_shape(sample["position"], (3,), "position")
        assert err is None, err

        # rotation: (2,)
        err = validate_tensor_shape(sample["rotation"], (2,), "rotation")
        assert err is None, err

    def test_sample_rotation_normalized(self, plcs_dataset: SceneDataset) -> None:
        """Test that rotation values are normalized sin/cos in [-1, 1]."""
        if len(plcs_dataset) == 0:
            pytest.skip("Dataset is empty")

        sample = plcs_dataset[0]
        err = validate_tensor_range(sample["rotation"], -1.0, 1.0, "rotation")
        assert err is None, err

    def test_sample_visibility_masks_valid(self, plcs_dataset: SceneDataset) -> None:
        """Test that visibility masks contain valid values."""
        if len(plcs_dataset) == 0:
            pytest.skip("Dataset is empty")

        sample = plcs_dataset[0]

        human_vis_err = validate_visibility_mask(sample["human_vis"], "human_vis")
        assert human_vis_err is None, human_vis_err

        court_vis_err = validate_visibility_mask(sample["court_vis"], "court_vis")
        assert court_vis_err is None, court_vis_err

    def test_sample_tensor_dtypes(self, plcs_dataset: SceneDataset) -> None:
        """Test that sample tensors have correct dtypes."""
        if len(plcs_dataset) == 0:
            pytest.skip("Dataset is empty")

        sample = plcs_dataset[0]

        # Float tensors
        float_keys = ["human_kp", "court_kp", "position", "rotation"]
        for key in float_keys:
            assert sample[key].dtype in (torch.float32, torch.float64), (
                f"{key} dtype: {sample[key].dtype}"
            )


@pytest.mark.e2e
class TestPLCSDataLoaderBatchValidation:
    """Tests for validating collated batches from DataLoader."""

    @pytest.fixture
    def plcs_dataloader(self, tmp_path: Path) -> torch.utils.data.DataLoader:
        """Create a minimal PLCS DataLoader."""
        dataset_dir = create_minimal_plcs_dataset(
            tmp_path / "plcs_data", num_scenes=5
        )
        dataset = SceneDataset(
            scene_dir=dataset_dir,
            config=None,
            augment=False,
        )
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=2,
            shuffle=False,
        )

    def test_batch_shapes_consistent(
        self, plcs_dataloader: torch.utils.data.DataLoader
    ) -> None:
        """Test that batch tensor shapes are consistent."""
        batch = next(iter(plcs_dataloader))

        B = batch["human_kp"].shape[0]

        # Verify batch dimension is consistent
        assert batch["court_kp"].shape[0] == B
        assert batch["human_vis"].shape[0] == B
        assert batch["court_vis"].shape[0] == B
        assert batch["position"].shape[0] == B
        assert batch["rotation"].shape[0] == B

    def test_batch_tensor_shapes(
        self, plcs_dataloader: torch.utils.data.DataLoader
    ) -> None:
        """Test that batched tensors have correct shapes."""
        batch = next(iter(plcs_dataloader))

        B = batch["human_kp"].shape[0]

        # human_kp: (B, 34)
        assert batch["human_kp"].shape == (B, 34), (
            f"human_kp shape: {batch['human_kp'].shape}"
        )

        # court_kp: (B, 40)
        assert batch["court_kp"].shape == (B, 40), (
            f"court_kp shape: {batch['court_kp'].shape}"
        )

        # position: (B, 3)
        assert batch["position"].shape == (B, 3), (
            f"position shape: {batch['position'].shape}"
        )

        # rotation: (B, 2)
        assert batch["rotation"].shape == (B, 2), (
            f"rotation shape: {batch['rotation'].shape}"
        )
