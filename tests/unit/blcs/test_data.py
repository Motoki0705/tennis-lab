"""Tests for BLCS data pipeline."""

import tempfile

import torch
from omegaconf import OmegaConf

from src.blcs.data.camera_projector import CameraProjector
from src.blcs.data.datamodule import BLCSDataModule
from src.blcs.data.dataset import collate_trajectories
from src.blcs.data.dataset_writer import BLCSDatasetWriter
from src.blcs.data.distribution_sampler import SamplingConfig
from src.blcs.data.scene_generator import (
    BLCSSceneData,
    BLCSSceneGenerator,
    GeneratorConfig,
)
from src.blcs.simulation.cell_manager import CellManager, ShotCategory
from src.blcs.simulation.shot_simulator import ShotSimulator


class TestShotSimulator:
    """Tests for ShotSimulator."""

    def test_generate_shot(self) -> None:
        """Test basic shot generation."""
        sim = ShotSimulator()
        shot = sim.generate_shot(from_cell=4, from_side="near")

        assert shot.trajectory.shape[1] == 3
        assert shot.velocities.shape[1] == 3
        assert shot.category in ShotCategory
        assert shot.from_cell == 4
        assert shot.from_side == "near"

    def test_shot_has_events(self) -> None:
        """Test that shots have event markers."""
        sim = ShotSimulator()

        for _ in range(5):
            shot = sim.generate_shot(from_cell=0, from_side="near")
            # At least one of these should occur
            has_event = (
                shot.t_net >= 0
                or shot.t_fence >= 0
                or shot.t_bounce1 >= 0
                or shot.t_bounce2 >= 0
            )
            assert has_event

    def test_shot_categories(self) -> None:
        """Test that different categories are generated."""
        sim = ShotSimulator()
        categories = set()

        for _ in range(50):
            shot = sim.generate_shot(from_cell=4, from_side="near")
            categories.add(shot.category)

        # Should see at least 2 different categories
        assert len(categories) >= 2


class TestCellManager:
    """Tests for CellManager."""

    def test_all_cells(self) -> None:
        """Test that we have 20 cells."""
        cm = CellManager()
        assert len(cm.get_all_cell_ids()) == 20

    def test_court_cells(self) -> None:
        """Test court interior cells."""
        cm = CellManager()
        assert len(cm.get_court_cell_ids()) == 9

    def test_exterior_cells(self) -> None:
        """Test exterior cells."""
        cm = CellManager()
        assert len(cm.get_exterior_cell_ids()) == 11

    def test_position_to_cell(self) -> None:
        """Test position to cell conversion."""
        cm = CellManager()

        # Center of far court should be cell 4
        pos = torch.tensor([0.0, 6.0, 1.0])
        cell_id = cm.position_to_cell_id(pos, "far")
        assert 0 <= cell_id <= 8  # Should be in court

    def test_sample_position(self) -> None:
        """Test sampling position in cell."""
        cm = CellManager()

        pos = cm.sample_position_in_cell(4, "near")
        assert pos.shape == (3,)
        assert pos[2] > 0  # Height should be positive


class TestCameraProjector:
    """Tests for CameraProjector."""

    def test_generate_view(self) -> None:
        """Test camera view generation."""
        sim = ShotSimulator()
        shot = sim.generate_shot(from_cell=4, from_side="near")

        cam = CameraProjector()
        view = cam.generate_camera_view(shot.trajectory)

        assert view.court_kp_uv.shape == (20, 2)
        assert view.ball_uv.shape[0] == shot.trajectory.shape[0]
        assert view.ball_uv.shape[1] == 2


class TestCollateTrajectories:
    """Tests for collate_trajectories."""

    def test_collate_same_length(self) -> None:
        """Test collate with same length sequences."""
        samples = [
            {
                "ball_uv": torch.randn(30, 2),
                "court_kp": torch.randn(20, 2),
                "court_vis": torch.ones(20),
                "position_3d": torch.randn(30, 3),
                "velocity_3d": torch.randn(30, 3),
                "ball_mask": torch.ones(30),
                "seq_len": 30,
            }
            for _ in range(4)
        ]

        batch = collate_trajectories(samples)

        assert batch["ball_uv"].shape == (4, 30, 2)
        assert batch["court_kp"].shape == (4, 20, 2)
        assert batch["position_3d"].shape == (4, 30, 3)

    def test_collate_different_lengths(self) -> None:
        """Test collate with different length sequences."""
        samples = [
            {
                "ball_uv": torch.randn(length, 2),
                "court_kp": torch.randn(20, 2),
                "court_vis": torch.ones(20),
                "position_3d": torch.randn(length, 3),
                "velocity_3d": torch.randn(length, 3),
                "ball_mask": torch.ones(length),
                "seq_len": length,
            }
            for length in [20, 30, 25, 35]
        ]

        batch = collate_trajectories(samples)

        # Should be padded to max length (35)
        assert batch["ball_uv"].shape == (4, 35, 2)
        assert batch["position_3d"].shape == (4, 35, 3)

        # Mask should indicate valid positions
        assert batch["ball_mask"].shape == (4, 35)
        assert batch["ball_mask"][0, :20].sum() == 20
        assert batch["ball_mask"][0, 20:].sum() == 0


class TestBLCSDataModule:
    """Tests for BLCSDataModule."""

    def test_init(self) -> None:
        """Test DataModule initialization."""
        config = OmegaConf.create(
            {
                "data": {
                    "scene_dir": "/nonexistent",
                    "batch_size": 16,
                    "num_workers": 0,
                },
            }
        )

        dm = BLCSDataModule(config)
        assert dm.batch_size == 16


class TestSceneGenerator:
    """Tests for BLCSSceneGenerator."""

    def test_generate_scenes(self) -> None:
        """Test scene generation with visibility filtering."""
        sampling_config = SamplingConfig(
            category_ratios={
                ShotCategory.DIRECT_NET: 0.1,
                ShotCategory.DIRECT_FENCE: 0.1,
                ShotCategory.IN_COURT: 0.5,
                ShotCategory.OUT_COURT: 0.3,
            },
            per_from_cell_samples=3,
        )
        config = GeneratorConfig(
            sampling=sampling_config,
            num_cameras_sampled=5,  # Try 5 cameras per scene
            ball_visibility_threshold=0.5,  # Lower threshold for test
            max_attempts_per_cell=50,
        )

        generator = BLCSSceneGenerator(config=config)
        scenes = list(generator.generate_scenes_for_cell(from_cell=4, side="near"))

        assert len(scenes) > 0
        for scene in scenes:
            assert isinstance(scene, BLCSSceneData)
            assert scene.from_cell == 4
            assert scene.from_side == "near"
            assert len(scene.cameras) > 0  # Must have at least 1 valid camera
            assert scene.num_cameras_sampled == 5

            # All cameras should meet visibility threshold
            for cam in scene.cameras:
                assert cam.ball_visibility_ratio >= 0.5


class TestDatasetWriter:
    """Tests for BLCSDatasetWriter (PLCS-unified format)."""

    def test_save_and_load(self) -> None:
        """Test saving and loading scenes with multiple cameras."""
        from src.blcs.data.scene_generator import CameraData
        from src.blcs.simulation.ball_physics import BallPhysics

        sim = ShotSimulator()
        cam_proj = CameraProjector()
        shot = sim.generate_shot(from_cell=4, from_side="near")
        physics = BallPhysics()

        # Generate multiple camera views
        cameras = []
        for _ in range(3):
            view = cam_proj.generate_camera_view(shot.trajectory)
            ball_vis = view.ball_visible.numpy()
            T = len(ball_vis)
            cameras.append(
                CameraData(
                    camera_params=view.camera_params,
                    ball_uv=view.ball_uv.numpy(),
                    ball_visible=ball_vis,
                    ball_visibility_ratio=float(ball_vis.sum()) / T,
                    court_kp_uv=view.court_kp_uv.numpy(),
                    court_kp_visible=view.court_kp_visible.numpy(),
                    court_visibility_count=float(view.court_kp_visible.sum()),
                )
            )

        scene = BLCSSceneData(
            scene_id="test123",
            from_cell=4,
            from_side="near",
            category=shot.category,
            to_cell=shot.to_cell,
            ball_pos_world=shot.trajectory,
            ball_pos_norm=physics.normalize_position(shot.trajectory),
            ball_vel_world=shot.velocities,
            t_net=shot.t_net,
            t_fence=shot.t_fence,
            t_bounce1=shot.t_bounce1,
            t_bounce2=shot.t_bounce2,
            cameras=cameras,
            num_cameras_sampled=5,
            fps_out=30,
            sim_fps=240,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            writer = BLCSDatasetWriter(tmpdir)
            filepath = writer.save_scene(scene)

            assert filepath.exists()

            # Load and verify
            from src.blcs.data.dataset_writer import load_scene

            loaded = load_scene(filepath)

            assert loaded["meta"]["scene_id"] == "test123"
            assert loaded["num_cameras"] == 3
            assert loaded["ball_pos_world"].shape == shot.trajectory.shape
            assert len(loaded["cameras"]) == 3

            # Verify camera data
            for i, cam in enumerate(loaded["cameras"]):
                assert "ball_uv" in cam
                assert "ball_visibility_ratio" in cam
                assert "court_kp_uv" in cam
