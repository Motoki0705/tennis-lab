"""Orchestrator for tennis scene 3D reconstruction pipeline.

This module provides the main orchestration class that coordinates
all pipeline modules (Court KP, GVHMR, WASB, PLCS, BLCS).

Example:
    >>> orchestrator = TennisSceneOrchestrator.from_config(cfg)
    >>> result = orchestrator.run("video.mp4")
    >>> result.save("output.npz")

Config entry point: `src/tennis_scene/configs/pipeline.yaml`
"""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING, Any

import cv2
import numpy as np

from src.tennis_scene.io import SceneResult
from src.tennis_scene.transforms import apply_plcs_transform_batch
from src.tennis_scene.pipeline.court_kp import CourtKPModule, CourtKPConfig
from src.tennis_scene.pipeline.gvhmr import GVHMRResult
from src.tennis_scene.pipeline.wasb import WASBModule, WASBConfig
from src.tennis_scene.pipeline.plcs import PLCSModule, PLCSConfig
from src.tennis_scene.pipeline.blcs import BLCSModule, BLCSConfig

if TYPE_CHECKING:
    from numpy.typing import NDArray
    from omegaconf import DictConfig

LOGGER = logging.getLogger(__name__)


class TennisSceneOrchestrator:
    """Orchestrator for tennis scene 3D reconstruction.

    Coordinates the execution of pipeline modules in the correct order:
    1. Court KP Detection (single frame)
    2. GVHMR via CLI subprocess (parallel with WASB)
    3. WASB ball detection (parallel with GVHMR)
    4. PLCS player localization (depends on Court KP + GVHMR)
    5. BLCS ball localization (depends on Court KP + WASB)
    6. SMPL coordinate transformation (depends on GVHMR + PLCS)

    """

    def __init__(
        self,
        court_kp_module: CourtKPModule,
        gvhmr_config: dict[str, Any] | None,
        wasb_module: WASBModule | None,
        plcs_module: PLCSModule,
        blcs_module: BLCSModule | None,
        device: str = "cuda",
    ) -> None:
        """Initialize the orchestrator.

        Args:
            court_kp_module: Court keypoint detection module.
            gvhmr_config: GVHMR configuration dict for subprocess execution.
            wasb_module: WASB module (None to skip ball detection).
            plcs_module: PLCS module.
            blcs_module: BLCS module (None to skip).
            device: Inference device.

        """
        self.court_kp_module = court_kp_module
        self.gvhmr_config = gvhmr_config
        self.wasb_module = wasb_module
        self.plcs_module = plcs_module
        self.blcs_module = blcs_module
        self.device = device

    @classmethod
    def from_config(cls, cfg: DictConfig) -> TennisSceneOrchestrator:
        """Create orchestrator from Hydra config.

        Args:
            cfg: Hydra configuration.

        Returns:
            Initialized TennisSceneOrchestrator.

        """
        from hydra.utils import to_absolute_path

        device = str(cfg.device)

        # Determine output directory for module results
        output_dir = Path(to_absolute_path(cfg.output_dir))

        # Helper to resolve load_path
        def get_load_path(section: str) -> str | None:
            load_path = cfg[section].get("load_path")
            if load_path is not None:
                return to_absolute_path(str(load_path))
            return None

        # Helper to get output_path
        def get_output_path(section: str, default_name: str) -> str:
            output_path = cfg[section].get("output_path")
            if output_path is not None:
                return to_absolute_path(str(output_path))
            return str(output_dir / default_name)

        court_kp_config = CourtKPConfig(
            checkpoint_path=to_absolute_path(cfg.court_kp.checkpoint),
            mode=str(cfg.court_kp.get("mode", "model")),
            device=device,
            save_result=cfg.court_kp.get("save_result", True),
            output_path=get_output_path("court_kp", "court_kp_result.json"),
            load_path=get_load_path("court_kp"),
        )
        court_kp_module = CourtKPModule(court_kp_config)

        # GVHMR runs as CLI subprocess - store config dict instead of module
        gvhmr_config = None
        if not cfg.gvhmr.get("skip", False):
            gvhmr_config = {
                "python_executable": to_absolute_path(cfg.gvhmr.python_executable),
                "model_checkpoint": to_absolute_path(cfg.gvhmr.checkpoint),
                "yolo_checkpoint": to_absolute_path(cfg.gvhmr.yolo_checkpoint),
                "vitpose_checkpoint": to_absolute_path(cfg.gvhmr.vitpose_checkpoint),
                "hmr2_checkpoint": to_absolute_path(cfg.gvhmr.hmr2_checkpoint),
                "output_path": get_output_path("gvhmr", "gvhmr_result.json"),
                "load_path": get_load_path("gvhmr"),
                "device": device,
            }

        wasb_module = None
        if not cfg.wasb.get("skip", False):
            wasb_config = WASBConfig(
                checkpoint=to_absolute_path(cfg.wasb.checkpoint),
                batch_size=int(cfg.wasb.batch_size),
                device=device,
                save_result=cfg.wasb.get("save_result", True),
                output_path=get_output_path("wasb", "wasb_result.json"),
                load_path=get_load_path("wasb"),
            )
            wasb_module = WASBModule(wasb_config)

        plcs_config = PLCSConfig(
            checkpoint_path=to_absolute_path(cfg.plcs.checkpoint),
            device=device,
            save_result=cfg.plcs.get("save_result", True),
            output_path=get_output_path("plcs", "plcs_result.json"),
            load_path=get_load_path("plcs"),
        )
        plcs_module = PLCSModule(plcs_config)

        blcs_module = None
        if wasb_module is not None:
            blcs_config = BLCSConfig(
                checkpoint_path=to_absolute_path(cfg.blcs.checkpoint),
                device=device,
                save_result=cfg.blcs.get("save_result", True),
                output_path=get_output_path("blcs", "blcs_result.json"),
                load_path=get_load_path("blcs"),
            )
            blcs_module = BLCSModule(blcs_config)

        return cls(
            court_kp_module=court_kp_module,
            gvhmr_config=gvhmr_config,
            wasb_module=wasb_module,
            plcs_module=plcs_module,
            blcs_module=blcs_module,
            device=device,
        )

    def _run_gvhmr(
        self,
        video_path: Path,
        max_frames: int | None = None,
    ) -> GVHMRResult:
        """Run GVHMR via CLI subprocess or load from cached result.

        Args:
            video_path: Path to input video.
            max_frames: Maximum frames to process.

        Returns:
            GVHMRResult with SMPL and keypoint data.

        """
        if self.gvhmr_config is None:
            raise RuntimeError("GVHMR config not set")

        load_path = self.gvhmr_config.get("load_path")
        output_path = self.gvhmr_config["output_path"]

        # If load_path is specified and exists, load from it
        if load_path is not None and Path(load_path).exists():
            LOGGER.info(f"Loading GVHMR result from: {load_path}")
            return GVHMRResult.load(load_path)

        # Run GVHMR CLI subprocess
        LOGGER.info("Running GVHMR via CLI subprocess...")
        python_exe = self.gvhmr_config["python_executable"]
        cmd = [
            python_exe,
            "-m",
            "src.tennis_scene.pipeline.gvhmr",
            f"--video={video_path}",
            f"--output={output_path}",
            f"--model-checkpoint={self.gvhmr_config['model_checkpoint']}",
            f"--yolo-checkpoint={self.gvhmr_config['yolo_checkpoint']}",
            f"--vitpose-checkpoint={self.gvhmr_config['vitpose_checkpoint']}",
            f"--hmr2-checkpoint={self.gvhmr_config['hmr2_checkpoint']}",
            f"--device={self.gvhmr_config['device']}",
        ]
        if max_frames is not None:
            cmd.append(f"--max_frames={max_frames}")

        LOGGER.info(f"Command: {' '.join(cmd)}")

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=str(Path(__file__).parents[3]),  # tennis-lab root
        )

        if result.returncode != 0:
            LOGGER.error(f"GVHMR subprocess failed:\n{result.stderr}")
            raise RuntimeError(f"GVHMR subprocess failed: {result.stderr}")

        LOGGER.info("GVHMR subprocess completed successfully")

        # Load result from output JSON
        return GVHMRResult.load(output_path)

    def load_all(self) -> None:
        """Pre-load all modules (except GVHMR which runs as subprocess)."""
        LOGGER.info("Pre-loading all modules...")
        self.court_kp_module.load()
        # GVHMR runs as subprocess, no pre-loading needed
        if self.wasb_module is not None:
            self.wasb_module.load()
        self.plcs_module.load()
        if self.blcs_module is not None:
            self.blcs_module.load()

    def _read_video_info(self, video_path: Path) -> dict[str, Any]:
        """Read video metadata."""
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video: {video_path}")

        try:
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        finally:
            cap.release()

        return {
            "fps": fps,
            "width": width,
            "height": height,
            "num_frames": num_frames,
        }

    def _read_frame(self, video_path: Path, frame_idx: int) -> NDArray[np.uint8]:
        """Read a single frame from video."""
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video: {video_path}")

        try:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                raise RuntimeError(f"Failed to read frame {frame_idx}")
            return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        finally:
            cap.release()

    def run(
        self,
        video_path: str | Path,
        max_frames: int | None = None,
        court_kp_frame: int = 0,
    ) -> SceneResult:
        """Run the full tennis scene reconstruction pipeline.

        Args:
            video_path: Path to input video.
            max_frames: Maximum frames to process.
            court_kp_frame: Frame index for court keypoint detection.

        Returns:
            SceneResult with all reconstruction data.

        """
        video_path = Path(video_path)
        video_info = self._read_video_info(video_path)
        width, height = video_info["width"], video_info["height"]

        LOGGER.info("=" * 60)
        LOGGER.info("Tennis Scene 3D Reconstruction")
        LOGGER.info(f"Video: {video_path}")
        LOGGER.info(f"Resolution: {width}x{height}, FPS: {video_info['fps']:.2f}")
        LOGGER.info("=" * 60)

        frame = self._read_frame(video_path, court_kp_frame)
        court_result = self.court_kp_module.process(
            frame,
            frame_index=court_kp_frame,
            image_width=width,
            image_height=height,
        )
        court_kp = court_result.keypoints
        court_vis = court_result.visibility

        if self.gvhmr_config is not None:
            gvhmr_result = self._run_gvhmr(video_path, max_frames)

            human_kp_2d_norm = gvhmr_result.human_kp_2d.copy()
            human_kp_2d_norm[..., 0] /= width
            human_kp_2d_norm[..., 1] /= height

            smpl_body_pose = gvhmr_result.smpl_body_pose
            smpl_global_orient = gvhmr_result.smpl_global_orient
            smpl_betas = gvhmr_result.smpl_betas
            smpl_vertices_local = gvhmr_result.smpl_vertices_local
            human_kp_vis = gvhmr_result.human_kp_vis
        else:
            T = max_frames or video_info["num_frames"]
            human_kp_2d_norm = np.zeros((T, 17, 2), dtype=np.float32)
            human_kp_vis = np.ones((T, 17), dtype=np.float32)
            smpl_body_pose = np.zeros((T, 63), dtype=np.float32)
            smpl_global_orient = np.zeros((T, 3), dtype=np.float32)
            smpl_betas = np.zeros(10, dtype=np.float32)
            smpl_vertices_local = None

        plcs_result = self.plcs_module.process(
            human_kp_2d=human_kp_2d_norm,
            court_kp=court_kp,
            human_kp_vis=human_kp_vis,
            court_vis=court_vis,
        )

        if smpl_vertices_local is not None:
            smpl_vertices_global = apply_plcs_transform_batch(
                smpl_vertices_local,
                plcs_result.position,
                plcs_result.yaw,
            )
        else:
            smpl_vertices_global = None

        ball_uv = None
        ball_visibility = None
        ball_3d = None

        if self.wasb_module is not None:
            wasb_result = self.wasb_module.process(
                video_path,
                max_frames=max_frames,
                image_width=width,
                image_height=height,
            )
            ball_uv = wasb_result.ball_uv
            ball_visibility = wasb_result.visibility

            if self.blcs_module is not None:
                blcs_result = self.blcs_module.process(
                    ball_uv=ball_uv,
                    court_kp=court_kp,
                    ball_vis=ball_visibility,
                    court_vis=court_vis,
                )
                ball_3d = blcs_result.ball_3d

        T = len(plcs_result.position)

        return SceneResult(
            num_frames=T,
            fps=video_info["fps"],
            width=width,
            height=height,
            court_kp=court_kp,
            court_vis=court_vis,
            player_position=plcs_result.position,
            player_yaw=plcs_result.yaw,
            smpl_body_pose=smpl_body_pose,
            smpl_global_orient=smpl_global_orient,
            smpl_betas=smpl_betas,
            smpl_vertices_local=smpl_vertices_local,
            smpl_vertices_global=smpl_vertices_global,
            ball_uv=ball_uv,
            ball_visibility=ball_visibility,
            ball_3d=ball_3d,
            human_kp_2d=human_kp_2d_norm,
            human_kp_vis=human_kp_vis,
            metadata={
                "video_path": str(video_path),
                "court_kp_frame": court_kp_frame,
            },
        )


if __name__ == "__main__":
    print("TennisSceneOrchestrator: pipeline orchestration module")
    print("Use TennisSceneOrchestrator.from_config(cfg) to create")
