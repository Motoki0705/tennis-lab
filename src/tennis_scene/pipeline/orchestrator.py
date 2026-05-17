"""Orchestrator for tennis scene 3D reconstruction pipeline."""

from __future__ import annotations

import logging
import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

import cv2
import numpy as np

from src.tennis_scene.io import SceneResult
from src.tennis_scene.pipeline.components.ball_detection import (
    BallDetectionConfig,
    BallDetectionModule,
)
from src.tennis_scene.pipeline.components.blcs import BLCSConfig, BLCSModule
from src.tennis_scene.pipeline.components.court_kp import CourtKPConfig, CourtKPModule
from src.tennis_scene.pipeline.components.plcs import PLCSConfig, PLCSModule
from src.tennis_scene.pipeline.dependency_graph import (
    ResolutionResult,
    Stage,
    build_default_dependency_graph,
)

if TYPE_CHECKING:
    from numpy.typing import NDArray
    from omegaconf import DictConfig

LOGGER = logging.getLogger(__name__)


class TennisSceneOrchestrator:
    """Orchestrator for tennis scene 3D reconstruction."""

    def __init__(
        self,
        court_kp_module: CourtKPModule,
        gvhmr_config: dict[str, Any] | None,
        ball_detection_module: BallDetectionModule | None,
        plcs_module: PLCSModule,
        blcs_module: BLCSModule | None,
        resolution: ResolutionResult,
        device: str = "cuda",
    ) -> None:
        self.court_kp_module = court_kp_module
        self.gvhmr_config = gvhmr_config
        self.ball_detection_module = ball_detection_module
        self.plcs_module = plcs_module
        self.blcs_module = blcs_module
        self.resolution = resolution
        self.enabled_stages = resolution.enabled_set
        self.execution_order = resolution.enabled_order
        self.device = device

    @classmethod
    def from_config(cls, cfg: DictConfig) -> TennisSceneOrchestrator:
        from hydra.utils import to_absolute_path

        device = str(cfg.device)
        output_dir = Path(to_absolute_path(cfg.output_dir))
        graph = build_default_dependency_graph()
        resolution = graph.resolve_from_config(cfg)
        for line in graph.format_resolution_messages(resolution):
            LOGGER.info(line)

        def get_load_path(section: str) -> str | None:
            load_path = cfg[section].get("load_path")
            if load_path is not None:
                return to_absolute_path(str(load_path))
            return None

        def get_output_path(section: str, default_name: str) -> str:
            output_path = cfg[section].get("output_path")
            if output_path is not None:
                return to_absolute_path(str(output_path))
            return str(output_dir / default_name)

        if Stage.COURT_KP not in resolution.enabled_set:
            raise ValueError("COURT_KP stage must be enabled.")
        court_kp_module = CourtKPModule(
            CourtKPConfig(
                checkpoint_path=to_absolute_path(cfg.court_kp.checkpoint),
                mode=str(cfg.court_kp.get("mode", "model")),
                device=device,
                save_result=cfg.court_kp.get("save_result", True),
                output_path=get_output_path("court_kp", "court_kp_result.json"),
                load_path=get_load_path("court_kp"),
            )
        )

        gvhmr_config = None
        if Stage.GVHMR in resolution.enabled_set:
            smplx_body_model_path = cfg.gvhmr.get("smplx_body_model_path")
            gvhmr_config = {
                "python_executable": to_absolute_path(cfg.gvhmr.python_executable),
                "model_checkpoint": to_absolute_path(cfg.gvhmr.checkpoint),
                "yolo_checkpoint": to_absolute_path(cfg.gvhmr.yolo_checkpoint),
                "vitpose_checkpoint": to_absolute_path(cfg.gvhmr.vitpose_checkpoint),
                "hmr2_checkpoint": to_absolute_path(cfg.gvhmr.hmr2_checkpoint),
                "smplx_model_type": str(
                    cfg.gvhmr.get("smplx_model_type", "supermotion")
                ),
                "smplx2smpl_path": str(
                    cfg.gvhmr.get(
                        "smplx2smpl_path",
                        "hmr4d/utils/body_model/smplx2smpl_sparse.pt",
                    )
                ),
                "smplx_body_model_path": (
                    to_absolute_path(str(smplx_body_model_path))
                    if smplx_body_model_path is not None
                    else None
                ),
                "output_path": get_output_path("gvhmr", "gvhmr_result.json"),
                "load_path": get_load_path("gvhmr"),
                "device": device,
            }

        ball_detection_module = None
        if Stage.BALL_DETECTION in resolution.enabled_set:
            ball_detection_cfg = cfg.ball_detection
            ball_detection_module = BallDetectionModule(
                BallDetectionConfig(
                    checkpoint=to_absolute_path(ball_detection_cfg.checkpoint),
                    batch_size=int(ball_detection_cfg.batch_size),
                    device=device,
                    image_size=tuple(
                        int(value)
                        for value in ball_detection_cfg.get("image_size", [288, 512])
                    ),
                    normalize_imagenet=bool(
                        ball_detection_cfg.get("normalize_imagenet", True)
                    ),
                    score_threshold=float(
                        ball_detection_cfg.get("score_threshold", 0.5)
                    ),
                    prefetch_batches=int(ball_detection_cfg.get("prefetch_batches", 2)),
                    window_stride=(
                        None
                        if ball_detection_cfg.get("window_stride", None) is None
                        else int(ball_detection_cfg.window_stride)
                    ),
                    tail_policy=str(ball_detection_cfg.get("tail_policy", "backfill")),
                    overlap_aggregation=str(
                        ball_detection_cfg.get(
                            "overlap_aggregation", "last_window_wins"
                        )
                    ),
                    pin_memory=bool(ball_detection_cfg.get("pin_memory", True)),
                    save_result=ball_detection_cfg.get("save_result", True),
                    output_path=get_output_path(
                        "ball_detection",
                        "ball_detection_result.json",
                    ),
                    load_path=get_load_path("ball_detection"),
                )
            )

        if Stage.PLCS not in resolution.enabled_set:
            raise ValueError("PLCS stage must be enabled.")
        plcs_module = PLCSModule(
            PLCSConfig(
                checkpoint_path=to_absolute_path(cfg.plcs.checkpoint),
                device=device,
                save_result=cfg.plcs.get("save_result", True),
                output_path=get_output_path("plcs", "plcs_result.json"),
                load_path=get_load_path("plcs"),
            )
        )

        blcs_module = None
        if Stage.BLCS in resolution.enabled_set:
            blcs_module = BLCSModule(
                BLCSConfig(
                    checkpoint_path=to_absolute_path(cfg.blcs.checkpoint),
                    device=device,
                    save_result=cfg.blcs.get("save_result", True),
                    output_path=get_output_path("blcs", "blcs_result.json"),
                    load_path=get_load_path("blcs"),
                )
            )

        return cls(
            court_kp_module=court_kp_module,
            gvhmr_config=gvhmr_config,
            ball_detection_module=ball_detection_module,
            plcs_module=plcs_module,
            blcs_module=blcs_module,
            resolution=resolution,
            device=device,
        )

    def _run_gvhmr(self, video_path: Path, max_frames: int | None = None):
        from src.tennis_scene.pipeline.components.gvhmr import GVHMRResult

        if self.gvhmr_config is None:
            raise RuntimeError("GVHMR config not set")

        load_path = self.gvhmr_config.get("load_path")
        output_path = self.gvhmr_config["output_path"]

        if load_path is not None and Path(load_path).exists():
            LOGGER.info(f"Loading GVHMR result from: {load_path}")
            return GVHMRResult.load(load_path)

        LOGGER.info("Running GVHMR via CLI subprocess...")
        python_exe = self.gvhmr_config["python_executable"]
        cmd = [
            python_exe,
            "-m",
            "src.tennis_scene.pipeline.components.gvhmr",
            f"--video={video_path}",
            f"--output={output_path}",
            f"--model-checkpoint={self.gvhmr_config['model_checkpoint']}",
            f"--yolo-checkpoint={self.gvhmr_config['yolo_checkpoint']}",
            f"--vitpose-checkpoint={self.gvhmr_config['vitpose_checkpoint']}",
            f"--hmr2-checkpoint={self.gvhmr_config['hmr2_checkpoint']}",
            f"--smplx-model-type={self.gvhmr_config['smplx_model_type']}",
            f"--smplx2smpl-path={self.gvhmr_config['smplx2smpl_path']}",
            f"--device={self.gvhmr_config['device']}",
        ]
        if self.gvhmr_config.get("smplx_body_model_path") is not None:
            cmd.append(
                f"--smplx-body-model-path={self.gvhmr_config['smplx_body_model_path']}"
            )
        if max_frames is not None:
            cmd.append(f"--max-frames={max_frames}")

        # Keep stdio attached so GVHMR tracker selection UI can read user input.
        result = subprocess.run(
            cmd,
            cwd=str(Path(__file__).parents[3]),
            stdin=sys.stdin,
            stdout=sys.stdout,
            stderr=sys.stderr,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"GVHMR subprocess failed with return code {result.returncode}"
            )

        return GVHMRResult.load(output_path)

    def load_all(self) -> None:
        LOGGER.info("Pre-loading all modules...")
        self.court_kp_module.load()
        if (
            Stage.BALL_DETECTION in self.enabled_stages
            and self.ball_detection_module is not None
        ):
            self.ball_detection_module.load()
        self.plcs_module.load()
        if Stage.BLCS in self.enabled_stages and self.blcs_module is not None:
            self.blcs_module.load()

    def _read_video_info(self, video_path: Path) -> dict[str, Any]:
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
        return {"fps": fps, "width": width, "height": height, "num_frames": num_frames}

    def _read_frame(self, video_path: Path, frame_idx: int) -> NDArray[np.uint8]:
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
        video_path = Path(video_path)
        video_info = self._read_video_info(video_path)
        width, height = video_info["width"], video_info["height"]

        frame = self._read_frame(video_path, court_kp_frame)
        court_result = self.court_kp_module.process(
            frame,
            frame_index=court_kp_frame,
            image_width=width,
            image_height=height,
        )
        court_kp = court_result.keypoints
        court_vis = None

        if Stage.GVHMR in self.enabled_stages and self.gvhmr_config is not None:
            gvhmr_result = self._run_gvhmr(video_path, max_frames)

            human_kp_2d_norm = gvhmr_result.human_kp_2d.copy()  # (P, T, 17, 2)
            human_kp_2d_norm[..., 0] /= width
            human_kp_2d_norm[..., 1] /= height

            human_kp_vis = gvhmr_result.human_kp_vis
            smpl_body_pose = gvhmr_result.smpl_body_pose
            smpl_global_orient = gvhmr_result.smpl_global_orient
            smpl_betas = gvhmr_result.smpl_betas
            smpl_vertices_local = gvhmr_result.smpl_vertices_local
            track_ids = gvhmr_result.track_ids
        else:
            raise RuntimeError("GVHMR stage is required because PLCS depends on GVHMR.")

        plcs_result = self.plcs_module.process(
            human_kp_2d=human_kp_2d_norm,
            court_kp=court_kp,
            human_kp_vis=human_kp_vis,
            court_vis=court_vis,
            track_ids=track_ids,
        )

        ball_uv = None
        ball_visibility = None
        ball_3d = None
        if (
            Stage.BALL_DETECTION in self.enabled_stages
            and self.ball_detection_module is not None
        ):
            ball_detection_result = self.ball_detection_module.process(
                video_path,
                max_frames=max_frames,
                image_width=width,
                image_height=height,
            )
            ball_uv = ball_detection_result.ball_uv
            ball_visibility = ball_detection_result.visibility
            ball_uv_for_downstream = ball_uv

            if Stage.BLCS in self.enabled_stages and self.blcs_module is not None:
                blcs_result = self.blcs_module.process(
                    ball_uv=ball_uv_for_downstream,
                    court_kp=court_kp,
                    ball_vis=ball_visibility,
                    court_vis=court_vis,
                )
                ball_3d = blcs_result.ball_3d

        T = plcs_result.position.shape[1]

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
            ball_uv=ball_uv,
            ball_visibility=ball_visibility,
            ball_3d=ball_3d,
            human_kp_2d=human_kp_2d_norm,
            human_kp_vis=human_kp_vis,
            player_track_ids=track_ids,
            metadata={
                "video_path": str(video_path),
                "court_kp_frame": court_kp_frame,
                "track_ids": track_ids.tolist(),
                "enabled_stages": [stage.value for stage in self.execution_order],
            },
        )


if __name__ == "__main__":
    print("TennisSceneOrchestrator: pipeline orchestration module")
