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
from src.tennis_scene.pipeline.components.blcs import BLCSConfig, BLCSModule
from src.tennis_scene.pipeline.components.court_kp import CourtKPConfig, CourtKPModule
from src.tennis_scene.pipeline.components.event_3d import Event3DConfig, Event3DModule
from src.tennis_scene.pipeline.components.event_uv import EventUVConfig, EventUVModule
from src.tennis_scene.pipeline.components.plcs import PLCSConfig, PLCSModule
from src.tennis_scene.pipeline.components.trajectory import TrajectoryConfig, TrajectoryModule
from src.tennis_scene.pipeline.components.wasb import WASBConfig, WASBModule
from src.tennis_scene.pipeline.dependency_graph import (
    ResolutionResult,
    Stage,
    build_default_dependency_graph,
)
from src.tennis_scene.utils.transforms import apply_plcs_transform_batch

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
        wasb_module: WASBModule | None,
        trajectory_module: TrajectoryModule | None,
        event_uv_module: EventUVModule | None,
        plcs_module: PLCSModule,
        blcs_module: BLCSModule | None,
        event_3d_module: Event3DModule | None,
        resolution: ResolutionResult,
        device: str = "cuda",
    ) -> None:
        self.court_kp_module = court_kp_module
        self.gvhmr_config = gvhmr_config
        self.wasb_module = wasb_module
        self.trajectory_module = trajectory_module
        self.event_uv_module = event_uv_module
        self.plcs_module = plcs_module
        self.blcs_module = blcs_module
        self.event_3d_module = event_3d_module
        self.resolution = resolution
        self.enabled_stages = resolution.enabled_set
        self.execution_order = resolution.enabled_order
        self.device = device

    @classmethod
    def from_config(cls, cfg: DictConfig) -> "TennisSceneOrchestrator":
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
                "smplx_model_type": str(cfg.gvhmr.get("smplx_model_type", "supermotion")),
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

        wasb_module = None
        if Stage.WASB in resolution.enabled_set:
            wasb_module = WASBModule(
                WASBConfig(
                    checkpoint=to_absolute_path(cfg.wasb.checkpoint),
                    batch_size=int(cfg.wasb.batch_size),
                    device=device,
                    save_result=cfg.wasb.get("save_result", True),
                    output_path=get_output_path("wasb", "wasb_result.json"),
                    load_path=get_load_path("wasb"),
                )
            )

        trajectory_module = None
        if Stage.TRAJECTORY in resolution.enabled_set:
            trajectory_module = TrajectoryModule(
                TrajectoryConfig(
                    checkpoint_path=to_absolute_path(cfg.trajectory.checkpoint),
                    device=device,
                    merge_observed=cfg.trajectory.get("merge_observed", True),
                    in_frame_threshold=float(cfg.trajectory.get("in_frame_threshold", 0.5)),
                    cut_out_of_frame=bool(cfg.trajectory.get("cut_out_of_frame", False)),
                    use_in_frame_pred_for_visibility=bool(
                        cfg.trajectory.get("use_in_frame_pred_for_visibility", True)
                    ),
                    save_result=cfg.trajectory.get("save_result", True),
                    output_path=get_output_path("trajectory", "trajectory_result.json"),
                    load_path=get_load_path("trajectory"),
                )
            )

        event_uv_module = None
        if Stage.EVENT_UV in resolution.enabled_set:
            event_uv_module = EventUVModule(
                EventUVConfig(
                    checkpoint_path=to_absolute_path(cfg.event_uv.checkpoint),
                    device=device,
                    threshold=float(cfg.event_uv.get("threshold", 0.5)),
                    min_distance=int(cfg.event_uv.get("min_distance", 1)),
                    top_k=cfg.event_uv.get("top_k"),
                    save_result=cfg.event_uv.get("save_result", True),
                    output_path=get_output_path("event_uv", "event_uv_result.json"),
                    load_path=get_load_path("event_uv"),
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

        event_3d_module = None
        if Stage.EVENT_3D in resolution.enabled_set:
            event_3d_module = Event3DModule(
                Event3DConfig(
                    checkpoint_path=to_absolute_path(cfg.event_3d.checkpoint),
                    device=device,
                    threshold=float(cfg.event_3d.get("threshold", 0.5)),
                    min_distance=int(cfg.event_3d.get("min_distance", 1)),
                    top_k=cfg.event_3d.get("top_k"),
                    save_result=cfg.event_3d.get("save_result", True),
                    output_path=get_output_path("event_3d", "event_3d_result.json"),
                    load_path=get_load_path("event_3d"),
                )
            )

        return cls(
            court_kp_module=court_kp_module,
            gvhmr_config=gvhmr_config,
            wasb_module=wasb_module,
            trajectory_module=trajectory_module,
            event_uv_module=event_uv_module,
            plcs_module=plcs_module,
            blcs_module=blcs_module,
            event_3d_module=event_3d_module,
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
        if Stage.WASB in self.enabled_stages and self.wasb_module is not None:
            self.wasb_module.load()
        if Stage.TRAJECTORY in self.enabled_stages and self.trajectory_module is not None:
            self.trajectory_module.load()
        if Stage.EVENT_UV in self.enabled_stages and self.event_uv_module is not None:
            self.event_uv_module.load()
        self.plcs_module.load()
        if Stage.BLCS in self.enabled_stages and self.blcs_module is not None:
            self.blcs_module.load()
        if Stage.EVENT_3D in self.enabled_stages and self.event_3d_module is not None:
            self.event_3d_module.load()

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
        court_vis = court_result.visibility

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

        if smpl_vertices_local is not None:
            smpl_vertices_global = np.stack(
                [
                    apply_plcs_transform_batch(
                        smpl_vertices_local[p],
                        plcs_result.position[p],
                        plcs_result.yaw[p],
                    )
                    for p in range(smpl_vertices_local.shape[0])
                ],
                axis=0,
            ).astype(np.float32)
        else:
            smpl_vertices_global = None

        ball_uv = None
        ball_uv_pred = None
        ball_uv_completed = None
        ball_visibility = None
        ball_3d = None
        event_uv_probs = None
        event_uv_peak_mask = None
        event_uv_names = None
        event_3d_probs = None
        event_3d_peak_mask = None
        event_3d_names = None
        if Stage.WASB in self.enabled_stages and self.wasb_module is not None:
            wasb_result = self.wasb_module.process(
                video_path,
                max_frames=max_frames,
                image_width=width,
                image_height=height,
            )
            ball_uv = wasb_result.ball_uv
            ball_visibility = wasb_result.visibility
            ball_uv_for_downstream = ball_uv

            if Stage.TRAJECTORY in self.enabled_stages and self.trajectory_module is not None:
                trajectory_result = self.trajectory_module.process(
                    ball_uv=ball_uv,
                    court_kp=court_kp,
                    ball_vis=ball_visibility,
                    court_vis=court_vis,
                )
                ball_uv_pred = trajectory_result.ball_uv_pred
                ball_uv_completed = trajectory_result.ball_uv_completed
                if (
                    self.trajectory_module.config.merge_observed
                    and ball_uv_completed is not None
                ):
                    ball_uv_for_downstream = ball_uv_completed
                else:
                    ball_uv_for_downstream = ball_uv_pred
                if (
                    self.trajectory_module.config.use_in_frame_pred_for_visibility
                    and trajectory_result.in_frame_pred is not None
                ):
                    ball_visibility = trajectory_result.in_frame_pred

            if Stage.EVENT_UV in self.enabled_stages and self.event_uv_module is not None:
                event_uv_result = self.event_uv_module.process(
                    ball_uv=ball_uv_for_downstream,
                    court_kp=court_kp,
                    ball_vis=ball_visibility,
                    court_vis=court_vis,
                )
                event_uv_probs = event_uv_result.event_probs[0]
                event_uv_peak_mask = event_uv_result.event_peak_mask[0]
                event_uv_names = event_uv_result.event_names

            if Stage.BLCS in self.enabled_stages and self.blcs_module is not None:
                blcs_result = self.blcs_module.process(
                    ball_uv=ball_uv_for_downstream,
                    court_kp=court_kp,
                    ball_vis=ball_visibility,
                    court_vis=court_vis,
                )
                ball_3d = blcs_result.ball_3d
                if Stage.EVENT_3D in self.enabled_stages and self.event_3d_module is not None:
                    event_3d_result = self.event_3d_module.process(ball_3d=ball_3d)
                    event_3d_probs = event_3d_result.event_probs[0]
                    event_3d_peak_mask = event_3d_result.event_peak_mask[0]
                    event_3d_names = event_3d_result.event_names

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
            smpl_vertices_global=smpl_vertices_global,
            ball_uv=ball_uv,
            ball_uv_pred=ball_uv_pred,
            ball_uv_completed=ball_uv_completed,
            ball_visibility=ball_visibility,
            ball_3d=ball_3d,
            event_uv_probs=event_uv_probs,
            event_uv_peak_mask=event_uv_peak_mask,
            event_uv_names=event_uv_names,
            event_3d_probs=event_3d_probs,
            event_3d_peak_mask=event_3d_peak_mask,
            event_3d_names=event_3d_names,
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
