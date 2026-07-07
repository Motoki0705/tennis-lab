"""Orchestrator for tennis scene 3D reconstruction pipeline."""

from __future__ import annotations

import logging
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

from src.tasks.ball_detection.inference.trajectory_gate import TrajectoryGateConfig
from src.tennis_scene.io import SceneResult
from src.tennis_scene.pipeline.components.ball_detection import (
    BallDetectionConfig,
    BallDetectionModule,
)
from src.tennis_scene.pipeline.components.blcs import BLCSConfig, BLCSModule
from src.tennis_scene.pipeline.components.court_kp import CourtKPConfig, CourtKPModule
from src.tennis_scene.pipeline.components.player_association import (
    PlayerAssociationConfig,
    PlayerAssociationModule,
)
from src.tennis_scene.pipeline.components.plcs import PLCSConfig, PLCSModule
from src.tennis_scene.pipeline.dependency_graph import (
    ResolutionResult,
    Stage,
    build_default_dependency_graph,
)
from src.utils.video import probe_video_info

if TYPE_CHECKING:
    from omegaconf import DictConfig

    from src.tennis_scene.pipeline.components.gvhmr import GVHMRResult
    from src.tennis_scene.pipeline.components.player_association import (
        PlayerAssociationApplied,
        PlayerAssociationResult,
    )
    from src.utils.video import VideoInfo

LOGGER = logging.getLogger(__name__)


def _size_hw(value: Any) -> tuple[int, int]:
    """Coerce a config size sequence into a validated (height, width) pair."""
    items = [int(v) for v in value]
    if len(items) != 2:
        raise ValueError(f"image_size must have exactly 2 entries, got {items!r}")
    return items[0], items[1]


class TennisSceneOrchestrator:
    """Orchestrator for tennis scene 3D reconstruction."""

    def __init__(
        self,
        court_kp_module: CourtKPModule,
        gvhmr_config: dict[str, Any] | None,
        player_association_module: PlayerAssociationModule,
        ball_detection_module: BallDetectionModule | None,
        plcs_module: PLCSModule,
        blcs_module: BLCSModule | None,
        resolution: ResolutionResult,
        device: str = "cuda",
    ) -> None:
        self.court_kp_module = court_kp_module
        self.gvhmr_config = gvhmr_config
        self.player_association_module = player_association_module
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
        from omegaconf import ListConfig

        device = str(cfg.device)
        output_dir = Path(to_absolute_path(cfg.output_dir))
        graph = build_default_dependency_graph()
        resolution = graph.resolve_from_config(cfg)
        for line in graph.format_resolution_messages(resolution):
            LOGGER.info(line)

        def load_path(section: str) -> str | None:
            value = cfg[section].get("load_path")
            return str(to_absolute_path(str(value))) if value is not None else None

        def load_paths(section: str) -> str | list[str] | None:
            value = cfg[section].get("load_path")
            if value is None:
                return None
            if isinstance(value, (list, tuple, ListConfig)):
                return [str(to_absolute_path(str(item))) for item in value]
            return str(to_absolute_path(str(value)))

        def output_path(section: str, default_name: str) -> str:
            value = cfg[section].get("output_path")
            if value is not None:
                return str(to_absolute_path(str(value)))
            return str(output_dir / default_name)

        if Stage.COURT_KP not in resolution.enabled_set:
            raise ValueError("COURT_KP stage must be enabled.")
        court_kp_module = CourtKPModule(
            CourtKPConfig(
                checkpoint_path=to_absolute_path(cfg.court_kp.checkpoint),
                mode=str(cfg.court_kp.get("mode", "model")),
                device=device,
                num_keypoints=int(cfg.court_kp.get("num_keypoints", 14)),
                save_result=cfg.court_kp.get("save_result", True),
                output_path=output_path("court_kp", "court_kp_result.json"),
                load_path=load_path("court_kp"),
            )
        )

        gvhmr_config = None
        if Stage.GVHMR in resolution.enabled_set:
            smplx_body_model_path = cfg.gvhmr.get("smplx_body_model_path")
            gvhmr_config = {
                "python_executable": to_absolute_path(cfg.gvhmr.python_executable),
                "gvhmr_checkpoint": to_absolute_path(cfg.gvhmr.gvhmr_checkpoint),
                "yolo_checkpoint": to_absolute_path(cfg.gvhmr.yolo_checkpoint),
                "vitpose_checkpoint": to_absolute_path(cfg.gvhmr.vitpose_checkpoint),
                "hmr2_checkpoint": to_absolute_path(cfg.gvhmr.hmr2_checkpoint),
                "smplx_body_model_path": (
                    to_absolute_path(str(smplx_body_model_path))
                    if smplx_body_model_path is not None
                    else None
                ),
                "track_selection": str(cfg.gvhmr.get("track_selection", "interactive")),
                "num_tracks": int(cfg.gvhmr.get("num_tracks", 2)),
                "output_path": output_path("gvhmr", "gvhmr_result.json"),
                "load_path": load_paths("gvhmr"),
                "device": device,
            }

        player_association_cfg = cfg.get("player_association", {})
        player_association_module = PlayerAssociationModule(
            PlayerAssociationConfig(
                mode=str(player_association_cfg.get("mode", "manual_ui")),
                initial_frame_index=int(player_association_cfg.get("frame_index", 0)),
                reference_camera=player_association_cfg.get("reference_camera", 0),
                save_result=bool(player_association_cfg.get("save_result", True)),
                output_path=output_path(
                    "player_association", "player_association_result.json"
                ),
                load_path=load_path("player_association"),
            )
        )

        ball_detection_module = None
        if Stage.BALL_DETECTION in resolution.enabled_set:
            bcfg = cfg.ball_detection
            trajectory_gate_cfg = bcfg.get("trajectory_gate", {}) or {}
            ball_detection_module = BallDetectionModule(
                BallDetectionConfig(
                    checkpoint=to_absolute_path(bcfg.checkpoint),
                    batch_size=int(bcfg.batch_size),
                    device=device,
                    image_size=_size_hw(bcfg.get("image_size", [288, 512])),
                    normalize_imagenet=bool(bcfg.get("normalize_imagenet", True)),
                    score_threshold=float(bcfg.get("score_threshold", 0.5)),
                    subpixel_refine=bool(bcfg.get("subpixel_refine", True)),
                    prefetch_batches=int(bcfg.get("prefetch_batches", 2)),
                    window_stride=(
                        None
                        if bcfg.get("window_stride", None) is None
                        else int(bcfg.window_stride)
                    ),
                    tail_policy=str(bcfg.get("tail_policy", "backfill")),
                    overlap_aggregation=str(
                        bcfg.get("overlap_aggregation", "last_window_wins")
                    ),
                    pin_memory=bool(bcfg.get("pin_memory", True)),
                    trajectory_gate=TrajectoryGateConfig(
                        enabled=bool(trajectory_gate_cfg.get("enabled", False)),
                        max_residual_px=float(
                            trajectory_gate_cfg.get("max_residual_px", 60.0)
                        ),
                        k_support=int(trajectory_gate_cfg.get("k_support", 2)),
                        max_support_gap=int(
                            trajectory_gate_cfg.get("max_support_gap", 5)
                        ),
                        max_passes=int(trajectory_gate_cfg.get("max_passes", 2)),
                    ),
                    save_result=bcfg.get("save_result", True),
                    output_path=output_path(
                        "ball_detection", "ball_detection_result.json"
                    ),
                    load_path=load_path("ball_detection"),
                )
            )

        if Stage.PLCS not in resolution.enabled_set:
            raise ValueError("PLCS stage must be enabled.")
        plcs_module = PLCSModule(
            PLCSConfig(
                checkpoint_path=to_absolute_path(cfg.plcs.checkpoint),
                device=device,
                save_result=cfg.plcs.get("save_result", True),
                output_path=output_path("plcs", "plcs_result.json"),
                load_path=load_path("plcs"),
                window_size=int(cfg.plcs.get("window_size", 256)),
                window_overlap=int(cfg.plcs.get("window_overlap", 64)),
                human_vis_threshold=float(cfg.plcs.get("human_vis_threshold", 0.35)),
            )
        )

        blcs_module = None
        if Stage.BLCS in resolution.enabled_set:
            blcs_module = BLCSModule(
                BLCSConfig(
                    checkpoint_path=to_absolute_path(cfg.blcs.checkpoint),
                    device=device,
                    save_result=cfg.blcs.get("save_result", True),
                    output_path=output_path("blcs", "blcs_result.json"),
                    load_path=load_path("blcs"),
                    window_size=int(cfg.blcs.get("window_size", 256)),
                    window_overlap=int(cfg.blcs.get("window_overlap", 64)),
                )
            )

        return cls(
            court_kp_module=court_kp_module,
            gvhmr_config=gvhmr_config,
            player_association_module=player_association_module,
            ball_detection_module=ball_detection_module,
            plcs_module=plcs_module,
            blcs_module=blcs_module,
            resolution=resolution,
            device=device,
        )

    @staticmethod
    def _camera_artifact_path(path: str | Path, camera_index: int) -> Path:
        path = Path(path)
        return path.with_name(f"{path.stem}_cam{camera_index}{path.suffix}")

    @classmethod
    def _select_camera_artifact_path(
        cls,
        path_config: Any,
        *,
        camera_index: int,
        num_cameras: int,
    ) -> Path:
        if isinstance(path_config, (list, tuple)):
            return Path(path_config[camera_index])
        path = Path(path_config)
        if num_cameras == 1:
            return path
        return cls._camera_artifact_path(path, camera_index)

    def _run_gvhmr(
        self,
        video_path: Path,
        *,
        camera_index: int,
        num_cameras: int,
        max_frames: int | None = None,
    ) -> GVHMRResult:
        from src.tennis_scene.pipeline.components.gvhmr import GVHMRResult

        if self.gvhmr_config is None:
            raise RuntimeError("GVHMR config not set")

        load_path_config = self.gvhmr_config.get("load_path")
        load_path = (
            self._select_camera_artifact_path(
                load_path_config,
                camera_index=camera_index,
                num_cameras=num_cameras,
            )
            if load_path_config is not None
            else None
        )
        output_path = self._select_camera_artifact_path(
            self.gvhmr_config["output_path"],
            camera_index=camera_index,
            num_cameras=num_cameras,
        )

        if load_path is not None and load_path.exists():
            LOGGER.info(f"Loading GVHMR result from: {load_path}")
            return GVHMRResult.load(load_path)

        LOGGER.info(f"Running GVHMR via CLI subprocess for camera {camera_index}...")
        cmd = [
            self.gvhmr_config["python_executable"],
            "-m",
            "src.tennis_scene.pipeline.components.gvhmr",
            f"--video={video_path}",
            f"--output={output_path}",
            f"--model-checkpoint={self.gvhmr_config['gvhmr_checkpoint']}",
            f"--yolo-checkpoint={self.gvhmr_config['yolo_checkpoint']}",
            f"--vitpose-checkpoint={self.gvhmr_config['vitpose_checkpoint']}",
            f"--hmr2-checkpoint={self.gvhmr_config['hmr2_checkpoint']}",
            f"--track-selection={self.gvhmr_config['track_selection']}",
            f"--num-tracks={self.gvhmr_config['num_tracks']}",
            f"--device={self.gvhmr_config['device']}",
        ]
        if self.gvhmr_config.get("smplx_body_model_path") is not None:
            cmd.append(
                f"--smplx-body-model-path={self.gvhmr_config['smplx_body_model_path']}"
            )
        if max_frames is not None:
            cmd.append(f"--max-frames={max_frames}")

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

    def run(
        self,
        video_paths: Sequence[str | Path],
        max_frames: int | None = None,
        frame_index: int = 0,
        camera_ids: Sequence[str] | None = None,
    ) -> SceneResult:
        resolved_video_paths = [Path(video_path) for video_path in video_paths]
        if not resolved_video_paths:
            raise ValueError("video_paths must contain at least one video")
        if camera_ids is None:
            camera_ids = [f"cam{idx}" for idx in range(len(resolved_video_paths))]
        else:
            camera_ids = [str(camera_id) for camera_id in camera_ids]
            if len(camera_ids) != len(resolved_video_paths):
                raise ValueError(
                    f"camera_ids length must match video_paths length, "
                    f"got {len(camera_ids)} and {len(resolved_video_paths)}"
                )

        video_infos = self._probe_synced_video_infos(
            resolved_video_paths, max_frames=max_frames
        )
        video_info = video_infos[0]
        width, height = video_info.width, video_info.height
        num_cameras = len(resolved_video_paths)

        court_result = self.court_kp_module.process(
            resolved_video_paths,
            max_frames=max_frames,
            annotation_frame_index=frame_index,
        )
        court_kp = court_result.keypoints
        court_vis = court_result.visibility

        if Stage.GVHMR in self.enabled_stages and self.gvhmr_config is not None:
            association_result, aligned_players = self._run_gvhmr_multicamera(
                video_paths=resolved_video_paths,
                video_infos=video_infos,
                camera_ids=camera_ids,
                max_frames=max_frames,
            )
            human_kp_2d_norm = aligned_players.human_kp_2d
            human_kp_vis = aligned_players.human_kp_vis
            smpl_body_pose = aligned_players.smpl_body_pose
            smpl_global_orient = aligned_players.smpl_global_orient
            smpl_betas = aligned_players.smpl_betas
            smpl_vertices_local = aligned_players.smpl_vertices_local
            track_ids = aligned_players.track_ids
            track_ids_by_camera = aligned_players.track_ids_by_camera
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
        ball_vis = None
        ball_3d = None
        if (
            Stage.BALL_DETECTION in self.enabled_stages
            and self.ball_detection_module is not None
        ):
            ball_detection_result = self.ball_detection_module.process(
                resolved_video_paths,
                max_frames=max_frames,
                image_width=width,
                image_height=height,
            )
            ball_uv = ball_detection_result.ball_uv
            ball_vis = ball_detection_result.visibility
            if Stage.BLCS in self.enabled_stages and self.blcs_module is not None:
                blcs_result = self.blcs_module.process(
                    ball_uv=ball_uv,
                    court_kp=court_kp,
                    ball_vis=ball_vis,
                    court_vis=court_vis,
                )
                ball_3d = blcs_result.ball_3d

        T = plcs_result.position.shape[1]
        return SceneResult(
            num_frames=T,
            fps=video_info.fps,
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
            ball_vis=ball_vis,
            ball_3d=ball_3d,
            human_kp_2d=human_kp_2d_norm,
            human_kp_vis=human_kp_vis,
            player_track_ids=track_ids,
            metadata={
                "video_paths": [str(video_path) for video_path in resolved_video_paths],
                "camera_ids": list(camera_ids),
                "num_cameras": num_cameras,
                "sync_assumption": "preprocessed",
                "frame_index": frame_index,
                "court_kp_frame_indices": court_result.frame_indices.tolist(),
                "track_ids": track_ids.tolist(),
                "track_ids_by_camera": [
                    camera_track_ids.tolist()
                    for camera_track_ids in track_ids_by_camera
                ],
                "player_association": association_result.to_dict(),
                "enabled_stages": [stage.value for stage in self.execution_order],
            },
        )

    def _probe_synced_video_infos(
        self,
        video_paths: Sequence[Path],
        *,
        max_frames: int | None,
    ) -> list[VideoInfo]:
        """Probe videos and enforce the synchronized multi-camera contract."""
        video_infos = [probe_video_info(video_path) for video_path in video_paths]
        first = video_infos[0]
        expected_frames = first.frame_count
        if max_frames is not None:
            expected_frames = min(expected_frames, int(max_frames))
        for camera_index, video_info in enumerate(video_infos):
            frame_count = video_info.frame_count
            if max_frames is not None:
                frame_count = min(frame_count, int(max_frames))
            if frame_count != expected_frames:
                raise ValueError(
                    f"video_paths[{camera_index}] has T={frame_count}, "
                    f"expected synchronized T={expected_frames}"
                )
            if abs(video_info.fps - first.fps) > 1e-6:
                raise ValueError(
                    f"video_paths[{camera_index}] fps={video_info.fps} does not "
                    f"match first camera fps={first.fps}"
                )
            if (video_info.width, video_info.height) != (first.width, first.height):
                raise ValueError(
                    f"video_paths[{camera_index}] resolution="
                    f"{video_info.width}x{video_info.height} does not match "
                    f"first camera resolution={first.width}x{first.height}"
                )
        return video_infos

    def _run_gvhmr_multicamera(
        self,
        *,
        video_paths: Sequence[Path],
        video_infos: Sequence[VideoInfo],
        camera_ids: Sequence[str],
        max_frames: int | None,
    ) -> tuple[PlayerAssociationResult, PlayerAssociationApplied]:
        """Run per-camera GVHMR and align players as (P, N, T, ...)."""
        gvhmr_results = [
            self._run_gvhmr(
                video_path,
                camera_index=camera_index,
                num_cameras=len(video_paths),
                max_frames=max_frames,
            )
            for camera_index, video_path in enumerate(video_paths)
        ]

        first_shape = gvhmr_results[0].human_kp_2d.shape[1:]
        for camera_index, gvhmr_result in enumerate(gvhmr_results):
            human_kp_2d = gvhmr_result.human_kp_2d
            if human_kp_2d.shape[1:] != first_shape:
                raise ValueError(
                    f"GVHMR camera {camera_index} human_kp_2d trailing shape "
                    f"{human_kp_2d.shape[1:]} does not match first camera {first_shape}"
                )
            if gvhmr_result.human_kp_vis.shape != human_kp_2d.shape[:3]:
                raise ValueError(
                    f"GVHMR camera {camera_index} human_kp_vis shape "
                    f"{gvhmr_result.human_kp_vis.shape} is invalid"
                )

        association_result = self.player_association_module.process(
            gvhmr_results=gvhmr_results,
            video_paths=video_paths,
            video_infos=video_infos,
            camera_ids=camera_ids,
        )
        aligned_players = self.player_association_module.apply(
            gvhmr_results=gvhmr_results,
            video_infos=video_infos,
            association=association_result,
        )
        return association_result, aligned_players


if __name__ == "__main__":
    print("TennisSceneOrchestrator: pipeline orchestration module")
