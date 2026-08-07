"""Orchestrator for tennis scene 3D reconstruction pipeline."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING

from src.tennis_scene.pipeline.components.ball_detection import (
    BallDetectionModule,
)
from src.tennis_scene.pipeline.components.blcs import BLCSModule
from src.tennis_scene.pipeline.components.court_kp import (
    CourtKPModule,
)
from src.tennis_scene.pipeline.components.gvhmr import GVHMRConfig, GVHMRModule
from src.tennis_scene.pipeline.components.player_association import (
    PlayerAssociationModule,
)
from src.tennis_scene.pipeline.components.plcs import PLCSModule
from src.tennis_scene.pipeline.dependency_graph import (
    ResolutionResult,
    Stage,
    build_default_dependency_graph,
)
from src.tennis_scene.pipeline.model_io.gvhmr import (
    GVHMRChain,
    GVHMRResult,
    build_gvhmr_chain,
)
from src.tennis_scene.schema import SceneResult
from src.utils.configuration import PathResolver, PathRole
from src.utils.video import probe_video_info

if TYPE_CHECKING:
    from omegaconf import DictConfig

    from src.tennis_scene.configuration import PipelineRuntimeConfig
    from src.tennis_scene.pipeline.components.player_association import (
        PlayerAssociationApplied,
        PlayerAssociationResult,
    )
    from src.utils.video import VideoInfo

LOGGER = logging.getLogger(__name__)


class TennisSceneOrchestrator:
    """Orchestrator for tennis scene 3D reconstruction."""

    def __init__(
        self,
        court_kp_module: CourtKPModule,
        gvhmr_config: GVHMRConfig | None,
        gvhmr_chain: GVHMRChain | None,
        player_association_module: PlayerAssociationModule,
        ball_detection_module: BallDetectionModule | None,
        plcs_module: PLCSModule,
        blcs_module: BLCSModule | None,
        resolution: ResolutionResult,
        device: str,
        resolver: PathResolver,
    ) -> None:
        self.court_kp_module = court_kp_module
        self.gvhmr_config = gvhmr_config
        self.gvhmr_chain = gvhmr_chain
        self.player_association_module = player_association_module
        self.ball_detection_module = ball_detection_module
        self.plcs_module = plcs_module
        self.blcs_module = blcs_module
        self.resolution = resolution
        self.enabled_stages = resolution.enabled_set
        self.execution_order = resolution.enabled_order
        self.device = device
        self.resolver = resolver

    @classmethod
    def from_runtime_config(cls, cfg: PipelineRuntimeConfig) -> TennisSceneOrchestrator:
        """Build every stage from an already validated runtime contract."""
        graph = build_default_dependency_graph()
        resolution = graph.resolve_from_enabled(cfg.enabled)
        for line in graph.format_resolution_messages(resolution):
            LOGGER.info(line)
        gvhmr_enabled = Stage.GVHMR in resolution.enabled_set
        gvhmr_config = cfg.gvhmr if gvhmr_enabled else None
        gvhmr_chain = (
            build_gvhmr_chain(cfg.gvhmr)
            if gvhmr_enabled and cfg.gvhmr.source == "execute"
            else None
        )
        return cls(
            court_kp_module=CourtKPModule(cfg.court_kp),
            gvhmr_config=gvhmr_config,
            gvhmr_chain=gvhmr_chain,
            player_association_module=PlayerAssociationModule(cfg.player_association),
            ball_detection_module=(
                BallDetectionModule(cfg.ball_detection)
                if Stage.BALL_DETECTION in resolution.enabled_set
                else None
            ),
            plcs_module=PLCSModule(cfg.plcs),
            blcs_module=(
                BLCSModule(cfg.blcs) if Stage.BLCS in resolution.enabled_set else None
            ),
            resolution=resolution,
            device=cfg.device,
            resolver=cfg.resolver,
        )

    @classmethod
    def from_config(cls, cfg: DictConfig) -> TennisSceneOrchestrator:
        """Validate a composed Hydra config before constructing any stage."""
        from src.tennis_scene.configuration import PipelineRuntimeConfig

        return cls.from_runtime_config(PipelineRuntimeConfig.from_config(cfg))

    @staticmethod
    def _camera_artifact_path(path: Path, camera_index: int) -> Path:
        return path.with_name(f"{path.stem}_cam{camera_index}{path.suffix}")

    @classmethod
    def _select_camera_artifact_path(
        cls,
        path: Path,
        *,
        camera_index: int,
        num_cameras: int,
    ) -> Path:
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
        if self.gvhmr_config is None:
            raise RuntimeError("GVHMR config not set")

        load_path_config = self.gvhmr_config.load_path
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
            self.gvhmr_config.output_path,
            camera_index=camera_index,
            num_cameras=num_cameras,
        )

        LOGGER.info(f"Running GVHMR in-process for camera {camera_index}...")
        module = GVHMRModule(
            replace(
                self.gvhmr_config,
                output_path=output_path,
                load_path=load_path,
            ),
            self.gvhmr_chain,
        )
        return module.process(video_path, max_frames=max_frames)

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
        video_paths: Sequence[Path],
        *,
        video_role: PathRole,
        max_frames: int | None,
        frame_index: int,
        camera_ids: Sequence[str],
    ) -> SceneResult:
        resolved_video_paths = [
            self.resolver.validate(video_role, video_path) for video_path in video_paths
        ]
        if not resolved_video_paths:
            raise ValueError("video_paths must contain at least one video")
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
