"""GVHMR module for 3D human mesh estimation.

Runs the src/submodules model chain (person detection/tracking -> ViTPose -> HMR2 features
-> GVHMR) in the main ``.venv``; there is no dependency on ``third_party/GVHMR``
code anymore (checkpoints are read via the ``ckpt/`` symlinks).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import torch

from src.submodules.configuration import BundledModelAssetPaths, SubmoduleRuntimeConfig
from src.tennis_scene.pipeline.components.base import BasePipelineModule
from src.utils.io import load_json, save_json

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from src.submodules.models import (
        DinoPersonTracker,
        GvhmrMeshRecovery,
        Hmr2FeatureExtractor,
        SmplVertexReconstructor,
        ViTPosePose2D,
        YoloPersonTracker,
    )

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class GVHMRConfig:
    """Configuration for GVHMR module."""

    gvhmr_checkpoint: Path
    source: Literal["execute", "load"]
    detector: str
    yolo_checkpoint: Path
    dino_checkpoint: Path
    dino_repository: Path
    vitpose_checkpoint: Path
    hmr2_checkpoint: Path
    body_models_dir: Path
    bundled_assets: BundledModelAssetPaths
    runtime: SubmoduleRuntimeConfig
    track_selection: str
    num_tracks: int
    save_result: bool
    output_path: Path
    load_path: Path | None

    def __post_init__(self) -> None:
        if (self.source == "load") != (self.load_path is not None):
            raise ValueError(
                "GVHMR source='load' requires load_path; execute forbids it"
            )
        if self.detector not in {"yolo", "dino"}:
            raise ValueError(
                f"detector must be 'yolo' or 'dino', got {self.detector!r}"
            )
        if self.track_selection not in {"interactive", "auto"}:
            raise ValueError(
                "track_selection must be 'interactive' or 'auto', got "
                f"{self.track_selection!r}"
            )


@dataclass
class GVHMRResult:
    """Result of GVHMR inference.

    Attributes:
        smpl_body_pose: SMPL body pose parameters, shape (P, T, 63).
        smpl_global_orient: SMPL global orientation, shape (P, T, 3).
        smpl_betas: SMPL shape parameters, shape (P, 10).
        smpl_vertices_local: GVHMR/SMPL vertices, shape (P, T, V, 3) or None.
            These remain in the SMPL body convention (Y-up) and are not court
            coordinates; visualization converts them to court Z-up explicitly.
        human_kp_2d: 2D keypoints in pixels, shape (P, T, 17, 2).
        human_kp_vis: Keypoint visibility/confidence, shape (P, T, 17).
        bbx_xys: Bounding boxes, shape (P, T, 3).
        track_ids: Track IDs aligned to player axis, shape (P,).
    """

    smpl_body_pose: NDArray[np.float32]
    smpl_global_orient: NDArray[np.float32]
    smpl_betas: NDArray[np.float32]
    smpl_vertices_local: NDArray[np.float32] | None
    human_kp_2d: NDArray[np.float32]
    human_kp_vis: NDArray[np.float32]
    bbx_xys: NDArray[np.float32]
    track_ids: NDArray[np.int32] | None = None

    def to_dict(self) -> dict:
        result = {
            "smpl_body_pose": self.smpl_body_pose.tolist(),
            "smpl_global_orient": self.smpl_global_orient.tolist(),
            "smpl_betas": self.smpl_betas.tolist(),
            "human_kp_2d": self.human_kp_2d.tolist(),
            "human_kp_vis": self.human_kp_vis.tolist(),
            "bbx_xys": self.bbx_xys.tolist(),
        }
        if self.smpl_vertices_local is not None:
            result["smpl_vertices_local"] = self.smpl_vertices_local.tolist()
        if self.track_ids is not None:
            result["track_ids"] = self.track_ids.tolist()
        return result

    @classmethod
    def from_dict(cls, data: dict) -> GVHMRResult:
        smpl_body_pose = np.array(data["smpl_body_pose"], dtype=np.float32)
        smpl_global_orient = np.array(data["smpl_global_orient"], dtype=np.float32)
        smpl_betas = np.array(data["smpl_betas"], dtype=np.float32)
        human_kp_2d = np.array(data["human_kp_2d"], dtype=np.float32)
        human_kp_vis = np.array(data["human_kp_vis"], dtype=np.float32)
        bbx_xys = np.array(data["bbx_xys"], dtype=np.float32)

        vertices = data.get("smpl_vertices_local")
        smpl_vertices_local = None
        if vertices is not None:
            smpl_vertices_local = np.array(vertices, dtype=np.float32)

        track_ids = data.get("track_ids")
        if track_ids is not None:
            track_ids = np.array(track_ids, dtype=np.int32)

        return cls(
            smpl_body_pose=smpl_body_pose,
            smpl_global_orient=smpl_global_orient,
            smpl_betas=smpl_betas,
            smpl_vertices_local=smpl_vertices_local,
            human_kp_2d=human_kp_2d,
            human_kp_vis=human_kp_vis,
            bbx_xys=bbx_xys,
            track_ids=track_ids,
        )

    def save(self, path: str | Path) -> None:
        save_json(self.to_dict(), path)
        LOGGER.info(f"Saved GVHMR result to {path}")

    @classmethod
    def load(cls, path: str | Path) -> GVHMRResult:
        return cls.from_dict(load_json(path))


class GVHMRModule(BasePipelineModule):
    """GVHMR module for 3D human mesh estimation (src/submodules based)."""

    def __init__(self, config: GVHMRConfig) -> None:
        self.config = config
        self._tracker: YoloPersonTracker | DinoPersonTracker | None = None
        self._pose_model: ViTPosePose2D | None = None
        self._feature_model: Hmr2FeatureExtractor | None = None
        self._mesh_model: GvhmrMeshRecovery | None = None
        self._vertex_reconstructor: SmplVertexReconstructor | None = None

    def load(self) -> None:
        if self.is_loaded:
            return

        from src.submodules.models import (
            DinoPersonTracker,
            GvhmrMeshRecovery,
            Hmr2FeatureExtractor,
            SmplVertexReconstructor,
            ViTPosePose2D,
            YoloPersonTracker,
        )

        runtime = self.config.runtime
        device = runtime.device
        allow_device_fallback = runtime.allow_device_fallback
        if self.config.detector == "yolo":
            self._tracker = YoloPersonTracker(
                checkpoint=self.config.yolo_checkpoint,
                device=device,
                allow_device_fallback=allow_device_fallback,
                confidence=runtime.tracking.yolo_confidence,
            )
        elif self.config.detector == "dino":
            self._tracker = DinoPersonTracker(
                checkpoint=self.config.dino_checkpoint,
                repository=self.config.dino_repository,
                device=device,
                allow_device_fallback=allow_device_fallback,
                confidence=runtime.dino_detector.confidence,
                short_side=runtime.dino_detector.short_side,
                max_long_side=runtime.dino_detector.max_long_side,
            )
        else:
            raise AssertionError(f"Unvalidated detector: {self.config.detector}")
        self._pose_model = ViTPosePose2D(
            checkpoint=self.config.vitpose_checkpoint,
            device=device,
            allow_device_fallback=allow_device_fallback,
            flip_test=runtime.vitpose.flip_test,
            batch_size=runtime.vitpose.batch_size,
            head_config=runtime.vitpose.head,
        )
        self._feature_model = Hmr2FeatureExtractor(
            checkpoint=self.config.hmr2_checkpoint,
            device=device,
            allow_device_fallback=allow_device_fallback,
            batch_size=runtime.hmr2.batch_size,
            mean_params_path=self.config.bundled_assets.hmr2_mean_params,
        )
        self._mesh_model = GvhmrMeshRecovery(
            checkpoint=self.config.gvhmr_checkpoint,
            body_models_dir=self.config.body_models_dir,
            device=device,
            allow_device_fallback=allow_device_fallback,
            bundled_assets=self.config.bundled_assets,
        )
        self._vertex_reconstructor = SmplVertexReconstructor(
            body_models_dir=self.config.body_models_dir,
            device=device,
            allow_device_fallback=allow_device_fallback,
            bundled_assets=self.config.bundled_assets,
        )
        self._mesh_model.load()

    @property
    def is_loaded(self) -> bool:
        return self._mesh_model is not None

    def process(
        self,
        video_path: Path,
        max_frames: int | None = None,
    ) -> GVHMRResult:
        """Run GVHMR preprocessing and inference.

        Returns:
            GVHMRResult with shapes based on (P, T, ...).
        """
        if self.config.source == "load":
            assert self.config.load_path is not None
            load_path = self.config.load_path
            if not load_path.is_file():
                raise FileNotFoundError(f"GVHMR artifact not found: {load_path}")
            LOGGER.info(f"Loading GVHMR result from {load_path}")
            return GVHMRResult.load(load_path)

        from src.submodules.models.tracker import TrackRequest

        if not self.is_loaded:
            self.load()
        assert self._tracker is not None

        track_result = self._tracker.predict(
            TrackRequest(
                video_path=video_path,
                num_tracks=self.config.num_tracks,
                interactive=self.config.track_selection == "interactive",
            )
        )
        track_ids = track_result.track_ids
        if not track_ids:
            raise RuntimeError("No tracks selected")

        players: list[dict[str, Any]] = []
        for track_id in track_ids:
            bbx_xys = track_result.bbx_xys(
                track_id,
                base_enlarge=self.config.runtime.tracking.bbox_enlarge,
            )
            if max_frames is not None and len(bbx_xys) > max_frames:
                bbx_xys = bbx_xys[:max_frames]

            LOGGER.info(f"Running GVHMR for track_id={track_id}")
            players.append(self._run_track(video_path, track_id, bbx_xys))

        frame_lengths = {p["human_kp_2d"].shape[0] for p in players}
        if len(frame_lengths) != 1:
            raise RuntimeError(
                f"Selected tracks have inconsistent frame lengths: {sorted(frame_lengths)}"
            )

        result = GVHMRResult(
            smpl_body_pose=np.stack([p["smpl_body_pose"] for p in players], axis=0),
            smpl_global_orient=np.stack(
                [p["smpl_global_orient"] for p in players], axis=0
            ),
            smpl_betas=np.stack([p["smpl_betas"] for p in players], axis=0),
            smpl_vertices_local=np.stack(
                [p["smpl_vertices_local"] for p in players], axis=0
            ),
            human_kp_2d=np.stack([p["human_kp_2d"] for p in players], axis=0),
            human_kp_vis=np.stack([p["human_kp_vis"] for p in players], axis=0),
            bbx_xys=np.stack([p["bbx_xys"] for p in players], axis=0),
            track_ids=np.array(track_ids, dtype=np.int32),
        )

        if self.config.save_result:
            result.save(self.config.output_path)

        return result

    def _run_track(
        self,
        video_path: Path,
        track_id: int,
        bbx_xys: torch.Tensor,
    ) -> dict[str, Any]:
        """Run pose/feature extraction and GVHMR for one person track."""
        from src.submodules.models.gvhmr import GvhmrRequest
        from src.submodules.models.hmr2 import ImageFeatureRequest
        from src.submodules.models.vitpose import Pose2DRequest
        from src.utils.video.reader import probe_video_info

        assert self._pose_model is not None
        assert self._feature_model is not None
        assert self._mesh_model is not None
        assert self._vertex_reconstructor is not None

        info = probe_video_info(video_path)

        pose = self._pose_model.predict(
            Pose2DRequest(video_path=video_path, bbx_xys=bbx_xys)
        )
        features = self._feature_model.predict(
            ImageFeatureRequest(video_path=video_path, bbx_xys=bbx_xys)
        )
        gvhmr = self._mesh_model.predict(
            GvhmrRequest(
                kp2d=pose.keypoints,
                bbx_xys=bbx_xys,
                f_imgseq=features.features,
                width=info.width,
                height=info.height,
                static_cam=self.config.runtime.static_cam,
            )
        )

        smpl_params = gvhmr.smpl_params_incam
        vertices = self._vertex_reconstructor.reconstruct(smpl_params)

        betas = smpl_params["betas"].numpy().astype(np.float32)
        if betas.ndim == 2:
            betas = betas[0]

        return {
            "track_id": track_id,
            "smpl_body_pose": smpl_params["body_pose"].numpy().astype(np.float32),
            "smpl_global_orient": smpl_params["global_orient"]
            .numpy()
            .astype(np.float32),
            "smpl_betas": betas,
            "smpl_vertices_local": vertices.numpy().astype(np.float32),
            "human_kp_2d": pose.keypoints[..., :2].numpy().astype(np.float32),
            "human_kp_vis": pose.keypoints[..., 2].numpy().astype(np.float32),
            "bbx_xys": bbx_xys.numpy().astype(np.float32),
        }
