"""Typed composition and I/O boundary for the complete GVHMR model chain."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

import numpy as np
import torch

import src.submodules.models as submodule_models
from src.utils.io import load_json, save_json
from src.utils.video import VideoInfo, probe_video_info

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from src.submodules.configuration import (
        BundledModelAssetPaths,
        SubmoduleRuntimeConfig,
    )

LOGGER = logging.getLogger(__name__)

_REQUIRED_SMPL_WIDTHS = {
    "body_pose": 63,
    "betas": 10,
    "global_orient": 3,
    "transl": 3,
}


class GVHMRContractError(ValueError):
    """A typed GVHMR chain request or model result violated its contract."""


@dataclass(frozen=True, slots=True)
class GVHMRChainRequest:
    """High-level request accepted by a resolved GVHMR chain."""

    video_path: Path
    max_frames: int | None
    num_tracks: int
    interactive: bool
    bbox_enlarge: float
    static_cam: bool

    def __post_init__(self) -> None:
        if not isinstance(self.video_path, Path):
            raise TypeError("GVHMRChainRequest.video_path must be a pathlib.Path.")
        if self.max_frames is not None:
            if type(self.max_frames) is not int:
                raise TypeError("GVHMRChainRequest.max_frames must be an integer.")
            if self.max_frames <= 0:
                raise GVHMRContractError(
                    "GVHMRChainRequest.max_frames must be positive."
                )
        if type(self.num_tracks) is not int:
            raise TypeError("GVHMRChainRequest.num_tracks must be an integer.")
        if self.num_tracks <= 0:
            raise GVHMRContractError(
                "GVHMRChainRequest.num_tracks must be positive."
            )
        if type(self.interactive) is not bool:
            raise TypeError("GVHMRChainRequest.interactive must be a bool.")
        if type(self.bbox_enlarge) is not float:
            raise TypeError("GVHMRChainRequest.bbox_enlarge must be a float.")
        if not math.isfinite(self.bbox_enlarge) or self.bbox_enlarge <= 0.0:
            raise GVHMRContractError(
                "GVHMRChainRequest.bbox_enlarge must be finite and positive."
            )
        if type(self.static_cam) is not bool:
            raise TypeError("GVHMRChainRequest.static_cam must be a bool.")


@dataclass
class GVHMRResult:
    """Decoded, pipeline-ready GVHMR inference result.

    Shapes use ``P`` selected people and ``T`` frames:
    ``smpl_body_pose (P,T,63)``, ``smpl_global_orient (P,T,3)``,
    ``smpl_betas (P,10)``, optional ``smpl_vertices_local (P,T,V,3)``,
    ``human_kp_2d (P,T,17,2)``, ``human_kp_vis (P,T,17)``,
    ``bbx_xys (P,T,3)``, and ``track_ids (P,)``.
    """

    smpl_body_pose: NDArray[np.float32]
    smpl_global_orient: NDArray[np.float32]
    smpl_betas: NDArray[np.float32]
    smpl_vertices_local: NDArray[np.float32] | None
    human_kp_2d: NDArray[np.float32]
    human_kp_vis: NDArray[np.float32]
    bbx_xys: NDArray[np.float32]
    track_ids: NDArray[np.int32]

    def to_dict(self) -> dict[str, object]:
        result: dict[str, object] = {
            "smpl_body_pose": self.smpl_body_pose.tolist(),
            "smpl_global_orient": self.smpl_global_orient.tolist(),
            "smpl_betas": self.smpl_betas.tolist(),
            "human_kp_2d": self.human_kp_2d.tolist(),
            "human_kp_vis": self.human_kp_vis.tolist(),
            "bbx_xys": self.bbx_xys.tolist(),
            "track_ids": self.track_ids.tolist(),
        }
        if self.smpl_vertices_local is not None:
            result["smpl_vertices_local"] = self.smpl_vertices_local.tolist()
        return result

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> GVHMRResult:
        missing = {
            "smpl_body_pose",
            "smpl_global_orient",
            "smpl_betas",
            "human_kp_2d",
            "human_kp_vis",
            "bbx_xys",
            "track_ids",
        } - set(data)
        if missing:
            raise GVHMRContractError(
                f"GVHMR result is missing required fields: {sorted(missing)}"
            )
        vertices = data.get("smpl_vertices_local")
        return cls(
            smpl_body_pose=np.asarray(data["smpl_body_pose"], dtype=np.float32),
            smpl_global_orient=np.asarray(
                data["smpl_global_orient"], dtype=np.float32
            ),
            smpl_betas=np.asarray(data["smpl_betas"], dtype=np.float32),
            smpl_vertices_local=(
                None if vertices is None else np.asarray(vertices, dtype=np.float32)
            ),
            human_kp_2d=np.asarray(data["human_kp_2d"], dtype=np.float32),
            human_kp_vis=np.asarray(data["human_kp_vis"], dtype=np.float32),
            bbx_xys=np.asarray(data["bbx_xys"], dtype=np.float32),
            track_ids=np.asarray(data["track_ids"], dtype=np.int32),
        )

    def save(self, path: str | Path) -> None:
        save_json(self.to_dict(), path)
        LOGGER.info("Saved GVHMR result to %s", path)

    @classmethod
    def load(cls, path: str | Path) -> GVHMRResult:
        payload = load_json(path)
        if not isinstance(payload, dict):
            raise GVHMRContractError("GVHMR result artifact must contain an object.")
        return cls.from_dict(payload)


class GVHMRChain(Protocol):
    """Resolved chain consumed by :class:`GVHMRModule`."""

    @property
    def is_loaded(self) -> bool:
        """Whether all loadable chain models are resident."""
        ...

    def load(self) -> None:
        """Load all chain models without choosing an implementation."""
        ...

    def unload(self) -> None:
        """Unload all chain models."""
        ...

    def predict(self, request: GVHMRChainRequest) -> GVHMRResult:
        """Validate, execute, and decode the full chain."""
        ...


class GVHMRChainConfig(Protocol):
    """Factory-facing subset of the validated tennis-scene configuration."""

    @property
    def detector(self) -> str: ...

    @property
    def yolo_checkpoint(self) -> Path: ...

    @property
    def dino_checkpoint(self) -> Path: ...

    @property
    def dino_repository(self) -> Path: ...

    @property
    def vitpose_checkpoint(self) -> Path: ...

    @property
    def hmr2_checkpoint(self) -> Path: ...

    @property
    def gvhmr_checkpoint(self) -> Path: ...

    @property
    def body_models_dir(self) -> Path: ...

    @property
    def bundled_assets(self) -> BundledModelAssetPaths: ...

    @property
    def runtime(self) -> SubmoduleRuntimeConfig: ...


@dataclass(frozen=True, slots=True)
class _DecodedTrack:
    track_id: int
    smpl_body_pose: NDArray[np.float32]
    smpl_global_orient: NDArray[np.float32]
    smpl_betas: NDArray[np.float32]
    smpl_vertices_local: NDArray[np.float32]
    human_kp_2d: NDArray[np.float32]
    human_kp_vis: NDArray[np.float32]
    bbx_xys: NDArray[np.float32]


@dataclass(slots=True)
class GVHMRChainAdapter:
    """Validate each typed submodule result before entering the next model."""

    tracker: submodule_models.YoloPersonTracker | submodule_models.DinoPersonTracker
    pose_model: submodule_models.ViTPosePose2D
    feature_model: submodule_models.Hmr2FeatureExtractor
    mesh_model: submodule_models.GvhmrMeshRecovery
    vertex_reconstructor: submodule_models.SmplVertexReconstructor

    @property
    def is_loaded(self) -> bool:
        return all(
            model.is_loaded
            for model in (
                self.tracker,
                self.pose_model,
                self.feature_model,
                self.mesh_model,
            )
        )

    def load(self) -> None:
        for model in (
            self.tracker,
            self.pose_model,
            self.feature_model,
            self.mesh_model,
        ):
            model.load()

    def unload(self) -> None:
        for model in (
            self.mesh_model,
            self.feature_model,
            self.pose_model,
            self.tracker,
        ):
            model.unload()

    def predict(self, request: GVHMRChainRequest) -> GVHMRResult:
        info = _validate_video(request.video_path)
        expected_frames = (
            info.frame_count
            if request.max_frames is None
            else min(info.frame_count, request.max_frames)
        )

        track_result = self.tracker.predict(
            submodule_models.TrackRequest(
                video_path=request.video_path,
                num_tracks=request.num_tracks,
                interactive=request.interactive,
            )
        )
        track_ids = _validate_track_result(track_result, info=info)

        decoded_tracks = [
            self._predict_track(
                request=request,
                info=info,
                expected_frames=expected_frames,
                track_result=track_result,
                track_id=track_id,
            )
            for track_id in track_ids
        ]
        frame_lengths = {track.human_kp_2d.shape[0] for track in decoded_tracks}
        if frame_lengths != {expected_frames}:
            raise GVHMRContractError(
                "Selected GVHMR tracks have inconsistent frame lengths: "
                f"{sorted(frame_lengths)}; expected {expected_frames}."
            )

        return GVHMRResult(
            smpl_body_pose=np.stack(
                [track.smpl_body_pose for track in decoded_tracks], axis=0
            ),
            smpl_global_orient=np.stack(
                [track.smpl_global_orient for track in decoded_tracks], axis=0
            ),
            smpl_betas=np.stack(
                [track.smpl_betas for track in decoded_tracks], axis=0
            ),
            smpl_vertices_local=np.stack(
                [track.smpl_vertices_local for track in decoded_tracks], axis=0
            ),
            human_kp_2d=np.stack(
                [track.human_kp_2d for track in decoded_tracks], axis=0
            ),
            human_kp_vis=np.stack(
                [track.human_kp_vis for track in decoded_tracks], axis=0
            ),
            bbx_xys=np.stack([track.bbx_xys for track in decoded_tracks], axis=0),
            track_ids=np.asarray(track_ids, dtype=np.int32),
        )

    def _predict_track(
        self,
        *,
        request: GVHMRChainRequest,
        info: VideoInfo,
        expected_frames: int,
        track_result: submodule_models.TrackResult,
        track_id: int,
    ) -> _DecodedTrack:
        boxes = track_result.bbx_xys(
            track_id,
            base_enlarge=request.bbox_enlarge,
        )[:expected_frames]
        _validate_float_tensor(
            boxes,
            name="selected track boxes",
            expected_shape=(expected_frames, 3),
        )
        if boxes.device.type != "cpu":
            raise GVHMRContractError(
                f"selected track boxes must be on CPU, got {boxes.device}."
            )
        if not bool((boxes[:, 2] > 0.0).all()):
            raise GVHMRContractError("selected track box sizes must be positive.")

        pose_result = self.pose_model.predict(
            submodule_models.Pose2DRequest(
                video_path=request.video_path,
                bbx_xys=boxes,
            )
        )
        if not isinstance(pose_result, submodule_models.Pose2DResult):
            raise TypeError("ViTPose must return Pose2DResult.")
        keypoints = pose_result.keypoints
        _validate_float_tensor(
            keypoints,
            name="ViTPose keypoints",
            expected_shape=(expected_frames, 17, 3),
            reference=boxes,
        )
        if not bool((keypoints[..., 2] >= 0.0).all()):
            raise GVHMRContractError(
                "ViTPose keypoint confidence values must be non-negative."
            )

        feature_result = self.feature_model.predict(
            submodule_models.ImageFeatureRequest(
                video_path=request.video_path,
                bbx_xys=boxes,
            )
        )
        if not isinstance(feature_result, submodule_models.ImageFeatureResult):
            raise TypeError("HMR2 must return ImageFeatureResult.")
        features = feature_result.features
        _validate_float_tensor(
            features,
            name="HMR2 features",
            expected_shape=(expected_frames, 1024),
            reference=boxes,
        )

        mesh_result = self.mesh_model.predict(
            submodule_models.GvhmrRequest(
                kp2d=keypoints,
                bbx_xys=boxes,
                f_imgseq=features,
                width=info.width,
                height=info.height,
                static_cam=request.static_cam,
            )
        )
        return self._decode_track_result(
            track_id=track_id,
            boxes=boxes,
            keypoints=keypoints,
            mesh_result=mesh_result,
            expected_frames=expected_frames,
        )

    def _decode_track_result(
        self,
        *,
        track_id: int,
        boxes: torch.Tensor,
        keypoints: torch.Tensor,
        mesh_result: submodule_models.GvhmrResult,
        expected_frames: int,
    ) -> _DecodedTrack:
        if not isinstance(mesh_result, submodule_models.GvhmrResult):
            raise TypeError("GVHMR mesh recovery must return GvhmrResult.")
        incam = _validate_smpl_parameters(
            mesh_result.smpl_params_incam,
            name="smpl_params_incam",
            expected_frames=expected_frames,
            reference=boxes,
        )
        _validate_smpl_parameters(
            mesh_result.smpl_params_global,
            name="smpl_params_global",
            expected_frames=expected_frames,
            reference=boxes,
        )
        _validate_float_tensor(
            mesh_result.K_fullimg,
            name="GVHMR K_fullimg",
            expected_shape=(expected_frames, 3, 3),
            reference=boxes,
        )

        vertices = self.vertex_reconstructor.reconstruct(incam)
        if not isinstance(vertices, torch.Tensor) or vertices.ndim != 3:
            shape = getattr(vertices, "shape", None)
            raise GVHMRContractError(
                "SMPL vertex reconstruction must return a tensor shaped "
                f"(F,V,3), got {shape}."
            )
        _validate_float_tensor(
            vertices,
            name="SMPL vertices",
            expected_shape=(expected_frames, int(vertices.shape[1]), 3),
            reference=boxes,
        )
        if vertices.shape[1] <= 0:
            raise GVHMRContractError(
                "SMPL vertex reconstruction must contain at least one vertex."
            )

        betas = incam["betas"]
        if not bool(torch.allclose(betas, betas[:1].expand_as(betas))):
            raise GVHMRContractError(
                "GVHMR betas must be constant across a selected track."
            )
        return _DecodedTrack(
            track_id=track_id,
            smpl_body_pose=_as_float32_numpy(incam["body_pose"]),
            smpl_global_orient=_as_float32_numpy(incam["global_orient"]),
            smpl_betas=_as_float32_numpy(betas[0]),
            smpl_vertices_local=_as_float32_numpy(vertices),
            human_kp_2d=_as_float32_numpy(keypoints[..., :2]),
            human_kp_vis=_as_float32_numpy(keypoints[..., 2]),
            bbx_xys=_as_float32_numpy(boxes),
        )


def build_gvhmr_chain(config: GVHMRChainConfig) -> GVHMRChainAdapter:
    """Select the detector once and construct the resolved model chain."""
    runtime = config.runtime
    common_device = {
        "device": runtime.device,
    }
    tracker: (
        submodule_models.YoloPersonTracker | submodule_models.DinoPersonTracker
    )
    if config.detector == "yolo":
        tracker = submodule_models.YoloPersonTracker(
            checkpoint=config.yolo_checkpoint,
            confidence=runtime.tracking.yolo_confidence,
            **common_device,
        )
    elif config.detector == "dino":
        tracker = submodule_models.DinoPersonTracker(
            checkpoint=config.dino_checkpoint,
            repository=config.dino_repository,
            confidence=runtime.dino_detector.confidence,
            short_side=runtime.dino_detector.short_side,
            max_long_side=runtime.dino_detector.max_long_side,
            **common_device,
        )
    else:
        raise GVHMRContractError(
            f"Cannot compose unknown GVHMR detector {config.detector!r}."
        )

    return GVHMRChainAdapter(
        tracker=tracker,
        pose_model=submodule_models.ViTPosePose2D(
            checkpoint=config.vitpose_checkpoint,
            flip_test=runtime.vitpose.flip_test,
            batch_size=runtime.vitpose.batch_size,
            head_config=runtime.vitpose.head,
            **common_device,
        ),
        feature_model=submodule_models.Hmr2FeatureExtractor(
            checkpoint=config.hmr2_checkpoint,
            batch_size=runtime.hmr2.batch_size,
            mean_params_path=config.bundled_assets.hmr2_mean_params,
            **common_device,
        ),
        mesh_model=submodule_models.GvhmrMeshRecovery(
            checkpoint=config.gvhmr_checkpoint,
            body_models_dir=config.body_models_dir,
            bundled_assets=config.bundled_assets,
            **common_device,
        ),
        vertex_reconstructor=submodule_models.SmplVertexReconstructor(
            body_models_dir=config.body_models_dir,
            bundled_assets=config.bundled_assets,
            **common_device,
        ),
    )


def _validate_video(video_path: Path) -> VideoInfo:
    if not video_path.is_file():
        raise FileNotFoundError(f"GVHMR input video not found: {video_path}")
    info = probe_video_info(video_path)
    if not math.isfinite(info.fps) or info.fps <= 0.0:
        raise GVHMRContractError(
            f"GVHMR input video FPS must be finite and positive, got {info.fps}."
        )
    for name, value in {
        "width": info.width,
        "height": info.height,
        "frame_count": info.frame_count,
    }.items():
        if type(value) is not int or value <= 0:
            raise GVHMRContractError(
                f"GVHMR input video {name} must be a positive integer, got {value}."
            )
    return info


def _validate_track_result(
    result: submodule_models.TrackResult,
    *,
    info: VideoInfo,
) -> list[int]:
    if not isinstance(result, submodule_models.TrackResult):
        raise TypeError("GVHMR tracker must return TrackResult.")
    if type(result.num_frames) is not int or result.num_frames != info.frame_count:
        raise GVHMRContractError(
            "Tracker frame count must match video metadata; "
            f"got {result.num_frames} and {info.frame_count}."
        )
    track_ids: list[int] = result.track_ids
    if not track_ids:
        raise GVHMRContractError("GVHMR tracker selected no person tracks.")
    if any(type(track_id) is not int or track_id < 0 for track_id in track_ids):
        raise GVHMRContractError(
            "GVHMR selected track IDs must be non-negative integers."
        )
    if len(set(track_ids)) != len(track_ids):
        raise GVHMRContractError("GVHMR selected track IDs must be unique.")
    for track_id in track_ids:
        track = result.tracks[track_id]
        _validate_float_tensor(
            track,
            name=f"selected track {track_id}",
            expected_shape=(info.frame_count, 4),
        )
        if track.device.type != "cpu":
            raise GVHMRContractError(
                f"selected track {track_id} must be on CPU, got {track.device}."
            )
        if not bool((track[:, 2:] > track[:, :2]).all()):
            raise GVHMRContractError(
                f"selected track {track_id} must contain positive-area xyxy boxes."
            )
    return track_ids


def _validate_smpl_parameters(
    parameters: dict[str, torch.Tensor],
    *,
    name: str,
    expected_frames: int,
    reference: torch.Tensor,
) -> dict[str, torch.Tensor]:
    if not isinstance(parameters, dict):
        raise TypeError(f"GVHMR {name} must be a tensor dictionary.")
    missing = set(_REQUIRED_SMPL_WIDTHS) - set(parameters)
    if missing:
        raise GVHMRContractError(
            f"GVHMR {name} is missing required SMPL keys: {sorted(missing)}."
        )
    for key, width in _REQUIRED_SMPL_WIDTHS.items():
        _validate_float_tensor(
            parameters[key],
            name=f"GVHMR {name}.{key}",
            expected_shape=(expected_frames, width),
            reference=reference,
        )
    return {key: parameters[key] for key in _REQUIRED_SMPL_WIDTHS}


def _validate_float_tensor(
    tensor: torch.Tensor,
    *,
    name: str,
    expected_shape: tuple[int, ...],
    reference: torch.Tensor | None = None,
) -> None:
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor.")
    if tuple(tensor.shape) != expected_shape:
        raise GVHMRContractError(
            f"{name} must have shape {expected_shape}, got {tuple(tensor.shape)}."
        )
    if tensor.dtype != torch.float32:
        raise TypeError(f"{name} must have dtype torch.float32, got {tensor.dtype}.")
    if reference is not None and tensor.device != reference.device:
        raise GVHMRContractError(
            f"{name} must be on {reference.device}, got {tensor.device}."
        )
    if not bool(torch.isfinite(tensor).all()):
        raise GVHMRContractError(f"{name} must contain only finite values.")


def _as_float32_numpy(tensor: torch.Tensor) -> NDArray[np.float32]:
    return np.asarray(tensor.detach().cpu().numpy(), dtype=np.float32)
