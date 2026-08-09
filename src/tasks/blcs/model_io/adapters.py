"""Validated task-local adapters for every active BLCS model family."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import Literal, cast

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor, nn

from src.tasks.base.model_io import (
    ModelCall,
    ModelInputContractError,
    ModelOutputContractError,
    TensorSpec,
    require_tensor,
)
from src.tasks.blcs.data.types import (
    BLCSBatch,
    BLCSMultiViewBatch,
    BLCSMultiViewSample,
)
from src.tasks.blcs.model_io.attention_masks import (
    prepare_axial_attention_masks,
    prepare_multiview_attention_masks,
    prepare_single_attention_mask,
    prepare_tracking_attention_masks,
)
from src.tasks.blcs.model_io.contracts import (
    BLCSTrackQueryPrediction,
    BLCSTrackQueryTrainingBatch,
    BLCSTrajectoryPrediction,
    BLCSTrajectoryTrainingBatch,
)
from src.tasks.blcs.models.blcs_model import BLCSModel
from src.tasks.blcs.models.blcs_multiview_axial_model import BLCSMultiViewAxialModel
from src.tasks.blcs.models.blcs_multiview_model import BLCSMultiViewModel
from src.tasks.blcs.models.blcs_track_query_model import BLCSTrackQueryModel

RawBLCSOutput = Mapping[str, Tensor]
FloatDtypes = frozenset({torch.float16, torch.float32, torch.float64, torch.bfloat16})
MaskDtypes = frozenset(
    {torch.bool, torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64, torch.float16, torch.float32, torch.float64, torch.bfloat16}
)
IndexDtypes = frozenset({torch.int8, torch.int16, torch.int32, torch.int64})


def _same_device(tensors: Mapping[str, Tensor]) -> None:
    devices = {tensor.device for tensor in tensors.values()}
    if len(devices) != 1:
        details = ", ".join(f"{name}={value.device}" for name, value in tensors.items())
        raise ModelInputContractError(
            f"All BLCS model inputs must share one device; got {details}."
        )


def _positive_axes(name: str, tensor: Tensor, axes: tuple[int, ...]) -> None:
    empty = [axis for axis in axes if tensor.shape[axis] <= 0]
    if empty:
        raise ModelInputContractError(f"{name} must be non-empty on axes {empty}.")


def _validate_uv(name: str, tensor: Tensor) -> None:
    if not bool(torch.isfinite(tensor).all()):
        raise ModelInputContractError(f"{name} must contain only finite UV values.")
    if not bool(((tensor >= 0.0) & (tensor <= 1.0)).all()):
        raise ModelInputContractError(
            f"{name} must contain normalized UV values within [0, 1]."
        )


def _validate_mask(name: str, tensor: Tensor) -> None:
    if not bool(((tensor == 0) | (tensor == 1)).all()):
        raise ModelInputContractError(f"{name} must contain only binary mask values.")


def _raw_output(output: object) -> Mapping[str, Tensor]:
    if not isinstance(output, Mapping):
        raise ModelOutputContractError(
            f"BLCS model output must be a mapping, got {type(output).__name__}."
        )
    if any(not isinstance(value, Tensor) for value in output.values()):
        raise ModelOutputContractError("Every BLCS model output value must be a tensor.")
    return cast("Mapping[str, Tensor]", output)


class TrajectoryModelIOAdapter(ABC):
    """Common validation/decode contract for one-ball trajectory models."""

    def __init__(
        self,
        *,
        num_court_tokens: int,
        max_seq_len: int,
        predict_velocity: bool,
        input_profile: Literal["single", "multiview"],
        max_num_cameras: int | None,
    ) -> None:
        self.num_court_tokens = num_court_tokens
        self.max_seq_len = max_seq_len
        self.predict_velocity = predict_velocity
        self.input_profile = input_profile
        self.max_num_cameras = max_num_cameras

    @property
    @abstractmethod
    def model_type(self) -> type[nn.Module]:
        """Return the sole model class supported by this adapter."""

    @abstractmethod
    def build_call(self, batch: Mapping[str, object]) -> ModelCall:
        """Validate and normalize one batch into a model call."""

    def decode_output(self, output: object) -> BLCSTrajectoryPrediction:
        """Validate exact raw keys and decode position/velocity semantics."""
        result = _raw_output(output)
        expected = {"position", "velocity"} if self.predict_velocity else {"position"}
        if set(result) != expected:
            raise ModelOutputContractError(
                f"BLCS trajectory output keys must be {sorted(expected)}, got {sorted(result)}."
            )
        position = result["position"]
        if position.ndim != 3 or position.shape[-1] != 3:
            raise ModelOutputContractError(
                f"position must have shape (B,T,3), got {tuple(position.shape)}."
            )
        if position.dtype not in FloatDtypes or not bool(torch.isfinite(position).all()):
            raise ModelOutputContractError(
                "position must use a floating dtype and contain only finite values."
            )
        velocity = result.get("velocity")
        if velocity is not None and velocity.shape != position.shape:
            raise ModelOutputContractError(
                "velocity must have exactly the same (B,T,3) shape as position."
            )
        if velocity is not None and (
            velocity.dtype != position.dtype
            or velocity.device != position.device
            or not bool(torch.isfinite(velocity).all())
        ):
            raise ModelOutputContractError(
                "velocity must match position dtype/device and contain finite values."
            )
        return BLCSTrajectoryPrediction(position=position, velocity=velocity)

    def build_training_batch(
        self, batch: Mapping[str, object]
    ) -> BLCSTrajectoryTrainingBatch:
        """Validate the complete supervised/reprojection training boundary."""
        call = self.build_call(batch)
        position = require_tensor(
            batch,
            "position_3d",
            spec=TensorSpec(shape=(None, None, 3), dtypes=FloatDtypes),
        )
        velocity = require_tensor(
            batch,
            "velocity_3d",
            spec=TensorSpec(shape=(None, None, 3), dtypes=FloatDtypes),
        )
        loss_mask = self._loss_mask(batch)
        target_uv = require_tensor(
            batch,
            "ball_uv_target" if "ball_uv_target" in batch else "ball_uv",
            spec=TensorSpec(dtypes=FloatDtypes),
        )
        target_vis = require_tensor(
            batch,
            "ball_vis_target" if "ball_vis_target" in batch else "ball_vis",
            spec=TensorSpec(dtypes=MaskDtypes),
        )
        if self.input_profile == "single":
            target_uv = target_uv.unsqueeze(1)
            target_vis = target_vis.unsqueeze(1)
        batch_size, frames = position.shape[:2]
        call_ball_uv = cast(Tensor, call.kwargs["ball_uv"])
        call_batch = call_ball_uv.shape[0]
        call_frames = call_ball_uv.shape[1 if self.input_profile == "single" else 2]
        if (batch_size, frames) != (call_batch, call_frames):
            raise ModelInputContractError(
                "position_3d batch/time axes must match the validated model call."
            )
        if velocity.shape != position.shape or loss_mask.shape != (batch_size, frames):
            raise ModelInputContractError(
                "position_3d, velocity_3d, and normalized loss mask must share (B,T)."
            )
        if (target_uv.shape[0], target_uv.shape[2]) != (batch_size, frames):
            raise ModelInputContractError("target UV batch/time axes must match position_3d.")
        if target_vis.shape != target_uv.shape[:-1]:
            raise ModelInputContractError("target visibility must match target UV without XY.")
        cameras = {
            "camera_R": require_tensor(batch, "camera_R", spec=TensorSpec(shape=(batch_size, None, 3, 3), dtypes=FloatDtypes)),
            "camera_C": require_tensor(batch, "camera_C", spec=TensorSpec(shape=(batch_size, None, 3), dtypes=FloatDtypes)),
            "camera_f": require_tensor(batch, "camera_f", spec=TensorSpec(shape=(batch_size, None), dtypes=FloatDtypes)),
            "camera_cx": require_tensor(batch, "camera_cx", spec=TensorSpec(shape=(batch_size, None), dtypes=FloatDtypes)),
            "camera_cy": require_tensor(batch, "camera_cy", spec=TensorSpec(shape=(batch_size, None), dtypes=FloatDtypes)),
            "camera_w": require_tensor(batch, "camera_w", spec=TensorSpec(shape=(batch_size, None), dtypes=FloatDtypes)),
            "camera_h": require_tensor(batch, "camera_h", spec=TensorSpec(shape=(batch_size, None), dtypes=FloatDtypes)),
        }
        num_cameras = target_uv.shape[1]
        if any(value.shape[1] != num_cameras for value in cameras.values()):
            raise ModelInputContractError(
                "All camera parameter tensors must match the target UV camera axis."
            )
        _validate_uv("target_uv", target_uv)
        _validate_mask("target_vis", target_vis)
        _validate_mask("loss_mask", loss_mask)
        _same_device(
            {
                "model_ball_uv": call_ball_uv,
                "position_3d": position,
                "velocity_3d": velocity,
                "loss_mask": loss_mask,
                "target_uv": target_uv,
                "target_vis": target_vis,
                **cameras,
            }
        )
        return BLCSTrajectoryTrainingBatch(
            call=call,
            position=position,
            velocity=velocity,
            loss_mask=loss_mask,
            target_uv=target_uv,
            target_vis=target_vis,
            camera_R=cameras["camera_R"],
            camera_C=cameras["camera_C"],
            camera_f=cameras["camera_f"],
            camera_cx=cameras["camera_cx"],
            camera_cy=cameras["camera_cy"],
            camera_w=cameras["camera_w"],
            camera_h=cameras["camera_h"],
        )

    @abstractmethod
    def _loss_mask(self, batch: Mapping[str, object]) -> Tensor:
        """Return a validated (B,T) loss mask."""

    @abstractmethod
    def collate_samples(
        self, samples: list[BLCSMultiViewSample]
    ) -> BLCSBatch | BLCSMultiViewBatch:
        """Collate canonical samples into this adapter's fixed input profile."""

    def build_inference_batch_from_arrays(
        self,
        *,
        ball_uv: NDArray[np.float32],
        court_kp: NDArray[np.float32],
        ball_vis: NDArray[np.bool_],
        court_vis: NDArray[np.bool_] | NDArray[np.float32],
    ) -> dict[str, Tensor]:
        """Build a batch from explicit scene arrays under this profile contract."""
        ball = torch.from_numpy(np.asarray(ball_uv, dtype=np.float32)).unsqueeze(0)
        court = torch.from_numpy(np.asarray(court_kp, dtype=np.float32)).unsqueeze(0)
        visible = torch.from_numpy(np.asarray(ball_vis, dtype=np.bool_)).unsqueeze(0)
        court_visible = torch.from_numpy(np.asarray(court_vis, dtype=np.bool_)).unsqueeze(0)
        batch: dict[str, Tensor] = {
            "ball_uv": ball,
            "court_kp": court,
            "ball_vis": visible,
            "court_vis": court_visible,
            "ball_mask": torch.ones_like(visible, dtype=torch.bool),
        }
        self.build_call(batch)
        return batch

    def build_inference_batch_from_scene(
        self,
        scene: Mapping[str, object],
        cameras: list[int],
    ) -> dict[str, Tensor]:
        """Build the configured profile from one canonical BLCS scene mapping."""
        if not cameras:
            raise ModelInputContractError("At least one camera must be selected.")
        raw_cameras = scene.get("cameras")
        if not isinstance(raw_cameras, list):
            raise ModelInputContractError("scene.cameras must be a list.")
        selected: list[Mapping[str, object]] = []
        for camera_index in cameras:
            if not 0 <= camera_index < len(raw_cameras):
                raise ModelInputContractError(
                    f"Camera index {camera_index} is outside scene.cameras."
                )
            camera = raw_cameras[camera_index]
            if not isinstance(camera, Mapping):
                raise ModelInputContractError("Each scene camera must be a mapping.")
            selected.append(camera)
        if self.input_profile == "single":
            if len(selected) != 1:
                raise ModelInputContractError(
                    "The single-view adapter requires exactly one selected camera."
                )
            camera = selected[0]
            batch = self.build_inference_batch_from_arrays(
                ball_uv=np.asarray(camera["ball_uv"], dtype=np.float32),
                court_kp=np.asarray(camera["court_kp_uv"], dtype=np.float32),
                ball_vis=np.asarray(camera["ball_visible"], dtype=np.bool_),
                court_vis=np.asarray(camera["court_kp_visible"], dtype=np.bool_),
            )
            return batch
        return self.build_inference_batch_from_arrays(
            ball_uv=np.stack(
                [np.asarray(camera["ball_uv"], dtype=np.float32) for camera in selected]
            ),
            court_kp=np.stack(
                [
                    np.asarray(camera["court_kp_uv"], dtype=np.float32)
                    for camera in selected
                ]
            ),
            ball_vis=np.stack(
                [
                    np.asarray(camera["ball_visible"], dtype=np.bool_)
                    for camera in selected
                ]
            ),
            court_vis=np.stack(
                [
                    np.asarray(camera["court_kp_visible"], dtype=np.bool_)
                    for camera in selected
                ]
            ),
        )

    @staticmethod
    def trajectory_arrays(
        batch: Mapping[str, object],
        prediction: BLCSTrajectoryPrediction,
        *,
        sample_index: int,
    ) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
        """Decode one validated qualitative pair without output-key inspection."""
        target = require_tensor(batch, "position_3d")
        if not 0 <= sample_index < target.shape[0]:
            raise ModelInputContractError(
                f"sample_index {sample_index} is outside batch size {target.shape[0]}."
            )
        if prediction.position.shape != target.shape:
            raise ModelOutputContractError(
                "Predicted and target trajectory shapes must match for rendering."
            )
        mask = require_tensor(batch, "ball_mask")
        normalized = mask if mask.ndim == 2 else mask.any(dim=1)
        valid = normalized[sample_index].bool()
        gt = target[sample_index][valid].detach().cpu().numpy().astype(np.float32)
        pred = prediction.position[sample_index][valid].detach().cpu().numpy().astype(np.float32)
        return gt, pred


class SingleTrajectoryModelIOAdapter(TrajectoryModelIOAdapter):
    """I/O adapter for :class:`BLCSModel`."""

    @property
    def model_type(self) -> type[nn.Module]:
        return BLCSModel

    def build_call(self, batch: Mapping[str, object]) -> ModelCall:
        ball_uv = require_tensor(batch, "ball_uv", spec=TensorSpec(shape=(None, None, 2), dtypes=FloatDtypes))
        court_kp = require_tensor(batch, "court_kp", spec=TensorSpec(shape=(None, self.num_court_tokens, 2), dtypes=FloatDtypes))
        ball_vis = require_tensor(batch, "ball_vis", spec=TensorSpec(shape=ball_uv.shape[:-1], dtypes=MaskDtypes))
        ball_mask = require_tensor(batch, "ball_mask", spec=TensorSpec(shape=ball_uv.shape[:-1], dtypes=MaskDtypes))
        court_vis = require_tensor(batch, "court_vis", spec=TensorSpec(shape=court_kp.shape[:-1], dtypes=MaskDtypes))
        _positive_axes("ball_uv", ball_uv, (0, 1))
        if ball_uv.shape[1] > self.max_seq_len:
            raise ModelInputContractError(
                f"ball_uv time axis {ball_uv.shape[1]} exceeds max_seq_len={self.max_seq_len}."
            )
        if court_kp.shape[0] != ball_uv.shape[0]:
            raise ModelInputContractError("court_kp batch axis must match ball_uv.")
        _validate_uv("ball_uv", ball_uv)
        _validate_uv("court_kp", court_kp)
        _validate_mask("ball_vis", ball_vis)
        _validate_mask("ball_mask", ball_mask)
        _validate_mask("court_vis", court_vis)
        ball_vis = ball_vis.bool()
        ball_mask = ball_mask.bool()
        court_vis = court_vis.bool()
        _same_device({"ball_uv": ball_uv, "court_kp": court_kp, "ball_vis": ball_vis, "ball_mask": ball_mask, "court_vis": court_vis})
        attention_mask = prepare_single_attention_mask(
            ball_mask,
            num_court_tokens=self.num_court_tokens,
        )
        return ModelCall(kwargs={"ball_uv": ball_uv, "court_kp": court_kp, "ball_vis": ball_vis, "court_vis": court_vis, "attention_mask": attention_mask})

    def _loss_mask(self, batch: Mapping[str, object]) -> Tensor:
        return require_tensor(batch, "ball_mask", spec=TensorSpec(shape=(None, None), dtypes=MaskDtypes)).bool()

    def collate_samples(self, samples: list[BLCSMultiViewSample]) -> BLCSBatch:
        """Collate canonical samples into the fixed single-view profile."""
        from src.tasks.blcs.data.dataset import collate_multiview_trajectories

        batch = collate_multiview_trajectories(samples)
        adapted: dict[str, Tensor] = {
            "ball_uv": batch["ball_uv"][:, 0],
            "ball_vis": batch["ball_vis"][:, 0],
            "ball_mask": batch["ball_mask"][:, 0],
            "court_kp": batch["court_kp"][:, 0, 0],
            "court_vis": batch["court_vis"][:, 0, 0],
            "position_3d": batch["position_3d"],
            "velocity_3d": batch["velocity_3d"],
            "seq_len": batch["seq_len"],
            "camera_R": batch["camera_R"][:, :1],
            "camera_C": batch["camera_C"][:, :1],
            "camera_f": batch["camera_f"][:, :1],
            "camera_cx": batch["camera_cx"][:, :1],
            "camera_cy": batch["camera_cy"][:, :1],
            "camera_w": batch["camera_w"][:, :1],
            "camera_h": batch["camera_h"][:, :1],
        }
        if "ball_uv_target" in batch and "ball_vis_target" in batch:
            adapted["ball_uv_target"] = batch["ball_uv_target"][:, 0]
            adapted["ball_vis_target"] = batch["ball_vis_target"][:, 0]
        result = cast("BLCSBatch", adapted)
        self.build_training_batch(result)
        return result


class _MultiviewTrajectoryModelIOAdapter(TrajectoryModelIOAdapter):
    @abstractmethod
    def _prepare_attention_kwargs(self, ball_mask: Tensor) -> dict[str, Tensor]:
        """Prepare the exact attention tensors required by the model family."""

    def build_call(self, batch: Mapping[str, object]) -> ModelCall:
        ball_uv = require_tensor(batch, "ball_uv", spec=TensorSpec(shape=(None, None, None, 2), dtypes=FloatDtypes))
        ball_vis = require_tensor(batch, "ball_vis", spec=TensorSpec(shape=ball_uv.shape[:-1], dtypes=MaskDtypes))
        ball_mask = require_tensor(batch, "ball_mask", spec=TensorSpec(shape=ball_uv.shape[:-1], dtypes=MaskDtypes))
        court_kp = require_tensor(batch, "court_kp", spec=TensorSpec(dtypes=FloatDtypes))
        court_vis = require_tensor(batch, "court_vis", spec=TensorSpec(dtypes=MaskDtypes))
        batch_size, cameras, frames = ball_uv.shape[:3]
        _positive_axes("ball_uv", ball_uv, (0, 1, 2))
        if frames > self.max_seq_len:
            raise ModelInputContractError(
                f"ball_uv time axis {frames} exceeds max_seq_len={self.max_seq_len}."
            )
        if self.max_num_cameras is not None and cameras > self.max_num_cameras:
            raise ModelInputContractError(
                f"ball_uv camera axis {cameras} exceeds max_num_cameras={self.max_num_cameras}."
            )
        if court_kp.ndim == 4:
            if court_kp.shape != (batch_size, cameras, self.num_court_tokens, 2):
                raise ModelInputContractError("static court_kp must have shape (B,V,K,2).")
            court_kp = court_kp.unsqueeze(2).expand(-1, -1, frames, -1, -1)
        if court_kp.shape != (batch_size, cameras, frames, self.num_court_tokens, 2):
            raise ModelInputContractError(
                f"court_kp must have shape (B,V,T,{self.num_court_tokens},2)."
            )
        if court_vis.ndim == 3:
            if court_vis.shape != (batch_size, cameras, self.num_court_tokens):
                raise ModelInputContractError("static court_vis must have shape (B,V,K).")
            court_vis = court_vis.unsqueeze(2).expand(-1, -1, frames, -1)
        if court_vis.shape != court_kp.shape[:-1]:
            raise ModelInputContractError("court_vis must match court_kp without XY.")
        _validate_uv("ball_uv", ball_uv)
        _validate_uv("court_kp", court_kp)
        _validate_mask("ball_vis", ball_vis)
        _validate_mask("ball_mask", ball_mask)
        _validate_mask("court_vis", court_vis)
        ball_vis = ball_vis.bool()
        ball_mask = ball_mask.bool()
        court_vis = court_vis.bool()
        _same_device({"ball_uv": ball_uv, "court_kp": court_kp, "ball_vis": ball_vis, "ball_mask": ball_mask, "court_vis": court_vis})
        kwargs = {"ball_uv": ball_uv, "court_kp": court_kp, "ball_vis": ball_vis, "court_vis": court_vis}
        kwargs.update(self._prepare_attention_kwargs(ball_mask))
        return ModelCall(kwargs=kwargs)

    def _loss_mask(self, batch: Mapping[str, object]) -> Tensor:
        mask = require_tensor(batch, "ball_mask", spec=TensorSpec(shape=(None, None, None), dtypes=MaskDtypes))
        return mask.bool().any(dim=1)

    def collate_samples(
        self, samples: list[BLCSMultiViewSample]
    ) -> BLCSMultiViewBatch:
        """Collate and validate the fixed multiview profile."""
        from src.tasks.blcs.data.dataset import collate_multiview_trajectories

        result = collate_multiview_trajectories(samples)
        self.build_training_batch(result)
        return result


class MultiViewTrajectoryModelIOAdapter(_MultiviewTrajectoryModelIOAdapter):
    """I/O adapter for the iterative multiview model."""

    @property
    def model_type(self) -> type[nn.Module]:
        return BLCSMultiViewModel

    def _prepare_attention_kwargs(self, ball_mask: Tensor) -> dict[str, Tensor]:
        query_mask, query_state_valid, cross_mask, frame_token_valid = (
            prepare_multiview_attention_masks(
                ball_mask,
                num_court_tokens=self.num_court_tokens,
            )
        )
        return {
            "query_attention_mask": query_mask,
            "query_state_valid": query_state_valid,
            "cross_attention_mask": cross_mask,
            "frame_token_valid": frame_token_valid,
        }


class AxialTrajectoryModelIOAdapter(_MultiviewTrajectoryModelIOAdapter):
    """I/O adapter for the axial multiview model."""

    def __init__(
        self,
        *,
        num_court_tokens: int,
        max_seq_len: int,
        predict_velocity: bool,
        input_profile: Literal["single", "multiview"],
        max_num_cameras: int | None,
        time_window_radius: int,
    ) -> None:
        super().__init__(
            num_court_tokens=num_court_tokens,
            max_seq_len=max_seq_len,
            predict_velocity=predict_velocity,
            input_profile=input_profile,
            max_num_cameras=max_num_cameras,
        )
        self.time_window_radius = time_window_radius

    @property
    def model_type(self) -> type[nn.Module]:
        return BLCSMultiViewAxialModel

    def _prepare_attention_kwargs(self, ball_mask: Tensor) -> dict[str, Tensor]:
        camera_mask, time_mask, sliding_mask = prepare_axial_attention_masks(
            ball_mask,
            time_window_radius=self.time_window_radius,
        )
        return {
            "camera_attention_mask": camera_mask,
            "time_attention_mask": time_mask,
            "sliding_attention_mask": sliding_mask,
        }


class TrackQueryModelIOAdapter:
    """Boundary adapter for lifecycle-query BLCS batches and outputs."""

    input_profile: Literal["tracking"] = "tracking"

    def __init__(
        self,
        *,
        num_court_tokens: int,
        num_queries: int,
        presence_threshold: float,
        mask_invisible_observations: bool,
    ) -> None:
        self.num_court_tokens = num_court_tokens
        self.num_queries = num_queries
        self.presence_threshold = presence_threshold
        self.mask_invisible_observations = mask_invisible_observations

    @property
    def model_type(self) -> type[nn.Module]:
        return BLCSTrackQueryModel

    def build_call(self, batch: Mapping[str, object]) -> ModelCall:
        ball_uv = require_tensor(batch, "ball_uv", spec=TensorSpec(shape=(None, None, None, None, 2), dtypes=FloatDtypes))
        ball_visible = require_tensor(batch, "ball_visible", spec=TensorSpec(shape=ball_uv.shape[:-1], dtypes=frozenset({torch.bool})))
        batch_size, views, frames, detections = ball_uv.shape[:4]
        court_kp = require_tensor(batch, "court_kp", spec=TensorSpec(shape=(batch_size, views, frames, self.num_court_tokens, 2), dtypes=FloatDtypes))
        court_vis = require_tensor(batch, "court_vis", spec=TensorSpec(shape=court_kp.shape[:-1], dtypes=frozenset({torch.bool})))
        frame_mask = require_tensor(batch, "frame_mask", spec=TensorSpec(shape=(batch_size, frames), dtypes=frozenset({torch.bool})))
        view_mask = require_tensor(batch, "view_mask", spec=TensorSpec(shape=(batch_size, views), dtypes=frozenset({torch.bool})))
        _positive_axes("ball_uv", ball_uv, (0, 1, 2, 3))
        _validate_uv("ball_uv", ball_uv)
        _validate_uv("court_kp", court_kp)
        if bool(
            (
                ball_visible
                & ~(view_mask[:, :, None, None] & frame_mask[:, None, :, None])
            ).any()
        ):
            raise ModelInputContractError(
                "ball_visible cannot be true in a padded view or frame."
            )
        _same_device({"ball_uv": ball_uv, "ball_visible": ball_visible, "court_kp": court_kp, "court_vis": court_vis, "frame_mask": frame_mask, "view_mask": view_mask})
        observation_state_valid, spatial_mask, temporal_mask, point_mask = (
            prepare_tracking_attention_masks(
                ball_visible=ball_visible,
                court_visible=court_vis,
                frame_mask=frame_mask,
                view_mask=view_mask,
                num_queries=self.num_queries,
                mask_invisible_observations=self.mask_invisible_observations,
            )
        )
        return ModelCall(kwargs={"ball_uv": ball_uv, "ball_visible": ball_visible, "court_kp": court_kp, "court_vis": court_vis, "frame_mask": frame_mask, "observation_state_valid": observation_state_valid, "spatial_attention_mask": spatial_mask, "temporal_attention_mask": temporal_mask, "point_attention_mask": point_mask})

    def decode_output(self, output: object) -> BLCSTrackQueryPrediction:
        result = _raw_output(output)
        if set(result) != {"position", "presence_logits"}:
            raise ModelOutputContractError(
                "BLCS track-query output requires exactly position and presence_logits."
            )
        position = result["position"]
        logits = result["presence_logits"]
        if position.ndim != 4 or position.shape[-2:] != (self.num_queries, 3):
            raise ModelOutputContractError(
                f"position must have shape (B,T,{self.num_queries},3), got {tuple(position.shape)}."
            )
        if logits.shape != position.shape[:-1]:
            raise ModelOutputContractError("presence_logits must match position without XYZ.")
        if (
            position.dtype not in FloatDtypes
            or logits.dtype != position.dtype
            or logits.device != position.device
            or not bool(torch.isfinite(position).all())
            or not bool(torch.isfinite(logits).all())
        ):
            raise ModelOutputContractError(
                "Tracking outputs must share one floating dtype/device and be finite."
            )
        probability = logits.sigmoid()
        return BLCSTrackQueryPrediction(
            position=position,
            presence_logits=logits,
            presence_probability=probability,
            presence=probability >= self.presence_threshold,
        )

    def build_training_batch(
        self, batch: Mapping[str, object]
    ) -> BLCSTrackQueryTrainingBatch:
        call = self.build_call(batch)
        frame_mask = cast(Tensor, call.kwargs["frame_mask"])
        batch_size, frames = frame_mask.shape
        position = require_tensor(batch, "target_position", spec=TensorSpec(shape=(batch_size, frames, self.num_queries, 3), dtypes=FloatDtypes))
        velocity = require_tensor(batch, "target_velocity", spec=TensorSpec(shape=position.shape, dtypes=FloatDtypes))
        presence = require_tensor(batch, "target_presence", spec=TensorSpec(shape=position.shape[:-1], dtypes=frozenset({torch.bool})))
        instance_id = require_tensor(batch, "target_instance_id", spec=TensorSpec(shape=presence.shape, dtypes=IndexDtypes))
        slot_mask = require_tensor(batch, "target_slot_mask", spec=TensorSpec(shape=(batch_size, self.num_queries), dtypes=frozenset({torch.bool})))
        if bool((instance_id[presence] < 0).any()) or bool((instance_id[~presence] != -1).any()):
            raise ModelInputContractError(
                "target_instance_id must be non-negative exactly where target_presence is true and -1 otherwise."
            )
        call_ball_uv = cast(Tensor, call.kwargs["ball_uv"])
        _same_device({"model_ball_uv": call_ball_uv, "target_position": position, "target_velocity": velocity, "target_presence": presence, "target_instance_id": instance_id, "target_slot_mask": slot_mask, "frame_mask": frame_mask})
        return BLCSTrackQueryTrainingBatch(
            call=call,
            target_position=position,
            target_velocity=velocity,
            target_presence=presence,
            target_instance_id=instance_id,
            target_slot_mask=slot_mask,
            frame_mask=frame_mask,
        )


__all__ = [
    "AxialTrajectoryModelIOAdapter",
    "MultiViewTrajectoryModelIOAdapter",
    "RawBLCSOutput",
    "SingleTrajectoryModelIOAdapter",
    "TrackQueryModelIOAdapter",
    "TrajectoryModelIOAdapter",
]
