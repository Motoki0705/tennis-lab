"""Boundary adapters for every active PLCS model input/output profile."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import replace
from typing import cast

import numpy as np
import torch
from torch import Tensor, nn

from src.tasks.base.data.court_peaks import (
    CourtObservationProfile,
    court_peak_batch_from_model_input,
    parse_court_observation_profile,
    reference_view_mask,
)
from src.tasks.base.data.reference_orientation import (
    reflect_court_vectors,
    reflect_heading,
    validate_declared_reference_orientation,
)
from src.tasks.base.model_io import (
    ModelCall,
    ModelInputContractError,
    ModelOutputContractError,
    TensorSpec,
    require_tensor,
)
from src.tasks.plcs.model_io.attention_masks import (
    prepare_axial_attention_masks,
    prepare_tracking_attention_masks,
)
from src.tasks.plcs.model_io.contracts import (
    PLCSDecodedPrediction,
    PLCSInputProfile,
    PLCSPreparedBatch,
    PLCSTrackingDecodedPrediction,
)
from src.tasks.plcs.models.plcs_multiview_axial_model import PLCSMultiViewAxialModel
from src.utils.schema.player import NUM_HUMAN_KP

_FLOAT_DTYPES = frozenset(
    {torch.float16, torch.bfloat16, torch.float32, torch.float64}
)
_MASK_DTYPES = frozenset(
    {
        torch.bool,
        torch.uint8,
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.float16,
        torch.bfloat16,
        torch.float32,
        torch.float64,
    }
)


def _finite(name: str, tensor: Tensor) -> None:
    if tensor.is_floating_point() and not bool(torch.isfinite(tensor).all().item()):
        raise ModelInputContractError(f"{name} must contain only finite values.")


def _normalized_uv(name: str, tensor: Tensor) -> None:
    _finite(name, tensor)
    if tensor.numel() and (
        bool((tensor < 0).any().item()) or bool((tensor > 1).any().item())
    ):
        raise ModelInputContractError(
            f"{name} must use normalized UV coordinates within [0, 1]."
        )


def _binary_mask(name: str, tensor: Tensor) -> None:
    _finite(name, tensor)
    if tensor.dtype != torch.bool and tensor.numel():
        binary = (tensor == 0) | (tensor == 1)
        if not bool(binary.all().item()):
            raise ModelInputContractError(
                f"{name} must be boolean or contain only explicit 0/1 values."
            )


def _prepare_player_geometry(
    human_kp: Tensor,
    joint_visibility: Tensor,
    detection_score: Tensor,
) -> tuple[Tensor, Tensor]:
    """Build a visibility-aware player anchor and centered pose descriptor."""
    weights = joint_visibility.to(dtype=human_kp.dtype)
    visible_count = weights.sum(dim=-1, keepdim=True)
    visible_center = (human_kp * weights.unsqueeze(-1)).sum(dim=-2) / (
        visible_count.clamp_min(1.0)
    )
    visible_center = torch.where(
        visible_count > 0,
        visible_center,
        torch.zeros_like(visible_center),
    )

    hip_weights = weights[..., 11:13]
    hip_count = hip_weights.sum(dim=-1, keepdim=True)
    hip_center = (human_kp[..., 11:13, :] * hip_weights.unsqueeze(-1)).sum(
        dim=-2
    ) / hip_count.clamp_min(1.0)
    anchor = torch.where(hip_count > 0, hip_center, visible_center)
    centered = (human_kp - anchor.unsqueeze(-2)) * weights.unsqueeze(-1)
    features = torch.cat(
        [centered.flatten(start_dim=-2), weights, detection_score.unsqueeze(-1)],
        dim=-1,
    )
    return anchor, features


def _required_output(
    output: Mapping[str, object],
    name: str,
    *,
    shape: tuple[int | None, ...],
) -> Tensor:
    if name not in output:
        raise ModelOutputContractError(f"Required PLCS model output {name!r} is missing.")
    value = output[name]
    if not isinstance(value, Tensor):
        raise ModelOutputContractError(
            f"PLCS model output {name!r} must be a Tensor, got "
            f"{type(value).__name__}."
        )
    try:
        TensorSpec(shape=shape, dtypes=_FLOAT_DTYPES).validate(name, value)
    except ModelInputContractError as error:
        raise ModelOutputContractError(str(error)) from error
    if not bool(torch.isfinite(value).all().item()):
        raise ModelOutputContractError(
            f"PLCS model output {name!r} must contain only finite values."
        )
    return value


class PLCSModelIOAdapter:
    """Typed standard-model adapter selected once by the PLCS factory."""

    def __init__(
        self,
        *,
        model_type: type[nn.Module],
        profile: PLCSInputProfile,
        num_court_tokens: int,
        camera_index: int,
        output_rank: int,
        predict_canonical_pose: bool,
        predict_auxiliary_position: bool,
        max_views: int | None = None,
        max_sequence_length: int | None = None,
        min_views: int = 1,
    ) -> None:
        if profile is PLCSInputProfile.TRACK_QUERY:
            raise ValueError("Use PLCSTrackQueryIOAdapter for track-query models.")
        if num_court_tokens <= 0 or camera_index < 0 or min_views <= 0:
            raise ValueError(
                "num_court_tokens/min_views must be positive and camera_index "
                "must be non-negative."
            )
        expected_output_rank = (
            3 if profile is PLCSInputProfile.MULTIVIEW else 2
        )
        if output_rank != expected_output_rank:
            raise ValueError(
                f"PLCS {profile.value!r} profile requires output_rank="
                f"{expected_output_rank}, got {output_rank}."
            )
        if max_views is not None and max_views <= 0:
            raise ValueError("max_views must be positive when configured.")
        if max_sequence_length is not None and max_sequence_length <= 0:
            raise ValueError(
                "max_sequence_length must be positive when configured."
            )
        if max_views is not None and min_views > max_views:
            raise ValueError("min_views cannot exceed max_views.")
        self._model_type = model_type
        self.profile = profile
        self.num_court_tokens = num_court_tokens
        self.camera_index = camera_index
        self.output_rank = output_rank
        self.predict_canonical_pose = predict_canonical_pose
        self.predict_auxiliary_position = predict_auxiliary_position
        self.max_views = max_views
        self.max_sequence_length = max_sequence_length
        self.min_views = min_views

    @property
    def model_type(self) -> type[nn.Module]:
        """Return the exact model class accepted by this adapter."""
        return self._model_type

    def require_profile(self, profile: PLCSInputProfile | str) -> None:
        """Reject a consumer/profile mismatch before tensor assembly."""
        requested = PLCSInputProfile(profile)
        if requested is not self.profile:
            raise ModelInputContractError(
                f"PLCS adapter profile is {self.profile.value!r}, not "
                f"{requested.value!r}."
            )

    def _validate_ready_inputs(
        self, batch: Mapping[str, object]
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        human_shape: tuple[int | None, ...]
        court_shape: tuple[int | None, ...]
        human_vis_shape: tuple[int | None, ...]
        human_mask_shape: tuple[int | None, ...]
        court_vis_shape: tuple[int | None, ...]
        if self.profile in {PLCSInputProfile.FRAME, PLCSInputProfile.SEQUENCE}:
            human_shape = (None, NUM_HUMAN_KP, 2)
            court_shape = (None, self.num_court_tokens, 2)
            human_vis_shape = (None, NUM_HUMAN_KP)
            human_mask_shape = (None,)
            court_vis_shape = (None, self.num_court_tokens)
        else:
            human_shape = (None, None, None, NUM_HUMAN_KP, 2)
            court_shape = (None, None, None, self.num_court_tokens, 2)
            human_vis_shape = (None, None, None, NUM_HUMAN_KP)
            human_mask_shape = (None, None, None)
            court_vis_shape = (None, None, None, self.num_court_tokens)

        human_kp = require_tensor(
            batch,
            "human_kp",
            spec=TensorSpec(shape=human_shape, dtypes=_FLOAT_DTYPES),
        )
        court_kp = require_tensor(
            batch,
            "court_kp",
            spec=TensorSpec(shape=court_shape, dtypes=_FLOAT_DTYPES),
        )
        human_vis = require_tensor(
            batch,
            "human_vis",
            spec=TensorSpec(shape=human_vis_shape, dtypes=_MASK_DTYPES),
        )
        human_mask = require_tensor(
            batch,
            "human_mask",
            spec=TensorSpec(shape=human_mask_shape, dtypes=_MASK_DTYPES),
        )
        court_vis = require_tensor(
            batch,
            "court_vis",
            spec=TensorSpec(shape=court_vis_shape, dtypes=_MASK_DTYPES),
        )

        if human_kp.shape[:-2] != court_kp.shape[:-2]:
            raise ModelInputContractError(
                "human_kp and court_kp must share every leading batch/view/time axis."
            )
        if human_vis.shape != human_kp.shape[:-1]:
            raise ModelInputContractError(
                "human_vis must match human_kp without its UV axis."
            )
        if court_vis.shape != court_kp.shape[:-1]:
            raise ModelInputContractError(
                "court_vis must match court_kp without its UV axis."
            )
        if any(dimension == 0 for dimension in human_kp.shape[:-2]):
            raise ModelInputContractError(
                "PLCS batch/view/time axes must all be non-empty."
            )
        if self.profile is PLCSInputProfile.MULTIVIEW:
            views = human_kp.shape[1]
            frames = human_kp.shape[2]
            if views < self.min_views:
                raise ModelInputContractError(
                    f"{self.model_type.__name__} requires at least "
                    f"{self.min_views} views, got {views}."
                )
            if self.max_views is not None and views > self.max_views:
                raise ModelInputContractError(
                    f"PLCS input has {views} views, exceeding max_views="
                    f"{self.max_views}."
                )
            if self.max_sequence_length is not None and frames > self.max_sequence_length:
                raise ModelInputContractError(
                    f"PLCS input has {frames} frames, exceeding max_seq_len="
                    f"{self.max_sequence_length}."
                )
        _normalized_uv("human_kp", human_kp)
        _normalized_uv("court_kp", court_kp)
        _binary_mask("human_vis", human_vis)
        _binary_mask("human_mask", human_mask)
        _binary_mask("court_vis", court_vis)
        return human_kp, court_kp, human_vis, human_mask, court_vis

    def build_call(self, batch: Mapping[str, object]) -> ModelCall:
        """Validate a model-ready batch and build one immutable model call."""
        human_kp, court_kp, human_vis, human_mask, court_vis = (
            self._validate_ready_inputs(batch)
        )
        kwargs = {
            "human_kp": human_kp,
            "court_kp": court_kp,
            "human_vis": human_vis,
            "human_mask": human_mask,
            "court_vis": court_vis,
        }
        if issubclass(self.model_type, PLCSMultiViewAxialModel):
            camera_mask, time_mask = prepare_axial_attention_masks(human_mask)
            kwargs.update(
                {
                    "camera_attention_mask": camera_mask,
                    "time_attention_mask": time_mask,
                }
            )
        return ModelCall(kwargs=kwargs)

    def prepare_training_batch(
        self, batch: Mapping[str, object]
    ) -> PLCSPreparedBatch:
        """Validate a canonical ``(B,V,T,...)`` batch and select its profile."""
        human_kp = require_tensor(
            batch,
            "human_kp",
            spec=TensorSpec(
                shape=(None, None, None, NUM_HUMAN_KP, 2), dtypes=_FLOAT_DTYPES
            ),
        )
        court_kp = require_tensor(
            batch,
            "court_kp",
            spec=TensorSpec(
                shape=(None, None, None, self.num_court_tokens, 2),
                dtypes=_FLOAT_DTYPES,
            ),
        )
        human_vis = require_tensor(
            batch,
            "human_vis",
            spec=TensorSpec(
                shape=(None, None, None, NUM_HUMAN_KP), dtypes=_MASK_DTYPES
            ),
        )
        human_mask = require_tensor(
            batch,
            "human_mask",
            spec=TensorSpec(shape=(None, None, None), dtypes=_MASK_DTYPES),
        )
        court_vis = require_tensor(
            batch,
            "court_vis",
            spec=TensorSpec(
                shape=(None, None, None, self.num_court_tokens),
                dtypes=_MASK_DTYPES,
            ),
        )
        canonical = {
            "human_kp": human_kp,
            "court_kp": court_kp,
            "human_vis": human_vis,
            "human_mask": human_mask,
            "court_vis": court_vis,
        }
        self._validate_canonical_axes(canonical)
        batch_size, views, frames = human_kp.shape[:3]
        if batch_size == 0 or views == 0 or frames == 0:
            raise ModelInputContractError(
                "Canonical PLCS (B,V,T) axes must all be non-empty."
            )
        target_position = require_tensor(
            batch,
            "position",
            spec=TensorSpec(shape=(batch_size, frames, 3), dtypes=_FLOAT_DTYPES),
        )
        target_rotation = require_tensor(
            batch,
            "rotation",
            spec=TensorSpec(shape=(batch_size, frames, 2), dtypes=_FLOAT_DTYPES),
        )
        _finite("position", target_position)
        _finite("rotation", target_rotation)
        target_human_kp_3d: Tensor | None = None
        if "human_kp_3d" in batch:
            target_human_kp_3d = require_tensor(
                batch,
                "human_kp_3d",
                spec=TensorSpec(
                    shape=(batch_size, frames, NUM_HUMAN_KP, 3),
                    dtypes=_FLOAT_DTYPES,
                ),
            )
            _finite("human_kp_3d", target_human_kp_3d)

        if self.profile is PLCSInputProfile.MULTIVIEW:
            return PLCSPreparedBatch(
                call=self.build_call(canonical),
                target_position=target_position,
                target_rotation=target_rotation,
                target_human_kp_3d=target_human_kp_3d,
                target_human_mask=human_mask,
            )
        if self.camera_index >= views:
            raise ModelInputContractError(
                f"adapter_camera_index={self.camera_index} is out of range for "
                f"a batch with {views} views."
            )
        if self.profile is PLCSInputProfile.FRAME and frames != 1:
            raise ModelInputContractError(
                f"Frame-profile PLCS requires exactly one frame, got {frames}."
            )

        ready = {
            "human_kp": human_kp[:, self.camera_index].reshape(
                batch_size * frames, NUM_HUMAN_KP, 2
            ),
            "court_kp": court_kp[:, self.camera_index].reshape(
                batch_size * frames, self.num_court_tokens, 2
            ),
            "human_vis": human_vis[:, self.camera_index].reshape(
                batch_size * frames, NUM_HUMAN_KP
            ),
            "human_mask": human_mask[:, self.camera_index].reshape(
                batch_size * frames
            ),
            "court_vis": court_vis[:, self.camera_index].reshape(
                batch_size * frames, self.num_court_tokens
            ),
        }
        sequence_shape = (
            (batch_size, frames)
            if self.profile is PLCSInputProfile.SEQUENCE
            else None
        )
        if self.profile is PLCSInputProfile.FRAME:
            target_position = target_position[:, 0]
            target_rotation = target_rotation[:, 0]
            if target_human_kp_3d is not None:
                target_human_kp_3d = target_human_kp_3d[:, 0]
            target_human_mask = human_mask[:, self.camera_index, 0]
        else:
            target_human_mask = human_mask[:, self.camera_index]
        return PLCSPreparedBatch(
            call=self.build_call(ready),
            sequence_shape=sequence_shape,
            target_position=target_position,
            target_rotation=target_rotation,
            target_human_kp_3d=target_human_kp_3d,
            target_human_mask=target_human_mask,
        )

    def _validate_canonical_axes(self, batch: Mapping[str, Tensor]) -> None:
        human_kp = batch["human_kp"]
        court_kp = batch["court_kp"]
        if human_kp.shape[:3] != court_kp.shape[:3]:
            raise ModelInputContractError(
                "Canonical human/court observations must share (B,V,T)."
            )
        if batch["human_vis"].shape != human_kp.shape[:-1]:
            raise ModelInputContractError(
                "human_vis must match human_kp without its UV axis."
            )
        if batch["court_vis"].shape != court_kp.shape[:-1]:
            raise ModelInputContractError(
                "court_vis must match court_kp without its UV axis."
            )
        if batch["human_mask"].shape != human_kp.shape[:3]:
            raise ModelInputContractError("human_mask must have shape (B,V,T).")
        _normalized_uv("human_kp", human_kp)
        _normalized_uv("court_kp", court_kp)
        _binary_mask("human_vis", batch["human_vis"])
        _binary_mask("human_mask", batch["human_mask"])
        _binary_mask("court_vis", batch["court_vis"])

    def decode_output(
        self, output: Mapping[str, object]
    ) -> PLCSDecodedPrediction:
        """Validate output keys, tensors, ranks, and cross-output shape semantics."""
        if not isinstance(output, Mapping):
            raise ModelOutputContractError(
                f"PLCS model output must be a mapping, got {type(output).__name__}."
            )
        expected = {"position", "rotation"}
        if self.predict_canonical_pose:
            expected.add("canonical_pose")
        if self.predict_auxiliary_position:
            expected.add("aux_position")
        unknown = set(output) - expected
        missing = expected - set(output)
        if missing or unknown:
            raise ModelOutputContractError(
                "PLCS model output keys do not match the paired adapter: "
                f"missing={sorted(missing)}, unknown={sorted(unknown)}."
            )
        prefix = (None,) * (self.output_rank - 1)
        position = _required_output(output, "position", shape=(*prefix, 3))
        rotation = _required_output(output, "rotation", shape=(*prefix, 2))
        if position.shape[:-1] != rotation.shape[:-1]:
            raise ModelOutputContractError(
                "PLCS position and rotation outputs must share leading axes."
            )
        canonical_pose = None
        if self.predict_canonical_pose:
            canonical_pose = _required_output(
                output,
                "canonical_pose",
                shape=(*prefix, NUM_HUMAN_KP, 3),
            )
            if canonical_pose.shape[: self.output_rank - 1] != position.shape[:-1]:
                raise ModelOutputContractError(
                    "PLCS canonical_pose must share position leading axes."
                )
        auxiliary_position = None
        if self.predict_auxiliary_position:
            auxiliary_position = _required_output(
                output, "aux_position", shape=(*prefix, 3)
            )
            if auxiliary_position.shape != position.shape:
                raise ModelOutputContractError(
                    "PLCS aux_position must match position shape."
                )
        return PLCSDecodedPrediction(
            position=position,
            rotation=rotation,
            canonical_pose=canonical_pose,
            auxiliary_position=auxiliary_position,
        )

    def decode_prepared_output(
        self,
        output: Mapping[str, object],
        prepared: PLCSPreparedBatch,
    ) -> PLCSDecodedPrediction:
        """Decode one output and restore a flattened sequence layout."""
        decoded = self.decode_output(output)
        if prepared.sequence_shape is None:
            return decoded
        batch_size, frames = prepared.sequence_shape

        def restore(value: Tensor | None) -> Tensor | None:
            if value is None:
                return None
            return value.reshape(batch_size, frames, *value.shape[1:])

        return replace(
            decoded,
            position=cast(Tensor, restore(decoded.position)),
            rotation=cast(Tensor, restore(decoded.rotation)),
            canonical_pose=restore(decoded.canonical_pose),
            auxiliary_position=restore(decoded.auxiliary_position),
        )

    def prepare_scene(
        self,
        scene: object,
        cameras: Sequence[int],
    ) -> PLCSPreparedBatch:
        """Build a validated frame/sequence/multiview call from a loaded scene."""
        if not cameras:
            raise ModelInputContractError("At least one PLCS camera is required.")
        scene_cameras = getattr(scene, "cameras", None)
        if not isinstance(scene_cameras, Sequence):
            raise ModelInputContractError("PLCS scene must expose a cameras sequence.")
        if any(index < 0 or index >= len(scene_cameras) for index in cameras):
            raise ModelInputContractError("PLCS camera selection is out of range.")

        if self.profile is PLCSInputProfile.MULTIVIEW:
            selected = list(cameras)
        else:
            selected = [cameras[0]]
        human = np.stack(
            [np.asarray(scene_cameras[index].human_kp_uv) for index in selected],
            axis=0,
        )
        court = np.stack(
            [np.asarray(scene_cameras[index].court_kp_uv) for index in selected],
            axis=0,
        )
        human_vis = np.stack(
            [
                np.asarray(scene_cameras[index].human_kp_visible, dtype=np.bool_)
                for index in selected
            ],
            axis=0,
        )
        court_vis = np.stack(
            [
                np.asarray(scene_cameras[index].court_kp_visible, dtype=np.bool_)
                for index in selected
            ],
            axis=0,
        )
        frames = human.shape[1]
        ready = {
            "human_kp": torch.as_tensor(human, dtype=torch.float32).unsqueeze(0),
            "court_kp": torch.as_tensor(court, dtype=torch.float32).unsqueeze(0),
            "human_vis": torch.as_tensor(human_vis, dtype=torch.bool).unsqueeze(0),
            "human_mask": torch.ones((1, len(selected), frames), dtype=torch.bool),
            "court_vis": torch.as_tensor(court_vis, dtype=torch.bool).unsqueeze(0),
        }
        if self.profile is PLCSInputProfile.MULTIVIEW:
            return PLCSPreparedBatch(call=self.build_call(ready))
        flattened = {
            "human_kp": ready["human_kp"][:, 0].reshape(
                frames, NUM_HUMAN_KP, 2
            ),
            "court_kp": ready["court_kp"][:, 0].reshape(
                frames, self.num_court_tokens, 2
            ),
            "human_vis": ready["human_vis"][:, 0].reshape(frames, NUM_HUMAN_KP),
            "human_mask": ready["human_mask"][:, 0].reshape(frames),
            "court_vis": ready["court_vis"][:, 0].reshape(
                frames, self.num_court_tokens
            ),
        }
        return PLCSPreparedBatch(
            call=self.build_call(flattened), sequence_shape=(1, frames)
        )

    def prepare_multiview_observations(
        self,
        *,
        human_kp: np.ndarray,
        court_kp: np.ndarray,
        human_vis: np.ndarray,
        human_mask: np.ndarray,
        court_vis: np.ndarray,
    ) -> PLCSPreparedBatch:
        """Convert explicit NumPy multiview observations at the task boundary."""
        self.require_profile(PLCSInputProfile.MULTIVIEW)
        if human_kp.ndim != 5:
            raise ModelInputContractError(
                "human_kp must have shape (B,V,T,17,2)."
            )
        batch_size, views, frames = human_kp.shape[:3]
        if court_kp.ndim == 4:
            court_kp = np.broadcast_to(court_kp[None], (batch_size, *court_kp.shape))
        if court_vis.ndim == 3:
            court_vis = np.broadcast_to(
                court_vis[None], (batch_size, *court_vis.shape)
            )
        expected_human_vis = (batch_size, views, frames, NUM_HUMAN_KP)
        expected_human_mask = (batch_size, views, frames)
        expected_court = (batch_size, views, frames, self.num_court_tokens, 2)
        expected_court_vis = (batch_size, views, frames, self.num_court_tokens)
        if human_vis.shape != expected_human_vis:
            raise ModelInputContractError(
                f"human_vis must have shape {expected_human_vis}, got {human_vis.shape}."
            )
        if human_mask.shape != expected_human_mask:
            raise ModelInputContractError(
                f"human_mask must have shape {expected_human_mask}, got {human_mask.shape}."
            )
        if court_kp.shape != expected_court:
            raise ModelInputContractError(
                f"court_kp must have shape {expected_court}, got {court_kp.shape}."
            )
        if court_vis.shape != expected_court_vis:
            raise ModelInputContractError(
                f"court_vis must have shape {expected_court_vis}, got {court_vis.shape}."
            )
        ready = {
            "human_kp": torch.as_tensor(np.array(human_kp, copy=True)),
            "court_kp": torch.as_tensor(np.array(court_kp, copy=True)),
            "human_vis": torch.as_tensor(np.array(human_vis, copy=True)),
            "human_mask": torch.as_tensor(np.array(human_mask, copy=True)),
            "court_vis": torch.as_tensor(np.array(court_vis, copy=True)),
        }
        return PLCSPreparedBatch(call=self.build_call(ready))


class PLCSTrackQueryIOAdapter:
    """Strict I/O adapter for the multi-person track-query model."""

    def __init__(
        self,
        *,
        model_type: type[nn.Module],
        num_queries: int,
        num_court_tokens: int,
        num_joints: int,
        mask_invisible_observations: bool,
        court_observation_profile: CourtObservationProfile = "kp14_reference_baseline",
    ) -> None:
        self._model_type = model_type
        self.profile = PLCSInputProfile.TRACK_QUERY
        self.num_queries = num_queries
        self.num_court_tokens = num_court_tokens
        self.num_joints = num_joints
        self.mask_invisible_observations = mask_invisible_observations
        self.court_observation_profile = parse_court_observation_profile(
            court_observation_profile
        )

    @property
    def model_type(self) -> type[nn.Module]:
        return self._model_type

    def require_profile(self, profile: PLCSInputProfile | str) -> None:
        requested = PLCSInputProfile(profile)
        if requested is not self.profile:
            raise ModelInputContractError(
                f"PLCS adapter profile is {self.profile.value!r}, not "
                f"{requested.value!r}."
            )

    def build_call(self, batch: Mapping[str, object]) -> ModelCall:
        human_kp = require_tensor(
            batch,
            "human_kp",
            spec=TensorSpec(
                shape=(None, None, None, None, self.num_joints, 2),
                dtypes=_FLOAT_DTYPES,
            ),
        )
        detection_mask = require_tensor(
            batch,
            "detection_mask",
            spec=TensorSpec(
                shape=(None, None, None, None), dtypes=frozenset({torch.bool})
            ),
        )
        joint_visibility = require_tensor(
            batch,
            "joint_visibility",
            spec=TensorSpec(
                shape=human_kp.shape[:-1],
                dtypes=frozenset({torch.bool}),
            ),
        )
        detection_score = require_tensor(
            batch,
            "detection_score",
            spec=TensorSpec(
                shape=human_kp.shape[:-2],
                dtypes=_FLOAT_DTYPES,
            ),
        )
        frame_mask = require_tensor(
            batch,
            "frame_mask",
            spec=TensorSpec(
                shape=(None, None), dtypes=frozenset({torch.bool})
            ),
        )
        view_mask = require_tensor(
            batch,
            "view_mask",
            spec=TensorSpec(
                shape=(None, None), dtypes=frozenset({torch.bool})
            ),
        )
        reference_index = require_tensor(
            batch,
            "reference_view_index",
            spec=TensorSpec(
                shape=(human_kp.shape[0],), dtypes=frozenset({torch.int64})
            ),
        )
        batch_size, views, frames, detections = human_kp.shape[:4]
        if min(batch_size, views, frames, detections) == 0:
            raise ModelInputContractError(
                "Tracking (B,V,T,P) axes must all be non-empty."
            )
        if detection_mask.shape != (batch_size, views, frames, detections):
            raise ModelInputContractError(
                "detection_mask must match human_kp through its detection axis."
            )
        if frame_mask.shape != (batch_size, frames):
            raise ModelInputContractError("frame_mask must have shape (B,T).")
        if view_mask.shape != (batch_size, views):
            raise ModelInputContractError("view_mask must have shape (B,V).")
        _normalized_uv("human_kp", human_kp)
        _finite("detection_score", detection_score)
        if bool(((detection_score < 0.0) | (detection_score > 1.0)).any()):
            raise ModelInputContractError("detection_score must be within [0,1].")
        if bool((detection_mask & ~joint_visibility.any(dim=-1)).any()):
            raise ModelInputContractError(
                "detection_mask cannot be true without a visible joint."
            )
        valid_observation = view_mask[:, :, None, None] & frame_mask[:, None, :, None]
        if bool((detection_mask & ~valid_observation).any().item()):
            raise ModelInputContractError(
                "detection_mask cannot be true in a padded view or frame."
            )
        try:
            reference_mask = reference_view_mask(reference_index, view_mask)
        except ValueError as error:
            raise ModelInputContractError(str(error)) from error
        camera_state_valid, spatial_mask, temporal_mask = (
            prepare_tracking_attention_masks(
                detection_mask=detection_mask,
                frame_mask=frame_mask,
                view_mask=view_mask,
                reference_view_mask=reference_mask,
                num_queries=self.num_queries,
                mask_invisible_observations=self.mask_invisible_observations,
            )
        )
        kwargs: dict[str, Tensor] = {
            "human_kp": human_kp,
            "detection_mask": detection_mask,
            "frame_mask": frame_mask,
            "camera_state_valid": camera_state_valid,
            "spatial_attention_mask": spatial_mask,
            "temporal_attention_mask": temporal_mask,
            "reference_view_mask": reference_mask,
        }
        if self.court_observation_profile == "kp14_reference_baseline":
            court_kp = require_tensor(
                batch,
                "court_kp",
                spec=TensorSpec(
                    shape=(batch_size, views, frames, self.num_court_tokens, 2),
                    dtypes=_FLOAT_DTYPES,
                ),
            )
            court_vis = require_tensor(
                batch,
                "court_vis",
                spec=TensorSpec(
                    shape=court_kp.shape[:-1], dtypes=frozenset({torch.bool})
                ),
            )
            _normalized_uv("court_kp", court_kp)
            kwargs.update({"court_kp": court_kp, "court_vis": court_vis})
        else:
            try:
                peaks = court_peak_batch_from_model_input(
                    batch,
                    expected_shape_bvt=(batch_size, views, frames),
                )
            except (TypeError, ValueError) as error:
                raise ModelInputContractError(str(error)) from error
            court_peak_uv = peaks.uv
            court_peak_score = peaks.score
            court_peak_covariance = peaks.covariance
            court_peak_valid = peaks.valid
            if court_peak_uv.dtype != human_kp.dtype:
                raise ModelInputContractError(
                    "Court peak floating tensors must match human_kp dtype."
                )
            player_anchor, player_features = _prepare_player_geometry(
                human_kp,
                joint_visibility,
                detection_score,
            )
            kwargs.update(
                {
                    "court_peak_uv": court_peak_uv,
                    "court_peak_score": court_peak_score,
                    "court_peak_covariance": court_peak_covariance,
                    "court_peak_valid": court_peak_valid,
                    "player_anchor": player_anchor,
                    "player_features": player_features,
                }
            )
        devices = {value.device for value in kwargs.values()}
        if len(devices) != 1:
            raise ModelInputContractError(
                "All PLCS tracking model inputs must share one device."
            )
        return ModelCall(kwargs=kwargs)

    def prepare_training_batch(
        self, batch: Mapping[str, object]
    ) -> PLCSPreparedBatch:
        call = self.build_call(batch)
        human_kp = cast(Tensor, call.kwargs["human_kp"])
        batch_size, _, frames = human_kp.shape[:3]
        target_position = require_tensor(
            batch,
            "target_position",
            spec=TensorSpec(
                shape=(batch_size, frames, None, 3), dtypes=_FLOAT_DTYPES
            ),
        )
        target_rotation = require_tensor(
            batch,
            "target_rotation",
            spec=TensorSpec(
                shape=(batch_size, frames, None, 2), dtypes=_FLOAT_DTYPES
            ),
        )
        source_target_position = require_tensor(
            batch,
            "source_target_position",
            spec=TensorSpec(shape=target_position.shape, dtypes=_FLOAT_DTYPES),
        )
        source_target_rotation = require_tensor(
            batch,
            "source_target_rotation",
            spec=TensorSpec(shape=target_rotation.shape, dtypes=_FLOAT_DTYPES),
        )
        target_human_kp_3d = require_tensor(
            batch,
            "target_human_kp_3d",
            spec=TensorSpec(
                shape=(*target_position.shape[:-1], 17, 3), dtypes=_FLOAT_DTYPES
            ),
        )
        source_target_human_kp_3d = require_tensor(
            batch,
            "source_target_human_kp_3d",
            spec=TensorSpec(shape=target_human_kp_3d.shape, dtypes=_FLOAT_DTYPES),
        )
        orientation_sign = require_tensor(
            batch,
            "orientation_sign",
            spec=TensorSpec(shape=(batch_size,), dtypes=_FLOAT_DTYPES),
        )
        views = human_kp.shape[1]
        view_mask = require_tensor(
            batch,
            "view_mask",
            spec=TensorSpec(
                shape=(batch_size, views), dtypes=frozenset({torch.bool})
            ),
        )
        reference_index = require_tensor(
            batch,
            "reference_view_index",
            spec=TensorSpec(
                shape=(batch_size,), dtypes=frozenset({torch.int64})
            ),
        )
        camera_center = require_tensor(
            batch,
            "camera_center",
            spec=TensorSpec(
                shape=(batch_size, views, 3), dtypes=_FLOAT_DTYPES
            ),
        )
        target_presence = require_tensor(
            batch,
            "target_presence",
            spec=TensorSpec(
                shape=(batch_size, frames, None),
                dtypes=frozenset({torch.bool}),
            ),
        )
        target_slot_mask = require_tensor(
            batch,
            "target_slot_mask",
            spec=TensorSpec(
                shape=(batch_size, None), dtypes=frozenset({torch.bool})
            ),
        )
        target_instance_id = require_tensor(
            batch,
            "target_instance_id",
            spec=TensorSpec(
                shape=(batch_size, frames, None),
                dtypes=frozenset({torch.int64}),
            ),
        )
        if target_position.shape[:-1] != target_rotation.shape[:-1]:
            raise ModelInputContractError(
                "Tracking target_position/target_rotation must share (B,T,S)."
            )
        if target_presence.shape != target_position.shape[:-1]:
            raise ModelInputContractError(
                "target_presence must match tracking target (B,T,S)."
            )
        if target_instance_id.shape != target_presence.shape:
            raise ModelInputContractError(
                "target_instance_id must match target_presence shape."
            )
        if target_slot_mask.shape != (
            batch_size,
            target_position.shape[2],
        ):
            raise ModelInputContractError(
                "target_slot_mask must match tracking target slot axis."
            )
        _finite("target_position", target_position)
        _finite("target_rotation", target_rotation)
        try:
            validate_declared_reference_orientation(
                camera_center,
                view_mask,
                reference_index,
                orientation_sign,
            )
        except (TypeError, ValueError) as error:
            raise ModelInputContractError(str(error)) from error
        if not bool(
            torch.allclose(
                target_position,
                reflect_court_vectors(source_target_position, orientation_sign),
            )
        ):
            raise ModelInputContractError(
                "target_position is inconsistent with reference orientation."
            )
        if not bool(
            torch.allclose(
                target_rotation,
                reflect_heading(source_target_rotation, orientation_sign),
            )
        ):
            raise ModelInputContractError(
                "target_rotation is inconsistent with reference orientation."
            )
        if not bool(
            torch.allclose(
                target_human_kp_3d,
                reflect_court_vectors(
                    source_target_human_kp_3d, orientation_sign
                ),
            )
        ):
            raise ModelInputContractError(
                "target_human_kp_3d is inconsistent with reference orientation."
            )
        invalid_inactive_ids = (~target_presence) & (target_instance_id != -1)
        if bool(invalid_inactive_ids.any().item()):
            raise ModelInputContractError(
                "Inactive tracking targets must use target_instance_id=-1."
            )
        return PLCSPreparedBatch(call=call)

    def decode_output(
        self, output: Mapping[str, object]
    ) -> PLCSTrackingDecodedPrediction:
        if not isinstance(output, Mapping):
            raise ModelOutputContractError(
                f"PLCS tracking output must be a mapping, got {type(output).__name__}."
            )
        expected = {"position", "rotation", "presence_logits"}
        if set(output) != expected:
            raise ModelOutputContractError(
                "PLCS tracking output keys do not match the paired adapter: "
                f"missing={sorted(expected - set(output))}, "
                f"unknown={sorted(set(output) - expected)}."
            )
        position = _required_output(
            output, "position", shape=(None, None, self.num_queries, 3)
        )
        rotation = _required_output(
            output, "rotation", shape=(None, None, self.num_queries, 2)
        )
        presence_logits = _required_output(
            output, "presence_logits", shape=(None, None, self.num_queries)
        )
        if position.shape[:-1] != rotation.shape[:-1]:
            raise ModelOutputContractError(
                "Tracking position and rotation must share (B,T,Q)."
            )
        if presence_logits.shape != position.shape[:-1]:
            raise ModelOutputContractError(
                "presence_logits must match tracking output (B,T,Q)."
            )
        return PLCSTrackingDecodedPrediction(
            position=position,
            rotation=rotation,
            presence_logits=presence_logits,
        )

    def decode_prepared_output(
        self,
        output: Mapping[str, object],
        prepared: PLCSPreparedBatch,
    ) -> PLCSTrackingDecodedPrediction:
        del prepared
        return self.decode_output(output)


PLCSAdapter = PLCSModelIOAdapter | PLCSTrackQueryIOAdapter

__all__ = ["PLCSAdapter", "PLCSModelIOAdapter", "PLCSTrackQueryIOAdapter"]
