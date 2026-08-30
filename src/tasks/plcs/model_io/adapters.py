"""Boundary adapters for every active PLCS model input/output profile."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import replace
from typing import cast

import numpy as np
import torch
from torch import Tensor, nn

from src.tasks.base.data import validate_reference_view_batch
from src.tasks.base.generate_dataset import (
    CAMERA_VIEW_V2_SELECTOR,
    PHYSICAL_V1_SELECTOR,
    CourtKeypointContract,
    CourtReferenceFrameProvenance,
    CourtViewRecord,
    align_court_keypoints_to_reference,
    build_physical_court_provenance,
    build_reference_frame_provenance,
)
from src.tasks.base.model_io import (
    ModelCall,
    ModelInputContractError,
    ModelOutputContractError,
    TensorSpec,
    TrackQueryReferenceContract,
    require_tensor,
    validate_model_artifact_court_keypoint_contract,
    validate_track_query_reference_contract,
)
from src.tasks.base.models import resolve_reference_selector_mode
from src.tasks.plcs.court_keypoint_contract import (
    court_keypoint_contract_document,
    provenance_from_value,
    validate_provenance_contract,
)
from src.tasks.plcs.model_io.attention_masks import (
    prepare_axial_attention_masks,
)
from src.tasks.plcs.model_io.contracts import (
    PLCSDecodedPrediction,
    PLCSInputProfile,
    PLCSPreparedBatch,
    PLCSReprojectionTarget,
    PLCSTrackingDecodedPrediction,
    plcs_reference_metadata_from_batch,
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
_REPROJECTION_KEYS = frozenset(
    {
        "human_kp_target",
        "human_vis_target",
        "camera_R",
        "camera_C",
        "camera_f",
        "camera_cx",
        "camera_cy",
        "camera_w",
        "camera_h",
    }
)
def _court_context(
    batch: Mapping[str, object],
    contract: CourtKeypointContract,
    *,
    batch_size: int,
) -> tuple[CourtReferenceFrameProvenance, ...]:
    documents_value = batch.get("court_keypoint_metadata")
    if documents_value is None:
        documents: tuple[Mapping[str, object], ...] = ({},)
    elif isinstance(documents_value, Mapping):
        documents = (documents_value,)
    elif isinstance(documents_value, Sequence) and not isinstance(
        documents_value, (str, bytes, bytearray)
    ):
        documents = tuple(
            value for value in documents_value if isinstance(value, Mapping)
        )
        if len(documents) != len(documents_value):
            raise ModelInputContractError(
                "court_keypoint_metadata must contain only metadata mappings."
            )
    else:
        raise ModelInputContractError(
            "court_keypoint_metadata must be a mapping or mapping sequence."
        )
    if not documents or batch_size % len(documents) != 0:
        raise ModelInputContractError(
            "court_keypoint_metadata cardinality must divide the model batch axis."
        )
    for index, document in enumerate(documents):
        try:
            validate_model_artifact_court_keypoint_contract(
                document,
                contract,
                location=f"PLCS input[{index}]",
            )
        except ValueError as error:
            raise ModelInputContractError(str(error)) from error

    provenance_value = batch.get("court_reference_provenance")
    if provenance_value is None:
        if contract.selector != PHYSICAL_V1_SELECTOR:
            raise ModelInputContractError(
                "camera_view_v2 input requires court_reference_provenance."
            )
        return tuple(build_physical_court_provenance() for _ in documents)
    if isinstance(provenance_value, (CourtReferenceFrameProvenance, Mapping)):
        raw_provenance: tuple[object, ...] = (provenance_value,)
    elif isinstance(provenance_value, Sequence) and not isinstance(
        provenance_value, (str, bytes, bytearray)
    ):
        raw_provenance = tuple(provenance_value)
    else:
        raise ModelInputContractError(
            "court_reference_provenance must be a record or record sequence."
        )
    if not raw_provenance or batch_size % len(raw_provenance) != 0:
        raise ModelInputContractError(
            "court_reference_provenance cardinality must divide the model batch axis."
        )
    parsed: list[CourtReferenceFrameProvenance] = []
    for index, value in enumerate(raw_provenance):
        try:
            provenance = provenance_from_value(
                value,
                location=f"PLCS input provenance[{index}]",
            )
            validate_provenance_contract(
                provenance,
                contract,
                location=f"PLCS input provenance[{index}]",
            )
        except ValueError as error:
            raise ModelInputContractError(str(error)) from error
        parsed.append(provenance)
    return tuple(parsed)


def _finite(name: str, tensor: Tensor) -> None:
    if tensor.is_floating_point() and not bool(torch.isfinite(tensor).all().item()):
        raise ModelInputContractError(f"{name} must contain only finite values.")


def _normalized_uv(name: str, tensor: Tensor, visibility: Tensor) -> None:
    _finite(name, tensor)
    if visibility.shape != tensor.shape[:-1]:
        raise ModelInputContractError(
            f"{name} visibility must match {name} without its UV axis."
        )
    if visibility.device != tensor.device:
        raise ModelInputContractError(
            f"{name} visibility must share the UV tensor device."
        )
    _binary_mask(f"{name} visibility", visibility)
    visible_uv = tensor[visibility.to(dtype=torch.bool)]
    if visible_uv.numel() and (
        bool((visible_uv < 0).any().item())
        or bool((visible_uv > 1).any().item())
    ):
        raise ModelInputContractError(
            f"Visible {name} must use normalized UV coordinates within [0, 1]."
        )


def _binary_mask(name: str, tensor: Tensor) -> None:
    _finite(name, tensor)
    if tensor.dtype != torch.bool and tensor.numel():
        binary = (tensor == 0) | (tensor == 1)
        if not bool(binary.all().item()):
            raise ModelInputContractError(
                f"{name} must be boolean or contain only explicit 0/1 values."
            )


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
        court_keypoint_contract: CourtKeypointContract,
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
        self.court_keypoint_contract = court_keypoint_contract

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
        padding_mask_shape: tuple[int | None, ...]
        court_vis_shape: tuple[int | None, ...]
        if self.profile in {PLCSInputProfile.FRAME, PLCSInputProfile.SEQUENCE}:
            human_shape = (None, NUM_HUMAN_KP, 2)
            court_shape = (None, self.num_court_tokens, 2)
            human_vis_shape = (None, NUM_HUMAN_KP)
            padding_mask_shape = (None,)
            court_vis_shape = (None, self.num_court_tokens)
        else:
            human_shape = (None, None, None, NUM_HUMAN_KP, 2)
            court_shape = (None, None, None, self.num_court_tokens, 2)
            human_vis_shape = (None, None, None, NUM_HUMAN_KP)
            padding_mask_shape = (None, None, None)
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
        if "human_mask" in batch:
            raise ModelInputContractError(
                "human_mask is no longer supported; use padding_mask with True=padding."
            )
        padding_mask = require_tensor(
            batch,
            "padding_mask",
            spec=TensorSpec(shape=padding_mask_shape, dtypes=frozenset({torch.bool})),
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
            if bool(padding_mask.all(dim=(1, 2)).any().item()):
                raise ModelInputContractError(
                    "Every multiview sequence must contain at least one "
                    "non-padding frame."
                )
        _normalized_uv("human_kp", human_kp, human_vis)
        _normalized_uv("court_kp", court_kp, court_vis)
        _binary_mask("padding_mask", padding_mask)
        return human_kp, court_kp, human_vis, padding_mask, court_vis

    def build_call(self, batch: Mapping[str, object]) -> ModelCall:
        """Validate a model-ready batch and build one immutable model call."""
        human_kp, court_kp, human_vis, padding_mask, court_vis = (
            self._validate_ready_inputs(batch)
        )
        _court_context(
            batch,
            self.court_keypoint_contract,
            batch_size=int(human_kp.shape[0]),
        )
        kwargs = {
            "human_kp": human_kp,
            "court_kp": court_kp,
            "human_vis": human_vis,
            "padding_mask": padding_mask,
            "court_vis": court_vis,
        }
        if issubclass(self.model_type, PLCSMultiViewAxialModel):
            camera_mask, time_mask = prepare_axial_attention_masks(padding_mask)
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
        if "human_mask" in batch:
            raise ModelInputContractError(
                "human_mask is no longer supported; use padding_mask with True=padding."
            )
        padding_mask = require_tensor(
            batch,
            "padding_mask",
            spec=TensorSpec(
                shape=(None, None, None), dtypes=frozenset({torch.bool})
            ),
        )
        court_vis = require_tensor(
            batch,
            "court_vis",
            spec=TensorSpec(
                shape=(None, None, None, self.num_court_tokens),
                dtypes=_MASK_DTYPES,
            ),
        )
        canonical_tensors = {
            "human_kp": human_kp,
            "court_kp": court_kp,
            "human_vis": human_vis,
            "padding_mask": padding_mask,
            "court_vis": court_vis,
        }
        canonical: dict[str, object] = {
            **canonical_tensors,
            "court_keypoint_metadata": batch.get("court_keypoint_metadata"),
            "court_reference_provenance": batch.get(
                "court_reference_provenance"
            ),
        }
        self._validate_canonical_axes(canonical_tensors)
        batch_size, views, frames = human_kp.shape[:3]
        provenance = _court_context(
            batch,
            self.court_keypoint_contract,
            batch_size=batch_size,
        )
        try:
            reference_metadata = plcs_reference_metadata_from_batch(batch)
        except (TypeError, ValueError) as error:
            raise ModelInputContractError(str(error)) from error
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
        reprojection_target = self._prepare_reprojection_target(
            batch,
            batch_size=batch_size,
            views=views,
            frames=frames,
            padding_mask=padding_mask,
        )

        if self.profile is PLCSInputProfile.MULTIVIEW:
            return PLCSPreparedBatch(
                call=self.build_call(canonical),
                target_position=target_position,
                target_rotation=target_rotation,
                target_human_kp_3d=target_human_kp_3d,
                target_padding_mask=padding_mask,
                reprojection_target=reprojection_target,
                court_reference_provenance=provenance,
                reference_metadata=reference_metadata,
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
            "padding_mask": padding_mask[:, self.camera_index].reshape(
                batch_size * frames
            ),
            "court_vis": court_vis[:, self.camera_index].reshape(
                batch_size * frames, self.num_court_tokens
            ),
            "court_keypoint_metadata": batch.get("court_keypoint_metadata"),
            "court_reference_provenance": batch.get(
                "court_reference_provenance"
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
            target_padding_mask = padding_mask[:, self.camera_index, 0]
        else:
            target_padding_mask = padding_mask[:, self.camera_index]
        return PLCSPreparedBatch(
            call=self.build_call(ready),
            sequence_shape=sequence_shape,
            target_position=target_position,
            target_rotation=target_rotation,
            target_human_kp_3d=target_human_kp_3d,
            target_padding_mask=target_padding_mask,
            court_reference_provenance=provenance,
            reference_metadata=reference_metadata,
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
        if batch["padding_mask"].shape != human_kp.shape[:3]:
            raise ModelInputContractError("padding_mask must have shape (B,V,T).")
        _normalized_uv("human_kp", human_kp, batch["human_vis"])
        _normalized_uv("court_kp", court_kp, batch["court_vis"])
        _binary_mask("padding_mask", batch["padding_mask"])

    def _prepare_reprojection_target(
        self,
        batch: Mapping[str, object],
        *,
        batch_size: int,
        views: int,
        frames: int,
        padding_mask: Tensor,
    ) -> PLCSReprojectionTarget | None:
        present = _REPROJECTION_KEYS.intersection(batch)
        if present and present != _REPROJECTION_KEYS:
            missing = sorted(_REPROJECTION_KEYS - present)
            raise ModelInputContractError(
                "PLCS reprojection fields are all-or-none; missing="
                f"{missing}."
            )
        if not present:
            return None

        target_uv = require_tensor(
            batch,
            "human_kp_target",
            spec=TensorSpec(
                shape=(batch_size, views, frames, NUM_HUMAN_KP, 2),
                dtypes=_FLOAT_DTYPES,
            ),
        )
        target_vis = require_tensor(
            batch,
            "human_vis_target",
            spec=TensorSpec(
                shape=(batch_size, views, frames, NUM_HUMAN_KP),
                dtypes=_MASK_DTYPES,
            ),
        )
        camera_R = require_tensor(
            batch,
            "camera_R",
            spec=TensorSpec(
                shape=(batch_size, views, 3, 3),
                dtypes=_FLOAT_DTYPES,
            ),
        )
        camera_C = require_tensor(
            batch,
            "camera_C",
            spec=TensorSpec(
                shape=(batch_size, views, 3),
                dtypes=_FLOAT_DTYPES,
            ),
        )

        camera_scalars: dict[str, Tensor] = {}
        for name in (
            "camera_f",
            "camera_cx",
            "camera_cy",
            "camera_w",
            "camera_h",
        ):
            camera_scalars[name] = require_tensor(
                batch,
                name,
                spec=TensorSpec(
                    shape=(batch_size, views),
                    dtypes=_FLOAT_DTYPES,
                ),
            )

        _normalized_uv("human_kp_target", target_uv, target_vis)
        _binary_mask("human_vis_target", target_vis)
        for name, tensor in {
            "camera_R": camera_R,
            "camera_C": camera_C,
            **camera_scalars,
        }.items():
            _finite(name, tensor)

        if bool(((target_vis > 0) & padding_mask.unsqueeze(-1)).any().item()):
            raise ModelInputContractError(
                "human_vis_target must be zero at padded view/time entries."
            )

        valid_views = ~padding_mask.all(dim=-1)
        for name in ("camera_f", "camera_w", "camera_h"):
            if bool((camera_scalars[name][valid_views] <= 0).any().item()):
                raise ModelInputContractError(
                    f"{name} must be positive for every non-padded camera view."
                )

        return PLCSReprojectionTarget(
            target_uv=target_uv,
            target_vis=target_vis,
            padding_mask=padding_mask,
            camera_R=camera_R,
            camera_C=camera_C,
            camera_f=camera_scalars["camera_f"],
            camera_cx=camera_scalars["camera_cx"],
            camera_cy=camera_scalars["camera_cy"],
            camera_w=camera_scalars["camera_w"],
            camera_h=camera_scalars["camera_h"],
        )

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
            return replace(
                decoded,
                court_reference_provenance=prepared.court_reference_provenance,
                reference_metadata=prepared.reference_metadata,
            )
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
            court_reference_provenance=prepared.court_reference_provenance,
            reference_metadata=prepared.reference_metadata,
        )

    def prepare_scene(
        self,
        scene: object,
        cameras: Sequence[int],
        *,
        reference_camera_id: str | None = None,
    ) -> PLCSPreparedBatch:
        """Build a validated frame/sequence/multiview call from a loaded scene."""
        if not cameras:
            raise ModelInputContractError("At least one PLCS camera is required.")
        scene_cameras = getattr(scene, "cameras", None)
        if not isinstance(scene_cameras, Sequence):
            raise ModelInputContractError("PLCS scene must expose a cameras sequence.")
        scene_contract = getattr(scene, "court_keypoint_contract", None)
        if not isinstance(scene_contract, CourtKeypointContract):
            raise ModelInputContractError(
                "PLCS direct scene input requires an explicit CourtKP20 contract."
            )
        if scene_contract != self.court_keypoint_contract:
            raise ModelInputContractError(
                "PLCS scene and adapter CourtKP20 contracts do not match."
            )
        if any(index < 0 or index >= len(scene_cameras) for index in cameras):
            raise ModelInputContractError("PLCS camera selection is out of range.")

        if self.profile is PLCSInputProfile.MULTIVIEW:
            selected = list(cameras)
        else:
            selected = [cameras[0]]
        selected_scene_cameras = [scene_cameras[index] for index in selected]
        if self.court_keypoint_contract.selector == CAMERA_VIEW_V2_SELECTOR:
            if reference_camera_id is None:
                raise ModelInputContractError(
                    "camera_view_v2 direct scene inference requires an explicit "
                    "reference_camera_id."
                )
            court_views = tuple(
                getattr(camera, "court_view", None)
                for camera in selected_scene_cameras
            )
            if any(view is None for view in court_views):
                raise ModelInputContractError(
                    "camera_view_v2 scene cameras require CourtKP20 metadata."
                )
            typed_views = cast(
                "tuple[CourtViewRecord, ...]",
                court_views,
            )
            try:
                provenance = build_reference_frame_provenance(
                    typed_views,
                    reference_camera_id=reference_camera_id,
                )
            except ValueError as error:
                raise ModelInputContractError(str(error)) from error
            reference_view = typed_views[
                cast(int, provenance.reference_camera_local_index)
            ]
        else:
            provenance = build_physical_court_provenance()
            typed_views = ()
            reference_view = None
        human = np.stack(
            [np.asarray(scene_cameras[index].human_kp_uv) for index in selected],
            axis=0,
        )
        court_arrays: list[np.ndarray] = []
        for local_index, camera in enumerate(selected_scene_cameras):
            court_array = np.asarray(camera.court_kp_uv)
            if reference_view is not None:
                court_array = align_court_keypoints_to_reference(
                    court_array,
                    typed_views[local_index],
                    reference_view,
                    keypoint_axis=-2,
                )
            court_arrays.append(court_array[..., : self.num_court_tokens, :])
        court = np.stack(court_arrays, axis=0)
        human_vis = np.stack(
            [
                np.asarray(scene_cameras[index].human_kp_vis, dtype=np.bool_)
                for index in selected
            ],
            axis=0,
        )
        court_vis_arrays: list[np.ndarray] = []
        for local_index, camera in enumerate(selected_scene_cameras):
            court_vis_array = np.asarray(camera.court_kp_vis, dtype=np.bool_)
            if reference_view is not None:
                court_vis_array = align_court_keypoints_to_reference(
                    court_vis_array,
                    typed_views[local_index],
                    reference_view,
                    keypoint_axis=-1,
                )
            court_vis_arrays.append(court_vis_array[..., : self.num_court_tokens])
        court_vis = np.stack(court_vis_arrays, axis=0)
        frames = human.shape[1]
        ready_human = torch.as_tensor(human, dtype=torch.float32).unsqueeze(0)
        ready_court = torch.as_tensor(court, dtype=torch.float32).unsqueeze(0)
        ready_human_vis = torch.as_tensor(human_vis, dtype=torch.bool).unsqueeze(0)
        ready_padding = torch.zeros((1, len(selected), frames), dtype=torch.bool)
        ready_court_vis = torch.as_tensor(court_vis, dtype=torch.bool).unsqueeze(0)
        ready: dict[str, object] = {
            "human_kp": ready_human,
            "court_kp": ready_court,
            "human_vis": ready_human_vis,
            "padding_mask": ready_padding,
            "court_vis": ready_court_vis,
            "court_keypoint_metadata": court_keypoint_contract_document(
                self.court_keypoint_contract
            ),
            "court_reference_provenance": provenance,
        }
        if self.profile is PLCSInputProfile.MULTIVIEW:
            return PLCSPreparedBatch(
                call=self.build_call(ready),
                court_reference_provenance=(provenance,),
            )
        flattened = {
            "human_kp": ready_human[:, 0].reshape(frames, NUM_HUMAN_KP, 2),
            "court_kp": ready_court[:, 0].reshape(
                frames, self.num_court_tokens, 2
            ),
            "human_vis": ready_human_vis[:, 0].reshape(frames, NUM_HUMAN_KP),
            "padding_mask": ready_padding[:, 0].reshape(frames),
            "court_vis": ready_court_vis[:, 0].reshape(
                frames, self.num_court_tokens
            ),
            "court_keypoint_metadata": ready["court_keypoint_metadata"],
            "court_reference_provenance": ready["court_reference_provenance"],
        }
        return PLCSPreparedBatch(
            call=self.build_call(flattened),
            sequence_shape=(1, frames),
            court_reference_provenance=(provenance,),
        )

    def prepare_multiview_observations(
        self,
        *,
        human_kp: np.ndarray,
        court_kp: np.ndarray,
        human_vis: np.ndarray,
        padding_mask: np.ndarray,
        court_vis: np.ndarray,
        court_keypoint_metadata: Mapping[str, object] | None = None,
        court_reference_provenance: CourtReferenceFrameProvenance | None = None,
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
        expected_padding_mask = (batch_size, views, frames)
        expected_court = (batch_size, views, frames, self.num_court_tokens, 2)
        expected_court_vis = (batch_size, views, frames, self.num_court_tokens)
        if human_vis.shape != expected_human_vis:
            raise ModelInputContractError(
                f"human_vis must have shape {expected_human_vis}, got {human_vis.shape}."
            )
        if padding_mask.shape != expected_padding_mask:
            raise ModelInputContractError(
                "padding_mask must have shape "
                f"{expected_padding_mask}, got {padding_mask.shape}."
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
            "padding_mask": torch.as_tensor(np.array(padding_mask, copy=True)),
            "court_vis": torch.as_tensor(np.array(court_vis, copy=True)),
            "court_keypoint_metadata": court_keypoint_metadata,
            "court_reference_provenance": court_reference_provenance,
        }
        provenance = _court_context(
            ready,
            self.court_keypoint_contract,
            batch_size=batch_size,
        )
        return PLCSPreparedBatch(
            call=self.build_call(ready),
            court_reference_provenance=provenance,
        )


class PLCSTrackQueryIOAdapter:
    """Strict I/O adapter for the multi-person track-query model."""

    def __init__(
        self,
        *,
        model_type: type[nn.Module],
        num_queries: int,
        num_court_tokens: int,
        num_joints: int,
        court_keypoint_contract: CourtKeypointContract,
        predict_canonical_pose: bool = False,
        reprojection_enabled: bool = False,
    ) -> None:
        self._model_type = model_type
        self.profile = PLCSInputProfile.TRACK_QUERY
        self.num_queries = num_queries
        self.num_court_tokens = num_court_tokens
        self.num_joints = num_joints
        self.court_keypoint_contract = court_keypoint_contract
        self.predict_canonical_pose = predict_canonical_pose
        self.reprojection_enabled = reprojection_enabled

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
        legacy_keys = {"detection_mask", "frame_mask", "view_mask"} & set(batch)
        if legacy_keys:
            raise ModelInputContractError(
                "Legacy PLCS tracking mask keys are not supported: "
                f"{sorted(legacy_keys)}. Use padding_mask with True=padding."
            )
        human_kp = require_tensor(
            batch,
            "human_kp",
            spec=TensorSpec(
                shape=(None, None, None, self.num_queries, self.num_joints, 2),
                dtypes=_FLOAT_DTYPES,
            ),
        )
        human_vis = require_tensor(
            batch,
            "human_vis",
            spec=TensorSpec(
                shape=(None, None, None, self.num_queries, self.num_joints),
                dtypes=frozenset({torch.bool}),
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
        court_vis = require_tensor(
            batch,
            "court_vis",
            spec=TensorSpec(
                shape=(None, None, None, self.num_court_tokens),
                dtypes=frozenset({torch.bool}),
            ),
        )
        padding_mask = require_tensor(
            batch,
            "padding_mask",
            spec=TensorSpec(
                shape=(None, None, None), dtypes=frozenset({torch.bool})
            ),
        )
        batch_size, views, frames, queries = human_kp.shape[:4]
        _court_context(
            batch,
            self.court_keypoint_contract,
            batch_size=batch_size,
        )
        if min(batch_size, views, frames, queries) == 0:
            raise ModelInputContractError(
                "Tracking (B,V,T,Q) axes must all be non-empty."
            )
        if queries != self.num_queries:
            raise ModelInputContractError(
                "human_kp query axis must equal num_queries."
            )
        if human_vis.shape != human_kp.shape[:-1]:
            raise ModelInputContractError("human_vis must match human_kp without UV.")
        if court_kp.shape[:3] != (batch_size, views, frames):
            raise ModelInputContractError(
                "court_kp must share human_kp (B,V,T) axes."
            )
        if court_vis.shape != court_kp.shape[:-1]:
            raise ModelInputContractError(
                "court_vis must match court_kp without its UV axis."
            )
        if padding_mask.shape != (batch_size, views, frames):
            raise ModelInputContractError("padding_mask must have shape (B,V,T).")
        _normalized_uv("human_kp", human_kp, human_vis)
        _normalized_uv("court_kp", court_kp, court_vis)
        _binary_mask("padding_mask", padding_mask)
        return ModelCall(
            kwargs={
                "human_kp": human_kp,
                "human_vis": human_vis,
                "court_kp": court_kp,
                "court_vis": court_vis,
                "padding_mask": padding_mask,
            }
        )

    def prepare_training_batch(
        self, batch: Mapping[str, object]
    ) -> PLCSPreparedBatch:
        call = self.build_call(batch)
        human_kp = cast(Tensor, call.kwargs["human_kp"])
        padding_mask = cast(Tensor, call.kwargs["padding_mask"])
        batch_size, views, frames = human_kp.shape[:3]
        provenance = _court_context(
            batch,
            self.court_keypoint_contract,
            batch_size=batch_size,
        )
        target_position = require_tensor(
            batch,
            "target_position",
            spec=TensorSpec(
                shape=(batch_size, frames, self.num_queries, 3),
                dtypes=_FLOAT_DTYPES,
            ),
        )
        target_rotation = require_tensor(
            batch,
            "target_rotation",
            spec=TensorSpec(
                shape=(batch_size, frames, self.num_queries, 2),
                dtypes=_FLOAT_DTYPES,
            ),
        )
        target_presence = require_tensor(
            batch,
            "target_presence",
            spec=TensorSpec(
                shape=(batch_size, frames, self.num_queries),
                dtypes=frozenset({torch.bool}),
            ),
        )
        target_slot_mask = require_tensor(
            batch,
            "target_slot_mask",
            spec=TensorSpec(
                shape=(batch_size, self.num_queries),
                dtypes=frozenset({torch.bool}),
            ),
        )
        target_instance_id = require_tensor(
            batch,
            "target_instance_id",
            spec=TensorSpec(
                shape=(batch_size, frames, self.num_queries),
                dtypes=frozenset({torch.int64}),
            ),
        )
        if self.predict_canonical_pose:
            target_human_kp_3d = require_tensor(
                batch,
                "target_human_kp_3d",
                spec=TensorSpec(
                    shape=(
                        batch_size,
                        frames,
                        self.num_queries,
                        self.num_joints,
                        3,
                    ),
                    dtypes=_FLOAT_DTYPES,
                ),
            )
            _finite("target_human_kp_3d", target_human_kp_3d)
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
        invalid_inactive_ids = (~target_presence) & (target_instance_id != -1)
        if bool(invalid_inactive_ids.any().item()):
            raise ModelInputContractError(
                "Inactive tracking targets must use target_instance_id=-1."
            )
        reprojection_target = self._prepare_reprojection_target(
            batch,
            batch_size=batch_size,
            views=views,
            frames=frames,
            padding_mask=padding_mask,
        )
        return PLCSPreparedBatch(
            call=call,
            reprojection_target=reprojection_target,
            court_reference_provenance=provenance,
            reference_metadata=plcs_reference_metadata_from_batch(batch),
        )

    def _prepare_reprojection_target(
        self,
        batch: Mapping[str, object],
        *,
        batch_size: int,
        views: int,
        frames: int,
        padding_mask: Tensor,
    ) -> PLCSReprojectionTarget | None:
        present = _REPROJECTION_KEYS.intersection(batch)
        if not present and not self.reprojection_enabled:
            return None
        if present != _REPROJECTION_KEYS:
            missing = sorted(_REPROJECTION_KEYS - present)
            requirement = (
                "required when reprojection_enabled=True"
                if self.reprojection_enabled
                else "all-or-none once any reprojection field is supplied"
            )
            raise ModelInputContractError(
                "PLCS tracking reprojection fields are "
                f"{requirement}; missing={missing}."
            )

        target_uv = require_tensor(
            batch,
            "human_kp_target",
            spec=TensorSpec(
                shape=(
                    batch_size,
                    views,
                    frames,
                    self.num_queries,
                    self.num_joints,
                    2,
                ),
                dtypes=_FLOAT_DTYPES,
            ),
        )
        target_vis = require_tensor(
            batch,
            "human_vis_target",
            spec=TensorSpec(
                shape=(
                    batch_size,
                    views,
                    frames,
                    self.num_queries,
                    self.num_joints,
                ),
                dtypes=frozenset({torch.bool}),
            ),
        )
        camera_R = require_tensor(
            batch,
            "camera_R",
            spec=TensorSpec(
                shape=(batch_size, views, 3, 3),
                dtypes=_FLOAT_DTYPES,
            ),
        )
        camera_C = require_tensor(
            batch,
            "camera_C",
            spec=TensorSpec(
                shape=(batch_size, views, 3),
                dtypes=_FLOAT_DTYPES,
            ),
        )

        camera_scalars: dict[str, Tensor] = {}
        for name in (
            "camera_f",
            "camera_cx",
            "camera_cy",
            "camera_w",
            "camera_h",
        ):
            camera_scalars[name] = require_tensor(
                batch,
                name,
                spec=TensorSpec(
                    shape=(batch_size, views),
                    dtypes=_FLOAT_DTYPES,
                ),
            )

        _normalized_uv("human_kp_target", target_uv, target_vis)
        _binary_mask("human_vis_target", target_vis)
        for name, tensor in {
            "camera_R": camera_R,
            "camera_C": camera_C,
            **camera_scalars,
        }.items():
            _finite(name, tensor)

        padded_targets = padding_mask.unsqueeze(-1).unsqueeze(-1)
        if bool((target_vis & padded_targets).any().item()):
            raise ModelInputContractError(
                "human_vis_target must be zero at padded view/time entries."
            )

        valid_views = ~padding_mask.all(dim=-1)
        for name in ("camera_f", "camera_w", "camera_h"):
            if bool((camera_scalars[name][valid_views] <= 0).any().item()):
                raise ModelInputContractError(
                    f"{name} must be positive for every non-padded camera view."
                )

        return PLCSReprojectionTarget(
            target_uv=target_uv,
            target_vis=target_vis,
            padding_mask=padding_mask,
            camera_R=camera_R,
            camera_C=camera_C,
            camera_f=camera_scalars["camera_f"],
            camera_cx=camera_scalars["camera_cx"],
            camera_cy=camera_scalars["camera_cy"],
            camera_w=camera_scalars["camera_w"],
            camera_h=camera_scalars["camera_h"],
        )

    def decode_output(
        self, output: Mapping[str, object]
    ) -> PLCSTrackingDecodedPrediction:
        if not isinstance(output, Mapping):
            raise ModelOutputContractError(
                f"PLCS tracking output must be a mapping, got {type(output).__name__}."
            )
        expected = {"position", "rotation", "presence_logits"}
        if self.predict_canonical_pose:
            expected.add("canonical_pose")
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
        canonical_pose = None
        if self.predict_canonical_pose:
            canonical_pose = _required_output(
                output,
                "canonical_pose",
                shape=(None, None, self.num_queries, self.num_joints, 3),
            )
        if position.shape[:-1] != rotation.shape[:-1]:
            raise ModelOutputContractError(
                "Tracking position and rotation must share (B,T,Q)."
            )
        if presence_logits.shape != position.shape[:-1]:
            raise ModelOutputContractError(
                "presence_logits must match tracking output (B,T,Q)."
            )
        if canonical_pose is not None and canonical_pose.shape[:3] != position.shape[:-1]:
            raise ModelOutputContractError(
                "canonical_pose must share tracking output (B,T,Q)."
            )
        return PLCSTrackingDecodedPrediction(
            position=position,
            rotation=rotation,
            presence_logits=presence_logits,
            canonical_pose=canonical_pose,
        )

    def decode_prepared_output(
        self,
        output: Mapping[str, object],
        prepared: PLCSPreparedBatch,
    ) -> PLCSTrackingDecodedPrediction:
        decoded = self.decode_output(output)
        return replace(
            decoded,
            court_reference_provenance=prepared.court_reference_provenance,
            reference_metadata=prepared.reference_metadata,
        )


class PLCSTrackQueryReferenceIOAdapter(PLCSTrackQueryIOAdapter):
    """Strict six-input adapter for reference-conditioned track-query v2."""

    def __init__(
        self,
        *,
        model_type: type[nn.Module],
        num_queries: int,
        num_court_tokens: int,
        num_joints: int,
        court_keypoint_contract: CourtKeypointContract,
        target_frame_contract: str,
        track_query_rope_contract: str,
        reference_selector_mode: str,
        predict_canonical_pose: bool = False,
        reprojection_enabled: bool = False,
    ) -> None:
        super().__init__(
            model_type=model_type,
            num_queries=num_queries,
            num_court_tokens=num_court_tokens,
            num_joints=num_joints,
            court_keypoint_contract=court_keypoint_contract,
            predict_canonical_pose=predict_canonical_pose,
            reprojection_enabled=reprojection_enabled,
        )
        selector_mode = resolve_reference_selector_mode(reference_selector_mode)
        reference_contract = TrackQueryReferenceContract.reference_v2(selector_mode)
        if court_keypoint_contract.contract_id != (
            reference_contract.court_keypoint_contract
        ):
            raise ValueError(
                "Reference track-query I/O CourtKP20 contract does not match "
                "the shared reference-v2 contract."
            )
        if target_frame_contract != reference_contract.target_frame_contract:
            raise ValueError(
                "Reference track-query I/O target-frame contract does not match "
                "the shared reference-v2 contract."
            )
        if track_query_rope_contract != (
            reference_contract.track_query_rope_contract.value
        ):
            raise ValueError(
                "Reference track-query I/O RoPE contract does not match the "
                "shared reference-v2 contract."
            )
        self.reference_contract = reference_contract
        self.target_frame_contract = reference_contract.target_frame_contract
        self.track_query_rope_contract = (
            reference_contract.track_query_rope_contract
        )
        self.reference_selector_mode = selector_mode

    def build_call(self, batch: Mapping[str, object]) -> ModelCall:
        """Build the exact six-tensor call after identity/index validation."""
        try:
            validate_track_query_reference_contract(
                batch,
                self.reference_contract,
                location="PLCS track-query input",
            )
        except ValueError as error:
            raise ModelInputContractError(str(error)) from error
        call = super().build_call(batch)
        human_kp = cast(Tensor, call.kwargs["human_kp"])
        padding_mask = cast(Tensor, call.kwargs["padding_mask"])
        batch_size, _num_views, num_frames = human_kp.shape[:3]
        try:
            reference_metadata = plcs_reference_metadata_from_batch(batch)
            if reference_metadata is None:
                raise ValueError(
                    "Reference-v2 PLCS input requires complete typed reference "
                    "metadata."
                )
            if reference_metadata.track_query_contract != self.reference_contract:
                raise ValueError(
                    "PLCS typed reference metadata and adapter contracts do not "
                    "match exactly."
                )
            validate_reference_view_batch(
                reference_view_index=reference_metadata.reference_view_index,
                view_camera_ids=reference_metadata.view_camera_ids,
                reference_camera_id=reference_metadata.reference_camera_id,
                stable_camera_id_tables=(
                    reference_metadata.stable_camera_id_tables
                ),
                reference_from_physical=(
                    reference_metadata.reference_from_physical
                ),
                physical_from_reference=(
                    reference_metadata.physical_from_reference
                ),
                expected_device=human_kp.device,
            )
        except (TypeError, ValueError) as error:
            raise ModelInputContractError(str(error)) from error
        reference_view_index = reference_metadata.reference_view_index
        reference_from_physical = reference_metadata.reference_from_physical
        if reference_from_physical.dtype != human_kp.dtype:
            raise ModelInputContractError(
                "reference_from_physical must share the model input floating dtype."
            )
        provenance = _court_context(
            batch,
            self.court_keypoint_contract,
            batch_size=batch_size,
        )
        if len(provenance) != batch_size:
            raise ModelInputContractError(
                "Reference-v2 court_reference_provenance must contain exactly "
                "one record per sample."
            )
        for sample_index, selection in enumerate(reference_metadata.selections):
            if provenance[sample_index] != selection.provenance:
                raise ModelInputContractError(
                    f"sample {sample_index} Court/target provenance does not "
                    "match its typed stable camera selection."
                )

        selected_padding = padding_mask.gather(
            1,
            reference_view_index[:, None, None].expand(
                batch_size,
                1,
                num_frames,
            ),
        ).squeeze(1)
        supervised_time = (~padding_mask).any(dim=1)
        if bool((selected_padding & supervised_time).any().item()):
            raise ModelInputContractError(
                "Every non-padding time must retain an unmasked reference-view "
                "context token."
            )
        return ModelCall(
            kwargs={
                **call.kwargs,
                "reference_view_index": reference_view_index,
            }
        )


PLCSAdapter = (
    PLCSModelIOAdapter
    | PLCSTrackQueryIOAdapter
    | PLCSTrackQueryReferenceIOAdapter
)

__all__ = [
    "PLCSAdapter",
    "PLCSModelIOAdapter",
    "PLCSTrackQueryIOAdapter",
    "PLCSTrackQueryReferenceIOAdapter",
]
