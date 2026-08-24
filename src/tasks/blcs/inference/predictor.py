"""Unified predictor class for BLCS inference."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any, Self, cast

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor

from src.tasks.base.data import ReferenceViewSelection, StableCameraIdTable
from src.tasks.base.generate_dataset import (
    PHYSICAL_V1_SELECTOR,
    CourtKeypointContract,
    CourtKeypointContractMismatchError,
    CourtReferenceFrameProvenance,
    CourtViewRecord,
    align_court_keypoints_to_reference,
    build_physical_court_provenance,
    resolve_court_keypoint_contract,
)
from src.tasks.base.inference.predictor import BasePredictor
from src.tasks.base.model_io import (
    ModelInputContractError,
    validate_model_artifact_court_keypoint_contract,
)
from src.tasks.blcs.model_io import (
    BLCSReferenceMetadata,
    BLCSTrajectoryPrediction,
    TrajectoryBoundModelIO,
    TrajectoryModelIOAdapter,
    blcs_reference_metadata_from_batch,
    compose_blcs_trajectory_model_io,
)
from src.tasks.blcs.model_io.checkpoints import load_checkpoint_runtime
from src.tasks.blcs.training.lightning_module import BLCSLightningModule
from src.utils.configuration import PathResolver
from src.utils.schema.court import COURT_COORD_SCALE_XYZ


class BLCSPredictor(BasePredictor[BLCSTrajectoryPrediction]):
    """Unified BLCS model inference predictor.

    Supports:
    - `blcs` (single-view)
    - `blcs_multiview` (multi-view)

    Attributes:
        model: The BLCS model.
        device: The inference device.

    Example:
        >>> predictor = BLCSPredictor.load_from_checkpoint("model.ckpt", device="cuda")
        >>> result = predictor.predict(
        ...     ball_uv, court_kp, ball_vis, padding_mask, court_vis
        ... )
        >>> print(result.position.shape)  # (B, T, 3)

    """

    def __init__(
        self,
        model_io: TrajectoryBoundModelIO,
        device: torch.device,
        norm_scale_xyz: tuple[float, float, float] = COURT_COORD_SCALE_XYZ,
        court_keypoint_contract: CourtKeypointContract | str = "physical_v1",
    ) -> None:
        """Initialize the predictor.

        Use load_from_checkpoint to create instances in most cases.

        Args:
            model: Initialized BLCS model.
            device: Inference device.

        """
        self.model_io = model_io
        self.model = model_io.model.to(device)
        self.io_adapter = cast("TrajectoryModelIOAdapter", model_io.adapter)
        self.device = device
        self.norm_scale_xyz = norm_scale_xyz
        self.court_keypoint_contract = (
            court_keypoint_contract
            if isinstance(court_keypoint_contract, CourtKeypointContract)
            else resolve_court_keypoint_contract(court_keypoint_contract)
        )
        self.model.eval()

    @property
    def input_profile(self) -> str:
        """Return the adapter-declared input profile without model class checks."""
        return str(self.io_adapter.input_profile)

    @classmethod
    def load_from_checkpoint(
        cls,
        checkpoint_path: str | Path | Iterable[str | Path],
        *,
        resolver: PathResolver,
        device: str | torch.device,
        court_keypoints: CourtKeypointContract | str | None = None,
        **kwargs: Any,
    ) -> Self:
        """Create a BLCSPredictor from a checkpoint file.

        Args:
            checkpoint_path: Path to checkpoint file (.ckpt).
            device: Inference device.
            **kwargs: Forwarded to `BLCSLightningModule.load_from_checkpoint`.

        Returns:
            Initialized BLCSPredictor instance.

        Raises:
            FileNotFoundError: If checkpoint file does not exist.

        """
        checkpoints = cls._ensure_checkpoint(checkpoint_path, resolver=resolver)
        if len(checkpoints) != 1:
            raise ValueError(
                f"{cls.__name__} expects exactly one checkpoint, got {len(checkpoints)}."
            )
        checkpoint_runtime = load_checkpoint_runtime(
            checkpoints[0],
            runtime_court_keypoints=court_keypoints,
        )
        binding = compose_blcs_trajectory_model_io(checkpoint_runtime.config)
        if "config" in kwargs:
            raise TypeError(
                "BLCSPredictor.load_from_checkpoint owns checkpoint config "
                "restoration; do not pass config in kwargs."
            )
        lightning_module, resolved_device = cls._load_single_lightning_module(
            checkpoints[0],
            BLCSLightningModule,
            resolver=resolver,
            device=device,
            model_io=binding,
            strict=True,
            weights_only=False,
            config=checkpoint_runtime.config,
            **kwargs,
        )
        return cls(
            model_io=lightning_module.model_io,
            device=resolved_device,
            court_keypoint_contract=checkpoint_runtime.court_keypoint_contract,
        )

    def _prediction_provenance(
        self,
        batch: Mapping[str, object],
        *,
        batch_size: int,
        explicit: tuple[CourtReferenceFrameProvenance, ...] | None = None,
        reference_metadata: BLCSReferenceMetadata | None = None,
    ) -> tuple[CourtReferenceFrameProvenance, ...]:
        batch_provenance = batch.get("court_reference_provenance")
        if (
            explicit is not None
            and batch_provenance is not None
            and (
                not isinstance(batch_provenance, (tuple, list))
                or tuple(batch_provenance) != explicit
            )
        ):
            raise ModelInputContractError(
                "Explicit BLCS prediction provenance does not match the batch."
            )
        raw = explicit if explicit is not None else batch_provenance
        if reference_metadata is not None:
            metadata_provenance = tuple(
                selection.provenance for selection in reference_metadata.selections
            )
            if raw is not None and (
                not isinstance(raw, (tuple, list))
                or tuple(raw) != metadata_provenance
            ):
                raise ModelInputContractError(
                    "BLCS prediction provenance and typed reference metadata do "
                    "not match."
                )
            raw = metadata_provenance
        if raw is None:
            if self.court_keypoint_contract.selector != PHYSICAL_V1_SELECTOR:
                raise ValueError(
                    "camera_view_v2 prediction requires explicit reference-frame provenance."
                )
            return tuple(build_physical_court_provenance() for _ in range(batch_size))
        if not isinstance(raw, (tuple, list)) or len(raw) != batch_size:
            raise ValueError(
                "Prediction provenance must contain exactly one record per batch item."
            )
        records = tuple(raw)
        if any(
            not isinstance(record, CourtReferenceFrameProvenance)
            for record in records
        ):
            raise TypeError("Prediction provenance entries must be validated records.")
        typed = records
        if any(
            record.contract_id != self.court_keypoint_contract.contract_id
            for record in typed
        ):
            raise CourtKeypointContractMismatchError(
                "Prediction provenance does not match the predictor CourtKP contract."
            )
        return typed

    @staticmethod
    def _same_reference_metadata(
        left: BLCSReferenceMetadata,
        right: BLCSReferenceMetadata,
    ) -> bool:
        """Compare typed metadata without tensor truth-value coercion."""
        return (
            left.selections == right.selections
            and left.stable_camera_id_tables == right.stable_camera_id_tables
            and left.track_query_contract == right.track_query_contract
            and torch.equal(left.reference_view_index, right.reference_view_index)
            and torch.equal(left.view_camera_ids, right.view_camera_ids)
            and torch.equal(left.reference_camera_id, right.reference_camera_id)
            and torch.equal(
                left.reference_from_physical,
                right.reference_from_physical,
            )
            and torch.equal(
                left.physical_from_reference,
                right.physical_from_reference,
            )
        )

    def _resolve_reference_metadata(
        self,
        batch: Mapping[str, object],
        explicit: BLCSReferenceMetadata | None,
    ) -> BLCSReferenceMetadata | None:
        parsed = blcs_reference_metadata_from_batch(batch)
        if explicit is not None and parsed is not None and not self._same_reference_metadata(
            explicit,
            parsed,
        ):
            raise ModelInputContractError(
                "Explicit BLCS reference metadata does not match the batch."
            )
        metadata = explicit if explicit is not None else parsed
        if self.court_keypoint_contract.selector == PHYSICAL_V1_SELECTOR:
            if metadata is not None:
                raise ModelInputContractError(
                    "physical_v1 BLCS inference cannot consume reference metadata."
                )
        elif metadata is None:
            raise ModelInputContractError(
                "camera_view_v2 BLCS inference requires explicit typed "
                "reference_metadata."
            )
        return metadata

    def _validate_direct_contract(
        self,
        document: Mapping[str, object] | None,
    ) -> None:
        validate_model_artifact_court_keypoint_contract(
            {} if document is None else document,
            self.court_keypoint_contract,
            location="BLCS direct inference input",
        )

    def predict_batch(
        self,
        batch: Mapping[str, object],
        *,
        denormalize: bool,
        court_reference_provenance: tuple[
            CourtReferenceFrameProvenance, ...
        ]
        | None = None,
        reference_metadata: BLCSReferenceMetadata | None = None,
    ) -> BLCSTrajectoryPrediction:
        """Validate, execute, and decode one typed trajectory batch."""
        metadata = self._resolve_reference_metadata(batch, reference_metadata)
        model_batch = dict(batch)
        if metadata is not None:
            model_batch.update(metadata.to_batch_fields())
        moved = {
            key: value.to(self.device) if isinstance(value, Tensor) else value
            for key, value in model_batch.items()
        }
        with torch.no_grad():
            prediction = self.model_io.run(moved)
        position = prediction.position
        velocity = prediction.velocity
        provenance = self._prediction_provenance(
            batch,
            batch_size=int(position.shape[0]),
            explicit=court_reference_provenance,
            reference_metadata=metadata,
        )
        if denormalize:
            position = self._denormalize_coords(position, self.norm_scale_xyz)
            if velocity is not None:
                velocity = self._denormalize_coords(velocity, self.norm_scale_xyz)
        return BLCSTrajectoryPrediction(
            position=position.detach().cpu(),
            velocity=None if velocity is None else velocity.detach().cpu(),
            court_reference_provenance=provenance,
            coordinates_in_metres=denormalize,
            reference_metadata=metadata.cpu() if metadata is not None else None,
        )

    def predict_multiview_arrays(
        self,
        *,
        ball_uv: NDArray[np.float32],
        court_kp: NDArray[np.float32],
        ball_vis: NDArray[np.bool_],
        court_vis: NDArray[np.bool_] | NDArray[np.float32],
        denormalize: bool,
        court_keypoint_document: Mapping[str, object] | None = None,
        court_reference_provenance: tuple[
            CourtReferenceFrameProvenance, ...
        ]
        | None = None,
        reference_metadata: BLCSReferenceMetadata | None = None,
    ) -> BLCSTrajectoryPrediction:
        """Build and predict one explicit multiview scene-array window."""
        if self.input_profile != "multiview":
            raise ValueError(
                "predict_multiview_arrays requires a multiview BLCS checkpoint."
            )
        self._validate_direct_contract(court_keypoint_document)
        batch = self.io_adapter.build_inference_batch_from_arrays(
            ball_uv=ball_uv,
            court_kp=court_kp,
            ball_vis=ball_vis,
            court_vis=court_vis,
        )
        return self.predict_batch(
            batch,
            denormalize=denormalize,
            court_reference_provenance=court_reference_provenance,
            reference_metadata=reference_metadata,
        )

    def predict_scene(
        self,
        scene: Mapping[str, object],
        cameras: list[int],
        *,
        denormalize: bool,
        reference_camera_id: str | None = None,
    ) -> BLCSTrajectoryPrediction:
        """Build the selected profile from a scene and return a typed decode."""
        scene_contract = scene.get("court_keypoint_contract")
        if scene_contract is None:
            if self.court_keypoint_contract.selector != PHYSICAL_V1_SELECTOR:
                raise ValueError(
                    "camera_view_v2 scene inference requires validated CourtKP metadata."
                )
        elif scene_contract != self.court_keypoint_contract:
            raise CourtKeypointContractMismatchError(
                "Scene and predictor CourtKP contracts do not match."
            )
        inference_scene: Mapping[str, object] = scene
        if self.court_keypoint_contract.selector == PHYSICAL_V1_SELECTOR:
            if reference_camera_id is not None:
                raise ModelInputContractError(
                    "physical_v1 scene inference must not specify a reference camera."
                )
            provenance = (build_physical_court_provenance(),)
            reference_metadata = None
        else:
            if reference_camera_id is None:
                raise ValueError(
                    "camera_view_v2 scene inference requires reference_camera_id."
                )
            raw_cameras = scene.get("cameras")
            if not isinstance(raw_cameras, list):
                raise ValueError("scene.cameras must be a list.")
            complete_views = tuple(
                camera.get("court_view")
                if isinstance(camera, Mapping)
                else None
                for camera in raw_cameras
            )
            if any(not isinstance(view, CourtViewRecord) for view in complete_views):
                raise ValueError(
                    "camera_view_v2 scene cameras require validated metadata."
                )
            typed_complete_views = cast(
                "tuple[CourtViewRecord, ...]",
                complete_views,
            )
            selected_views: list[CourtViewRecord] = []
            selected_cameras: list[dict[str, object]] = []
            for camera_index in cameras:
                raw_camera = raw_cameras[camera_index]
                if not isinstance(raw_camera, Mapping):
                    raise TypeError("Each scene camera must be a mapping.")
                view = typed_complete_views[camera_index]
                selected_views.append(view)
                selected_cameras.append(dict(raw_camera))
            table = StableCameraIdTable.from_complete_scene_camera_ids(
                tuple(view.camera_id for view in typed_complete_views)
            )
            selection = ReferenceViewSelection.create(
                stable_camera_id_table=table,
                selected_views=tuple(selected_views),
                reference_camera_id=reference_camera_id,
            )
            frame = selection.provenance
            reference_view = selected_views[selection.reference_view_index]
            for camera, source_view in zip(
                selected_cameras, selected_views, strict=True
            ):
                for key, axis in (("court_kp_uv", 0), ("court_kp_vis", 0)):
                    aligned = align_court_keypoints_to_reference(
                        np.asarray(camera[key]),
                        source_view,
                        reference_view,
                        keypoint_axis=axis,
                    )
                    camera[key] = aligned
            inference_scene = {**scene, "cameras": selected_cameras}
            cameras = list(range(len(selected_cameras)))
            provenance = (frame,)
            fields = selection.to_tensor_fields(dtype=torch.float32)
            forward = fields["reference_from_physical"].unsqueeze(0)
            reference_metadata = BLCSReferenceMetadata(
                selections=(selection,),
                stable_camera_id_tables=(table,),
                reference_view_index=fields["reference_view_index"].unsqueeze(0),
                view_camera_ids=fields["view_camera_ids"].unsqueeze(0),
                reference_camera_id=fields["reference_camera_id"].unsqueeze(0),
                reference_from_physical=forward,
                physical_from_reference=forward.transpose(-1, -2),
            )
        batch = self.io_adapter.build_inference_batch_from_scene(
            inference_scene,
            cameras,
        )
        return self.predict_batch(
            batch,
            denormalize=denormalize,
            court_reference_provenance=provenance,
            reference_metadata=reference_metadata,
        )

    def predict(
        self,
        ball_uv: Tensor,
        court_kp: Tensor,
        ball_vis: Tensor,
        padding_mask: Tensor,
        court_vis: Tensor,
        denormalize: bool = True,
        court_keypoint_document: Mapping[str, object] | None = None,
        court_reference_provenance: tuple[
            CourtReferenceFrameProvenance, ...
        ]
        | None = None,
        reference_metadata: BLCSReferenceMetadata | None = None,
    ) -> BLCSTrajectoryPrediction:
        """Predict and return the adapter's typed trajectory decode.

        Args:
            ball_uv: Ball 2D trajectory tensor accepted by the loaded model.
            court_kp: Court keypoint tensor accepted by the loaded model.
            ball_vis: Ball visibility tensor.
            padding_mask: Ball padding tensor where ``True`` marks padding.
            court_vis: Court keypoint visibility tensor.
            denormalize: If True, convert positions to meters.

        """
        self._validate_direct_contract(court_keypoint_document)
        return self.predict_batch(
            {
                "ball_uv": ball_uv,
                "court_kp": court_kp,
                "ball_vis": ball_vis,
                "padding_mask": padding_mask,
                "court_vis": court_vis,
            },
            denormalize=denormalize,
            court_reference_provenance=court_reference_provenance,
            reference_metadata=reference_metadata,
        )
