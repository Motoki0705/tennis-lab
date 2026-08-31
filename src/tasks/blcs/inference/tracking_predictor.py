"""Inference wrapper for lifecycle-aware multi-ball track queries."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any, Self, cast

import torch
from torch import Tensor

from src.tasks.base.generate_dataset import (
    PHYSICAL_V1_SELECTOR,
    CourtKeypointContract,
    CourtKeypointContractMismatchError,
    CourtReferenceFrameProvenance,
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
    BLCSTrackQueryPrediction,
    TrackQueryBoundModelIO,
    blcs_reference_metadata_from_batch,
    compose_blcs_track_query_model_io,
)
from src.tasks.blcs.model_io.adapters import (
    TrackQueryModelIOAdapter,
    TrackQueryReferenceModelIOAdapter,
)
from src.tasks.blcs.model_io.checkpoints import load_checkpoint_runtime
from src.tasks.blcs.training.tracking_lightning_module import (
    BLCSTrackingLightningModule,
)
from src.utils.configuration import PathResolver
from src.utils.schema.court import COURT_COORD_SCALE_XYZ


class BLCSTrackingPredictor(BasePredictor[BLCSTrackQueryPrediction]):
    """Predict fixed queries from caller-provided per-camera slot observations."""

    def __init__(
        self,
        model_io: TrackQueryBoundModelIO,
        device: torch.device,
        court_keypoint_contract: CourtKeypointContract | str = "physical_v1",
    ) -> None:
        self.model_io = model_io
        if not isinstance(model_io.adapter, TrackQueryModelIOAdapter):
            raise TypeError("BLCSTrackingPredictor requires TrackQueryModelIOAdapter.")
        self.num_queries = model_io.adapter.num_queries
        self.model = model_io.model.to(device).eval()
        self.device = device
        self.court_keypoint_contract = (
            court_keypoint_contract
            if isinstance(court_keypoint_contract, CourtKeypointContract)
            else resolve_court_keypoint_contract(court_keypoint_contract)
        )

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
        checkpoints = cls._ensure_checkpoint(checkpoint_path, resolver=resolver)
        if len(checkpoints) != 1:
            raise ValueError(
                f"{cls.__name__} expects exactly one checkpoint, got {len(checkpoints)}."
            )
        checkpoint_runtime = load_checkpoint_runtime(
            checkpoints[0],
            runtime_court_keypoints=court_keypoints,
        )
        binding = compose_blcs_track_query_model_io(checkpoint_runtime.config)
        if "config" in kwargs:
            raise TypeError(
                "BLCSTrackingPredictor.load_from_checkpoint owns checkpoint "
                "config restoration; do not pass config in kwargs."
            )
        lightning_module, resolved_device = cls._load_single_lightning_module(
            checkpoints[0],
            BLCSTrackingLightningModule,
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
        explicit: tuple[CourtReferenceFrameProvenance, ...] | None,
        reference_metadata: BLCSReferenceMetadata | None,
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
        if any(
            record.contract_id != self.court_keypoint_contract.contract_id
            for record in records
        ):
            raise CourtKeypointContractMismatchError(
                "Prediction provenance does not match the predictor CourtKP contract."
            )
        return records

    @staticmethod
    def _same_reference_metadata(
        left: BLCSReferenceMetadata,
        right: BLCSReferenceMetadata,
    ) -> bool:
        """Compare typed metadata without relying on tensor dataclass equality."""
        return cast(
            "bool",
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
            ),
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
        if isinstance(self.model_io.adapter, TrackQueryReferenceModelIOAdapter):
            if metadata is None:
                raise ModelInputContractError(
                    "Reference-conditioned BLCS tracking inference requires "
                    "explicit typed reference_metadata."
                )
            if (
                metadata.track_query_contract
                != self.model_io.adapter.track_query_reference_contract
            ):
                raise ModelInputContractError(
                    "BLCS tracking reference metadata and adapter contracts do "
                    "not exactly match."
                )
        elif metadata is not None:
            raise ModelInputContractError(
                "Legacy BLCS tracking inference cannot consume v2 reference "
                "metadata."
            )
        return metadata

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
    ) -> BLCSTrackQueryPrediction:
        """Run one validated tracking call and return its typed decode."""
        with torch.no_grad():
            metadata = self._resolve_reference_metadata(batch, reference_metadata)
            model_batch = dict(batch)
            if metadata is not None:
                model_batch.update(metadata.to_batch_fields())
            moved = {
                key: value.to(self.device) if isinstance(value, Tensor) else value
                for key, value in model_batch.items()
            }
            prediction = self.model_io.run(moved)
            position = prediction.position
            provenance = self._prediction_provenance(
                batch,
                batch_size=int(position.shape[0]),
                explicit=court_reference_provenance,
                reference_metadata=metadata,
            )
            if denormalize:
                position = self._denormalize_coords(position, COURT_COORD_SCALE_XYZ)
            return BLCSTrackQueryPrediction(
                position=position.detach().cpu(),
                presence_logits=prediction.presence_logits.detach().cpu(),
                presence_probability=prediction.presence_probability.detach().cpu(),
                presence=prediction.presence.detach().cpu(),
                court_reference_provenance=provenance,
                coordinates_in_metres=denormalize,
                reference_metadata=metadata.cpu() if metadata is not None else None,
            )

    def predict(
        self,
        *,
        ball_uv: Tensor,
        ball_vis: Tensor,
        court_kp: Tensor,
        court_vis: Tensor,
        padding_mask: Tensor,
        denormalize: bool,
        court_keypoint_document: Mapping[str, object] | None = None,
        court_reference_provenance: tuple[
            CourtReferenceFrameProvenance, ...
        ]
        | None = None,
        reference_metadata: BLCSReferenceMetadata | None = None,
    ) -> BLCSTrackQueryPrediction:
        """Pad an explicit short candidate set, then run the strict adapter."""
        validate_model_artifact_court_keypoint_contract(
            {} if court_keypoint_document is None else court_keypoint_document,
            self.court_keypoint_contract,
            location="BLCS tracking direct inference input",
        )
        if ball_uv.ndim != 5:
            raise ValueError("ball_uv must have shape (B,V,T,P,2).")
        if ball_vis.shape != ball_uv.shape[:-1]:
            raise ValueError("ball_vis must match ball_uv without UV.")
        candidate_width = int(ball_uv.shape[3])
        if candidate_width > self.num_queries:
            raise ValueError(
                "Inference candidates exceed model.num_queries "
                f"({candidate_width} > {self.num_queries})."
            )
        if candidate_width < self.num_queries:
            padding_width = self.num_queries - candidate_width
            uv_padding = torch.zeros(
                *ball_uv.shape[:3],
                padding_width,
                2,
                dtype=ball_uv.dtype,
                device=ball_uv.device,
            )
            mask_padding = torch.zeros(
                *ball_vis.shape[:3],
                padding_width,
                dtype=torch.bool,
                device=ball_vis.device,
            )
            ball_uv = torch.cat((ball_uv, uv_padding), dim=3)
            ball_vis = torch.cat((ball_vis, mask_padding), dim=3)
        inputs = {
            "ball_uv": ball_uv,
            "ball_vis": ball_vis,
            "court_kp": court_kp,
            "court_vis": court_vis,
            "padding_mask": padding_mask,
        }
        return self.predict_batch(
            inputs,
            denormalize=denormalize,
            court_reference_provenance=court_reference_provenance,
            reference_metadata=reference_metadata,
        )


__all__ = ["BLCSTrackingPredictor"]
