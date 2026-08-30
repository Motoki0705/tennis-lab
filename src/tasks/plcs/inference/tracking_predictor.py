"""Inference boundary for a once-bound PLCS track-query model."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any, Self

import torch
from torch import Tensor, nn

from src.tasks.base.generate_dataset import (
    PHYSICAL_V1_SELECTOR,
    CourtKeypointContract,
    CourtReferenceFrameProvenance,
    build_physical_court_provenance,
)
from src.tasks.base.inference.predictor import BasePredictor
from src.tasks.base.model_io import ModelCall, ModelInputContractError
from src.tasks.base.training.tracking_metrics import TrackingMetricConfig
from src.tasks.plcs.court_keypoint_contract import (
    headings_target_to_physical,
    normalized_points_target_to_physical,
)
from src.tasks.plcs.model_io import (
    PLCSPreparedBatch,
    PLCSReferenceMetadata,
    PLCSTrackingBoundModelIO,
    PLCSTrackQueryIOAdapter,
    PLCSTrackQueryReferenceIOAdapter,
    bind_plcs_model_io,
    prepare_plcs_checkpoint_court_keypoint_config,
)
from src.tasks.plcs.training.tracking_lightning_module import (
    PLCSTrackingLightningModule,
)
from src.utils.configuration import PathResolver
from src.utils.schema.court_normalization import load_and_validate_checkpoint


class PLCSTrackingPredictor(BasePredictor):
    """Predict fixed lifecycle queries through the track-query adapter."""

    def __init__(
        self,
        *,
        model: nn.Module,
        adapter: PLCSTrackQueryIOAdapter,
        device: torch.device,
        court_keypoint_contract: CourtKeypointContract | None = None,
    ) -> None:
        bound = bind_plcs_model_io(model, adapter)
        self.model_io: PLCSTrackingBoundModelIO = bound
        self.model = self.model_io.model.to(device).eval()
        self.io_adapter = adapter
        self.device = device
        self.court_keypoint_contract = (
            court_keypoint_contract or adapter.court_keypoint_contract
        )
        if self.court_keypoint_contract != adapter.court_keypoint_contract:
            raise ModelInputContractError(
                "PLCS tracking predictor and adapter CourtKP20 contracts do not match."
            )

    @classmethod
    def load_from_checkpoint(
        cls,
        checkpoint_path: str | Path | Iterable[str | Path],
        *,
        resolver: PathResolver,
        device: str | torch.device,
        court_keypoint_contract: CourtKeypointContract | None = None,
        **kwargs: Any,
    ) -> Self:
        checkpoints = cls._ensure_checkpoint(checkpoint_path, resolver=resolver)
        if len(checkpoints) != 1:
            raise ValueError(
                f"{cls.__name__} expects a single checkpoint, "
                f"got {len(checkpoints)} checkpoints."
            )
        if "config" in kwargs:
            raise TypeError(
                "PLCSTrackingPredictor restores checkpoint config internally; "
                "do not pass config."
            )
        checkpoint = load_and_validate_checkpoint(checkpoints[0])
        checkpoint_config, keypoint_contract = (
            prepare_plcs_checkpoint_court_keypoint_config(
                checkpoint,
                court_keypoint_contract,
                location=str(checkpoints[0]),
            )
        )
        lightning_module, resolved_device = cls._load_single_lightning_module(
            checkpoints[0],
            PLCSTrackingLightningModule,
            resolver=resolver,
            device=device,
            config=checkpoint_config,
            strict=bool(kwargs.pop("strict", True)),
            weights_only=bool(kwargs.pop("weights_only", False)),
            **kwargs,
        )
        adapter = lightning_module.io_adapter
        if not isinstance(adapter, PLCSTrackQueryIOAdapter):
            raise ModelInputContractError(
                "Loaded checkpoint does not contain a PLCS track-query adapter."
            )
        return cls(
            model=lightning_module.model,
            adapter=adapter,
            device=resolved_device,
            court_keypoint_contract=keypoint_contract,
        )

    def predict(
        self,
        *,
        human_kp: Tensor,
        human_vis: Tensor,
        court_kp: Tensor,
        court_vis: Tensor,
        padding_mask: Tensor,
        tracking_metrics: TrackingMetricConfig,
        denormalize: bool,
        court_keypoint_metadata: dict[str, object] | None = None,
        court_reference_provenance: CourtReferenceFrameProvenance
        | Sequence[CourtReferenceFrameProvenance]
        | None = None,
        reference_metadata: PLCSReferenceMetadata | None = None,
    ) -> dict[str, object]:
        """Return query tracking outputs and optional per-query canonical pose."""
        with torch.no_grad():
            if isinstance(self.io_adapter, PLCSTrackQueryReferenceIOAdapter):
                if reference_metadata is None:
                    raise ModelInputContractError(
                        "Reference-conditioned PLCS tracking inference requires "
                        "explicit typed reference_metadata."
                    )
                if (
                    reference_metadata.track_query_contract
                    != self.io_adapter.reference_contract
                ):
                    raise ModelInputContractError(
                        "PLCS tracking reference metadata and adapter contracts "
                        "do not exactly match."
                    )
            elif reference_metadata is not None:
                raise ModelInputContractError(
                    "Legacy PLCS tracking inference cannot consume v2 reference "
                    "metadata."
                )
            metadata_provenance = (
                tuple(selection.provenance for selection in reference_metadata.selections)
                if reference_metadata is not None
                else None
            )
            effective_provenance = court_reference_provenance
            if metadata_provenance is not None:
                if effective_provenance is not None:
                    explicit = (
                        (effective_provenance,)
                        if isinstance(
                            effective_provenance,
                            CourtReferenceFrameProvenance,
                        )
                        else tuple(effective_provenance)
                    )
                    if explicit != metadata_provenance:
                        raise ModelInputContractError(
                            "PLCS tracking provenance and typed reference metadata "
                            "do not match."
                        )
                effective_provenance = metadata_provenance
            if (
                effective_provenance is None
                and self.court_keypoint_contract.selector == PHYSICAL_V1_SELECTOR
            ):
                effective_provenance = build_physical_court_provenance()
            model_batch: dict[str, object] = {
                "human_kp": human_kp,
                "human_vis": human_vis,
                "court_kp": court_kp,
                "court_vis": court_vis,
                "padding_mask": padding_mask,
                "court_keypoint_metadata": court_keypoint_metadata,
                "court_reference_provenance": effective_provenance,
            }
            if reference_metadata is not None:
                model_batch.update(reference_metadata.to_batch_fields())
            prepared_provenance: tuple[CourtReferenceFrameProvenance, ...] | None
            if isinstance(
                effective_provenance,
                CourtReferenceFrameProvenance,
            ):
                prepared_provenance = (effective_provenance,)
            elif effective_provenance is None:
                prepared_provenance = None
            else:
                prepared_provenance = tuple(effective_provenance)
            prepared = PLCSPreparedBatch(
                call=self.io_adapter.build_call(model_batch),
                court_reference_provenance=prepared_provenance,
                reference_metadata=reference_metadata,
            )
            moved = ModelCall(
                kwargs={
                    key: value.to(self.device) if isinstance(value, Tensor) else None
                    for key, value in prepared.call.kwargs.items()
                }
            )
            raw_output = self.model_io.execute_call(moved)
            decoded = self.io_adapter.decode_prepared_output(raw_output, prepared)
            presence_logits = decoded.presence_logits
            probability = presence_logits.sigmoid()
            frame_valid = (~padding_mask).any(dim=1).to(probability.device)
            result: dict[str, object] = {
                "position": decoded.position,
                "rotation": decoded.rotation,
                "presence_logits": presence_logits,
                "presence_probability": probability,
                "presence": (
                    probability >= tracking_metrics.presence_threshold
                )
                & frame_valid.unsqueeze(-1),
            }
            if decoded.canonical_pose is not None:
                result["canonical_pose"] = decoded.canonical_pose
            if denormalize:
                provenance = prepared.court_reference_provenance
                if provenance is None:
                    raise ModelInputContractError(
                        "PLCS tracking physical output requires provenance."
                    )
                physical_position, physical_heading = self._physical_outputs(
                    decoded.position,
                    decoded.rotation,
                    provenance,
                )
                result["position_meters"] = physical_position
                result["yaw_radians"] = torch.atan2(
                    physical_heading[..., 1], physical_heading[..., 0]
                )
            cpu_result = {
                key: value.detach().cpu() if isinstance(value, Tensor) else value
                for key, value in result.items()
            }
            if reference_metadata is not None:
                cpu_result["reference_metadata"] = reference_metadata.cpu()
            return cpu_result

    def _physical_outputs(
        self,
        position: Tensor,
        rotation: Tensor,
        provenance: Sequence[CourtReferenceFrameProvenance],
    ) -> tuple[Tensor, Tensor]:
        records = tuple(provenance)
        if not records:
            raise ModelInputContractError(
                "PLCS tracking physical output provenance must not be empty."
            )
        if len(records) == 1:
            return (
                normalized_points_target_to_physical(
                    position,
                    records[0],
                ),
                headings_target_to_physical(rotation, records[0]),
            )
        if position.shape[0] != len(records):
            raise ModelInputContractError(
                "PLCS tracking prediction batch and provenance cardinality do "
                "not match."
            )
        return (
            torch.stack(
                [
                    normalized_points_target_to_physical(
                        position[index],
                        record,
                    )
                    for index, record in enumerate(records)
                ]
            ),
            torch.stack(
                [
                    headings_target_to_physical(rotation[index], record)
                    for index, record in enumerate(records)
                ]
            ),
        )


__all__ = ["PLCSTrackingPredictor"]
