"""Inference boundary for a once-bound PLCS track-query model."""

from __future__ import annotations

from collections.abc import Iterable
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
    PLCSTrackingBoundModelIO,
    PLCSTrackQueryIOAdapter,
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
        court_reference_provenance: CourtReferenceFrameProvenance | None = None,
    ) -> dict[str, Tensor]:
        """Return query position/rotation and lifecycle presence outputs."""
        with torch.no_grad():
            effective_provenance = court_reference_provenance
            if (
                effective_provenance is None
                and self.court_keypoint_contract.selector == PHYSICAL_V1_SELECTOR
            ):
                effective_provenance = build_physical_court_provenance()
            prepared = PLCSPreparedBatch(
                call=self.io_adapter.build_call(
                    {
                        "human_kp": human_kp,
                        "human_vis": human_vis,
                        "court_kp": court_kp,
                        "court_vis": court_vis,
                        "padding_mask": padding_mask,
                        "court_keypoint_metadata": court_keypoint_metadata,
                        "court_reference_provenance": effective_provenance,
                    }
                ),
                court_reference_provenance=(
                    (effective_provenance,)
                    if effective_provenance is not None
                    else None
                ),
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
            result = {
                "position": decoded.position,
                "rotation": decoded.rotation,
                "presence_logits": presence_logits,
                "presence_probability": probability,
                "presence": probability >= tracking_metrics.presence_threshold,
            }
            if denormalize:
                if effective_provenance is None:
                    raise ModelInputContractError(
                        "PLCS tracking physical output requires provenance."
                    )
                result["position_meters"] = normalized_points_target_to_physical(
                    decoded.position,
                    effective_provenance,
                )
                physical_heading = headings_target_to_physical(
                    decoded.rotation,
                    effective_provenance,
                )
                result["yaw_radians"] = torch.atan2(
                    physical_heading[..., 1], physical_heading[..., 0]
                )
            return {key: value.detach().cpu() for key, value in result.items()}


__all__ = ["PLCSTrackingPredictor"]
