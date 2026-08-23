"""Profile one explicit Court query-model composition under a fixed input contract.

Usage:
    python -m src.tasks.court_detection.scripts.profile_query_model
    python -m src.tasks.court_detection.scripts.profile_query_model model/decoder=query_dpt_base profile.candidate.family=dpt

Notes:
    - Hydra loads ``src/tasks/court_detection/configs/profile_query_model.yaml``.
    - GPU latency and peak memory are adoption evidence; CPU mode must be explicitly
      enabled and is labeled diagnostic-only.
    - This script profiles one model and never starts training or bypasses the
      repository training queue.
"""

from __future__ import annotations

import torch
from omegaconf import DictConfig

from src.synthetic_data_generation.dataset.court.schema import (
    COURT_SEMANTIC_CLASS_NAMES_V3,
)
from src.tasks.court_detection.configuration import CourtQueryLossConfig
from src.tasks.court_detection.data.contracts import (
    CourtTargetBundleSpec,
    CourtTargetKind,
    CourtTargetSpec,
)
from src.tasks.court_detection.experiments.configuration import (
    QueryProfileConfig,
    validate_profile_boundary,
)
from src.tasks.court_detection.model_io.adapters import (
    CourtQueryDINOv3ExecutionBoundary,
    CourtQueryModelIOAdapter,
)
from src.tasks.court_detection.model_io.contracts import CourtQueryModelSpec
from src.tasks.court_detection.models.query_encoder.model import CourtQueryEncoderModel
from src.tasks.court_detection.models.query_encoder.profiling import (
    profile_query_model,
)
from src.utils.hydra import hydra_main, register_boundary_validator
from src.utils.io import save_json_atomic

_BOUNDARY = "court_detection.profile_query_model"
register_boundary_validator(_BOUNDARY, validate_profile_boundary)


def _target_bundle(config: QueryProfileConfig) -> CourtTargetBundleSpec:
    specs: dict[CourtTargetKind, CourtTargetSpec] = {}
    for kind in config.model.heads.dense_targets:
        if kind == "kp":
            specs[kind] = CourtTargetSpec(
                kind="kp",
                schema="synthetic_camera_view_kp14_v3_target_court:gaussian_max_v1",
                output_channels=14,
                channel_names=tuple(COURT_SEMANTIC_CLASS_NAMES_V3),
                target_dtype=torch.float32,
                precomputed=False,
            )
        elif kind == "seg":
            specs[kind] = CourtTargetSpec(
                kind="seg",
                schema="court_cell_segmentation_v1",
                output_channels=7,
                channel_names=(
                    "background",
                    "service_left",
                    "service_right",
                    "back_left",
                    "back_right",
                    "doubles_left",
                    "doubles_right",
                ),
                target_dtype=torch.long,
                precomputed=True,
            )
        else:
            specs[kind] = CourtTargetSpec(
                kind="line",
                schema="court_line_binary_v1",
                output_channels=1,
                channel_names=("court_line",),
                target_dtype=torch.float32,
                precomputed=True,
            )
    return CourtTargetBundleSpec(specs)


@hydra_main(
    version_base="1.3",
    config_path="../configs",
    config_name="profile_query_model",
    validation_boundary=_BOUNDARY,
)
def main(cfg: DictConfig) -> None:
    """Load and profile exactly one configured query-model candidate."""
    runtime = QueryProfileConfig.from_config(cfg)
    if runtime.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("profile.device=cuda requires an available CUDA device.")
    device = torch.device(runtime.device)
    bundle = _target_bundle(runtime)
    model = CourtQueryEncoderModel.from_config(runtime.model, bundle).to(device)
    if not isinstance(runtime.loss, CourtQueryLossConfig):  # pragma: no cover
        raise TypeError("Query profiler requires a query loss contract.")
    adapter = CourtQueryModelIOAdapter(
        CourtQueryModelSpec(
            target_bundle=bundle,
            in_channels=runtime.channels,
            short_side=runtime.height,
        ),
        execution_boundary=CourtQueryDINOv3ExecutionBoundary(
            frozen_backbone=model.backbone.frozen_execution
        ),
        loss_config=runtime.loss,
    ).to(device)
    adapter.validate_model_pair(model)
    images = torch.zeros(
        runtime.batch_size,
        runtime.channels,
        runtime.height,
        runtime.width,
        device=device,
        dtype=torch.float32,
    )
    record = profile_query_model(
        model,
        adapter,
        images,
        family=runtime.candidate_family,
        size=runtime.candidate_size,
        warmup=runtime.warmup,
        repeats=runtime.repeats,
    )
    save_json_atomic(record, runtime.output_path)
    print(f"Saved Court query profile to {runtime.output_path}")


if __name__ == "__main__":
    main()
