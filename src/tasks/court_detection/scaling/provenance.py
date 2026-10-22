"""Pin accepted training inventories and real evaluation registries in checkpoints."""

from __future__ import annotations

from collections import Counter
from typing import Any

from src.tasks.court_detection.configuration import SyntheticCourtStoresSourceConfig
from src.tasks.court_detection.data.datamodule import CourtDetectionDataModule
from src.tasks.court_detection.data.inputs.synthetic_court_dataset_version import (
    SyntheticCourtDatasetVersionInput,
)
from src.tasks.court_detection.scaling.real_split import (
    TennisCourtEvaluationInput,
    canonical_sha256,
)
from src.utils.data.image_record_store import file_sha256


def data_condition(dm: CourtDetectionDataModule) -> dict[str, Any]:
    source = dm.mixed_config.sources["synthetic_court"]
    pipeline = dm._train_pipelines["synthetic_court"]
    input_layer = pipeline.input_layer
    identity: dict[str, Any]
    if isinstance(input_layer, SyntheticCourtDatasetVersionInput):
        version = input_layer.dataset_version
        roots = input_layer.store_roots
        identity = {
            "data_version": version["dataset_version"],
            "generation_seed": version["generation_seed"],
            "dataset_version_manifest": {
                "sha256": input_layer.version_manifest_sha256,
                **{
                    key: version[key]
                    for key in (
                        "schema",
                        "request_sha256",
                        "plan_sha256",
                        "method",
                        "recipe_sha256",
                    )
                },
            },
        }
    elif isinstance(source, SyntheticCourtStoresSourceConfig):
        roots = source.store_roots
        identity = {
            "data_version": source.data_version,
            "generation_seed": source.generation_seed,
        }
    else:
        raise ValueError(
            "Data conditions require an explicit dataset version or training stores."
        )
    records = pipeline.input_layer.records("train")
    real = dm._eval_pipelines["tennis_court_detector"].input_layer
    if not isinstance(real, TennisCourtEvaluationInput):
        raise TypeError("Data conditions require a fixed real evaluation registry.")
    return {
        **identity,
        "training_seed": dm.seed,
        "accepted_images": len(records),
        "scene_images": dict(
            sorted(
                Counter(str(record.payload["scene_id"]) for record in records).items()
            )
        ),
        "ordered_training_ids_sha256": canonical_sha256(
            [record.sample_id for record in records]
        ),
        "stores": [
            {
                "descriptor_sha256": file_sha256(root / "dataset.json"),
                "index_sha256": file_sha256(root / "samples/index.npz"),
            }
            for root in roots
        ],
        "evaluation_registry_sha256": real.registry["sha256"],
        "evaluation_registry_version": real.registry["version"],
    }
