"""Resumable orchestration for manifest-defined evaluation jobs."""

from __future__ import annotations

import hashlib
import json
import traceback
from dataclasses import asdict
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

from src.tasks.ball_detection.evaluation.configuration import load_data_config
from src.tasks.ball_detection.evaluation.contracts import (
    DatasetSpec,
    EvaluationManifest,
    ModelSpec,
)
from src.tasks.ball_detection.evaluation.dataset_provenance import sha256_file
from src.tasks.ball_detection.evaluation.evaluator import JobEvaluator
from src.tasks.ball_detection.evaluation.reporting import (
    write_comparison_reports,
)
from src.utils.configuration import PathRole
from src.utils.io import ensure_dir, load_json, save_json

_EVALUATOR_SCHEMA = "ball_detection_evaluator_v1"


class EvaluationPipeline:
    """Execute, resume, and report all jobs in one evaluation manifest."""

    def __init__(
        self,
        manifest: EvaluationManifest,
        *,
        evaluator: JobEvaluator,
    ) -> None:
        self.manifest = manifest
        self.output_dir = ensure_dir(manifest.output_dir)
        self.results_dir = ensure_dir(self.output_dir / "results")
        self.resume = manifest.resume
        self.fail_fast = manifest.fail_fast
        self.evaluator = evaluator
        self._checkpoint_hashes: dict[Path, str] = {}
        self._dataset_fingerprints: dict[tuple[str, str], dict[str, Any]] = {}

    def run(self) -> dict[str, int]:
        """Run unfinished jobs, reuse successful ones, and refresh reports."""
        save_json(_manifest_payload(self.manifest), self.output_dir / "manifest.json")
        results: list[dict[str, Any]] = []
        executed = 0
        reused = 0
        failed = 0

        for model in self.manifest.models:
            if not model.enabled:
                continue
            for dataset_id in model.datasets:
                dataset = self.manifest.datasets[dataset_id]
                for split in dataset.splits:
                    job_id = f"{model.id}--{dataset.id}--{split}"
                    result_path = self.results_dir / f"{job_id}.json"
                    fingerprint = self._job_fingerprint(
                        model=model,
                        dataset=dataset,
                        split=split,
                    )
                    previous = _load_reusable_result(
                        result_path,
                        fingerprint=fingerprint,
                        resume=self.resume,
                    )
                    if previous is not None:
                        results.append(previous)
                        reused += 1
                        continue

                    executed += 1
                    base_result = {
                        "schema": "ball_detection_evaluation_result_v1",
                        "job_id": job_id,
                        "fingerprint": fingerprint,
                        "category": model.category,
                        "model_id": model.id,
                        "model_name": model.expected_model_name,
                        "dataset_id": dataset.id,
                        "split": split,
                    }
                    try:
                        payload = self.evaluator.evaluate(
                            model=model,
                            dataset=dataset,
                            split=split,
                            manifest=self.manifest,
                        )
                        result = {
                            **base_result,
                            "status": "success",
                            "result": payload,
                        }
                    except Exception as error:
                        failed += 1
                        result = {
                            **base_result,
                            "status": "failed",
                            "error": {
                                "type": type(error).__name__,
                                "message": str(error),
                                "traceback": traceback.format_exc(),
                            },
                        }
                        save_json(result, result_path)
                        results.append(result)
                        if self.fail_fast:
                            write_comparison_reports(
                                results,
                                output_dir=self.output_dir,
                            )
                            raise
                        continue

                    save_json(result, result_path)
                    results.append(result)

        write_comparison_reports(results, output_dir=self.output_dir)
        return {
            "jobs": len(results),
            "executed": executed,
            "reused": reused,
            "failed": failed,
        }

    def _job_fingerprint(
        self,
        *,
        model: ModelSpec,
        dataset: DatasetSpec,
        split: str,
    ) -> str:
        checkpoint = model.checkpoint
        if checkpoint.is_file():
            if checkpoint not in self._checkpoint_hashes:
                self._checkpoint_hashes[checkpoint] = sha256_file(checkpoint)
            checkpoint_hash = self._checkpoint_hashes[checkpoint]
        else:
            checkpoint_hash = "missing"
        payload = {
            "evaluator_schema": _EVALUATOR_SCHEMA,
            "manifest_schema": self.manifest.schema,
            "device": self.manifest.device,
            "model": {
                **asdict(model),
                "checkpoint": str(checkpoint),
            },
            "checkpoint_sha256": checkpoint_hash,
            "dataset": self._dataset_fingerprint(dataset, split),
            "split": split,
            "metrics": asdict(self.manifest.metrics),
            "performance": asdict(self.manifest.performance),
        }
        serialized = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        )
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()

    def _dataset_fingerprint(
        self,
        dataset: DatasetSpec,
        split: str,
    ) -> dict[str, Any]:
        key = (dataset.id, split)
        if key in self._dataset_fingerprints:
            return self._dataset_fingerprints[key]

        data_config = load_data_config(
            dataset.config,
            overrides=dataset.overrides,
            resolver=self.manifest.resolver,
        )
        resolved_config = OmegaConf.to_container(data_config, resolve=True)
        source = str(data_config.source)
        data_dir = self.manifest.resolver.resolve(
            PathRole.DATA, str(data_config.data_dir)
        )
        if source == "web":
            split_artifact = data_dir / "manifest.json"
        else:
            split_role = PathRole(str(data_config.split.root_role))
            split_artifact = self.manifest.resolver.resolve(
                split_role, str(data_config.split[f"{split}_file"])
            )
        fingerprint = {
            "spec": asdict(dataset),
            "resolved_config": resolved_config,
            "split_artifact": _file_identity(split_artifact),
        }
        self._dataset_fingerprints[key] = fingerprint
        return fingerprint


def _load_reusable_result(
    path: Path,
    *,
    fingerprint: str,
    resume: bool,
) -> dict[str, Any] | None:
    if not resume or not path.is_file():
        return None
    result = load_json(path)
    if not isinstance(result, dict):
        return None
    if result.get("status") != "success":
        return None
    if result.get("fingerprint") != fingerprint:
        return None
    return result


def _file_identity(path: Path) -> dict[str, str | None]:
    return {
        "path": str(path),
        "sha256": sha256_file(path) if path.is_file() else None,
    }


def _manifest_payload(manifest: EvaluationManifest) -> dict[str, Any]:
    return {
        "schema": manifest.schema,
        "output_dir": str(manifest.output_dir),
        "device": manifest.device,
        "resume": manifest.resume,
        "fail_fast": manifest.fail_fast,
        "metrics": asdict(manifest.metrics),
        "performance": asdict(manifest.performance),
        "datasets": {key: asdict(value) for key, value in manifest.datasets.items()},
        "models": [
            {**asdict(model), "checkpoint": str(model.checkpoint)}
            for model in manifest.models
        ],
    }


__all__ = ["EvaluationPipeline"]
