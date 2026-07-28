"""
Run one or more court-alignment jobs through the four immutable stages.

Usage:
    python -m src.synthetic_data_generation.scripts.alignment.run_pipeline jobs=b00 stages=all

Notes:
    - Hydra loads `src/synthetic_data_generation/configs/alignment/pipeline.yaml`.
    - Jobs execute serially because line inference shares one GPU.
    - Resume always strict-loads artifacts and verifies provider and upstream identity.
"""

from __future__ import annotations

import hashlib
import json
import sys
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

from hydra.utils import to_absolute_path
from omegaconf import DictConfig, ListConfig, OmegaConf

from src.synthetic_data_generation.alignment.artifacts.acceptance_decision import (
    load_alignment_acceptance_decision,
    verify_machine_evidence,
)
from src.synthetic_data_generation.alignment.artifacts.calibration import (
    load_calibration_artifact,
)
from src.synthetic_data_generation.alignment.artifacts.common import (
    canonical_json_bytes,
)
from src.synthetic_data_generation.alignment.artifacts.court_geometry import (
    load_court_geometry_artifact,
)
from src.synthetic_data_generation.alignment.artifacts.ground_line_map import (
    load_ground_line_map_artifact,
)
from src.synthetic_data_generation.alignment.artifacts.holdout_validation import (
    load_holdout_validation_artifact,
)
from src.synthetic_data_generation.alignment.scene_provider.bundle import (
    load_scene_provider_bundle,
    sha256_file,
)
from src.synthetic_data_generation.scene_contract import load_scene_contract
from src.synthetic_data_generation.scripts.alignment import (
    calibrate_court_alignment,
    finalize_court_alignment,
    fit_ground_courts,
    infer_ground_line_map,
)
from src.synthetic_data_generation.scripts.alignment.common import (
    AlignmentJob,
    AlignmentStageError,
    StageResult,
    directory_artifact_handle,
    find_matching_artifact,
    json_artifact_handle,
    write_json_summary,
)
from src.utils.hydra import hydra_main

_STAGE_ORDER = ("ground_line", "court_fit", "calibration", "finalization")
_STAGE_ALIASES = {
    "ground": "ground_line",
    "ground_line": "ground_line",
    "fit": "court_fit",
    "court_fit": "court_fit",
    "calibrate": "calibration",
    "calibration": "calibration",
    "finalize": "finalization",
    "finalization": "finalization",
}


@hydra_main(
    version_base="1.3",
    config_path="../../configs/alignment",
    config_name="pipeline",
)
def main(cfg: DictConfig) -> int:
    """Run the configured pipeline and print its strict summary path."""
    summary, summary_path = run(cfg)
    print(summary_path)
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True, allow_nan=False))
    if summary["status"] == "failed":
        raise SystemExit(1)
    return 0


def run(cfg: DictConfig) -> tuple[dict[str, Any], Path]:
    """Execute configured alignment jobs serially and write one summary."""
    if int(cfg.max_parallel_jobs) != 1:
        raise ValueError(
            "Initial alignment pipeline supports max_parallel_jobs=1 only."
        )
    repo_root = Path(to_absolute_path(".")).resolve()
    job_names = _parse_names(cfg.jobs, name="jobs")
    selected_stages = _parse_stages(cfg.stages)
    catalog = cfg.job_catalog
    if not isinstance(catalog, DictConfig):
        raise TypeError("job_catalog must be a mapping.")
    results: list[dict[str, Any]] = []
    for job_name in job_names:
        if job_name not in catalog:
            raise ValueError(f"Unknown alignment job {job_name!r}.")
        raw_job = catalog[job_name]
        if not isinstance(raw_job, DictConfig):
            raise TypeError(f"job_catalog.{job_name} must be a mapping.")
        job = _alignment_job(raw_job, repo_root=repo_root)
        result = _run_job(
            job_name,
            job,
            raw_job,
            selected_stages=selected_stages,
            resume=bool(cfg.resume),
            repo_root=repo_root,
        )
        results.append(result)
        if bool(cfg.fail_fast) and result["status"] in {
            "failed",
            "fit_calibration_failed",
        }:
            break

    statuses = {str(result["status"]) for result in results}
    if any(status in {"failed", "fit_calibration_failed"} for status in statuses):
        pipeline_status = "failed"
    elif any(status == "rejected" for status in statuses):
        pipeline_status = "completed_with_rejections"
    else:
        pipeline_status = "completed"
    created_at = datetime.now(UTC).isoformat()
    summary: dict[str, Any] = {
        "pipeline_id": str(cfg.pipeline_id),
        "created_at_utc": created_at,
        "status": pipeline_status,
        "execution": {
            "jobs": list(job_names),
            "stages": list(selected_stages),
            "resume": bool(cfg.resume),
            "fail_fast": bool(cfg.fail_fast),
            "max_parallel_jobs": 1,
        },
        "jobs": results,
    }
    summary["summary_fingerprint"] = hashlib.sha256(
        canonical_json_bytes(summary)
    ).hexdigest()
    summary_path = _path(cfg.summary_path)
    write_json_summary(summary_path, summary)
    return summary, summary_path


def _run_job(
    job_name: str,
    job: AlignmentJob,
    raw_job: DictConfig,
    *,
    selected_stages: tuple[str, ...],
    resume: bool,
    repo_root: Path,
) -> dict[str, Any]:
    selected = set(selected_stages)
    last_selected_index = max(_STAGE_ORDER.index(stage) for stage in selected_stages)
    required = set(_STAGE_ORDER[: last_selected_index + 1])
    stage_states = {stage: "skipped" for stage in _STAGE_ORDER}
    stage_results: dict[str, StageResult] = {}
    bundle = load_scene_provider_bundle(job.provider_bundle, verify_files=False)
    holdout_group_ids = tuple(int(value) for value in raw_job.holdout_group_ids)

    ground_cfg = _stage_config(repo_root, "infer_ground_line_map")
    ground_cfg.provider_bundle = str(job.provider_bundle)
    ground_cfg.output_dir = str(job.output_root / "ground_line_maps")
    ground_cfg.holdout_group_ids = list(holdout_group_ids)
    _merge_stage_override(ground_cfg, raw_job, "ground_line")
    try:
        ground_result, was_resumed = _resolve_ground_line(
            ground_cfg,
            bundle_fingerprint=bundle.manifest.bundle_fingerprint,
            holdout_group_ids=holdout_group_ids,
            allow_execute="ground_line" in selected,
            resume=resume,
            repo_root=repo_root,
        )
    except Exception as error:
        return _failed_job(
            job,
            stage_states,
            failed_stage="ground_line",
            error=error,
        )
    stage_results["ground_line"] = ground_result
    stage_states["ground_line"] = (
        "resumed"
        if was_resumed and "ground_line" in selected
        else "skipped"
        if was_resumed
        else "executed"
    )
    ground_path = _required_primary(ground_result)
    ground_manifest, _ = load_ground_line_map_artifact(ground_path)
    ground_handle = directory_artifact_handle(ground_path, ground_manifest)
    if "court_fit" not in required:
        return _partial_job(job, stage_states, stage_results)

    fit_cfg = _stage_config(repo_root, "fit_ground_courts")
    fit_cfg.ground_line_artifact = str(ground_path)
    fit_cfg.output_dir = str(job.output_root / "court_geometry")
    _merge_stage_override(fit_cfg, raw_job, "court_fit")
    try:
        fit_result, was_resumed = _resolve_court_fit(
            fit_cfg,
            ground_fingerprint=ground_handle.fingerprint,
            allow_execute="court_fit" in selected,
            resume=resume,
            repo_root=repo_root,
        )
    except Exception as error:
        return _failed_job(
            job,
            stage_states,
            failed_stage="court_fit",
            error=error,
        )
    stage_results["court_fit"] = fit_result
    stage_states["court_fit"] = (
        "resumed"
        if was_resumed and "court_fit" in selected
        else "skipped"
        if was_resumed
        else "executed"
    )
    geometry_path = _required_primary(fit_result)
    geometry = load_court_geometry_artifact(geometry_path)
    geometry_handle = json_artifact_handle(geometry_path, geometry)
    if "calibration" not in required:
        return _partial_job(job, stage_states, stage_results)

    calibration_cfg = _stage_config(repo_root, "calibrate_court_alignment")
    calibration_cfg.alignment_id = job.alignment_id
    calibration_cfg.provider_bundle = str(job.provider_bundle)
    calibration_cfg.ground_line_artifact = str(ground_path)
    calibration_cfg.ground_line_fingerprint = ground_handle.fingerprint
    calibration_cfg.geometry_artifact = str(geometry_path)
    calibration_cfg.geometry_file_sha256 = geometry_handle.file_sha256
    calibration_cfg.geometry_fingerprint = geometry_handle.fingerprint
    calibration_cfg.output_dir = str(job.output_root / "calibration")
    calibration_cfg.holdout_group_ids = list(holdout_group_ids)
    _merge_stage_override(calibration_cfg, raw_job, "calibration")
    try:
        calibration_result, was_resumed = _resolve_calibration(
            calibration_cfg,
            provider_fingerprint=bundle.manifest.bundle_fingerprint,
            geometry_fingerprint=geometry_handle.fingerprint,
            holdout_group_ids=holdout_group_ids,
            allow_execute="calibration" in selected,
            resume=resume,
            repo_root=repo_root,
        )
    except AlignmentStageError as error:
        stage_states["calibration"] = "failed"
        stage_states["finalization"] = "blocked"
        return {
            "alignment_id": job.alignment_id,
            "scene_id": job.scene_id,
            "status": "fit_calibration_failed",
            "scene_contract": None,
            "stages": stage_states,
            "preserved_artifacts": [str(path) for path in error.preserved_artifacts],
            "error": _error_record(error),
        }
    except Exception as error:
        return _failed_job(
            job,
            stage_states,
            failed_stage="calibration",
            error=error,
        )
    stage_results["calibration"] = calibration_result
    stage_states["calibration"] = (
        "resumed"
        if was_resumed and "calibration" in selected
        else "skipped"
        if was_resumed
        else "executed"
    )
    calibration_path = _required_primary(calibration_result)
    calibration = load_calibration_artifact(calibration_path)
    calibration_handle = json_artifact_handle(calibration_path, calibration)
    if calibration["status"] != "fit_calibration_passed":
        stage_states["finalization"] = "blocked"
        return {
            "alignment_id": job.alignment_id,
            "scene_id": job.scene_id,
            "status": "fit_calibration_failed",
            "scene_contract": None,
            "stages": stage_states,
            "preserved_artifacts": [str(calibration_path)],
            "error": None,
        }
    if "finalization" not in required:
        return _partial_job(job, stage_states, stage_results)

    final_cfg = _stage_config(repo_root, "finalize_court_alignment")
    final_cfg.alignment_id = job.alignment_id
    final_cfg.scene_id = job.scene_id
    final_cfg.provider_bundle = str(job.provider_bundle)
    final_cfg.ground_line_artifact = str(ground_path)
    final_cfg.ground_line_fingerprint = ground_handle.fingerprint
    final_cfg.calibration_artifact = str(calibration_path)
    final_cfg.calibration_file_sha256 = calibration_handle.file_sha256
    final_cfg.calibration_fingerprint = calibration_handle.fingerprint
    final_cfg.output_dir = str(job.output_root / "holdout_validation")
    final_cfg.holdout_group_ids = list(holdout_group_ids)
    final_cfg.scene_contract_path = str(
        job.provider_bundle / Path(str(final_cfg.scene_contract_path)).name
    )
    final_cfg.override.provider_bundle_fingerprint = bundle.manifest.bundle_fingerprint
    final_cfg.override.decision_output_dir = str(
        job.output_root / "acceptance_decisions"
    )
    final_cfg.override.scene_contract_path = str(
        job.provider_bundle / Path(str(final_cfg.override.scene_contract_path)).name
    )
    if "override" in raw_job:
        final_cfg.override.enabled = bool(raw_job.override.enabled)
    _merge_stage_override(final_cfg, raw_job, "finalization")
    try:
        final_result, was_resumed = _resolve_finalization(
            final_cfg,
            provider_fingerprint=bundle.manifest.bundle_fingerprint,
            calibration_fingerprint=calibration_handle.fingerprint,
            holdout_group_ids=holdout_group_ids,
            allow_execute="finalization" in selected,
            resume=resume,
            repo_root=repo_root,
        )
    except Exception as error:
        return _failed_job(
            job,
            stage_states,
            failed_stage="finalization",
            error=error,
        )
    stage_results["finalization"] = final_result
    stage_states["finalization"] = (
        "resumed"
        if was_resumed and "finalization" in selected
        else "skipped"
        if was_resumed
        else "executed"
    )
    outcome = str(final_result.metadata.get("outcome", final_result.status))
    scene_contract = final_result.metadata.get("scene_contract")
    return {
        "alignment_id": job.alignment_id,
        "scene_id": job.scene_id,
        "status": outcome,
        "scene_contract": scene_contract,
        "stages": stage_states,
        "artifacts": _stage_artifacts(stage_results),
        "error": None,
    }


def _resolve_ground_line(
    cfg: DictConfig,
    *,
    bundle_fingerprint: str,
    holdout_group_ids: tuple[int, ...],
    allow_execute: bool,
    resume: bool,
    repo_root: Path,
) -> tuple[StageResult, bool]:
    if resume:
        output_dir = _path(cfg.output_dir)
        match = find_matching_artifact(
            tuple(output_dir.glob(f"{cfg.artifact_id}-*"))
            if output_dir.is_dir()
            else (),
            load=lambda path: load_ground_line_map_artifact(path)[0],
            matches=lambda payload: (
                payload["provider"]["bundle_fingerprint"] == bundle_fingerprint
                and payload["split"]["holdout_group_ids"] == list(holdout_group_ids)
                and payload["ground_plane"]["fit_settings"]
                == _plain_mapping(cfg.ground_plane)
                and payload["projection"]["settings"]
                == _plain_mapping(cfg.line_projection)
                and payload["detector"]["checkpoint_sha256"]
                == str(cfg.line_checkpoint_sha256)
                and payload["detector"]["backbone_checkpoint_sha256"]
                == str(cfg.backbone_checkpoint_sha256)
                and _provenance_matches(payload, repo_root=repo_root)
            ),
        )
        if match is not None:
            path, manifest = match
            handle = directory_artifact_handle(path, manifest)
            return _resumed_result("ground_line", handle), True
    if not allow_execute:
        raise FileNotFoundError("No resumable ground-line artifact matched the job.")
    return infer_ground_line_map.run(cfg), False


def _resolve_court_fit(
    cfg: DictConfig,
    *,
    ground_fingerprint: str,
    allow_execute: bool,
    resume: bool,
    repo_root: Path,
) -> tuple[StageResult, bool]:
    if resume:
        output_dir = _path(cfg.output_dir)
        match = find_matching_artifact(
            tuple(output_dir.glob(f"{cfg.artifact_id}-*.json"))
            if output_dir.is_dir()
            else (),
            load=load_court_geometry_artifact,
            matches=lambda payload: (
                payload["ground_line_artifact"]["artifact_fingerprint"]
                == ground_fingerprint
                and payload["fit_settings"] == _plain_mapping(cfg.fit)
                and _provenance_matches(payload, repo_root=repo_root)
            ),
        )
        if match is not None:
            path, payload = match
            return _resumed_result(
                "court_fit",
                json_artifact_handle(path, payload),
            ), True
    if not allow_execute:
        raise FileNotFoundError("No resumable court-fit artifact matched the job.")
    return fit_ground_courts.run(cfg), False


def _resolve_calibration(
    cfg: DictConfig,
    *,
    provider_fingerprint: str,
    geometry_fingerprint: str,
    holdout_group_ids: tuple[int, ...],
    allow_execute: bool,
    resume: bool,
    repo_root: Path,
) -> tuple[StageResult, bool]:
    if resume:
        output_dir = _path(cfg.output_dir)
        match = find_matching_artifact(
            tuple(output_dir.glob(f"{cfg.artifact_id}-*.json"))
            if output_dir.is_dir()
            else (),
            load=load_calibration_artifact,
            matches=lambda payload: (
                payload["provider"]["bundle_fingerprint"] == provider_fingerprint
                and payload["geometry"]["artifact_fingerprint"] == geometry_fingerprint
                and payload["split"]["holdout_group_ids"] == list(holdout_group_ids)
                and payload["evaluation_settings"] == _plain_mapping(cfg.evaluation)
                and payload["gates"]["fit"] == _plain_mapping(cfg.fit_gates)
                and payload["gates"]["holdout_frozen"]
                == _plain_mapping(cfg.holdout_gates)
                and _provenance_matches(payload, repo_root=repo_root)
            ),
        )
        if match is not None:
            path, payload = match
            result = _resumed_result(
                "calibration",
                json_artifact_handle(path, payload),
                metadata={"calibration_status": payload["status"]},
            )
            return result, True
    if not allow_execute:
        raise FileNotFoundError("No resumable calibration artifact matched the job.")
    return calibrate_court_alignment.run(cfg), False


def _resolve_finalization(
    cfg: DictConfig,
    *,
    provider_fingerprint: str,
    calibration_fingerprint: str,
    holdout_group_ids: tuple[int, ...],
    allow_execute: bool,
    resume: bool,
    repo_root: Path,
) -> tuple[StageResult, bool]:
    if resume:
        output_dir = _path(cfg.output_dir)
        match = find_matching_artifact(
            tuple(output_dir.glob(f"{cfg.artifact_id}-*.json"))
            if output_dir.is_dir()
            else (),
            load=load_holdout_validation_artifact,
            matches=lambda payload: (
                payload["provider"]["bundle_fingerprint"] == provider_fingerprint
                and payload["calibration"]["artifact_fingerprint"]
                == calibration_fingerprint
                and payload["split"]["holdout_group_ids"] == list(holdout_group_ids)
                and _provenance_matches(payload, repo_root=repo_root)
            ),
        )
        if match is not None:
            path, payload = match
            resumed = _resumed_final_result(
                cfg,
                path=path,
                validation=payload,
                repo_root=repo_root,
            )
            if resumed is not None:
                return resumed, True
    if not allow_execute:
        raise FileNotFoundError("No resumable finalization artifact matched the job.")
    return finalize_court_alignment.run(cfg), False


def _resumed_final_result(
    cfg: DictConfig,
    *,
    path: Path,
    validation: Mapping[str, Any],
    repo_root: Path,
) -> StageResult | None:
    handle = json_artifact_handle(path, validation)
    status = str(validation["status"])
    if status == "accepted":
        contract_path = _path(cfg.scene_contract_path)
        if not _contract_references(contract_path, artifact_path=path):
            return None
        return _resumed_result(
            "finalization",
            handle,
            artifact_paths=(path, contract_path),
            metadata={"outcome": "accepted", "scene_contract": str(contract_path)},
        )
    override = cfg.get("override")
    if not isinstance(override, DictConfig) or not bool(override.get("enabled", False)):
        return _resumed_result(
            "finalization",
            handle,
            metadata={"outcome": "rejected", "scene_contract": None},
        )
    decision_dir = _path(override.decision_output_dir)
    decision_id = str(override.decision_id)
    for decision_path in sorted(decision_dir.glob(f"{decision_id}-*.json")):
        decision, fingerprint = load_alignment_acceptance_decision(decision_path)
        if decision.holdout_validation.sha256 != str(sha256_file(path)):
            continue
        calibration = load_calibration_artifact(_path(cfg.calibration_artifact))
        verify_machine_evidence(
            decision,
            calibration=calibration,
            holdout_validation=validation,
        )
        contract_path = _path(override.scene_contract_path)
        if not _contract_references(contract_path, artifact_path=decision_path):
            continue
        return _resumed_result(
            "finalization",
            handle,
            artifact_paths=(path, decision_path, contract_path),
            metadata={
                "outcome": "accepted_by_user_override",
                "scene_contract": str(contract_path),
                "decision_fingerprint": fingerprint,
            },
        )
    return None


def _contract_references(contract_path: Path, *, artifact_path: Path) -> bool:
    if not contract_path.is_file():
        return False
    contract = load_scene_contract(contract_path)
    return contract.alignment is not None and contract.alignment.manifest.sha256 == str(
        sha256_file(artifact_path)
    )


def _resumed_result(
    stage: str,
    handle: Any,
    *,
    artifact_paths: tuple[Path, ...] | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> StageResult:
    merged_metadata = {"artifact": handle.to_dict(), **dict(metadata or {})}
    return StageResult(
        stage=stage,
        status="resumed",
        artifact_paths=artifact_paths or (handle.path,),
        primary_artifact=handle.path,
        fingerprint=handle.fingerprint,
        metadata=merged_metadata,
    )


def _alignment_job(raw: DictConfig, *, repo_root: Path) -> AlignmentJob:
    overrides = OmegaConf.to_container(raw, resolve=True)
    if not isinstance(overrides, dict):
        raise TypeError("Alignment job config must resolve to a mapping.")
    return AlignmentJob(
        alignment_id=str(raw.alignment_id),
        scene_id=str(raw.scene_id),
        provider_bundle=_path(raw.provider_bundle),
        output_root=_path(raw.output_root),
        config_overrides=cast(dict[str, Any], overrides),
    )


def _stage_config(repo_root: Path, name: str) -> DictConfig:
    path = (
        repo_root / "src/synthetic_data_generation/configs/alignment" / f"{name}.yaml"
    )
    loaded = OmegaConf.load(path)
    if not isinstance(loaded, DictConfig):
        raise TypeError(f"Stage config {path} must be a mapping.")
    OmegaConf.set_struct(loaded, False)
    return loaded


def _merge_stage_override(
    stage_cfg: DictConfig,
    raw_job: DictConfig,
    stage: str,
) -> None:
    overrides = raw_job.get("stage_overrides")
    if not isinstance(overrides, DictConfig) or stage not in overrides:
        return
    stage_override = overrides[stage]
    if not isinstance(stage_override, DictConfig):
        raise TypeError(f"stage_overrides.{stage} must be a mapping.")
    stage_cfg.merge_with(stage_override)


def _provenance_matches(
    payload: Mapping[str, Any],
    *,
    repo_root: Path,
) -> bool:
    provenance = payload.get("provenance")
    if not isinstance(provenance, Mapping):
        return False
    records: list[Mapping[str, Any]] = []
    code_files = provenance.get("code_files")
    if isinstance(code_files, Sequence) and not isinstance(code_files, (str, bytes)):
        records.extend(item for item in code_files if isinstance(item, Mapping))
    config = provenance.get("config")
    if isinstance(config, Mapping):
        records.append(config)
    code = provenance.get("code")
    if isinstance(code, Mapping):
        script = code.get("script")
        if isinstance(script, Mapping):
            records.append(script)
    if not records:
        return False
    for record in records:
        path_value = record.get("path")
        expected = record.get("sha256")
        if not isinstance(path_value, str) or not isinstance(expected, str):
            return False
        path = Path(path_value)
        resolved = path if path.is_absolute() else repo_root / path
        if not resolved.is_file() or str(sha256_file(resolved)) != expected:
            return False
    return True


def _parse_stages(value: Any) -> tuple[str, ...]:
    names = _parse_names(value, name="stages")
    if names == ("all",):
        return _STAGE_ORDER
    normalized: list[str] = []
    for name in names:
        if name not in _STAGE_ALIASES:
            raise ValueError(f"Unknown alignment stage {name!r}.")
        stage = _STAGE_ALIASES[name]
        if stage not in normalized:
            normalized.append(stage)
    if not normalized:
        raise ValueError("At least one stage must be selected.")
    return tuple(stage for stage in _STAGE_ORDER if stage in normalized)


def _parse_names(value: Any, *, name: str) -> tuple[str, ...]:
    if isinstance(value, str):
        names = tuple(item.strip() for item in value.split(",") if item.strip())
    elif isinstance(value, (list, tuple, ListConfig)):
        names = tuple(str(item).strip() for item in value if str(item).strip())
    else:
        raise TypeError(f"{name} must be a comma-separated string or sequence.")
    if not names:
        raise ValueError(f"{name} must not be empty.")
    return names


def _plain_mapping(value: Any) -> dict[str, Any]:
    raw = OmegaConf.to_container(value, resolve=True)
    if not isinstance(raw, dict):
        raise TypeError("Expected a mapping config.")
    return cast(dict[str, Any], raw)


def _stage_artifacts(results: Mapping[str, StageResult]) -> dict[str, list[str]]:
    return {
        stage: [str(path) for path in result.artifact_paths]
        for stage, result in results.items()
    }


def _partial_job(
    job: AlignmentJob,
    stages: Mapping[str, str],
    results: Mapping[str, StageResult],
) -> dict[str, Any]:
    return {
        "alignment_id": job.alignment_id,
        "scene_id": job.scene_id,
        "status": "completed_partial",
        "scene_contract": None,
        "stages": dict(stages),
        "artifacts": _stage_artifacts(results),
        "error": None,
    }


def _failed_job(
    job: AlignmentJob,
    stages: Mapping[str, str],
    *,
    failed_stage: str,
    error: Exception,
) -> dict[str, Any]:
    failed = dict(stages)
    failed[failed_stage] = "failed"
    failed_index = _STAGE_ORDER.index(failed_stage)
    for stage in _STAGE_ORDER[failed_index + 1 :]:
        failed[stage] = "blocked"
    preserved = (
        [str(path) for path in error.preserved_artifacts]
        if isinstance(error, AlignmentStageError)
        else []
    )
    return {
        "alignment_id": job.alignment_id,
        "scene_id": job.scene_id,
        "status": "failed",
        "scene_contract": None,
        "stages": failed,
        "preserved_artifacts": preserved,
        "error": _error_record(error),
    }


def _error_record(error: Exception) -> dict[str, str]:
    return {"type": type(error).__name__, "message": str(error)}


def _required_primary(result: StageResult) -> Path:
    primary = result.primary_artifact
    if not isinstance(primary, Path):
        raise ValueError(f"Stage {result.stage} returned no primary artifact.")
    return primary


def _path(value: Any) -> Path:
    return Path(to_absolute_path(str(value))).resolve()


def _normalize_csv_overrides(argv: list[str]) -> None:
    """Quote design-level CSV overrides before Hydra parses sweep syntax."""
    for index, argument in enumerate(argv):
        key, separator, value = argument.partition("=")
        if (
            separator
            and key in {"jobs", "stages"}
            and "," in value
            and not value.startswith(("[", "'", '"'))
        ):
            argv[index] = f"{key}='{value}'"


if __name__ == "__main__":
    _normalize_csv_overrides(sys.argv)
    cast(Any, main)()
