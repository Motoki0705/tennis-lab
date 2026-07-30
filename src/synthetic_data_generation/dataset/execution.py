"""Execute generic path-driven synthetic-data stages."""

from __future__ import annotations

import json
import math
import shutil
import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

from src.synthetic_data_generation.dataset.pipeline import PathPipelineManifest
from src.synthetic_data_generation.visualization.summary import (
    write_pipeline_visualization,
)
from src.utils.io import save_json_atomic


def _load_json_object(path: Path, *, description: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"{description} does not exist: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"{description} is malformed JSON: {path}") from error
    if not isinstance(value, dict):
        raise TypeError(f"{description} must be a JSON object: {path}")
    return value


def _finite_numbers(value: object, *, name: str) -> list[float]:
    if not isinstance(value, list) or not value:
        raise TypeError(f"{name} must be a non-empty list.")
    numbers: list[float] = []
    for item in value:
        if (
            not isinstance(item, (int, float))
            or isinstance(item, bool)
            or not math.isfinite(float(item))
        ):
            raise TypeError(f"{name} must contain only finite numbers.")
        numbers.append(float(item))
    return numbers


def compute_alignment_metrics(manifest: PathPipelineManifest) -> Path:
    """Compute alignment measurements without turning quality into a gate."""
    observations = _load_json_object(
        manifest.alignment_observations,
        description="Alignment observations",
    )
    residuals = _finite_numbers(
        observations.get("residuals"),
        name="alignment residuals",
    )
    absolute = [abs(item) for item in residuals]
    return cast(
        Path,
        save_json_atomic(
            {
                "source": str(manifest.alignment_observations),
                "observation_count": len(residuals),
                "mean_absolute_error": sum(absolute) / len(absolute),
                "root_mean_square_error": math.sqrt(
                    sum(item * item for item in residuals) / len(residuals)
                ),
                "maximum_absolute_error": max(absolute),
            },
            manifest.alignment_metrics,
        ),
    )


def _resolve_job_path(value: object, *, root: Path, name: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise TypeError(f"{name} must be a non-empty path string.")
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = root / path
    return path.resolve()


def _string_list(value: object, *, name: str) -> list[str]:
    if not isinstance(value, list) or not all(
        isinstance(item, str) and item for item in value
    ):
        raise TypeError(f"{name} must be a list of non-empty strings.")
    return value


def build_dataset_plan(
    manifest: PathPipelineManifest,
    *,
    renderer_command: Sequence[str],
) -> Path:
    """Resolve configured job paths and write the plan consumed by rendering."""
    if not all(isinstance(item, str) and item for item in renderer_command):
        raise TypeError("renderer.command must contain non-empty strings.")
    payload = _load_json_object(manifest.render_jobs, description="Render jobs")
    raw_jobs = payload.get("jobs")
    if not isinstance(raw_jobs, list) or not raw_jobs:
        raise TypeError("Render jobs must contain a non-empty jobs list.")
    jobs: list[dict[str, object]] = []
    names: set[str] = set()
    for index, raw_job in enumerate(raw_jobs):
        if not isinstance(raw_job, Mapping):
            raise TypeError(f"Render job {index} must be a JSON object.")
        allowed = {"name", "input", "output", "reference", "arguments"}
        unexpected = set(raw_job) - allowed
        if unexpected:
            raise ValueError(
                f"Render job {index} has unexpected fields: {sorted(unexpected)}."
            )
        name = raw_job.get("name")
        if not isinstance(name, str) or not name.strip():
            raise TypeError(f"Render job {index} requires a non-empty name.")
        if name in names:
            raise ValueError(f"Render job names must be unique: {name!r}.")
        names.add(name)
        input_path = _resolve_job_path(
            raw_job.get("input"),
            root=manifest.source_root,
            name=f"jobs[{index}].input",
        )
        if not input_path.is_file():
            raise FileNotFoundError(f"Render input does not exist: {input_path}")
        output_path = _resolve_job_path(
            raw_job.get("output"),
            root=manifest.dataset_root,
            name=f"jobs[{index}].output",
        )
        reference_value = raw_job.get("reference")
        reference_path: Path | None = None
        if reference_value is not None:
            reference_path = _resolve_job_path(
                reference_value,
                root=manifest.source_root,
                name=f"jobs[{index}].reference",
            )
            if not reference_path.is_file():
                raise FileNotFoundError(
                    f"Quality reference does not exist: {reference_path}"
                )
        jobs.append(
            {
                "name": name,
                "input": str(input_path),
                "output": str(output_path),
                "reference": (None if reference_path is None else str(reference_path)),
                "arguments": _string_list(
                    raw_job.get("arguments", []),
                    name=f"jobs[{index}].arguments",
                ),
            }
        )
    return cast(
        Path,
        save_json_atomic(
            {
                "path_manifest": str(manifest.pipeline_manifest),
                "alignment_metrics": str(manifest.alignment_metrics),
                "renderer_command": list(renderer_command),
                "jobs": jobs,
            },
            manifest.dataset_plan,
        ),
    )


def _substitute_paths(
    value: str,
    *,
    manifest: PathPipelineManifest,
    job: Mapping[str, object],
) -> str:
    replacements = {
        "{input}": str(job["input"]),
        "{output}": str(job["output"]),
        "{source_root}": str(manifest.source_root),
        "{artifact_root}": str(manifest.artifact_root),
        "{dataset_root}": str(manifest.dataset_root),
    }
    reference = job.get("reference")
    if isinstance(reference, str):
        replacements["{reference}"] = reference
    result = value
    for placeholder, replacement in replacements.items():
        result = result.replace(placeholder, replacement)
    return result


def render_dataset(
    manifest: PathPipelineManifest,
    *,
    working_directory: Path,
) -> Path:
    """Run the configured renderer and record its path-based outputs."""
    plan = _load_json_object(manifest.dataset_plan, description="Dataset plan")
    renderer_command = _string_list(
        plan.get("renderer_command"),
        name="dataset plan renderer_command",
    )
    raw_jobs = plan.get("jobs")
    if not isinstance(raw_jobs, list) or not raw_jobs:
        raise TypeError("Dataset plan must contain a non-empty jobs list.")
    log_root = manifest.execution_root / "renderer-logs"
    log_root.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, object]] = []
    for index, raw_job in enumerate(raw_jobs):
        if not isinstance(raw_job, dict):
            raise TypeError(f"Dataset plan job {index} must be a JSON object.")
        name = raw_job.get("name")
        input_value = raw_job.get("input")
        output_value = raw_job.get("output")
        if (
            not isinstance(name, str)
            or not isinstance(input_value, str)
            or not isinstance(output_value, str)
        ):
            raise TypeError(f"Dataset plan job {index} has invalid paths or name.")
        input_path = Path(input_value)
        output_path = Path(output_value)
        if not input_path.is_file():
            raise FileNotFoundError(f"Render input does not exist: {input_path}")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        arguments = _string_list(
            raw_job.get("arguments"),
            name=f"dataset plan job {index} arguments",
        )
        stdout = ""
        stderr = ""
        if renderer_command:
            command = [
                _substitute_paths(item, manifest=manifest, job=raw_job)
                for item in (*renderer_command, *arguments)
            ]
            completed = subprocess.run(
                command,
                cwd=working_directory,
                check=False,
                capture_output=True,
                text=True,
            )
            stdout = completed.stdout
            stderr = completed.stderr
            if completed.returncode != 0:
                raise RuntimeError(
                    f"Renderer failed for job {name!r} with exit code "
                    f"{completed.returncode}: {stderr.strip()}"
                )
            returncode = completed.returncode
        else:
            shutil.copyfile(input_path, output_path)
            command = []
            returncode = 0
        if not output_path.is_file():
            raise RuntimeError(
                f"Renderer did not produce configured output: {output_path}"
            )
        stdout_path = log_root / f"{index:04d}-{name}.stdout.log"
        stderr_path = log_root / f"{index:04d}-{name}.stderr.log"
        stdout_path.write_text(stdout, encoding="utf-8")
        stderr_path.write_text(stderr, encoding="utf-8")
        records.append(
            {
                "name": name,
                "input": str(input_path),
                "output": str(output_path),
                "reference": raw_job.get("reference"),
                "command": command,
                "returncode": returncode,
                "stdout": str(stdout_path),
                "stderr": str(stderr_path),
            }
        )
    return cast(
        Path,
        save_json_atomic(
            {"dataset_plan": str(manifest.dataset_plan), "renders": records},
            manifest.render_manifest,
        ),
    )


def _mean_absolute_byte_error(left: bytes, right: bytes) -> float:
    width = max(len(left), len(right))
    if width == 0:
        return 0.0
    padded_left = left.ljust(width, b"\0")
    padded_right = right.ljust(width, b"\0")
    total = sum(abs(a - b) for a, b in zip(padded_left, padded_right, strict=True))
    return total / (255.0 * width)


def compute_quality_metrics(manifest: PathPipelineManifest) -> Path:
    """Compute render metrics and publish them regardless of their values."""
    render_payload = _load_json_object(
        manifest.render_manifest,
        description="Render manifest",
    )
    raw_renders = render_payload.get("renders")
    if not isinstance(raw_renders, list) or not raw_renders:
        raise TypeError("Render manifest must contain a non-empty renders list.")
    records: list[dict[str, object]] = []
    measured: list[float] = []
    for index, raw_render in enumerate(raw_renders):
        if not isinstance(raw_render, Mapping):
            raise TypeError(f"Render record {index} must be a JSON object.")
        name = raw_render.get("name")
        output_value = raw_render.get("output")
        if not isinstance(name, str) or not isinstance(output_value, str):
            raise TypeError(f"Render record {index} has invalid name or output.")
        output = Path(output_value)
        if not output.is_file():
            raise FileNotFoundError(f"Rendered output does not exist: {output}")
        reference_value = raw_render.get("reference")
        error: float | None = None
        if reference_value is not None:
            if not isinstance(reference_value, str):
                raise TypeError(f"Render record {index} reference is invalid.")
            reference = Path(reference_value)
            if not reference.is_file():
                raise FileNotFoundError(
                    f"Quality reference does not exist: {reference}"
                )
            error = _mean_absolute_byte_error(
                output.read_bytes(),
                reference.read_bytes(),
            )
            measured.append(error)
        records.append(
            {
                "name": name,
                "output": str(output),
                "reference": reference_value,
                "mean_absolute_byte_error": error,
            }
        )
    mean_error = None if not measured else sum(measured) / len(measured)
    return cast(
        Path,
        save_json_atomic(
            {
                "render_manifest": str(manifest.render_manifest),
                "render_count": len(records),
                "measured_render_count": len(measured),
                "mean_absolute_byte_error": mean_error,
                "renders": records,
            },
            manifest.quality_metrics,
        ),
    )


def execute_pipeline(
    manifest: PathPipelineManifest,
    *,
    renderer_command: Sequence[str],
    working_directory: Path,
) -> Path:
    """Run every stage and return the generated visualization path."""
    compute_alignment_metrics(manifest)
    build_dataset_plan(manifest, renderer_command=renderer_command)
    render_dataset(manifest, working_directory=working_directory.resolve())
    compute_quality_metrics(manifest)
    return write_pipeline_visualization(manifest)
