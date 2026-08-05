"""Execute generic path-driven synthetic-data stages."""

from __future__ import annotations

import json
import math
import shutil
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from string import Formatter
from typing import Any

from src.synthetic_data_generation.dataset.pipeline import PathPipelineManifest
from src.synthetic_data_generation.visualization.summary import (
    write_pipeline_visualization,
)
from src.utils.configuration import PathContractError, PathRole
from src.utils.io import save_json_atomic

_RENDER_JOB_FIELDS = {"name", "input", "output", "reference", "arguments"}
_PLAN_FIELDS = {"path_manifest", "alignment_metrics", "renderer_command", "jobs"}
_PLAN_JOB_FIELDS = _RENDER_JOB_FIELDS
_RENDER_MANIFEST_FIELDS = {"dataset_plan", "renders"}
_RENDER_RECORD_FIELDS = {
    "name",
    "input",
    "output",
    "reference",
    "command",
    "returncode",
    "stdout",
    "stderr",
}
_PATH_PLACEHOLDERS = {
    "input",
    "output",
    "reference",
    "source_root",
    "artifact_root",
    "dataset_root",
}


def _save_json_path(value: object, path: Path) -> Path:
    """Publish JSON while narrowing an untyped changed-file import to Path."""
    published: object = save_json_atomic(value, path)
    if not isinstance(published, Path):
        raise TypeError("save_json_atomic must return pathlib.Path.")
    return published


@dataclass(frozen=True, slots=True)
class _PlanJob:
    name: str
    input: Path
    output: Path
    reference: Path | None
    arguments: tuple[str, ...]


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


def _require_exact_fields(
    value: Mapping[str, object],
    *,
    expected: set[str],
    name: str,
) -> None:
    if any(type(key) is not str for key in value):
        raise TypeError(f"{name} keys must be strings.")
    actual = set(value)
    if actual != expected:
        raise ValueError(
            f"{name} fields differ: missing={sorted(expected - actual)}, "
            f"unexpected={sorted(actual - expected)}."
        )


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


def _path_safe_name(value: object, *, name: str) -> str:
    if (
        type(value) is not str
        or not value
        or value != value.strip()
        or value in {".", ".."}
        or "/" in value
        or "\\" in value
        or "\x00" in value
    ):
        raise ValueError(f"{name} must be a non-empty trimmed path-safe name.")
    return value


def _string_sequence(value: object, *, name: str) -> tuple[str, ...]:
    if (
        isinstance(value, (str, bytes))
        or not isinstance(value, Sequence)
        or not all(
            type(item) is str and bool(item) and item == item.strip()
            for item in value
        )
    ):
        raise TypeError(f"{name} must be a sequence of non-empty trimmed strings.")
    return tuple(value)


def _validate_template(value: str, *, reference_available: bool) -> None:
    try:
        parsed = tuple(Formatter().parse(value))
    except ValueError as error:
        raise ValueError(f"Invalid renderer path template {value!r}: {error}") from error
    for _, field_name, format_spec, conversion in parsed:
        if field_name is None:
            continue
        if field_name not in _PATH_PLACEHOLDERS:
            raise ValueError(
                f"Unknown renderer path placeholder {field_name!r} in {value!r}."
            )
        if format_spec or conversion:
            raise ValueError("Renderer path placeholders do not accept formatting.")
        if field_name == "reference" and not reference_available:
            raise ValueError(
                "Renderer path template uses {reference} for a job without one."
            )


def _alignment_residuals(manifest: PathPipelineManifest) -> list[float]:
    observations = _load_json_object(
        manifest.alignment_observations,
        description="Alignment observations",
    )
    _require_exact_fields(
        observations,
        expected={"residuals"},
        name="Alignment observations",
    )
    return _finite_numbers(
        observations["residuals"],
        name="alignment residuals",
    )


def compute_alignment_metrics(manifest: PathPipelineManifest) -> Path:
    """Compute alignment measurements without turning quality into a gate."""
    residuals = _alignment_residuals(manifest)
    absolute = [abs(item) for item in residuals]
    return _save_json_path(
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
    )


def _dataset_plan_payload(
    manifest: PathPipelineManifest,
    *,
    renderer_command: Sequence[str],
) -> dict[str, object]:
    command = _string_sequence(renderer_command, name="renderer.command")
    for token in command:
        _validate_template(token, reference_available=True)

    payload = _load_json_object(manifest.render_jobs, description="Render jobs")
    _require_exact_fields(payload, expected={"jobs"}, name="Render jobs")
    raw_jobs = payload["jobs"]
    if not isinstance(raw_jobs, list) or not raw_jobs:
        raise TypeError("Render jobs must contain a non-empty jobs list.")
    jobs: list[dict[str, object]] = []
    names: set[str] = set()
    outputs: set[Path] = set()
    for index, raw_job in enumerate(raw_jobs):
        if not isinstance(raw_job, Mapping):
            raise TypeError(f"Render job {index} must be a JSON object.")
        _require_exact_fields(
            raw_job,
            expected=_RENDER_JOB_FIELDS,
            name=f"Render job {index}",
        )
        name = _path_safe_name(raw_job["name"], name=f"Render job {index} name")
        if name in names:
            raise ValueError(f"Render job names must be unique: {name!r}.")
        names.add(name)
        input_path = manifest.resolve_render_job_path("input", raw_job["input"])
        if not input_path.is_file():
            raise FileNotFoundError(f"Render input does not exist: {input_path}")
        output_path = manifest.resolve_render_job_path("output", raw_job["output"])
        if output_path in outputs:
            raise ValueError(f"Render job outputs must be unique: {output_path}.")
        outputs.add(output_path)

        reference_value = raw_job["reference"]
        reference_path: Path | None = None
        if reference_value is not None:
            reference_path = manifest.resolve_render_job_path(
                "reference", reference_value
            )
            if not reference_path.is_file():
                raise FileNotFoundError(
                    f"Quality reference does not exist: {reference_path}"
                )
        arguments = _string_sequence(
            raw_job["arguments"],
            name=f"jobs[{index}].arguments",
        )
        for token in (*command, *arguments):
            _validate_template(token, reference_available=reference_path is not None)
        jobs.append(
            {
                "name": name,
                "input": str(input_path),
                "output": str(output_path),
                "reference": (None if reference_path is None else str(reference_path)),
                "arguments": list(arguments),
            }
        )
    return {
        "path_manifest": str(manifest.pipeline_manifest),
        "alignment_metrics": str(manifest.alignment_metrics),
        "renderer_command": list(command),
        "jobs": jobs,
    }


def build_dataset_plan(
    manifest: PathPipelineManifest,
    *,
    renderer_command: Sequence[str],
) -> Path:
    """Resolve configured job paths and write the plan consumed by rendering."""
    return _save_json_path(
        _dataset_plan_payload(manifest, renderer_command=renderer_command),
        manifest.dataset_plan,
    )


def _substitute_paths(
    value: str,
    *,
    manifest: PathPipelineManifest,
    job: _PlanJob,
) -> str:
    _validate_template(value, reference_available=job.reference is not None)
    replacements = {
        "input": str(job.input),
        "output": str(job.output),
        "source_root": str(manifest.source_root),
        "artifact_root": str(manifest.artifact_root),
        "dataset_root": str(manifest.dataset_root),
    }
    if job.reference is not None:
        replacements["reference"] = str(job.reference)
    return value.format_map(replacements)


def _validated_plan_jobs(
    manifest: PathPipelineManifest,
    raw_jobs: object,
) -> tuple[_PlanJob, ...]:
    if not isinstance(raw_jobs, list) or not raw_jobs:
        raise TypeError("Dataset plan must contain a non-empty jobs list.")
    jobs: list[_PlanJob] = []
    names: set[str] = set()
    outputs: set[Path] = set()
    for index, raw_job in enumerate(raw_jobs):
        if not isinstance(raw_job, Mapping):
            raise TypeError(f"Dataset plan job {index} must be a JSON object.")
        _require_exact_fields(
            raw_job,
            expected=_PLAN_JOB_FIELDS,
            name=f"Dataset plan job {index}",
        )
        name = _path_safe_name(
            raw_job["name"], name=f"Dataset plan job {index} name"
        )
        if name in names:
            raise ValueError(f"Dataset plan job names must be unique: {name!r}.")
        names.add(name)
        input_path = manifest.validate_render_job_path("input", raw_job["input"])
        output_path = manifest.validate_render_job_path("output", raw_job["output"])
        if output_path in outputs:
            raise ValueError(f"Dataset plan outputs must be unique: {output_path}.")
        outputs.add(output_path)
        reference_value = raw_job["reference"]
        reference = (
            None
            if reference_value is None
            else manifest.validate_render_job_path("reference", reference_value)
        )
        arguments = _string_sequence(
            raw_job["arguments"],
            name=f"dataset plan job {index} arguments",
        )
        jobs.append(
            _PlanJob(
                name=name,
                input=input_path,
                output=output_path,
                reference=reference,
                arguments=arguments,
            )
        )
    return tuple(jobs)


def _validated_renderer_boundary(
    manifest: PathPipelineManifest,
    *,
    renderer_mode: str,
    renderer_command: Sequence[str],
    working_directory: Path,
) -> tuple[tuple[str, ...], Path]:
    if renderer_mode not in {"execute", "prepared_outputs"}:
        raise ValueError(f"Unsupported renderer mode: {renderer_mode!r}.")
    command = _string_sequence(renderer_command, name="renderer command")
    if renderer_mode == "execute" and not command:
        raise ValueError("Execute renderer mode requires a non-empty command.")
    if renderer_mode == "prepared_outputs" and command:
        raise ValueError("Prepared-output renderer mode does not accept a command.")
    resolved_working_directory = manifest.resolver.validate(
        PathRole.EXTERNAL_ASSET,
        working_directory,
    )
    if resolved_working_directory == manifest.runtime_roots.external_asset_root:
        raise PathContractError(
            "Renderer working_directory must be below external_asset_root."
        )
    if renderer_mode == "execute" and not resolved_working_directory.is_dir():
        raise FileNotFoundError(
            f"Renderer working directory does not exist: {resolved_working_directory}"
        )
    return command, resolved_working_directory


def validate_pipeline_inputs(
    manifest: PathPipelineManifest,
    *,
    renderer_mode: str,
    renderer_command: Sequence[str],
    working_directory: Path,
) -> None:
    """Validate every configured input before publishing any pipeline output."""
    _alignment_residuals(manifest)
    _dataset_plan_payload(manifest, renderer_command=renderer_command)
    _validated_renderer_boundary(
        manifest,
        renderer_mode=renderer_mode,
        renderer_command=renderer_command,
        working_directory=working_directory,
    )


def render_dataset(
    manifest: PathPipelineManifest,
    *,
    renderer_mode: str,
    working_directory: Path,
) -> Path:
    """Run the configured renderer and record its path-based outputs."""
    plan = _load_json_object(manifest.dataset_plan, description="Dataset plan")
    _require_exact_fields(plan, expected=_PLAN_FIELDS, name="Dataset plan")
    if plan["path_manifest"] != str(manifest.pipeline_manifest):
        raise ValueError("Dataset plan path_manifest differs from the active manifest.")
    if plan["alignment_metrics"] != str(manifest.alignment_metrics):
        raise ValueError("Dataset plan alignment_metrics differs from the manifest.")
    renderer_command = _string_sequence(
        plan["renderer_command"],
        name="dataset plan renderer_command",
    )
    renderer_command, resolved_working_directory = _validated_renderer_boundary(
        manifest,
        renderer_mode=renderer_mode,
        renderer_command=renderer_command,
        working_directory=working_directory,
    )
    jobs = _validated_plan_jobs(manifest, plan["jobs"])
    for job in jobs:
        for token in (*renderer_command, *job.arguments):
            _validate_template(token, reference_available=job.reference is not None)
        if not job.input.is_file():
            raise FileNotFoundError(f"Render input does not exist: {job.input}")
        if job.reference is not None and not job.reference.is_file():
            raise FileNotFoundError(
                f"Quality reference does not exist: {job.reference}"
            )

    log_root = manifest.resolver.resolve_beneath(
        PathRole.OUTPUT,
        manifest.execution_root,
        "renderer-logs",
    )
    log_root.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, object]] = []
    for index, job in enumerate(jobs):
        job.output.parent.mkdir(parents=True, exist_ok=True)
        stdout = ""
        stderr = ""
        if renderer_mode == "execute":
            command = [
                _substitute_paths(item, manifest=manifest, job=job)
                for item in (*renderer_command, *job.arguments)
            ]
            completed = subprocess.run(
                command,
                cwd=resolved_working_directory,
                check=False,
                capture_output=True,
                text=True,
            )
            stdout = completed.stdout
            stderr = completed.stderr
            if completed.returncode != 0:
                raise RuntimeError(
                    f"Renderer failed for job {job.name!r} with exit code "
                    f"{completed.returncode}: {stderr.strip()}"
                )
            returncode = completed.returncode
        else:
            shutil.copyfile(job.input, job.output)
            command = []
            returncode = 0
        if not job.output.is_file():
            raise RuntimeError(
                f"Renderer did not produce configured output: {job.output}"
            )
        stdout_path = manifest.resolver.resolve_beneath(
            PathRole.OUTPUT,
            log_root,
            f"{index:04d}-{job.name}.stdout.log",
        )
        stderr_path = manifest.resolver.resolve_beneath(
            PathRole.OUTPUT,
            log_root,
            f"{index:04d}-{job.name}.stderr.log",
        )
        stdout_path.write_text(stdout, encoding="utf-8")
        stderr_path.write_text(stderr, encoding="utf-8")
        records.append(
            {
                "name": job.name,
                "input": str(job.input),
                "output": str(job.output),
                "reference": (
                    None if job.reference is None else str(job.reference)
                ),
                "command": command,
                "returncode": returncode,
                "stdout": str(stdout_path),
                "stderr": str(stderr_path),
            }
        )
    return _save_json_path(
        {"dataset_plan": str(manifest.dataset_plan), "renders": records},
        manifest.render_manifest,
    )


def _mean_absolute_byte_error(left: bytes, right: bytes) -> float:
    width = max(len(left), len(right))
    if width == 0:
        return 0.0
    padded_left = left.ljust(width, b"\0")
    padded_right = right.ljust(width, b"\0")
    total = sum(abs(a - b) for a, b in zip(padded_left, padded_right, strict=True))
    return total / (255.0 * width)


def _validate_execution_output_path(
    manifest: PathPipelineManifest,
    value: object,
    *,
    name: str,
) -> Path:
    if type(value) is not str or not value or value != value.strip():
        raise TypeError(f"{name} must be a non-empty trimmed absolute path.")
    path = Path(value)
    if not path.is_absolute():
        raise PathContractError(f"{name} must be absolute: {path}.")
    resolved: Path = manifest.resolver.validate(PathRole.OUTPUT, path)
    if resolved == manifest.execution_root or not resolved.is_relative_to(
        manifest.execution_root
    ):
        raise PathContractError(
            f"{name} must be below the pipeline execution_root: {resolved}."
        )
    return resolved


def compute_quality_metrics(manifest: PathPipelineManifest) -> Path:
    """Compute render metrics and publish them regardless of their values."""
    render_payload = _load_json_object(
        manifest.render_manifest,
        description="Render manifest",
    )
    _require_exact_fields(
        render_payload,
        expected=_RENDER_MANIFEST_FIELDS,
        name="Render manifest",
    )
    if render_payload["dataset_plan"] != str(manifest.dataset_plan):
        raise ValueError("Render manifest dataset_plan differs from the manifest.")
    raw_renders = render_payload["renders"]
    if not isinstance(raw_renders, list) or not raw_renders:
        raise TypeError("Render manifest must contain a non-empty renders list.")
    records: list[dict[str, object]] = []
    measured: list[float] = []
    for index, raw_render in enumerate(raw_renders):
        if not isinstance(raw_render, Mapping):
            raise TypeError(f"Render record {index} must be a JSON object.")
        _require_exact_fields(
            raw_render,
            expected=_RENDER_RECORD_FIELDS,
            name=f"Render record {index}",
        )
        name = _path_safe_name(
            raw_render["name"], name=f"Render record {index} name"
        )
        manifest.validate_render_job_path("input", raw_render["input"])
        output = manifest.validate_render_job_path("output", raw_render["output"])
        if not output.is_file():
            raise FileNotFoundError(f"Rendered output does not exist: {output}")
        reference_value = raw_render["reference"]
        reference: Path | None = None
        if reference_value is not None:
            reference = manifest.validate_render_job_path(
                "reference", reference_value
            )
            if not reference.is_file():
                raise FileNotFoundError(
                    f"Quality reference does not exist: {reference}"
                )
        _string_sequence(
            raw_render["command"],
            name=f"Render record {index} command",
        )
        returncode = raw_render["returncode"]
        if type(returncode) is not int or returncode != 0:
            raise ValueError(f"Render record {index} returncode must be zero.")
        _validate_execution_output_path(
            manifest,
            raw_render["stdout"],
            name=f"Render record {index} stdout",
        )
        _validate_execution_output_path(
            manifest,
            raw_render["stderr"],
            name=f"Render record {index} stderr",
        )
        error: float | None = None
        if reference is not None:
            error = _mean_absolute_byte_error(
                output.read_bytes(),
                reference.read_bytes(),
            )
            measured.append(error)
        records.append(
            {
                "name": name,
                "output": str(output),
                "reference": None if reference is None else str(reference),
                "mean_absolute_byte_error": error,
            }
        )
    mean_error = None if not measured else sum(measured) / len(measured)
    return _save_json_path(
        {
            "render_manifest": str(manifest.render_manifest),
            "render_count": len(records),
            "measured_render_count": len(measured),
            "mean_absolute_byte_error": mean_error,
            "renders": records,
        },
        manifest.quality_metrics,
    )


def execute_pipeline(
    manifest: PathPipelineManifest,
    *,
    renderer_mode: str,
    renderer_command: Sequence[str],
    working_directory: Path,
) -> Path:
    """Run every stage and return the generated visualization path."""
    validate_pipeline_inputs(
        manifest,
        renderer_mode=renderer_mode,
        renderer_command=renderer_command,
        working_directory=working_directory,
    )
    compute_alignment_metrics(manifest)
    build_dataset_plan(manifest, renderer_command=renderer_command)
    render_dataset(
        manifest,
        renderer_mode=renderer_mode,
        working_directory=working_directory,
    )
    compute_quality_metrics(manifest)
    visualization: Path = write_pipeline_visualization(manifest)
    return visualization
