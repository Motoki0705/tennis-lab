"""Relocate and enqueue the strict Issue #790 scaling-grid shard.

The canonical manifest remains the scientific authority.  This helper changes
only runtime-specific interpreter and role-root values, verifies the resulting
execution plan, and passes each argv through the repository training queue.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

from src.tasks.court_detection.experiments.query_consistency import (
    V3_DERIVED_TARGET_ROOT,
    V3_WORKSPACE_ROOT,
    validate_query_consistency_manifest,
)

JobKind = Literal["train", "profile", "both"]
JsonMapping = Mapping[str, Any]
_SCALING_SEEDS = (42,)
_INPUT_LONG_SIDES = (256, 384, 512)
_ENCODER_DEPTHS = (1, 8)
_DECODER_FAMILIES = ("dpt",)
_DECODER_SIZES = ("tiny", "small", "base", "large")


@dataclass(frozen=True, slots=True)
class RuntimeRelocation:
    """Absolute runtime roots allowed to differ from the canonical manifest."""

    python_executable: Path
    data_root: Path
    external_asset_root: Path
    output_root: Path
    checkpoint_root: Path

    def validate(self) -> None:
        values = {
            "python_executable": self.python_executable,
            "data_root": self.data_root,
            "external_asset_root": self.external_asset_root,
            "output_root": self.output_root,
            "checkpoint_root": self.checkpoint_root,
        }
        for name, value in values.items():
            if not value.is_absolute():
                raise ValueError(f"{name} must be absolute: {value}")
        if not self.python_executable.is_file() or not os.access(
            self.python_executable, os.X_OK
        ):
            raise ValueError(
                "python_executable must be an existing executable file: "
                f"{self.python_executable}"
            )


def _mapping(value: object, *, name: str) -> JsonMapping:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping.")
    return cast(JsonMapping, value)


def _string(value: object, *, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise TypeError(f"{name} must be a non-empty string.")
    return value


def _replace_once(argv: list[str], old: str, new: str, *, name: str) -> None:
    matches = [index for index, token in enumerate(argv) if token == old]
    if len(matches) != 1:
        raise ValueError(
            f"Canonical {name} token must occur exactly once; found {len(matches)}."
        )
    argv[matches[0]] = new


def relocate_run_argv(
    manifest: JsonMapping,
    run: JsonMapping,
    *,
    job_kind: Literal["train", "profile"],
    runtime: RuntimeRelocation,
) -> tuple[str, ...]:
    """Relocate one manifest argv while preserving every scientific override."""
    runtime.validate()
    fixed = _mapping(manifest["fixed_contract"], name="fixed_contract")
    run_id = _string(run["run_id"], name="run.run_id")
    field = "command_argv" if job_kind == "train" else "profile_command_argv"
    raw_argv = run[field]
    if not isinstance(raw_argv, Sequence) or isinstance(raw_argv, (str, bytes)):
        raise TypeError(f"Ready run {field} must be an argv sequence.")
    argv = [_string(token, name=f"run.{field} token") for token in raw_argv]

    canonical_python = _string(
        fixed["python_executable"], name="fixed_contract.python_executable"
    )
    if not argv or argv[0] != canonical_python:
        raise ValueError(f"Canonical {field} lost its interpreter authority.")
    argv[0] = str(runtime.python_executable)

    canonical_external = _string(
        fixed["external_asset_root"], name="fixed_contract.external_asset_root"
    )
    _replace_once(
        argv,
        f"paths.external_asset_root={canonical_external}",
        f"paths.external_asset_root={runtime.external_asset_root}",
        name="external_asset_root",
    )

    relative_dir = f"court_detection/query_consistency_ablation/{run_id}"
    if job_kind == "train":
        canonical_data = _string(fixed["data_root"], name="fixed_contract.data_root")
        _replace_once(
            argv,
            f"paths.data_root={canonical_data}",
            f"paths.data_root={runtime.data_root}",
            name="data_root",
        )
        _replace_once(
            argv,
            f"paths.artifact_root=outputs/{relative_dir}/artifacts",
            f"paths.artifact_root={runtime.output_root / relative_dir / 'artifacts'}",
            name="artifact_root",
        )
        if f"data.source.workspace_root={V3_WORKSPACE_ROOT}" not in argv:
            raise ValueError("Training argv lost the frozen V3 workspace descendant.")
        if f"data.processing.derived_target_root={V3_DERIVED_TARGET_ROOT}" not in argv:
            raise ValueError(
                "Training argv lost the frozen V3 derived-target descendant."
            )
        if f"run.output_dir={relative_dir}" not in argv:
            raise ValueError("Training argv lost its run-specific output descendant.")
        argv.extend(
            (
                f"paths.output_root={runtime.output_root}",
                f"paths.checkpoint_root={runtime.checkpoint_root}",
            )
        )
    else:
        if f"profile.output_path={relative_dir}/capacity_profile.json" not in argv:
            raise ValueError("Profile argv lost its run-specific evidence path.")
        argv.append(f"paths.output_root={runtime.output_root}")
    return tuple(argv)


def select_scaling_grid_runs(
    manifest: JsonMapping, *, seeds: Sequence[int]
) -> tuple[JsonMapping, ...]:
    """Select the complete seed-42 Cartesian grid in canonical order."""
    validate_query_consistency_manifest(manifest, require_resolved=False)
    requested = tuple(seeds)
    if requested != _SCALING_SEEDS:
        raise ValueError(f"Scaling-grid seeds must be exactly {_SCALING_SEEDS}.")
    fixed = _mapping(manifest["fixed_contract"], name="fixed_contract")
    canonical_seeds = fixed["seeds"]
    if not isinstance(canonical_seeds, list) or any(
        type(seed) is not int for seed in canonical_seeds
    ):
        raise TypeError("fixed_contract.seeds must be an integer list.")
    if tuple(canonical_seeds) != _SCALING_SEEDS:
        raise ValueError("Canonical manifest seeds changed from the scaling contract.")

    raw_runs = manifest["runs"]
    if not isinstance(raw_runs, list):
        raise TypeError("manifest.runs must be a list.")
    selected = tuple(
        _mapping(run, name="run")
        for run in raw_runs
        if _mapping(run, name="run")["phase"] == "scaling_grid"
        and _mapping(run, name="run")["seed"] in requested
    )
    expected_architectures = tuple(
        (input_long_side, depth, family, size)
        for input_long_side in _INPUT_LONG_SIDES
        for depth in _ENCODER_DEPTHS
        for family in _DECODER_FAMILIES
        for size in _DECODER_SIZES
    )
    actual_architectures = tuple(
        (
            architecture["input_long_side"],
            architecture["encoder_depth"],
            architecture["decoder_family"],
            architecture["decoder_size"],
        )
        for run in selected
        for architecture in (_mapping(run["architecture"], name="run.architecture"),)
    )
    if actual_architectures != expected_architectures or any(
        run["queue_ready"] is not True for run in selected
    ):
        raise ValueError(
            "Seed 42 must resolve exactly the 24-member input/depth/DPT-size grid."
        )
    return selected


def build_shard_plan(
    manifest: JsonMapping,
    *,
    seeds: Sequence[int],
    job_kind: JobKind,
    runtime: RuntimeRelocation,
) -> dict[str, object]:
    """Build a candidate-bound plan of queue commands without mutating state."""
    runs = select_scaling_grid_runs(manifest, seeds=seeds)
    kinds: tuple[Literal["train", "profile"], ...] = (
        ("train", "profile") if job_kind == "both" else (job_kind,)
    )
    jobs: list[dict[str, object]] = []
    for kind in kinds:
        for run in runs:
            run_id = _string(run["run_id"], name="run.run_id")
            jobs.append(
                {
                    "name": run_id if kind == "train" else f"{run_id}-profile",
                    "kind": kind,
                    "run_id": run_id,
                    "seed": run["seed"],
                    "argv": list(
                        relocate_run_argv(
                            manifest,
                            run,
                            job_kind=kind,
                            runtime=runtime,
                        )
                    ),
                }
            )
    return {
        "schema": "court_query_consistency_shard_plan_v2",
        "source_manifest_schema": manifest["schema"],
        "source_manifest_sha256": manifest["manifest_sha256"],
        "phase": "scaling_grid",
        "seeds": list(seeds),
        "job_kind": job_kind,
        "runtime": {
            "python_executable": str(runtime.python_executable),
            "data_root": str(runtime.data_root),
            "external_asset_root": str(runtime.external_asset_root),
            "output_root": str(runtime.output_root),
            "checkpoint_root": str(runtime.checkpoint_root),
        },
        "jobs": jobs,
    }


def _queue_status(queue_script: Path, *, queue_dir: Path, repository_root: Path) -> str:
    completed = subprocess.run(
        ["bash", str(queue_script), "status"],
        cwd=repository_root,
        env={**os.environ, "TRAINING_QUEUE_DIR": str(queue_dir)},
        check=True,
        capture_output=True,
        text=True,
    )
    status = completed.stdout
    if "worker: stopped" not in status or "queued=0 running=0" not in status:
        raise RuntimeError(
            "Shard queue must be stopped with no pending/running jobs before enqueue:\n"
            f"{status}"
        )
    return status


def enqueue_shard_plan(
    plan: Mapping[str, object],
    *,
    queue_script: Path,
    queue_dir: Path,
    repository_root: Path,
    provider: str,
    session: str,
    issue: int,
) -> None:
    """Enqueue every planned argv through the shared queue implementation."""
    if not queue_script.is_file():
        raise FileNotFoundError(f"Training queue script is missing: {queue_script}")
    if not repository_root.is_absolute() or not (repository_root / ".git").exists():
        raise ValueError("repository_root must be an absolute Git checkout/worktree.")
    if not queue_dir.is_absolute():
        raise ValueError("queue_dir must be absolute.")
    if not provider or not session or issue <= 0:
        raise ValueError("provider, session, and positive issue are required.")
    _queue_status(queue_script, queue_dir=queue_dir, repository_root=repository_root)
    jobs = plan["jobs"]
    if not isinstance(jobs, list) or not jobs:
        raise ValueError("Shard plan must contain at least one job.")
    environment = {**os.environ, "TRAINING_QUEUE_DIR": str(queue_dir)}
    for raw_job in jobs:
        job = _mapping(raw_job, name="job")
        raw_argv = job["argv"]
        if not isinstance(raw_argv, list) or any(
            not isinstance(token, str) or not token for token in raw_argv
        ):
            raise TypeError("job.argv must be a non-empty string list.")
        command = shlex.join(
            [
                "env",
                "MPLBACKEND=Agg",
                "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True",
                *cast(list[str], raw_argv),
            ]
        )
        subprocess.run(
            [
                "bash",
                str(queue_script),
                "add",
                command,
                "--name",
                _string(job["name"], name="job.name"),
                "--provider",
                provider,
                "--session",
                session,
                "--issue",
                str(issue),
            ],
            cwd=repository_root,
            env=environment,
            check=True,
        )

    queue_dir.mkdir(parents=True, exist_ok=True)
    plan_path = queue_dir / "shard-plan.json"
    temporary = plan_path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(plan_path)
    print(f"Enqueued {len(jobs)} jobs; shard plan: {plan_path}")


def _absolute_path(value: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        raise argparse.ArgumentTypeError(f"path must be absolute: {value}")
    return path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--seed", type=int, action="append", required=True)
    parser.add_argument(
        "--job-kind", choices=("train", "profile", "both"), required=True
    )
    parser.add_argument("--python-executable", type=_absolute_path, required=True)
    parser.add_argument("--data-root", type=_absolute_path, required=True)
    parser.add_argument("--external-asset-root", type=_absolute_path, required=True)
    parser.add_argument("--output-root", type=_absolute_path, required=True)
    parser.add_argument("--checkpoint-root", type=_absolute_path, required=True)
    parser.add_argument("--queue-script", type=Path, required=True)
    parser.add_argument("--queue-dir", type=_absolute_path, required=True)
    parser.add_argument("--repository-root", type=_absolute_path, required=True)
    parser.add_argument("--provider", required=True)
    parser.add_argument("--session", required=True)
    parser.add_argument("--issue", type=int, default=790)
    return parser.parse_args()


def main() -> None:
    """Validate, relocate, record, and enqueue the scaling-grid shard."""
    args = _parse_args()
    manifest_value = json.loads(args.manifest.read_text(encoding="utf-8"))
    manifest = _mapping(manifest_value, name="manifest")
    runtime = RuntimeRelocation(
        python_executable=args.python_executable,
        data_root=args.data_root,
        external_asset_root=args.external_asset_root,
        output_root=args.output_root,
        checkpoint_root=args.checkpoint_root,
    )
    plan = build_shard_plan(
        manifest,
        seeds=tuple(args.seed),
        job_kind=cast(JobKind, args.job_kind),
        runtime=runtime,
    )
    enqueue_shard_plan(
        plan,
        queue_script=args.queue_script,
        queue_dir=args.queue_dir,
        repository_root=args.repository_root,
        provider=args.provider,
        session=args.session,
        issue=args.issue,
    )


if __name__ == "__main__":
    main()
