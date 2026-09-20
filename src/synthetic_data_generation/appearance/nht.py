"""NHT public command boundary and shared training-queue submission."""

from __future__ import annotations

import json
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

from .contracts import Manifest
from .validation import validate_ready
from .workspace import load_manifest, save_manifest, sha256, variant_lock, write_json


def _environment(manifest: Manifest) -> dict[str, str]:
    environment = dict(os.environ)
    # Bind only this subprocess to the edited NHT checkout, not the shared install.
    environment["PYTHONPATH"] = str(manifest.config.nht_source_root)
    return environment


def _command(root: Path, manifest: Manifest) -> list[str]:
    return [
        manifest.config.nht_executable,
        "--scene-id",
        manifest.config.scene_id,
        "--workspace",
        str(root / "reconstruction"),
        "--config",
        str(root / "nht-config.yaml"),
        "--from-stage",
        "nht_training",
    ]


def _verify_prepared(root: Path, manifest: Manifest) -> dict[str, Any]:
    workspace = root / "reconstruction"
    provenance = json.loads((workspace / "import-provenance.json").read_text())
    if provenance["image_names"] != [frame.name for frame in manifest.frames]:
        raise ValueError("NHT prepared a different image selection")
    expected_validation = [
        frame.name for frame in manifest.frames if frame.split == "validation"
    ]
    if provenance["validation_names"] != expected_validation:
        raise ValueError("NHT changed the original train/validation assignment")
    for frame in manifest.frames:
        if sha256(workspace / "frames/images" / frame.name) != sha256(
            root / "generation/accepted" / frame.name
        ):
            raise ValueError(f"NHT replacement mismatch: {frame.name}")
    for relative, expected in manifest.source_files.items():
        if (
            relative.startswith("sfm/model/")
            and sha256(workspace / relative) != expected
        ):
            raise ValueError(f"NHT changed SfM geometry: {relative}")
    return dict(provenance)


def finalize(root: Path) -> dict[str, Any]:
    with variant_lock(root):
        manifest = load_manifest(root)
        validate_ready(root, manifest)
        if manifest.status in {"finalized", "training", "complete"}:
            _verify_prepared(root, manifest)
            return {
                "status": manifest.status,
                "workspace": str(root / "reconstruction"),
            }
        config = yaml.safe_load(
            (root / "provenance/source/resolved-config.yaml").read_text()
        )
        config["nht_training"].update(
            image_names=[frame.name for frame in manifest.frames],
            max_steps=manifest.config.max_steps,
            python=str(manifest.config.training_python),
            trainer=str(
                manifest.config.nht_source_root
                / "gsplat/examples/simple_trainer_nht.py"
            ),
            adapter=None,
            extra_args=["--disable_video"],
        )
        (root / "nht-config.yaml").write_text(
            yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
        )
        workspace = root / "reconstruction"
        if (workspace / "import-provenance.json").is_file():
            # Recover an interruption after NHT publication and before manifest save.
            _verify_prepared(root, manifest)
        else:
            command = _command(root, manifest) + [
                "--source-workspace",
                str(manifest.config.source_workspace),
                "--replacement-images",
                str(root / "generation/accepted"),
                "--prepare-only",
            ]
            subprocess.run(command, check=True, env=_environment(manifest))
            _verify_prepared(root, manifest)
        manifest.status = "finalized"
        save_manifest(root, manifest)
        return {"status": manifest.status, "workspace": str(workspace)}


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[3]


def save_code_provenance(root: Path, nht_root: Path) -> None:
    """Snapshot all three editable repositories, including untracked source files."""
    repositories = {
        "tennis-lab": _repository_root(),
        "nht": nht_root,
        "gsplat": nht_root / "gsplat",
    }
    for name, repository in repositories.items():
        destination = root / "provenance/code" / name
        destination.mkdir(parents=True, exist_ok=True)
        revision = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repository, text=True
        ).strip()
        diff = subprocess.check_output(
            ["git", "diff", "--binary", "HEAD"], cwd=repository
        )
        (destination / "uncommitted.patch").write_bytes(diff)
        untracked = subprocess.check_output(
            ["git", "ls-files", "--others", "--exclude-standard", "-z"], cwd=repository
        ).split(b"\0")
        names = []
        for encoded in untracked:
            if not encoded:
                continue
            relative = Path(os.fsdecode(encoded))
            source = repository / relative
            if not source.is_file() or source.is_symlink():
                raise ValueError(f"Unsupported untracked source artifact: {relative}")
            target = destination / "untracked" / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(source.read_bytes())
            names.append(str(relative))
        write_json(
            destination / "revision.json",
            {
                "repository": str(repository),
                "commit": revision,
                "untracked_files": names,
            },
        )


def enqueue_training(root: Path, *, retry_failed_job: bool = False) -> dict[str, Any]:
    with variant_lock(root):
        manifest = load_manifest(root)
        validate_ready(root, manifest)
        queue_record = root / "queue.json"
        recovering = retry_failed_job and manifest.status in {"training", "failed"}
        if manifest.status != "finalized" and not (
            recovering and queue_record.is_file()
        ):
            raise ValueError("Finalize the variant before enqueueing training")
        _verify_prepared(root, manifest)
        if queue_record.exists():
            existing_record = dict(json.loads(queue_record.read_text()))
            job_name = (
                existing_record["enqueue_output"].strip().removeprefix("queued: ")
            )
            if Path(job_name).name != job_name or not job_name.endswith(".job"):
                raise ValueError("Malformed queue receipt")
            queue_dir = Path(existing_record["queue_dir"])
            states = [
                state
                for state in ("jobs", "running", "done", "failed", "cancelled")
                if (queue_dir / state / job_name).is_file()
            ]
            if not states:
                raise ValueError(
                    "Queue receipt has no matching job; inspect queue state"
                )
            if len(states) > 1:
                raise ValueError("Queue receipt has conflicting job states")
            terminal_failure = states in (["failed"], ["cancelled"])
            if not terminal_failure:
                if retry_failed_job:
                    raise ValueError("Only a failed/cancelled job can be retried")
                _start_worker(root, existing_record)
                return existing_record
            if not retry_failed_job:
                raise ValueError(
                    "Previous training job failed; use training_retry=true after fixing its cause"
                )
            write_json(
                root / "provenance/queue-attempts" / f"{job_name}.json", existing_record
            )
            if recovering:
                # The queue publishes failed/cancelled only after verified process
                # teardown. A stale running receipt (for example after host reboot)
                # is deliberately insufficient proof to launch another GPU job.
                archive = (
                    root / "provenance/queue-attempts" / f"{job_name}.manifest.json"
                )
                if not archive.exists():
                    write_json(archive, manifest.model_dump(mode="json"))
                manifest.status = "finalized"
                save_manifest(root, manifest)
        session = (
            os.environ.get("CODEX_THREAD_ID")
            or os.environ.get("CODEX_SESSION_ID")
            or manifest.provider_session
        )
        if not session:
            raise ValueError("CODEX_THREAD_ID is required for queue attribution")
        save_code_provenance(root, manifest.config.nht_source_root)
        repository = _repository_root()
        common_git = Path(
            subprocess.check_output(
                ["git", "rev-parse", "--git-common-dir"], cwd=repository, text=True
            ).strip()
        )
        if not common_git.is_absolute():
            common_git = repository / common_git
        common_root = common_git.resolve().parent
        queue = common_root / ".agents/skills/training-queue/scripts/training_queue.sh"
        environment = dict(
            os.environ, TRAINING_QUEUE_DIR=str(common_root / ".training_queue")
        )
        command = shlex.join(
            [
                str(Path(sys.executable).absolute()),
                "-m",
                "src.synthetic_data_generation.scripts.run_appearance_variant",
                "action=execute_training",
                f"variant.output_root={root}",
            ]
        )
        name = (
            f"appearance-{manifest.config.scene_id.lower()}-{manifest.config.max_steps}"
        )
        completed = subprocess.run(
            [
                "bash",
                str(queue),
                "add",
                command,
                "--name",
                name,
                "--provider",
                "codex",
                "--session",
                session,
                "--resource",
                "all",
            ],
            cwd=repository,
            env=environment,
            check=True,
            capture_output=True,
            text=True,
        )
        record: dict[str, Any] = {
            "name": name,
            "command": command,
            "queue_dir": environment["TRAINING_QUEUE_DIR"],
            "enqueue_output": completed.stdout,
        }
        write_json(queue_record, record)
        _start_worker(root, record)
        return record


def _start_worker(root: Path, record: dict[str, Any]) -> None:
    queue_dir = Path(record["queue_dir"])
    script = (
        queue_dir.parent / ".agents/skills/training-queue/scripts/training_queue.sh"
    )
    worker = subprocess.run(
        ["bash", str(script), "start"],
        cwd=_repository_root(),
        env=dict(os.environ, TRAINING_QUEUE_DIR=str(queue_dir)),
        check=False,
        capture_output=True,
        text=True,
    )
    record["worker_output"] = worker.stdout + worker.stderr
    record["worker_returncode"] = worker.returncode
    write_json(root / "queue.json", record)
    if (
        worker.returncode != 0
        and "worker already running" not in record["worker_output"]
    ):
        raise RuntimeError(
            f"Job was enqueued but worker startup failed; see {root / 'queue.json'}"
        )


def execute_training(root: Path) -> dict[str, Any]:
    if not os.environ.get("TENNIS_RUN_ID"):
        raise ValueError("GPU execution must be launched by the shared training queue")
    with variant_lock(root):
        manifest = load_manifest(root)
        validate_ready(root, manifest)
        if manifest.status != "finalized":
            raise ValueError(
                "Training requires finalized, not already running/completed inputs"
            )
        _verify_prepared(root, manifest)
        manifest.status = "training"
        save_manifest(root, manifest)
        command = _command(root, manifest)
        write_json(
            root / "training-command.json",
            {
                "argv": command,
                "run_id": os.environ["TENNIS_RUN_ID"],
                "pythonpath": str(manifest.config.nht_source_root),
            },
        )
        try:
            subprocess.run(command, check=True, env=_environment(manifest))
            scene = json.loads((root / "reconstruction/export/scene.json").read_text())
            summary = json.loads(
                (root / "reconstruction/3dgs/training.json").read_text()
            )
            if (
                scene["camera_count"] != len(manifest.frames)
                or summary["max_steps"] != manifest.config.max_steps
            ):
                raise ValueError(
                    "NHT result camera count/step count does not match the experiment"
                )
            validate_ready(root, manifest)
        except BaseException:
            manifest.status = "failed"
            save_manifest(root, manifest)
            raise
        manifest.status = "complete"
        save_manifest(root, manifest)
        return {
            "status": "complete",
            "scene": str(root / "reconstruction/export/scene.json"),
            "training": summary,
        }
