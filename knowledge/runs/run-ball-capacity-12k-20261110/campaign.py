"""Finite preparation and idempotent nine-model submission for #1675."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import shlex
import subprocess
from datetime import UTC, datetime
from pathlib import Path

WORKSPACE = Path("/content/tennis-lab")
RELATIVE = Path("outputs/ball_detection/train/capacity_scaling/s42-20261110-12k")
ROOT = WORKSPACE / RELATIVE
DRIVE = Path("/content/drive/MyDrive/tennis_lab")
QUEUE = WORKSPACE / ".training_queue"
QUEUE_CLI = WORKSPACE / ".agents/skills/training-queue/scripts/training_queue.sh"
HERE = Path(__file__).resolve().parent
CONDITIONS = [
    (256, 4),
    (384, 4),
    (512, 4),
    (768, 4),
    (1024, 4),
    (512, 6),
    (512, 8),
    (512, 12),
    (512, 24),
]
DISTRIBUTION = "f8425ec2cc6f4d0ae018a3c9387cb3d48b3bd8353794a82888936c3d0bb7477b"
DOWNLOAD = (
    WORKSPACE
    / "outputs/ball_detection/precompute/sequential_length/drive-download-20261025"
)


def now():
    return datetime.now(UTC).isoformat()


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for data in iter(lambda: stream.read(4 * 1024**2), b""):
            digest.update(data)
    return digest.hexdigest()


def definitions(old, parents):
    jobs = []
    by_name = {p["name"]: p for p in parents}
    for index, (dim, layers) in enumerate(CONDITIONS):
        key = f"d{dim}-l{layers}"
        original_name = f"query-{key}-T64"
        template = next(
            s
            for s in old["stages"]
            if s["name"] == (original_name if index < 7 else "query-d512-l4-T64")
        )
        stages = []
        for offset in [6000] if index < 7 else [0, 6000]:
            name = f"query-{key}-T64-q{offset:05d}-q{offset + 6000:05d}"
            model = dict(template["model"], layers=layers)
            budget = dict(template["budget"], schedule_offset=offset)
            if index < 7:
                parent = by_name[original_name]
            elif offset == 0:
                parent = by_name["common-dpt-T32"]
            else:
                parent = dict(previous_stage=stages[-1]["name"], update=6000)
            stages.append(
                dict(
                    name=name,
                    model=model,
                    budget=budget,
                    parent=parent,
                    output=str(ROOT / name),
                )
            )
        jobs.append(
            dict(
                model=key,
                order=index + 1,
                updates=sum(s["budget"]["updates"] for s in stages),
                stages=stages,
            )
        )
    if (
        len(jobs) != 9
        or sum(j["updates"] for j in jobs) != 66000
        or sum(len(j["stages"]) for j in jobs) != 11
    ):
        raise ValueError("Nine-model/eleven-stage/66k invariant failed")
    return jobs


def prepare():
    old = read(Path("/tmp/ball-capacity-inputs-v1/plan.json"))
    parents = read(ROOT / "parents-verified.json")
    if not parents["complete"] or len(parents["parents"]) != 8:
        raise ValueError("Eight verified full-state parents are required")
    inputs = read(Path("/tmp/ball-capacity-inputs-v1/prepared/experiments.json"))
    del inputs  # The exact six manifests are verified separately, never regenerated.
    sample_run = Path(parents["parents"][1]["checkpoint"]).parent / "run.json"
    recipe = read(sample_run)["recipe"]
    plan = dict(
        schema="ball_capacity_12k_campaign.v1",
        issue=1675,
        requirement_comment="https://github.com/Motoki0705/tennis-lab/issues/1675#issuecomment-6758820955",
        requirement_sha256=sha(ROOT / "requirements-comment.json"),
        seed=42,
        additional_updates=66000,
        jobs=definitions(old, parents["parents"]),
        fixed_source_sha256=old["code"]["source_sha256"],
        torch_version=old["code"]["torch_version"],
        runtime=recipe["runtime"],
        augmentation=recipe["augmentation"],
        prepared=str(
            WORKSPACE
            / "outputs/ball_detection/precompute/sequential_length/observed-packs-20261025"
        ),
        distribution_sha256=DISTRIBUTION,
        input_frames=128,
        rclone_config="/tmp/ball-drive-mount.conf",
        source_bundle_original_sha256=sha(
            Path("/tmp/ball-capacity-inputs-v1/source.bundle")
        ),
        lock_sha256=sha(Path("uv.lock")),
        controller_sha256={p.name: sha(p) for p in sorted(HERE.glob("*.py"))},
        rng_policy="Continue full-state RNG at same-model stage boundaries; preserve seed42 fresh query initialization for DPT-to-new-query.",
        failure_policy="Publish FAILED and queue stop before returning nonzero; no automatic retries or successors.",
        local_checkpoint_cache="Retain latest local full checkpoint/weights per new stage; release older local copies only after verified immutable Drive publication; retain all Drive generations and original inputs.",
    )
    if (ROOT / "plan.json").exists() and read(ROOT / "plan.json") != plan:
        raise ValueError("Existing immutable plan differs")
    write(ROOT / "plan.json", plan)
    return plan


def validate_download():
    progress = read(DOWNLOAD / "progress.json")
    if not progress["complete"] or (
        progress["verified_files"],
        progress["verified_bytes"],
    ) != (2925, 157028755547):
        raise ValueError(
            "The complete verified dataset is required before queue publication"
        )
    if progress["distribution_sha256"] != DISTRIBUTION:
        raise ValueError("Distribution identity changed")
    data = Path(progress["destination"])
    if sha(data / "distribution.json") != DISTRIBUTION:
        raise ValueError("Local distribution identity changed")
    distribution = read(data / "distribution.json")
    records = read(DOWNLOAD / "files.json")
    expected = {
        e["path"]: (e["size_bytes"], e["sha256"]) for e in distribution["files"]
    }
    actual = {e["path"]: (e["size_bytes"], e["sha256"]) for e in records}
    if len(actual) != len(records) or actual != expected:
        raise ValueError(
            "Per-file dual-SHA download receipts differ from the fixed membership"
        )
    paths = {str(p.relative_to(data)) for p in data.rglob("*") if p.is_file()}
    if paths != set(expected) | {"distribution.json"}:
        raise ValueError("Local dataset membership differs")
    for name, (size, _) in expected.items():
        if (data / name).is_symlink() or (data / name).stat().st_size != size:
            raise ValueError("Local dataset size or file type changed: " + name)
    write(
        ROOT / "data-verified.json",
        dict(
            schema="ball_capacity_data_acceptance.v1",
            verified_at=now(),
            complete=True,
            download=progress,
            payload_files=len(actual),
            payload_bytes=sum(v[0] for v in actual.values()),
            verification="Original downloader checked every byte with hashlib and _sha256, source stability, exact membership; launch rechecked receipts/manifest/membership/sizes.",
            files_receipt_sha256=sha(DOWNLOAD / "files.json"),
        ),
    )


def verify_plan_sources(plan):
    for name, digest in plan["fixed_source_sha256"].items():
        if sha(Path(name)) != digest:
            raise ValueError("Pinned training source changed: " + name)
    for name, digest in plan["controller_sha256"].items():
        if sha(HERE / name) != digest:
            raise ValueError("Pinned campaign source changed: " + name)
    if sha(Path("uv.lock")) != plan["lock_sha256"]:
        raise ValueError("Pinned environment lock changed")


def enqueue_jobs(plan, session, attempt, queue=QUEUE, queue_cli=QUEUE_CLI, root=ROOT):
    env = dict(os.environ, TRAINING_QUEUE_DIR=str(queue))
    registrations = []
    for job in plan["jobs"]:
        command = shlex.join(
            [
                str(Path.cwd() / ".venv/bin/python"),
                "-u",
                "-B",
                str(HERE / "train_model.py"),
                "--root",
                str(root),
                "--model",
                job["model"],
                "--attempt",
                str(attempt),
            ]
        )
        command = (
            "env OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONWARNINGS=ignore PATH="
            + shlex.quote("/tmp/ball-tools:" + os.environ["PATH"])
            + " "
            + command
        )
        key = f"ball-capacity/s42-20261110-12k/{job['model']}/attempt-{attempt}"
        result = subprocess.check_output(
            [
                "bash",
                str(queue_cli),
                "add",
                command,
                "--name",
                f"i1675-12k-{job['model']}-a{attempt}",
                "--provider",
                "codex",
                "--session",
                session,
                "--issue",
                "1675",
                "--resource",
                "all",
                "--idempotency-key",
                key,
            ],
            text=True,
            env=env,
        )
        lines = [
            line.removeprefix("queued: ").strip()
            for line in result.splitlines()
            if line.startswith("queued: ")
        ]
        if len(lines) != 1 or not lines[0].endswith(".job"):
            raise ValueError(
                "Queue did not return a unique persisted job ID: " + result
            )
        registrations.append(
            dict(
                **job,
                queue_id=Path(lines[0]).stem,
                queue_file=Path(lines[0]).name,
                idempotency_key=key,
                attempt=attempt,
                command=command,
                resource="all",
            )
        )
        write(
            root / "queue-registration.json",
            dict(
                schema="ball_capacity_queue.v1",
                session=session,
                complete=len(registrations) == 9,
                plan_sha256=sha(root / "plan.json"),
                jobs=registrations,
            ),
        )
    return registrations


def request_service(registrations):
    if (ROOT / "queue-service-state.json").exists():
        state = read(ROOT / "queue-service-state.json")
        if (
            state["state"] == "running"
            and Path(f"/proc/{state['service_pid']}").exists()
        ):
            return
    command = [
        str(Path.cwd() / ".venv/bin/python"),
        "-u",
        "-B",
        str(HERE / "queue_service.py"),
        "--root",
        str(ROOT),
        "--",
        "bash",
        str(QUEUE_CLI),
        "serve",
        "--idle-timeout",
        "3600",
    ]
    request = dict(
        schema="ball_capacity_queue_service_request.v1",
        ready=True,
        requested_at=now(),
        cwd=str(Path.cwd()),
        argv=command,
        env=dict(
            TRAINING_QUEUE_DIR=str(QUEUE),
            OMP_NUM_THREADS="2",
            MKL_NUM_THREADS="2",
            PYTHONWARNINGS="ignore",
            PATH="/tmp/ball-tools:" + os.environ["PATH"],
        ),
        log=str(ROOT / "queue-service.log"),
        all_nine_registered=True,
        queue_ids=[r["queue_id"] for r in registrations],
        finite_worker_pid=os.getppid(),
        instruction="Parent must start this exact foreground service outside finite agent_exec ancestry. Do not start a second service.",
    )
    write(ROOT / "queue-service-request.json", request)
    progress = read(ROOT / "worker-progress.json")
    progress.update(
        updated_at=now(),
        phase="await_external_service",
        queue_service_request=str(ROOT / "queue-service-request.json"),
        queue_ids=request["queue_ids"],
    )
    write(ROOT / "worker-progress.json", progress)


def stop_failed(root, queue, model, error):
    # The stop is visible before the active all-GPU job returns or releases capacity.
    queue.mkdir(parents=True, exist_ok=True)
    (queue / "stop").touch()
    record = dict(
        schema="ball_capacity_failure.v1",
        failed_at=now(),
        model=model,
        error=error,
        queue_id=os.environ.get("TENNIS_RUN_ID"),
        successors_held=True,
        automatic_retry=False,
    )
    write(root / "FAILED.json", record)
    write(root / "notifications" / f"failed-{model}-{os.getpid()}.json", record)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["prepare", "launch", "recover", "status"])
    parser.add_argument("--session", default=os.environ.get("CODEX_THREAD_ID"))
    parser.add_argument("--attempt", type=int, default=1)
    args = parser.parse_args()
    ROOT.mkdir(parents=True, exist_ok=True)
    if args.action == "prepare":
        print(json.dumps(prepare(), indent=2))
    elif args.action in {"launch", "recover"}:
        if not args.session or args.attempt < 1:
            parser.error("An explicit Codex session and positive attempt are required")
        with (ROOT / "registration.lock").open("a") as stream:
            fcntl.flock(stream, fcntl.LOCK_EX)
            plan = read(ROOT / "plan.json")
            verify_plan_sources(plan)
            input_receipt = read(ROOT / "inputs-receipt.json")
            if (
                input_receipt.get("status") != "verified"
                or input_receipt.get("generation") != "inputs-v1"
            ):
                raise ValueError(
                    "The committed source/config inputs must be saved before queue admission"
                )
            validate_download()
            if args.action == "recover":
                recover_registration(args.attempt)
            elif (ROOT / "queue-registration.json").exists():
                prior = read(ROOT / "queue-registration.json")
                if any(j["attempt"] != args.attempt for j in prior["jobs"]):
                    raise ValueError(
                        "A new attempt requires explicit recover, not launch"
                    )
            if (ROOT / "FAILED.json").exists():
                raise ValueError(
                    "Inspect and explicitly recover the recorded failure before submission"
                )
            registrations = enqueue_jobs(plan, args.session, args.attempt)
            request_service(registrations)
            print(
                json.dumps(
                    dict(
                        registered=9,
                        queue_service_request=str(ROOT / "queue-service-request.json"),
                    )
                )
            )
    else:
        result = subprocess.check_output(
            ["bash", str(QUEUE_CLI), "list"],
            env=dict(os.environ, TRAINING_QUEUE_DIR=str(QUEUE)),
            text=True,
        )
        print(result)
        for path in sorted((ROOT / "job-status").glob("*.json")):
            print(path.name, path.read_text())


def recover_registration(attempt):
    """Explicitly supersede this campaign's queue records after safe recovery."""
    pid_file = QUEUE / "worker.pid"
    if pid_file.exists():
        pid = pid_file.read_text().strip()
        if pid.isdecimal() and Path(f"/proc/{pid}").exists():
            raise ValueError(
                "The old queue worker must finish teardown before recovery"
            )
    if list((QUEUE / "running").glob("*.job")):
        raise ValueError(
            "Running queue records require inspection, never automatic reset"
        )
    prior = read(ROOT / "queue-registration.json")
    if not prior["complete"] or attempt <= max(j["attempt"] for j in prior["jobs"]):
        raise ValueError("Recovery requires a new attempt number")
    if not read(ROOT / "recovery-verified.json")["complete"]:
        raise ValueError("Restore and validate committed stage snapshots first")
    history = ROOT / "attempt-history" / f"attempt-{prior['jobs'][0]['attempt']}"
    history.mkdir(parents=True, exist_ok=False)
    write(history / "queue-registration.json", prior)
    for job in prior["jobs"]:
        if (QUEUE / "jobs" / job["queue_file"]).exists():
            subprocess.run(
                ["bash", str(QUEUE_CLI), "cancel", job["queue_file"]],
                env=dict(os.environ, TRAINING_QUEUE_DIR=str(QUEUE)),
                check=True,
            )
    if (ROOT / "FAILED.json").exists():
        (ROOT / "FAILED.json").replace(history / "FAILED.json")


if __name__ == "__main__":
    main()
