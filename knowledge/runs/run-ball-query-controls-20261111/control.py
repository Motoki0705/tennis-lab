"""Frozen #1899 controls, preceded only by the remaining #1675 eight-layer stage."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import os
import shlex
import subprocess
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

WORKSPACE = Path("/content/tennis-lab")
RELATIVE = Path("outputs/ball_detection/train/query_controls/s42-20261111-main")
ROOT = WORKSPACE / RELATIVE
DRIVE = Path("/content/drive/MyDrive/tennis_lab")
QUEUE = WORKSPACE / ".training_queue"
QUEUE_CLI = (
    Path(__file__).resolve().parents[3]
    / ".agents/skills/training-queue/scripts/training_queue.sh"
)
HERE = Path(__file__).resolve().parent
LEGACY_HERE = HERE.parent / "run-ball-capacity-12k-20261110"
CAPACITY = Path("outputs/ball_detection/train/capacity_scaling/s42-20261110-12k")
OLD_ROOT = WORKSPACE / CAPACITY
OLD_STAGE = "query-d512-l8-T64-q06000-q12000"
OLD_CWD = WORKSPACE / ".claude/worktrees/ball-capacity-12k"
REFERENCE = (
    WORKSPACE / "outputs/ball_detection/train/sequential_length/s42-20261025-main"
)
PREPARED = (
    WORKSPACE
    / "outputs/ball_detection/precompute/sequential_length/observed-packs-20261025"
)
CONFIG = Path("/tmp/ball-drive-mount.conf")
DISTRIBUTION = "f8425ec2cc6f4d0ae018a3c9387cb3d48b3bd8353794a82888936c3d0bb7477b"
CONDITIONS = ("queries", "cnn", "length")
FIELDS = ("mean_error_px", "median_error_px", "p95_error_px", "within_distance_rate")


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
        for chunk in iter(lambda: stream.read(4 * 1024**2), b""):
            digest.update(chunk)
    return digest.hexdigest()


def store(root=ROOT):
    from src.utils.artifact_store import RcloneArtifactStore

    return RcloneArtifactStore(
        root,
        remote="mounted_drive",
        remote_root=root.relative_to(WORKSPACE).as_posix(),
        config_path=CONFIG,
    )


def snapshot_store(local, relative):
    from src.utils.artifact_store import RcloneSnapshotStore

    return RcloneSnapshotStore(
        local,
        remote="mounted_drive",
        remote_root=relative.as_posix(),
        config_path=CONFIG,
    )


def select_receipt(directory):
    import re

    entries = []
    for path in directory.glob("*.json"):
        match = re.fullmatch(r"(completed|checkpoint)-(\d{8})-[0-9a-f]{8}", path.stem)
        if match:
            entries.append(((int(match[2]), match[1] == "completed"), path))
    if not entries:
        raise ValueError(f"No committed full checkpoint: {directory}")
    highest = max(key for key, _ in entries)
    paths = [p for key, p in entries if key == highest]
    if len(paths) != 1:
        raise ValueError(f"Ambiguous newest checkpoint: {directory}")
    receipt = read(paths[0])
    if (
        receipt.get("status") != "verified"
        or receipt.get("generation") != paths[0].stem
    ):
        raise ValueError("Invalid newest checkpoint receipt")
    return paths[0], receipt


def definitions(conv, x3d, first_budget):
    """Use explicit stage index; T64 in the first stage never implies offset6000."""
    jobs = []
    for index, condition in enumerate(CONDITIONS):
        stages = []
        for stage_index in range(2):
            frames = 64 if condition == "length" or stage_index == 1 else 32
            name = f"{condition}-s{stage_index + 1}-T{frames}"
            model = dict(conv["model"])
            if condition == "cnn":
                for key in (
                    "stem_channels",
                    "mixed_channels",
                    "residual_blocks",
                    "encoder_variant",
                    "temporal_mixing",
                ):
                    model[key] = x3d["model"][key]
            model.update(
                frames=frames,
                dim=256,
                heads=8,
                layers=4,
                ffn_dim=704,
                activation_checkpointing=False,
                frame_chunk_size=None,
            )
            if condition == "queries":
                model.update(
                    num_streams=4,
                    mhc_coefficient_dim=64,
                    mhc_sinkhorn_iters=20,
                    mhc_eps=1e-6,
                    mhc_residual_identity_bias=4.0,
                    mhc_update_scale_init=0.0,
                )
            parent = (
                (x3d if condition == "cnn" else conv)["parent"]
                if stage_index == 0
                else dict(previous_stage=stages[0]["name"], update=6000)
            )
            stages.append(
                dict(
                    name=name,
                    model=model,
                    budget=dict(first_budget, schedule_offset=6000 * stage_index),
                    parent=parent,
                    output=str(ROOT / name),
                )
            )
        jobs.append(
            dict(model=condition, order=index + 1, updates=12000, stages=stages)
        )
    return jobs


def verify_plan_sources(plan):
    import torch

    for name, digest in plan["code"]["source_sha256"].items():
        if sha(Path(name)) != digest:
            raise ValueError("Pinned training source changed: " + name)
    for name, digest in plan["controller_sha256"].items():
        if sha(Path(name)) != digest:
            raise ValueError("Pinned controller/runtime changed: " + name)
    if (
        sha(Path("uv.lock")) != plan["lock_sha256"]
        or str(torch.__version__) != plan["torch_version"]
    ):
        raise ValueError("Frozen environment changed")


def reference_report(frames=64):
    path = REFERENCE / f"query-T{frames}" / "validation-update-00006000.json"
    return read(path)


def validate_report(report):
    reference = reference_report()
    if (
        report["identities"] != reference["identities"]
        or report["snapshot"] != reference["snapshot"]
    ):
        raise ValueError(
            "Validation frame/teacher IDs or snapshot differ from the baseline"
        )
    if report["metrics"]["support"] != reference["metrics"]["support"]:
        raise ValueError("Validation support differs from the baseline")
    if (
        report["metrics"]["support"]["observed"] != 85473
        or report["metrics"]["support"]["frames"] != 128668
    ):
        raise ValueError("Unexpected fixed validation population")
    if report["metrics"]["distance_threshold_source_px"] != 20:
        raise ValueError("Distance threshold changed")
    for field in FIELDS:
        if not math.isfinite(report["metrics"][field]):
            raise ValueError("Nonfinite validation metric")


def observed_identity(path):
    digest = hashlib.sha256()
    count = 0
    with path.open() as stream:
        for line in stream:
            row = json.loads(line)
            if row["kind"] == "observed":
                key = [
                    row[k]
                    for k in (
                        "clip_id",
                        "frame_index",
                        "native_fps",
                        "pts",
                        "time_base",
                    )
                ]
                digest.update((json.dumps(key, separators=(",", ":")) + "\n").encode())
                count += 1
    if count != 85473:
        raise ValueError("Observed prediction population changed")
    return digest.hexdigest()


def validate_log(stage, directory):
    from src.tasks.ball_detection.training.temporal.configuration import UpdateBudget

    bud = UpdateBudget(
        **{k: stage["budget"][k] for k in UpdateBudget.__dataclass_fields__}
    )
    ref = (
        REFERENCE
        / ("query-T32" if bud.schedule_offset == 0 else "query-T64")
        / "train.jsonl"
    )
    expected_rows = [json.loads(line) for line in ref.read_text().splitlines()]
    rows = [
        json.loads(line)
        for line in (directory / "train.jsonl").read_text().splitlines()
    ]
    keys = (
        "pack_ids",
        "pack_indices",
        "sample_position",
        "input_frames",
        "cumulative_input_frames",
        "augmentation_sha256",
        "valid_teachers",
        "teachers_by_kind",
        "original_teachers_by_kind",
    )
    for index, row in enumerate(rows):
        if row["update"] != index + 1 or row["input_frames"] != 128:
            raise ValueError("Discontinuous update or frame budget")
        for key in keys:
            if row[key] != expected_rows[index][key]:
                raise ValueError(
                    f"Baseline pack/augmentation mismatch: {stage['name']} update{index + 1} {key}"
                )
        for key in ("loss", "gradient_norm", "learning_rate", "encoder_learning_rate"):
            if not math.isfinite(row[key]):
                raise ValueError("Nonfinite training value: " + key)
        rate = bud.lr(index)
        frozen = index < bud.freeze_updates
        if (
            not math.isclose(row["learning_rate"], rate, rel_tol=1e-12, abs_tol=1e-15)
            or row["encoder_frozen"] != frozen
            or not math.isclose(
                row["encoder_learning_rate"],
                0.0 if frozen else rate * 0.1,
                rel_tol=1e-12,
                abs_tol=1e-15,
            )
        ):
            raise ValueError(
                "Actual LR/freeze boundary differs from the explicit stage schedule"
            )
    return rows


def verify_stage_result(root, stage):
    directory = root / stage["name"]
    if not read(directory / "COMPLETED.json")["complete"]:
        raise ValueError("Incomplete stage cannot admit its successor")
    rows = validate_log(stage, directory)
    if len(rows) != 6000 or read(directory / "latest.json")["update"] != 6000:
        raise ValueError("Stage has not completed6000 updates")
    report = read(directory / "validation-update-00006000.json")
    validate_report(report)
    predictions = directory / report["predictions"]["file"]
    if sha(predictions) != report["predictions"]["sha256"]:
        raise ValueError("Saved validation predictions changed")
    if (
        observed_identity(predictions)
        != read(root / "plan.json")["baseline_observed_identity_sha256"]
    ):
        raise ValueError("Position teacher IDs differ")
    evidence = dict(
        stage=stage["name"],
        validated_at=now(),
        updates=6000,
        baseline_inputs_exact=True,
        identities=report["identities"],
        observed_identity_sha256=observed_identity(predictions),
        metrics={k: report["metrics"][k] for k in FIELDS},
        freeze_boundary=[rows[i] for i in (0, 599, 600, 5999)],
    )
    write(directory / "acceptance.json", evidence)
    store(root).publish_file(directory / "acceptance.json")
    return evidence


def verify_prerequisite():
    """Read back the eight-layer terminal full state; never start12/24-layer jobs."""
    import torch

    status = read(OLD_ROOT / "job-status/d512-l8.json")
    _, receipt = select_receipt(DRIVE / CAPACITY / "persistence" / OLD_STAGE)
    if not status["complete"] or not receipt["generation"].startswith(
        "completed-00006000-"
    ):
        raise ValueError(
            "The #1675 eight-layer terminal checkpoint/validation is not complete"
        )
    accepted = ROOT / "predecessor-acceptance.json"
    if accepted.exists():
        if read(accepted)["receipt"] != receipt:
            raise ValueError("The prerequisite terminal generation changed")
        marker = DRIVE / receipt["remote"].split(":", 1)[1] / "_SNAPSHOT.json"
        if sha(marker) != receipt["manifest_sha256"]:
            raise ValueError("Prerequisite marker changed")
        return
    with tempfile.TemporaryDirectory(prefix="ball-eight-layer-readback-") as temporary:
        dest = Path(temporary) / "restored"
        restored = snapshot_store(dest, CAPACITY / OLD_STAGE).restore(
            receipt["generation"], manifest_sha256=receipt["manifest_sha256"]
        )
        latest = read(dest / "latest.json")
        checkpoint = torch.load(
            dest / latest["checkpoint"], map_location="cpu", weights_only=True
        )
        report = read(dest / "validation-update-00006000.json")
        validate_report(report)
        if (
            latest["update"] != 6000
            or checkpoint["updates_completed"] != 6000
            or not checkpoint["optimizer"]["state"]
        ):
            raise ValueError("Prerequisite is not a full terminal checkpoint")
        if checkpoint["model_artifact"]["training"]["cumulative_updates"] != 22000:
            raise ValueError("Prerequisite query cumulative updates are not12000")
        write(
            accepted,
            dict(
                complete=True,
                accepted_at=now(),
                receipt=receipt,
                restore=restored,
                query_updates=12000,
                report=report,
                deeper_models_held=[12, 24],
            ),
        )
    store().publish_file(accepted)


def prepare():
    import torch

    from src.tasks.ball_detection.checkpoints.provenance import source_identity
    from src.tasks.ball_detection.training.temporal.configuration import UpdateBudget

    if not read(ROOT / "initialization-audit.json")["complete"]:
        raise ValueError(
            "Historical query initialization must be independently reproduced"
        )
    baseline = read(REFERENCE / "query-T32/run.json")["recipe"]
    parents = []
    for path in (
        REFERENCE / "common-dpt-T32/checkpoint-update-00010000.pt",
        WORKSPACE
        / "outputs/ball_detection/train/capacity_scaling/s42-20261103-main/common-dpt-T32/checkpoint-update-00010000.pt",
    ):
        saved = torch.load(path, map_location="cpu", weights_only=True)
        if (
            saved["updates_completed"] != 10000
            or saved["sample_position"] != 10000
            or not saved["optimizer"]["state"]
        ):
            raise ValueError("Common DPT initialization must be a full10k checkpoint")
        for key in ("budget", "runtime", "augmentation", "data", "image_decode"):
            if parents and saved["recipe"][key] != parents[0]["recipe"][key]:
                raise ValueError("The two DPT pretraining recipes differ: " + key)
        parents.append(
            dict(
                model=saved["recipe"]["model"],
                parent=dict(checkpoint=str(path), sha256=sha(path), update=10000),
                recipe=saved["recipe"],
                code=saved["code"],
            )
        )
    first_budget = baseline["budget"]
    jobs = definitions(parents[0], parents[1], first_budget)
    for job in jobs:
        for s in job["stages"]:
            budget = UpdateBudget(
                **{k: s["budget"][k] for k in UpdateBudget.__dataclass_fields__}
            )
            if (
                budget.updates != 6000
                or budget.checkpoint_interval != 500
                or budget.input_frames != 128
                or budget.freeze_updates != 600
            ):
                raise ValueError("Production budget changed")
    for frames in (32, 64):
        if (
            sha(PREPARED / f"train_T{frames}.json")
            != read(REFERENCE / f"query-T{frames}/run.json")["recipe"][
                "manifest_sha256"
            ]
        ):
            raise ValueError(
                "Prepared training packs differ from the historical baseline"
            )
        if (
            sha(PREPARED / f"evaluation_T{frames}.json")
            != reference_report(frames)["manifest_sha256"]
        ):
            raise ValueError("Prepared validation differs from the baseline")
    controller_paths = sorted(HERE.glob("*.py")) + [
        LEGACY_HERE / "campaign.py",
        LEGACY_HERE / "train_model.py",
        QUEUE_CLI,
        QUEUE_CLI.with_name("queue_history.py"),
        Path("src/utils/training_queue.py"),
    ]
    controller_paths += sorted(Path("src/utils/artifact_store").glob("*.py"))
    plan = dict(
        schema="ball_query_controls.v1",
        issue=1899,
        created_at=now(),
        code=source_identity(),
        jobs=jobs,
        query_updates=36000,
        runtime=baseline["runtime"],
        augmentation=baseline["augmentation"],
        input_frames=128,
        seed=42,
        prepared=str(PREPARED),
        rclone_config=str(CONFIG),
        torch_version=str(torch.__version__),
        lock_sha256=sha(Path("uv.lock")),
        controller_sha256={
            str(p.resolve().relative_to(Path.cwd()))
            if p.resolve().is_relative_to(Path.cwd())
            else str(p): sha(p)
            for p in controller_paths
        },
        initial_parents=parents,
        baseline_report=reference_report(),
        baseline_observed_identity_sha256=observed_identity(
            REFERENCE / "query-T64/validation-update-00006000.frames.jsonl"
        ),
        rng_policy="Legacy seed42 construction at each new stage (#1514); full Python/NumPy/Torch/CUDA RNG on same-stage recovery.",
        held_models=[12, 24],
        prerequisite=dict(
            root=str(OLD_ROOT),
            model="d512-l8",
            source_commit="270702138e34c580c8fad703157d051cebc519a0",
        ),
        failure_policy="Record failed condition/latest durable state and stop the shared queue before releasing the GPU lock; no skips or automatic retries.",
    )
    if (ROOT / "plan.json").exists():
        raise ValueError(
            "Plan already exists; verify it instead of rewriting the frozen run"
        )
    ROOT.mkdir(parents=True, exist_ok=True)
    write(ROOT / "plan.json", plan)
    write(
        ROOT / "pretraining-audit.json",
        dict(
            complete=True,
            same_recipe_fields=[
                "budget",
                "runtime",
                "augmentation",
                "data",
                "image_decode",
            ],
            parents=parents,
            verified_at=now(),
        ),
    )
    write(
        ROOT / "scope.json",
        dict(
            issue=1899,
            execution_order=["d512-l8", *CONDITIONS],
            held_models=[12, 24],
            initial_gpu_processes=[],
            old_queue_present=False,
            unrelated_issue1859_untouched=True,
            observed_at=now(),
        ),
    )
    return plan


def stop_failed(root, queue, model, error):
    # Stop admission before the failing all-GPU job returns to the queue.
    queue.mkdir(parents=True, exist_ok=True)
    (queue / "stop").touch()
    durable = []
    for directory in (DRIVE / RELATIVE / "persistence").glob("*"):
        if directory.is_dir():
            try:
                _, receipt = select_receipt(directory)
                durable.append(receipt)
            except ValueError:
                pass  # An initialization-only stage has no resumable checkpoint.
    record = dict(
        schema="ball_query_controls_failure.v1",
        at=now(),
        model=model,
        error=error,
        successors_held=True,
        automatic_retry=False,
        latest_durable=durable,
    )
    write(root / "FAILED.json", record)
    write(root / "notifications" / f"failed-{model}-{os.getpid()}.json", record)
    return record


def compare():
    baseline = reference_report()
    validate_report(baseline)
    rows = [
        dict(
            condition="baseline",
            values={k: baseline["metrics"][k] for k in FIELDS},
            delta={k: 0.0 for k in FIELDS},
        )
    ]
    for condition in CONDITIONS:
        path = ROOT / f"{condition}-s2-T64/validation-update-00006000.json"
        status = ROOT / "job-status" / f"{condition}.json"
        if (
            not path.exists()
            or not status.exists()
            or not read(status).get("complete")
            or not (path.parent / "acceptance.json").exists()
        ):
            rows.append(dict(condition=condition, values=None, delta=None))
            continue
        report = read(path)
        validate_report(report)
        values = {k: report["metrics"][k] for k in FIELDS}
        delta = {
            k: (values[k] - baseline["metrics"][k])
            * (100 if k == "within_distance_rate" else 1)
            for k in FIELDS
        }
        rows.append(dict(condition=condition, values=values, delta=delta))
    result = dict(
        schema="ball_query_control_comparison.v1",
        updated_at=now(),
        complete=all(r["values"] is not None for r in rows),
        frames=128668,
        observed=85473,
        rate_delta_unit="percentage_points",
        error_delta_unit="source_px",
        rows=rows,
    )
    write(ROOT / "comparison.json", result)
    lines = [
        "| 条件 | mean px (差) | median px (差) | P95 px (差) | 20px以内 % (差pt) |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        cells = (
            ["未完了"] * 4
            if row["values"] is None
            else [
                f"{row['values'][k] * (100 if k == 'within_distance_rate' else 1):.4f} ({row['delta'][k]:+.4f})"
                for k in FIELDS
            ]
        )
        lines.append("| " + row["condition"] + " | " + " | ".join(cells) + " |")
    (ROOT / "comparison.md").write_text("\n".join(lines) + "\n")
    store().publish_file(ROOT / "comparison.json")
    store().publish_file(ROOT / "comparison.md")
    if result["complete"]:
        write(ROOT / "COMPLETED.json", result)
        store().publish_file(ROOT / "COMPLETED.json")
    return result


def launch(session, attempt):
    plan = read(ROOT / "plan.json")
    verify_plan_sources(plan)
    if (
        not read(ROOT / "inputs-receipt.json")["status"] == "verified"
        or not read(ROOT / "data-verified.json")["complete"]
    ):
        raise ValueError(
            "Verified inputs/source and full dataset are required before launch"
        )
    if (ROOT / "FAILED.json").exists():
        raise ValueError(
            "Archive the diagnosed failure explicitly before a new attempt"
        )
    registrations = []
    definitions = [
        (
            "d512-l8",
            OLD_CWD,
            [
                str(OLD_CWD / ".venv/bin/python"),
                "-u",
                "-B",
                str(
                    OLD_CWD
                    / "knowledge/runs/run-ball-capacity-12k-20261110/train_model.py"
                ),
                "--root",
                str(OLD_ROOT),
                "--model",
                "d512-l8",
                "--attempt",
                str(max(2, attempt)),
            ],
        )
    ]
    definitions += [
        (
            c,
            Path.cwd(),
            [
                str(Path.cwd() / ".venv/bin/python"),
                "-u",
                "-B",
                str(HERE / "train.py"),
                "--root",
                str(ROOT),
                "--model",
                c,
                "--attempt",
                str(attempt),
            ],
        )
        for c in CONDITIONS
    ]
    for model, cwd, argv in definitions:
        command = shlex.join(
            ["env", f"PATH=/tmp/ball-tools:{os.environ['PATH']}", *argv]
        )
        request = [
            "bash",
            str(QUEUE_CLI),
            "add",
            command,
            "--name",
            f"query-controls-{model}-a{attempt}",
            "--provider",
            "codex",
            "--session",
            session,
            "--issue",
            "1899",
            "--resource",
            "all",
            "--idempotency-key",
            f"query-controls/s42-20261111/{model}/attempt-{attempt}",
        ]
        output = subprocess.check_output(
            request,
            cwd=cwd,
            env=dict(os.environ, TRAINING_QUEUE_DIR=str(QUEUE)),
            text=True,
        ).strip()
        name = output.removeprefix("queued: ").strip()
        if "/" in name or not name.endswith(".job"):
            raise ValueError("Unexpected queue registration: " + output)
        registrations.append(
            dict(
                model=model,
                queue_file=name,
                queue_id=name[:-4],
                cwd=str(cwd),
                argv=argv,
                command=command,
                attempt=attempt,
            )
        )
    record = dict(
        schema="ball_query_controls_queue.v1",
        complete=True,
        registered_at=now(),
        jobs=registrations,
    )
    write(ROOT / "queue-registration.json", record)
    store().publish_file(ROOT / "queue-registration.json")
    request = dict(
        ready=True,
        cwd=str(Path.cwd()),
        argv=[
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
            "30",
        ],
        env=dict(
            TRAINING_QUEUE_DIR=str(QUEUE), PATH="/tmp/ball-tools:" + os.environ["PATH"]
        ),
        log=str(ROOT / "queue-service.log"),
    )
    write(ROOT / "queue-service-request.json", request)
    store().publish_file(ROOT / "queue-service-request.json")
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="operation", required=True)
    sub.add_parser("prepare")
    sub.add_parser("compare")
    sub.add_parser("status")
    p = sub.add_parser("launch")
    p.add_argument("--session", required=True)
    p.add_argument("--attempt", type=int, required=True)
    args = parser.parse_args()
    if args.operation == "prepare":
        result = prepare()
    elif args.operation == "compare":
        result = compare()
    elif args.operation == "launch":
        with (ROOT / "registration.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            result = launch(args.session, args.attempt)
    else:
        result = {p.name: read(p) for p in (ROOT / "job-status").glob("*.json")}
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
