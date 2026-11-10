"""Independently accept >=601 updates and read back the first Drive checkpoint."""

from __future__ import annotations

import ctypes
import json
import math
import os
import select
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))

import torch
from campaign import DRIVE, QUEUE, RELATIVE, ROOT, now, read, write
from train_model import named_optimizer, same

from src.tasks.ball_detection.training.temporal.configuration import UpdateBudget
from src.tasks.ball_detection.training.temporal.packs import PackSampler
from src.utils.artifact_store import RcloneSnapshotStore
from src.utils.checksum import dual_sha256


def wait_for_601(output):
    """Use filesystem events rather than periodic progress polling."""
    libc = ctypes.CDLL(None, use_errno=True)
    descriptor = libc.inotify_init1(os.O_CLOEXEC | os.O_NONBLOCK)
    if descriptor < 0:
        raise OSError(ctypes.get_errno(), "Cannot watch training events")
    watched = set()
    deadline = time.monotonic() + 7200
    last_report = -1
    try:
        while True:
            for directory in (ROOT, output):
                if directory.exists() and directory not in watched:
                    if (
                        libc.inotify_add_watch(
                            descriptor,
                            os.fsencode(directory),
                            0x00000008 | 0x00000080 | 0x00000100,
                        )
                        < 0
                    ):
                        raise OSError(
                            ctypes.get_errno(), "Cannot watch " + str(directory)
                        )
                    watched.add(directory)
            if (ROOT / "FAILED.json").exists():
                raise ValueError(
                    "The campaign recorded a failure: "
                    + (ROOT / "FAILED.json").read_text()
                )
            log = output / "train.jsonl"
            rows = []
            if log.exists():
                raw = log.read_bytes()
                rows = [
                    json.loads(line) for line in raw.splitlines() if line.endswith(b"}")
                ]
            update = len(rows)
            milestone = 601 if update >= 601 else update // 100 * 100
            if milestone > last_report:
                print(
                    json.dumps(
                        dict(event="startup_progress", updates=update, at=now())
                    ),
                    flush=True,
                )
                progress = read(ROOT / "worker-progress.json")
                progress.update(
                    updated_at=now(),
                    phase="verify_first_model_stability",
                    first_model_updates=update,
                )
                write(ROOT / "worker-progress.json", progress)
                last_report = milestone
            if update >= 601:
                return rows
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(
                    "No 601-update acceptance within the finite startup window"
                )
            ready, _, _ = select.select([descriptor], [], [], min(remaining, 30))
            if ready:
                os.read(descriptor, 65536)
    finally:
        os.close(descriptor)


def main():
    plan = read(ROOT / "plan.json")
    stage = plan["jobs"][0]["stages"][0]
    output = Path(stage["output"])
    rows = wait_for_601(output)
    budget = UpdateBudget(
        **{k: stage["budget"][k] for k in UpdateBudget.__dataclass_fields__}
    )
    for index, row in enumerate(rows, 1):
        if (
            row["update"] != index
            or row["sample_position"] != 16000 + index
            or row["cumulative_updates"] != 16000 + index
            or row["input_frames"] != 128
        ):
            raise ValueError("Update/sample/global/frame continuity failed")
        if not all(
            math.isfinite(row[k])
            for k in (
                "loss",
                "loss_sum",
                "gradient_norm",
                "learning_rate",
                "encoder_learning_rate",
            )
        ):
            raise ValueError("Nonfinite training observation")
        if not math.isclose(row["learning_rate"], budget.lr(index - 1), rel_tol=1e-12):
            raise ValueError("Continuous schedule did not match the fixed LR")
        expected_encoder = 0.0 if index <= 600 else budget.lr(index - 1) * 0.1
        if row["encoder_frozen"] != (index <= 600) or not math.isclose(
            row["encoder_learning_rate"], expected_encoder, rel_tol=1e-12
        ):
            raise ValueError("CNN freeze boundary did not occur at 600/601")
    expected_pack = next(iter(PackSampler(30478, seed=42, position=16000, count=1)))
    if rows[0]["pack_indices"] != [expected_pack]:
        raise ValueError("The first sample did not continue the counter-seeded sampler")
    candidates = list(
        (ROOT / "persistence" / output.name).glob("checkpoint-00000500-*.json")
    )
    if len(candidates) != 1:
        raise ValueError("The first 500-update receipt is missing or ambiguous")
    receipt = read(candidates[0])
    scratch = ROOT / "acceptance" / "first500-readback"
    restored = RcloneSnapshotStore(
        scratch,
        remote="mounted_drive",
        remote_root=f"{RELATIVE}/{output.name}",
        config_path=Path(plan["rclone_config"]),
    ).restore(receipt["generation"], manifest_sha256=receipt["manifest_sha256"])
    marker = DRIVE / RELATIVE / output.name / receipt["generation"] / "_SNAPSHOT.json"
    if dual_sha256(marker) != receipt["manifest_sha256"]:
        raise ValueError("Readback marker no longer matches the committed receipt")
    latest = read(scratch / "latest.json")
    checkpoint = scratch / latest["checkpoint"]
    if latest["update"] != 500 or dual_sha256(checkpoint) != latest["sha256"]:
        raise ValueError("Readback checkpoint identity/update differs")
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    parent = torch.load(
        stage["parent"]["checkpoint"], map_location="cpu", weights_only=True
    )
    if (
        saved["schema"] != "ball_temporal_checkpoint.v2"
        or saved["updates_completed"] != 500
        or saved["sample_position"] != 16500
        or saved["recipe"]["budget"] != stage["budget"]
        or saved["code"]["source_sha256"] != plan["fixed_source_sha256"]
        or saved["code"]["torch_version"] != plan["torch_version"]
    ):
        raise ValueError(
            "The durable full-state checkpoint does not match this experiment"
        )
    committed = [
        json.loads(line) for line in (scratch / "train.jsonl").read_text().splitlines()
    ]
    if len(committed) != 500 or saved["train_log"]["sha256"] != dual_sha256(
        scratch / "train.jsonl"
    ):
        raise ValueError("The committed 500-update log does not match its checkpoint")
    if committed != rows[:500]:
        raise ValueError("Live and independently restored log prefixes differ")
    old = named_optimizer(parent["optimizer"], parent["optimizer_parameter_names"])
    new = named_optimizer(saved["optimizer"], saved["optimizer_parameter_names"])
    encoder_states, head_states = 0, 0
    for name, state in new.items():
        if name.startswith("encoder."):
            if not same(state, old[name]):
                raise ValueError("Frozen CNN AdamW state changed: " + name)
            encoder_states += state is not None
        elif state is not None:
            if int(state["step"]) != int(old[name]["step"]) + 500:
                raise ValueError("Head AdamW step did not continue: " + name)
            head_states += 1
    for name, value in saved["model_artifact"]["state_dict"].items():
        if value.is_floating_point() and not bool(torch.isfinite(value).all()):
            raise ValueError("A durable model tensor is nonfinite")
        if name.startswith("encoder.") and not same(
            value, parent["model_artifact"]["state_dict"][name]
        ):
            raise ValueError("The frozen CNN model state changed")
    continuity = read(scratch / "continuity.json")
    if (
        not continuity["named_states_exact"]
        or not continuity["rng_exact_before_first_batch"]
        or continuity["new_model_keys"] != 0
    ):
        raise ValueError("The model/optimizer/RNG transfer audit failed")
    registration = read(ROOT / "queue-registration.json")
    jobs = registration["jobs"]
    if not registration["complete"] or len(jobs) != 9:
        raise ValueError("Nine queue identities are required")
    if not (QUEUE / "running" / jobs[0]["queue_file"]).is_file() or not all(
        (QUEUE / "jobs" / j["queue_file"]).is_file() for j in jobs[1:]
    ):
        raise ValueError("Expected first running plus eight queued jobs")
    service = read(ROOT / "queue-service-state.json")
    training = read(ROOT / "job-status" / "d256-l4.json")
    for pid in (service["service_pid"], service["worker_pid"], training["pid"]):
        if not Path(f"/proc/{pid}").is_dir():
            raise ValueError("A required external process is no longer alive")
    from queue_service import ancestors

    lineage = ancestors(training["pid"])
    if service["service_pid"] not in [p["pid"] for p in lineage]:
        raise ValueError("Training is not owned by the external service")
    if any(
        "agent_runtime.py" in Path(f"/proc/{p['pid']}/cmdline").read_text()
        for p in lineage
    ):
        raise ValueError("Training remains under a finite agent supervisor")
    gpu = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,used_memory",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    ).strip()
    result = dict(
        schema="ball_capacity_startup_acceptance.v1",
        complete=True,
        verified_at=now(),
        observed_updates=len(rows),
        query_cumulative_updates=6000 + len(rows),
        global_updates=16000 + len(rows),
        loss_range=[min(r["loss"] for r in rows), max(r["loss"] for r in rows)],
        gradient_range=[
            min(r["gradient_norm"] for r in rows),
            max(r["gradient_norm"] for r in rows),
        ],
        first_lr=rows[0]["learning_rate"],
        last_lr=rows[-1]["learning_rate"],
        update600=rows[599],
        update601=rows[600],
        data=read(ROOT / "data-verified.json"),
        checkpoint500_sha256=latest["sha256"],
        receipt=receipt,
        readback=restored,
        restored_payloads=len(read(marker)["files"]),
        optimizer_encoder_states_unchanged=encoder_states,
        optimizer_head_states_advanced500=head_states,
        continuity=continuity,
        queue_ids=[j["queue_id"] for j in jobs],
        remaining_queued=8,
        service=service,
        training_pid=training["pid"],
        training_lineage=lineage,
        gpu_processes=gpu,
    )
    write(ROOT / "startup-acceptance.json", result)
    print(
        json.dumps(
            dict(
                complete=True,
                observed_updates=len(rows),
                checkpoint500_sha256=latest["sha256"],
                remaining_queued=8,
            )
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
