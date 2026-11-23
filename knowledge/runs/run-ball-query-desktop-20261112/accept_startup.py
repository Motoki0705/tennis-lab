"""Finite, event-driven acceptance of the already registered desktop attempt."""

import argparse
import ctypes
import json
import math
import os
import select
import statistics
import struct
import time
from pathlib import Path

from desktop import GPU, REMOTE, load_frozen, read, sha, validate_result, write


def last_row(path):
    if not path.exists() or not path.stat().st_size:
        return None
    with path.open("rb") as stream:
        stream.seek(max(0, path.stat().st_size - 65536))
        lines = stream.read().splitlines()
    return json.loads(lines[-1]) if lines else None


def ready(root, target):
    for name in ("production-FAILED.json", "production-RESOURCE_STOP.json"):
        if (root / name).exists():
            raise RuntimeError(name + ": " + (root / name).read_text())
    production = root / "production"
    if not (production / "length-s1-T64/acceptance.json").exists():
        return False
    row = last_row(production / "length-s2-T64/train.jsonl")
    # Include a few ordinary unfrozen updates for an estimate that excludes JIT.
    required = 611 if target == 601 else target
    if row is None or row["update"] < required:
        return False
    receipts = list(
        (production / "readback-receipts").glob(
            "length-s1-T64-checkpoint-00006000-*.json"
        )
    )
    return len(receipts) > 0


def accept(root, source, target):
    import torch

    control, frozen, plan = load_frozen(source, root)
    production = root / "production"
    stages = next(j for j in plan["jobs"] if j["model"] == "length")["stages"]
    first, second = (production / stage["name"] for stage in stages)
    validate_result(control, plan, root, first, stages[0])
    rows1 = control.validate_log(stages[0], first)
    rows2 = control.validate_log(stages[1], second)
    if len(rows2) < target:
        raise ValueError("Second stage has not reached the declared acceptance point")
    continuity1, continuity2 = (
        read(first / "continuity.json"),
        read(second / "continuity.json"),
    )
    if not all(
        continuity1.get(key) is True
        for key in ("model_exact", "adamw_exact", "rng_exact_before_first_batch")
    ):
        raise ValueError("First resumed update has no complete state audit")
    if (
        continuity1["parent_update"] != 5500
        or continuity1["parent_sample_position"] != 15500
    ):
        raise ValueError("Wrong resume update or sample position")
    if (
        not continuity2["named_states_exact"]
        or not continuity2["same_model_continuation"]
        or continuity2["inherited_states"] != 201
    ):
        raise ValueError("The second stage did not inherit every named AdamW state")
    receipt_paths = list(
        (production / "persistence/length-s1-T64").glob("checkpoint-00006000-*.json")
    )
    if len(receipt_paths) != 1:
        raise ValueError("Ambiguous first periodic snapshot")
    receipt = read(receipt_paths[0])
    if receipt["status"] != "verified" or not receipt["remote"].startswith(
        "gdrive:" + REMOTE + "/production/length-s1-T64/"
    ):
        raise ValueError(
            "Periodic snapshot was not verified at the declared desktop destination"
        )
    generation = receipt["generation"]
    readback_receipt = read(
        production / "readback-receipts" / ("length-s1-T64-" + generation + ".json")
    )
    if (
        readback_receipt["status"] != "verified"
        or readback_receipt["manifest_sha256"] != receipt["manifest_sha256"]
    ):
        raise ValueError("Independent readback marker differs")
    restored = root / "readback/length-s1-T64" / generation
    latest = read(restored / "latest.json")
    current_latest = read(first / "latest.json")
    if (
        latest != current_latest
        or latest["update"] != 6000
        or sha(restored / latest["checkpoint"]) != latest["sha256"]
    ):
        raise ValueError("Readback latest/full checkpoint changed")
    saved = torch.load(
        restored / latest["checkpoint"], map_location="cpu", weights_only=True
    )
    actual = torch.load(
        first / latest["checkpoint"], map_location="cpu", weights_only=True
    )
    if not frozen.same(saved, actual):
        raise ValueError("Readback model/AdamW/RNG/schedule/sample differs")
    if saved["sample_position"] != 16000 or saved["updates_completed"] != 6000:
        raise ValueError("First stage did not complete exactly 6000 shared packs")
    histogram = {}
    for state in saved["optimizer"]["state"].values():
        step = str(int(state["step"]))
        histogram[step] = histogram.get(step, 0) + 1
    if histogram != {"15400": 116, "6000": 85}:
        raise ValueError("CNN/query AdamW step counts changed")
    for name in (
        "train.jsonl",
        "run.json",
        "train_manifest.json",
        "evaluation_manifest.json",
    ):
        if sha(restored / name) != sha(first / name):
            raise ValueError("Independent readback log/recipe/manifests differ")
    report = read(restored / "validation-update-00006000.json")
    reference = read(root / "reference/query-T64/validation-update-00006000.json")
    if report["identities"] != reference["identities"]:
        raise ValueError("Validation frame and teacher identities differ")
    expected_rows = (rows1[5500], rows2[0]) + (
        (rows2[599], rows2[600]) if target >= 601 else ()
    )
    for row in expected_rows:
        if not all(
            math.isfinite(row[key])
            for key in (
                "loss",
                "gradient_norm",
                "learning_rate",
                "encoder_learning_rate",
            )
        ):
            raise ValueError("Nonfinite accepted training row")
    latest2 = rows2[-1]
    steady = [
        row for row in rows2 if row["update"] > 601 and not row["encoder_frozen"]
    ][-50:]
    speed_source = "second_stage_unfrozen_updates_after_601"
    if not steady:
        steady = rows1[-50:]
        speed_source = "first_stage_unfrozen_last_50_updates"
    step_seconds = statistics.median(row["seconds"] for row in steady)
    environment = read(root / "production-environment.json")
    if environment["gpu"] != GPU or environment["torch"] != plan["torch_version"]:
        raise ValueError("Production GPU or torch version changed")
    queue = source.parents[2] / ".training_queue"
    job = read(root / "production-registration.json")["job"]
    state_path = queue / "state" / (job + ".state")
    process = Path(f"/proc/{environment['pid']}/stat").read_text()
    fields = process[process.rindex(")") + 2 :].split()
    status = Path(f"/proc/{environment['pid']}/status").read_text().splitlines()
    vmhwm_kib = int(
        next(line.split()[1] for line in status if line.startswith("VmHWM:"))
    )
    memory = read(root / "production-host-memory.json")
    result = dict(
        schema="ball_query_desktop_startup_acceptance.v1",
        complete=True,
        accepted_at=time.time(),
        target_stage2_update=target,
        source_commit=plan["code"]["base_commit"],
        adapter_training_commit="eda5ee08073da498347d411e075905f4ac94e25e",
        queue_id=job,
        resource="all",
        queue_state=state_path.read_text(),
        trainer=dict(pid=environment["pid"], ppid=int(fields[1]), pgid=int(fields[2])),
        environment=environment,
        resume_continuity=continuity1,
        stage2_continuity=continuity2,
        first_resumed_row=rows1[5500],
        stage1_terminal_row=rows1[-1],
        stage2_first_row=rows2[0],
        stage2_latest_row=latest2,
        cnn_unfreeze_rows=[rows2[i] for i in (599, 600)] if target >= 601 else None,
        baseline_pack_teacher_augmentation_exact=True,
        validation=dict(
            support=report["metrics"]["support"],
            identities=report["identities"],
            metrics=report["metrics"],
            observed_identity_sha256=control.observed_identity(
                restored / report["predictions"]["file"]
            ),
        ),
        first_periodic_snapshot=receipt,
        independent_readback=readback_receipt,
        readback_full_state_exact=True,
        optimizer_step_histogram=histogram,
        local_full_checkpoint=str(first / latest["checkpoint"]),
        validation_through_checkpoint_seconds=(first / latest["checkpoint"])
        .stat()
        .st_mtime
        - (first / "weights-update-00006000.pt").stat().st_mtime,
        resource_measurements=dict(
            peak_training_allocated_bytes=max(
                row["peak_vram_bytes"] for row in rows1[5500:] + rows2
            ),
            peak_training_reserved_bytes=max(
                row["peak_reserved_vram_bytes"] for row in rows1[5500:] + rows2
            ),
            trainer_peak_rss_bytes=vmhwm_kib * 1024,
            minimum_host_available_bytes=memory["minimum_available_bytes"],
        ),
        speed=dict(
            median_unfrozen_update_seconds=step_seconds,
            measurement_source=speed_source,
            updates_remaining=6000 - latest2["update"],
            training_only_seconds_remaining=(6000 - latest2["update"]) * step_seconds,
            excludes="future checkpoint publication/readback, fixed terminal validation, compilation",
        ),
        validation_final_12k_pending=True,
        terminal_event_service="tennis-i1899-length-desktop-a1.service",
        validator_requested=0,
        validator_attempted=0,
        validator_completed=0,
    )
    write(root / "startup-acceptance.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--stage2-update", type=int, choices=(1, 601), default=601)
    parser.add_argument("--timeout-seconds", type=int, default=3600)
    args = parser.parse_args()
    root, source = args.root.resolve(), args.source.resolve()
    libc = ctypes.CDLL(None, use_errno=True)
    descriptor = libc.inotify_init1(os.O_CLOEXEC | os.O_NONBLOCK)
    if descriptor < 0:
        raise OSError(ctypes.get_errno(), "Cannot watch acceptance artifacts")
    pid = read(root / "production-environment.json")["pid"]
    pidfd = os.pidfd_open(pid)
    watched = {}

    def refresh():
        directories = [root, root / "production"] + [
            p for p in (root / "production").glob("*") if p.is_dir()
        ]
        for path in directories:
            if path in watched.values():
                continue
            wd = libc.inotify_add_watch(
                descriptor, os.fsencode(path), 0x8 | 0x80 | 0x100
            )
            if wd < 0:
                raise OSError(ctypes.get_errno(), "Cannot watch " + str(path))
            watched[wd] = path

    start = time.monotonic()
    try:
        while True:
            refresh()
            if ready(root, args.stage2_update):
                result = accept(root, source, args.stage2_update)
                print(
                    json.dumps(
                        dict(
                            complete=True,
                            accepted_at=result["accepted_at"],
                            stage2_update=result["stage2_latest_row"]["update"],
                        )
                    ),
                    flush=True,
                )
                return
            remaining = args.timeout_seconds - (time.monotonic() - start)
            if remaining <= 0:
                raise TimeoutError("Startup acceptance deadline reached")
            events, _, _ = select.select(
                [descriptor, pidfd], [], [], min(60, remaining)
            )
            if pidfd in events:
                if ready(root, args.stage2_update):
                    continue
                raise RuntimeError(
                    "Production trainer exited before startup acceptance"
                )
            if descriptor in events:
                data = os.read(descriptor, 65536)
                # Drain kernel events; a following pass reads atomically published artifacts.
                offset = 0
                while offset < len(data):
                    _, _, _, length = struct.unpack_from("iIII", data, offset)
                    offset += 16 + length
    finally:
        os.close(pidfd)
        os.close(descriptor)


if __name__ == "__main__":
    main()
