"""Verify terminal local/Drive readback artifacts without rerunning inference."""

import argparse
import hashlib
import json
import math
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch


def read(path):
    return json.loads(path.read_text())


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    args = parser.parse_args()
    root = args.run_root.resolve()
    evidence = Path(__file__).resolve().parent
    sys.path.insert(0, str(args.adapter.resolve()))
    import desktop

    control, frozen, plan = desktop.load_frozen(args.source.resolve(), root)
    stages = next(job for job in plan["jobs"] if job["model"] == "length")["stages"]
    baseline = read(root / "reference/query-T64/validation-update-00006000.json")
    results = []
    for number, stage in enumerate(stages, 1):
        local = root / "production" / stage["name"]
        rows = control.validate_log(stage, local)
        if len(rows) != 6000 or not read(local / "COMPLETED.json")["complete"]:
            raise ValueError("Incomplete stage")
        receipt = read(evidence / f"stage{number}-publish-receipt.json")
        restored_receipt = read(evidence / f"stage{number}-readback-receipt.json")
        marker_path = evidence / f"stage{number}-SNAPSHOT.json"
        marker = read(marker_path)
        if (
            receipt["status"] != "verified"
            or restored_receipt["status"] != "verified"
            or sha(marker_path) != receipt["manifest_sha256"]
            or restored_receipt["manifest_sha256"] != receipt["manifest_sha256"]
            or marker["generation"] != receipt["generation"]
        ):
            raise ValueError("Terminal snapshot receipt/marker mismatch")
        restored = root / "readback" / stage["name"] / receipt["generation"]
        expected_names = set(marker["files"])
        actual_names = {
            str(path.relative_to(restored))
            for path in restored.rglob("*")
            if path.is_file()
        }
        if actual_names != expected_names:
            raise ValueError("Independent readback file membership differs")
        for name, expected in marker["files"].items():
            for directory in (restored, local):
                path = directory / name
                if (
                    path.stat().st_size != expected["size_bytes"]
                    or sha(path) != expected["sha256"]
                ):
                    raise ValueError("Terminal payload changed: " + str(path))
        latest = read(local / "latest.json")
        if (
            latest["update"] != 6000
            or sha(local / latest["checkpoint"]) != latest["sha256"]
        ):
            raise ValueError("Latest terminal checkpoint differs")
        actual = torch.load(
            local / latest["checkpoint"], map_location="cpu", weights_only=True
        )
        saved = torch.load(
            restored / latest["checkpoint"], map_location="cpu", weights_only=True
        )
        if not frozen.same(saved, actual):
            raise ValueError("Model/AdamW/RNG/schedule/sample readback differs")
        expected_histogram = (
            {15400: 116, 6000: 85} if number == 1 else {20800: 116, 12000: 85}
        )
        histogram = dict(
            Counter(
                int(state["step"]) for state in actual["optimizer"]["state"].values()
            )
        )
        if (
            actual["updates_completed"] != 6000
            or actual["sample_position"] != 10000 + 6000 * number
            or actual["model_artifact"]["training"]["cumulative_updates"]
            != 10000 + 6000 * number
            or histogram != expected_histogram
        ):
            raise ValueError("Terminal optimizer/update/sample accounting differs")
        report = read(local / "validation-update-00006000.json")
        expected_snapshot = dict(
            baseline["snapshot"], directory=desktop.HISTORICAL_STORE
        )
        if (
            report["identities"] != baseline["identities"]
            or report["metrics"]["support"] != baseline["metrics"]["support"]
            or desktop.normalize_report_snapshot(report, root / "dataset")
            != expected_snapshot
            or report["metrics"]["support"]["observed"] != 85473
        ):
            raise ValueError("Validation teacher/frame/snapshot identity differs")
        predictions = local / report["predictions"]["file"]
        if sha(predictions) != report["predictions"]["sha256"]:
            raise ValueError("Prediction stream changed")
        errors = []
        kinds = Counter()
        observed = hashlib.sha256()
        with predictions.open() as stream:
            for line in stream:
                row = json.loads(line)
                kinds[row["kind"]] += 1
                if row["kind"] == "observed":
                    errors.append(row["error_px"])
                    key = [
                        row[name]
                        for name in (
                            "clip_id",
                            "frame_index",
                            "native_fps",
                            "pts",
                            "time_base",
                        )
                    ]
                    observed.update(
                        (json.dumps(key, separators=(",", ":")) + "\n").encode()
                    )
        values = np.asarray(errors, dtype=np.float64)
        calculated = dict(
            mean_error_px=float(values.mean()),
            median_error_px=float(np.median(values)),
            p95_error_px=float(np.percentile(values, 95)),
            within_distance_rate=float((values <= 20).mean()),
        )
        if (
            len(values) != 85473
            or sum(kinds.values()) != 128668
            or not np.isfinite(values).all()
        ):
            raise ValueError("Prediction support/nonfinite error mismatch")
        if observed.hexdigest() != plan["baseline_observed_identity_sha256"]:
            raise ValueError("Observed PTS/frame identity differs")
        for key, value in calculated.items():
            if not math.isclose(
                value, report["metrics"][key], rel_tol=1e-11, abs_tol=1e-9
            ):
                raise ValueError("Recomputed metric differs: " + key)
        write(evidence / f"stage{number}-validation.json", report)
        write(
            evidence / f"stage{number}-COMPLETED.json", read(local / "COMPLETED.json")
        )
        write(
            evidence / f"stage{number}-continuity.json", read(local / "continuity.json")
        )
        results.append(
            dict(
                stage=stage["name"],
                updates=6000,
                full_state_readback_exact=True,
                manifest_members_and_all_bytes_exact=True,
                file_count=len(expected_names),
                size_bytes=receipt["size_bytes"],
                sample_position=actual["sample_position"],
                optimizer_step_histogram=histogram,
                validation_identities=report["identities"],
                observed_pts_identity_matches_baseline=True,
                metrics_recomputed=calculated,
                prediction_kind_counts=dict(kinds),
                checkpoint_sha256=latest["sha256"],
                freeze_boundary=[rows[i] for i in (0, 599, 600, 5999)],
            )
        )
        del actual, saved
    readbacks = list((root / "production/readback-receipts").glob("*.json"))
    if len(readbacks) != 15 or any(
        read(path)["status"] != "verified" for path in readbacks
    ):
        raise ValueError("Missing periodic/completed independent readbacks")
    registration = read(root / "production-registration.json")
    event = read(root / "notifications" / (registration["job"] + ".json"))
    upload = read(root / "terminal-event-upload.json")
    if (
        not read(root / "production/COMPLETED.json")["complete"]
        or event["state"] != "done"
        or event["queue_returncode"] != 0
        or upload["returncode"] != 0
        or not read(root / "startup-acceptance.json")["complete"]
    ):
        raise ValueError("Missing job completion/startup/persistence evidence")
    write(
        evidence / "completion-verification.json",
        dict(
            complete=True,
            source_commit=desktop.COMMIT,
            adapter_training_commit="eda5ee08073da498347d411e075905f4ac94e25e",
            stages=results,
            independent_readbacks=15,
            query_updates=12000,
            terminal_event=event,
            terminal_event_drive_verified=True,
            metric_method="all observed frame errors, float64 aggregation, linear 95th percentile",
            does_not_rerun_inference=True,
        ),
    )
    print(
        json.dumps(
            dict(
                complete=True,
                query_updates=12000,
                independent_readbacks=15,
                final_metrics=results[-1]["metrics_recomputed"],
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
