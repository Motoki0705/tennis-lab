"""Separate probe subprocess, QA, archive and final worker-save measurements."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any


def read(path: Path) -> Any:
    return json.loads(path.read_text())


def terminal(path: Path) -> dict[str, Any]:
    receipt = read(path)
    assert receipt["returncode"] == 0
    job = receipt["response"]["jobs"][0]
    assert job["status"] == "completed" and job["returncode"] == 0
    assert job["drive_saved_at"]
    return dict(job)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recovered", required=True, type=Path)
    parser.add_argument("--render-status", required=True, type=Path)
    parser.add_argument("--collect-status", required=True, type=Path)
    parser.add_argument("--drive-inspect", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    root = args.recovered
    state = read(root / "probe/state.json")
    ledger = read(root / "probe/nht-submissions.json")
    assert state["outcome"] == "completed"
    assert state["completed_frames"] == ledger["submitted_cameras"] == 192
    profile = read(root / "profile-summary.json")

    def stage(name: str, contains: str) -> dict[str, Any]:
        rows = [
            row
            for row in profile["rows"]
            if row["function"] == name and contains in row["file"]
        ]
        assert len(rows) == 1, (name, contains, len(rows))
        return dict(rows[0])

    render_status = terminal(args.render_status)
    collect_status = terminal(args.collect_status)
    process = read(root / "render-timing.json")
    process_finished = datetime.fromisoformat(process["started_at"]) + timedelta(
        seconds=process["wall_seconds"]
    )
    setup = {
        name: read(root / f"setup/{name}.json")["wall_seconds"]
        for name in ("uv", "nht", "asset-download", "asset-verify")
    }
    reports = [
        read(root / "probe/reports" / f"{s['variant']}-{s['scene']}.json")
        for s in read(root / "probe/selection.json")["scenarios"]
    ]
    inventory = read(root / "probe/inventory.json")["files"]
    sizes = {
        label: sum(entry["bytes"] for entry in inventory if entry["path"].endswith(end))
        for label, end in (
            ("raw_npz_bytes", "/raw.npz"),
            ("jpeg_bytes", "/rgb.jpg"),
            ("sparse_record_bytes", "/record.json"),
        )
    }
    result = {
        "schema": "court_d2_probe_timing_summary_v1",
        "setup_seconds": setup,
        "total_setup_seconds": sum(setup.values()),
        "probe_process": process,
        "nht_invocations": len(ledger["invocations"]),
        "nht_requested_cameras": ledger["submitted_cameras"],
        "nht_subprocess_seconds": sum(
            row["subprocess_seconds"] for row in ledger["invocations"]
        ),
        "scenario_render_qa_archive_seconds": sum(
            report["render_qa_archive_seconds"] for report in reports
        ),
        "profile_stages": {
            "load_four_scene_inputs": stage("load_scene_inputs", "scale/runtime.py"),
            "reconstruct_all_192_cameras": stage("reconstruct", "render_probe.py"),
            "nht_public_client_including_subprocess_and_file_validation": stage(
                "render", "rendering/nht/client.py"
            ),
            "qa": stage("evaluate_render", "scale/quality.py"),
            "jpeg_encoding": stage("encode_rgb_jpeg", "image_record_store.py"),
            "raw_npz_archive": stage("savez_compressed", "numpy"),
        },
        "worker_final_save_seconds": {
            name: (
                datetime.fromisoformat(status["finished_at"])
                - datetime.fromisoformat(status["updated_at"])
            ).total_seconds()
            for name, status in (("render", render_status), ("collect", collect_status))
        },
        "post_process_to_render_job_finished_seconds": (
            datetime.fromisoformat(render_status["finished_at"]) - process_finished
        ).total_seconds(),
        "post_process_before_finalizing_seconds": (
            datetime.fromisoformat(render_status["updated_at"]) - process_finished
        ).total_seconds(),
        "drive_inspect": read(args.drive_inspect)["response"],
        "probe_storage": sizes,
        "renderer_release": state["renderer_release"],
        "notes": [
            "The NHT subprocess interval includes its own scene/checkpoint initialization, rendering and public-file writes; it is not pure rasterizer time.",
            "The public-client profile contains that subprocess interval. Do not add those two nested measurements.",
            "QA, JPEG encoding and NPZ compression are separate profile entrances; unlisted JSON/filesystem/telemetry overhead remains in scenario wall time.",
            "Periodic worker saving overlaps the process and is not isolated. Final-save seconds run from finalizing updated_at to finished_at, including final status writes.",
            "Post-process time includes shell profile summarization, draining an in-progress periodic save and, after a save failure, inspection and explicit recovery latency before finalizing. It is not pure transfer time; process end is derived from its UTC start plus monotonic elapsed measurement.",
            "Raw NPZ and diagnostic JPEGs include QA rejects. This is not JPEG-only accepted production publication throughput or capacity.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(
        json.dumps(
            {
                k: result[k]
                for k in (
                    "total_setup_seconds",
                    "nht_requested_cameras",
                    "nht_subprocess_seconds",
                    "scenario_render_qa_archive_seconds",
                    "probe_storage",
                )
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
