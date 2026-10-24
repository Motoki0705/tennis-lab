"""Run the unchanged frozen probe with an absolute shared temporary root."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any
from unittest.mock import patch

FROZEN = Path(__file__).resolve().parents[1] / "run-court-d2-support-cpu-20261024"
sys.path.insert(0, str(FROZEN))

import render_probe as probe  # noqa: E402

from src.synthetic_data_generation.rendering.nht import client  # noqa: E402


@contextmanager
def absolute_scratch() -> Iterator[Path]:
    with tempfile.TemporaryDirectory(prefix="court-d2-probe-", dir="tmp") as temporary:
        yield Path(temporary).resolve()


def load_inputs(asset_root: Path) -> dict[str, Any]:
    return {
        scene: probe.load_scene_inputs(
            asset_root / "scenes" / scene,
            replay=False,
            asset_manifest=asset_root / "assets.json",
            asset_manifest_sha256=probe.ASSET_SHA,
        )
        for scene in probe.SCENES
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", required=True, type=Path)
    parser.add_argument("--asset-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if not Path("/content/tennis-lab").exists() or not Path(
        "/content/tennis-lab"
    ).samefile(Path.cwd()):
        raise RuntimeError("Run from the managed Colab checkout")
    selection = probe.load_selection(args.selection)
    inputs = load_inputs(args.asset_root)
    frozen = probe.reconstruct(selection, inputs)
    args.output.mkdir(parents=True, exist_ok=False)
    probe.write(args.output / "selection.json", selection)
    submissions: dict[str, Any] = {
        "schema": "court_d2_nht_submissions_v1",
        "submitted_cameras": 0,
        "invocations": [],
        "accounting": "Cameras passed to the actual nht-render subprocess; no automatic retry.",
    }
    ledger = args.output / "nht-submissions.json"
    probe.atomic_json(ledger, submissions)
    original_run = subprocess.run

    def record_submission(argv: list[str], **kwargs: Any) -> Any:
        if Path(argv[0]).name != "nht-render":
            return original_run(argv, **kwargs)
        request = Path(argv[argv.index("--cameras") + 1])
        cameras = json.loads(request.read_text())["cameras"]
        row: dict[str, Any] = {
            "camera_ids": [camera["camera_id"] for camera in cameras],
            "camera_count": len(cameras),
            "outcome": "submitted",
        }
        submissions["submitted_cameras"] += len(cameras)
        assert submissions["submitted_cameras"] <= 192
        submissions["invocations"].append(row)
        probe.atomic_json(ledger, submissions)
        started = time.monotonic()
        try:
            result = original_run(argv, **kwargs)
            row.update(outcome="completed", returncode=result.returncode)
            return result
        except BaseException as error:
            row.update(outcome="failed", error=f"{type(error).__name__}: {error}")
            raise
        finally:
            row["subprocess_seconds"] = time.monotonic() - started
            probe.atomic_json(ledger, submissions)

    with (
        absolute_scratch() as scratch,
        patch.object(client.subprocess, "run", side_effect=record_submission),
    ):
        probe.run(selection, inputs, frozen, scratch, args.output)
    assert submissions["submitted_cameras"] == 192
    assert len(submissions["invocations"]) == 16
    probe.atomic_json(
        args.output / "inventory.json",
        {
            "files": [
                {
                    "path": path.relative_to(args.output).as_posix(),
                    "bytes": path.stat().st_size,
                    "sha256": probe.file_sha256(path),
                }
                for path in sorted(args.output.rglob("*"))
                if path.is_file()
            ]
        },
    )


if __name__ == "__main__":
    main()
