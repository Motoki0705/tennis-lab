"""The public Hydra command exports each artifact using CPU-only evidence."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import av
from PIL import Image

from src.utils.paths import PROJECT_ROOT


def test_learning_stages_cli_exports_video_gif_stills_and_manifest(
    tmp_path: Path,
) -> None:
    segments = (
        "intro",
        "phase1",
        "handoff12",
        "phase2",
        "handoff23",
        "phase3",
        "outro",
    )
    command = [
        sys.executable,
        "-m",
        "src.tasks.ball_detection.scripts.visualize_learning_stages",
        "input_panel=schematic",
        f"paths.output_root={tmp_path}",
        "run_id=smoke",
        "width=320",
        "height=180",
        "fps=10",
        "gif_width=160",
        "gif_fps=10",
        *(f"durations.{name}=0.1" for name in segments),
    ]
    result = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
        timeout=90,
    )
    assert result.returncode == 0, result.stderr
    run_dir = tmp_path / "ball_detection/visualize/learning_stages/smoke"
    output = run_dir / "presentation"
    assert (run_dir / "hydra").is_dir()
    with av.open(str(output / "learning_stages.mp4")) as video:
        frames = list(video.decode(video=0))
        assert len(frames) == 7
        assert (frames[0].width, frames[0].height) == (320, 180)
    with Image.open(output / "learning_stages.gif") as gif:
        assert gif.size == (160, 90)
    assert {path.stem for path in (output / "stills").glob("*.png")} == {
        "intro",
        "phase1",
        "handoff12",
        "phase2",
        "phase3",
    }
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["elements"]["phase1"]["origin"] == "measured"
    assert manifest["elements"]["phase2"]["origin"] == "measured"
    assert manifest["elements"]["phase3"] == {"origin": "concept", "metrics": None}
    assert manifest["elements"]["input_panel"]["origin"] == "schematic"
    assert manifest["handoff"]["parent_global_step"] == 48000
    assert manifest["handoff"]["phase2_axis_start"] == 60000
    assert (
        manifest["elements"]["phase2"]["final_full_mean_error_px"] == 15.384655889330702
    )
    assert len(manifest["input_files"]) == 8
    assert len(manifest["git_commit"]) == 40
    repeated = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
        timeout=90,
    )
    assert repeated.returncode != 0
    assert "output_dir already exists" in repeated.stderr
