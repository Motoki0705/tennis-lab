"""Command-level contracts for the B01-B03 Colab alignment launcher."""

from __future__ import annotations

import subprocess
from pathlib import Path

ROOT = Path(__file__).parents[3]
SCRIPT = ROOT / "scripts/colab/train/20260829T150257Z/run_b01_b03_alignment.sh"
REPORT = SCRIPT.with_name("REPORT.md")


def test_b01_b03_alignment_script_has_valid_bash_syntax() -> None:
    subprocess.run(["bash", "-n", str(SCRIPT)], check=True)


def test_dry_run_fixes_scene_order_terminal_stage_and_input_hashes() -> None:
    completed = subprocess.run(
        ["bash", str(SCRIPT), "--dry-run"],
        check=False,
        capture_output=True,
        text=True,
        cwd=ROOT,
    )

    assert completed.returncode == 0, completed.stderr
    output = completed.stdout
    scene_positions = [
        output.index(f"scene={scene} gpu-command=") for scene in ("B01", "B02", "B03")
    ]
    assert scene_positions == sorted(scene_positions)
    for profile in ("b01", "b02", "b03"):
        command = (
            ".venv/bin/python -m "
            "src.synthetic_data_generation.scripts.run_scene_pipeline "
            f"profile={profile} request.from_stage=ingest "
            "request.through_stage=reconstruction"
        )
        assert output.count(command) == 1
        alignment_command = (
            "TENNIS_LAB_ALIGNMENT_INFERENCE_MIRROR_ROOT=<run>/"
            f"{profile.upper()}/court-line-inference "
            "TENNIS_LAB_ALIGNMENT_HOLDOUT_CAMERA_PREFIX_COUNT=72 "
            ".venv/bin/python -m "
            "src.synthetic_data_generation.scripts.run_scene_pipeline "
            f"profile={profile} request.from_stage=alignment "
            "request.through_stage=alignment"
        )
        assert output.count(alignment_command) == 1
    assert "TENNIS_LAB_ALIGNMENT_LINE_DEVICE=cpu" not in output
    assert "TENNIS_LAB_ALIGNMENT_MAXIMUM_UNEXPLAINED" not in output
    assert output.count("save-after=reconstruction,alignment") == 3
    for expected_hash in (
        "c9608e911f86274a862a289927ff9d0cc587543f836ffbdcad127f8ce61b5d56",
        "035a3e79637583d0794e598808fcdd46aac9d3f8e374599f453718a3d6c8615a",
        "80ec1676b420b05f22fc9c4ed5db9257e1c35b9e9bb9596dd1be3f479c7287ac",
        "81914bc58ba08824061b4509f54fcb2637a99b5c505cd5c28780cd4c1e88bfd4",
        "73cec8be7427c8655ceced13ce62f6e20a1fa90d1b4d4a550df17a1144081a7c",
    ):
        assert expected_hash in output
    assert output.count("verify=alignment,line-heatmaps,no-datasets,no-report") == 3


def test_unknown_argument_fails_before_environment_setup() -> None:
    completed = subprocess.run(
        ["bash", str(SCRIPT), "--unknown"],
        check=False,
        capture_output=True,
        text=True,
        cwd=ROOT,
    )

    assert completed.returncode == 2
    assert completed.stdout == ""
    assert "Usage:" in completed.stderr


def test_timestamped_run_keeps_its_auditable_execution_report() -> None:
    report = REPORT.read_text(encoding="utf-8")

    for required in (
        "B01–B03 3DGS学習・コートアライメント改善 実施報告書",
        "20260829T150257Z-f72f860df71e",
        "boundary-lattice-assisted identifiability",
        "B01 | 3 / 3 | 32 / 40",
        "#829は同じ自動コート数対応の旧PR",
        "`.spin/cmds.py`はmainの`cu130`版へ戻し",
    ):
        assert required in report
