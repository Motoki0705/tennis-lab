"""Launch a new L4 workflow, optionally staging durable state from a prior run."""

from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
import tomllib
from pathlib import Path

from scripts.colab.workflow.common import validate_run_id

REPO = Path(__file__).resolve().parents[4]


def resume_inputs(
    files: list[dict[str, object]], prior_run: str
) -> list[dict[str, object]]:
    inputs: list[dict[str, object]] = []
    seen: set[str] = set()
    for item in files:
        path = str(item["Path"])
        parts = Path(path).parts
        if len(parts) != 5 or parts[0] not in {"b", "s", "splus", "l"}:
            continue
        if (
            parts[1] != "logs"
            or parts[3] != "checkpoints"
            or parts[4] not in {"last.ckpt", "recovery.ckpt"}
        ):
            continue
        if path in seen:
            raise ValueError(f"Ambiguous duplicate Drive checkpoint: {path}")
        seen.add(path)
        inputs.append(
            {
                "source": f"colab-live/{prior_run}/training/{path}",
                "destination": f"ckpt/court_vit_ablation/{path}",
                "writable": False,
            }
        )
    if not inputs:
        raise ValueError(
            "No durable checkpoint found in the previous run; refusing to restart silently"
        )
    return inputs


def encode_job(job: dict[str, object]) -> str:
    # This schema uses strings, integers, booleans and lists only.
    def value(v: object) -> str:
        return json.dumps(v, ensure_ascii=False)

    lines = [
        f"{k} = {value(v)}" for k, v in job.items() if k not in {"command", "inputs"}
    ]
    lines.append("\n[command]")
    command = job["command"]
    assert isinstance(command, dict)
    lines.extend(f"{k} = {value(v)}" for k, v in command.items())
    inputs = job["inputs"]
    assert isinstance(inputs, list)
    for item in inputs:
        lines.append("\n[[inputs]]")
        lines.extend(f"{k} = {value(v)}" for k, v in item.items())
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--resume-from")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    validate_run_id(args.run_id)
    job = tomllib.loads(
        (REPO / "scripts/colab/workflows/jobs/court_vit_ablation.toml").read_text()
    )
    if args.resume_from:
        validate_run_id(args.resume_from)
        result = subprocess.run(
            [
                "rclone",
                "lsjson",
                f"gdrive:tennis_lab/colab-live/{args.resume_from}/training",
                "--recursive",
                "--files-only",
                "--include",
                "*.ckpt",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        job["inputs"].extend(resume_inputs(json.loads(result.stdout), args.resume_from))
        job["command"]["args"].append("--continuation")
    with tempfile.TemporaryDirectory(prefix="court-vit-job-") as directory:
        Path(directory, "court_vit_ablation.toml").write_text(encode_job(job))
        command = [
            "bash",
            str(REPO / "scripts/colab/run.sh"),
            "--jobs-dir",
            directory,
            "run",
            "court_vit_ablation",
            "--run-id",
            args.run_id,
            "--gpu",
            "L4",
            "--source",
            "snapshot",
        ]
        if args.dry_run:
            command.append("--dry-run")
        subprocess.run(command, cwd=REPO, check=True)


if __name__ == "__main__":
    main()
