#!/usr/bin/env python3
"""Launch one immutable NHT training run inside the isolated third-party stack."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def _git_head(path: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        text=True,
    ).strip()


def _git_is_clean(path: Path) -> bool:
    return (
        subprocess.check_output(
            ["git", "-C", str(path), "status", "--short"],
            text=True,
        ).strip()
        == ""
    )


def _load_pins(path: Path) -> dict[str, str]:
    pins: dict[str, str] = {}
    for line in path.read_text().splitlines():
        if not line or line.startswith("#"):
            continue
        key, separator, value = line.partition("=")
        if not separator or not key or not value:
            raise ValueError(f"Invalid pins.env record: {line!r}")
        pins[key] = value
    return pins


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(path)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--data-factor", type=int, default=2)
    parser.add_argument("--max-steps", type=int, default=30_000)
    parser.add_argument("--test-every", type=int, default=8)
    parser.add_argument("--cap-max", type=int, default=1_000_000)
    parser.add_argument("--lpips-net", choices=("alex", "vgg"), default="alex")
    parser.add_argument(
        "--trainer-arg",
        action="append",
        default=[],
        help="One additional explicit argument passed to simple_trainer_nht.py.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    nht_root = Path(__file__).resolve().parent
    upstream = nht_root / "upstream"
    venv_python = nht_root / ".venv" / "bin" / "python"
    pins = _load_pins(nht_root / "pins.env")

    if Path(sys.executable).resolve() != venv_python.resolve():
        raise SystemExit(f"Use the isolated interpreter: {venv_python}")
    if _git_head(upstream) != pins["NHT_COMMIT"]:
        raise SystemExit("NHT checkout does not match pins.env")
    if _git_head(upstream / "gsplat") != pins["GSPLAT_COMMIT"]:
        raise SystemExit("gsplat checkout does not match pins.env")
    if not _git_is_clean(upstream):
        raise SystemExit("Refusing training with a modified NHT checkout")

    data_dir = args.data_dir.resolve()
    result_dir = args.result_dir.resolve()
    if not (data_dir / "sparse" / "0").is_dir():
        raise SystemExit(f"Missing COLMAP sparse/0 directory: {data_dir}")
    if not (data_dir / f"images_{args.data_factor}").is_dir():
        raise SystemExit(
            f"Missing native image directory: {data_dir / f'images_{args.data_factor}'}"
        )
    if result_dir.exists() and any(result_dir.iterdir()):
        raise SystemExit(f"Refusing to overwrite non-empty result dir: {result_dir}")
    if args.data_factor <= 0 or args.max_steps <= 0:
        raise SystemExit("data-factor and max-steps must be positive")
    if args.test_every <= 0 or args.cap_max <= 0:
        raise SystemExit("test-every and cap-max must be positive")

    result_dir.mkdir(parents=True, exist_ok=True)
    trainer = upstream / "gsplat" / "examples" / "simple_trainer_nht.py"
    command = [
        str(venv_python),
        str(trainer),
        "default",
        "--data_dir",
        str(data_dir),
        "--data_factor",
        str(args.data_factor),
        "--native_images_factor",
        "--result_dir",
        str(result_dir),
        "--test_every",
        str(args.test_every),
        "--max_steps",
        str(args.max_steps),
        "--eval_steps",
        str(args.max_steps),
        "--save_steps",
        str(args.max_steps),
        "--save_ply",
        "--ply_steps",
        str(args.max_steps),
        "--render_traj_path",
        "interp",
        "--lpips_net",
        args.lpips_net,
        "--use_color_correction_metric",
        "--strategy.cap-max",
        str(args.cap_max),
        "--disable_viewer",
        *args.trainer_arg,
    ]
    manifest_path = result_dir / "nht-run.json"
    with tempfile.TemporaryDirectory(
        prefix="tennis-lab-nht-",
        dir="/tmp",
    ) as runtime_temp:
        environment = os.environ.copy()
        environment.setdefault("CUDA_VISIBLE_DEVICES", "0")
        environment.setdefault("OMP_NUM_THREADS", "4")
        environment.setdefault("OPENBLAS_NUM_THREADS", "4")
        environment["TMPDIR"] = runtime_temp
        environment["TEMP"] = runtime_temp
        environment["TMP"] = runtime_temp
        manifest: dict[str, Any] = {
            "schema": "tennis_lab_nht_training_run_v1",
            "status": "running",
            "started_at_utc": datetime.now(UTC).isoformat(),
            "finished_at_utc": None,
            "data_dir": str(data_dir),
            "result_dir": str(result_dir),
            "command": command,
            "pins": pins,
            "environment": {
                "CUDA_VISIBLE_DEVICES": environment["CUDA_VISIBLE_DEVICES"],
                "OMP_NUM_THREADS": environment["OMP_NUM_THREADS"],
                "OPENBLAS_NUM_THREADS": environment["OPENBLAS_NUM_THREADS"],
                "temporary_filesystem": "/tmp",
            },
            "returncode": None,
        }
        _write_json_atomic(manifest_path, manifest)

        try:
            completed = subprocess.run(
                command,
                cwd=trainer.parent,
                env=environment,
                check=False,
            )
        except KeyboardInterrupt:
            manifest["status"] = "interrupted"
            manifest["finished_at_utc"] = datetime.now(UTC).isoformat()
            manifest["returncode"] = 130
            _write_json_atomic(manifest_path, manifest)
            raise SystemExit(130) from None
        manifest["status"] = "completed" if completed.returncode == 0 else "failed"
        manifest["finished_at_utc"] = datetime.now(UTC).isoformat()
        manifest["returncode"] = completed.returncode
        _write_json_atomic(manifest_path, manifest)
        if completed.returncode != 0:
            raise SystemExit(completed.returncode)


if __name__ == "__main__":
    main()
