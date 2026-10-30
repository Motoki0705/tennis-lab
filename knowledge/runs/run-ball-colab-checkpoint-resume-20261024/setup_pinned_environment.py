"""Rebuild the proof environment with explicit uv and CPython patch versions."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--uv-version", required=True)
    parser.add_argument("--python-version", required=True)
    parser.add_argument("--tool-dir", type=Path, required=True)
    parser.add_argument("--receipts", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    args = parser.parse_args()
    if args.receipts.exists() or args.tool_dir.exists():
        raise FileExistsError("Use new receipt and tool directories")
    args.receipts.mkdir(parents=True)
    started = time.monotonic()
    installer = args.receipts / "uv-install.sh"
    url = f"https://astral.sh/uv/{args.uv_version}/install.sh"
    subprocess.run([
        "curl", "--fail", "--location", "--silent", "--show-error", "--max-time", "60",
        url, "--output", str(installer),
    ], check=True)
    environment = dict(os.environ, UV_UNMANAGED_INSTALL=str(args.tool_dir))
    # Colab's UV_INSTALL_DIR takes precedence over the unmanaged destination.
    for name in ("UV_INSTALL_DIR", "CARGO_DIST_FORCE_INSTALL_DIR"):
        environment.pop(name, None)
    with (args.receipts / "installer.log").open("wb") as log:
        subprocess.run(["sh", str(installer)], env=environment, stdout=log, stderr=log, check=True)
    uv = args.tool_dir / "uv"
    version = subprocess.check_output([str(uv), "--version"], text=True).strip()
    if version.split()[1] != args.uv_version:
        raise RuntimeError("Installed uv version differs from the requested pin")
    installer_seconds = time.monotonic() - started
    sync_started = time.monotonic()
    argv = [str(uv), "sync", "--locked", "--python", args.python_version, "--no-cache"]
    with (args.receipts / "uv.stdout.log").open("wb") as out, (args.receipts / "uv.stderr.log").open("wb") as err:
        result = subprocess.run(argv, stdout=out, stderr=err, check=False)
    summary = dict(schema_version=1, observed_at=datetime.now(UTC).isoformat(), installer_url=url,
                   installer_sha256=hashlib.sha256(installer.read_bytes()).hexdigest(),
                   uv_version=version, python_request=args.python_version, argv=argv,
                   installer_seconds=installer_seconds, sync_seconds=time.monotonic() - sync_started,
                   total_seconds=time.monotonic() - started, returncode=result.returncode,
                   code_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                   bootstrap_python=sys.version, disk=shutil.disk_usage(Path.cwd())._asdict(),
                   kernel_boot_id=Path("/proc/sys/kernel/random/boot_id").read_text().strip())
    if result.returncode == 0:
        probe = subprocess.check_output([
            ".venv/bin/python", "-c", "import _sha256, importlib.metadata, json, sys; "
            "print(json.dumps(dict(python=sys.version, independent_sha256_module=_sha256.__name__, "
            "packages={name:importlib.metadata.version(name) for name in "
            "['torch','torchvision','opencv-python','numpy','pandas','pytorch-lightning']})))",
        ], text=True)
        summary["environment"] = json.loads(probe)
        summary["python_matches_control"] = summary["environment"]["python"] == json.loads(args.reference.read_text())["python"]
    (args.receipts / "summary.json").write_text(json.dumps(summary, sort_keys=True, indent=2) + "\n")
    print(json.dumps(summary), flush=True)
    if result.returncode or not summary.get("python_matches_control"):
        raise RuntimeError("Pinned environment failed or Python build differs from the control")


if __name__ == "__main__":
    main()
