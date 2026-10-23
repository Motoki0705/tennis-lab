"""Measure the locked Colab environment independently of dataset retrieval."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def tree_usage(directory: Path) -> dict[str, int]:
    files = size = allocated = 0
    for root, _, names in os.walk(directory):
        for name in names:
            path = Path(root) / name
            if path.is_symlink():
                continue
            stat = path.stat()
            files += 1
            size += stat.st_size
            allocated += stat.st_blocks * 512
    return dict(files=files, size_bytes=size, allocated_bytes=allocated)


def main(receipts: Path) -> None:
    if receipts.exists() or Path(".venv").exists():
        raise FileExistsError("Measure environment creation in a fresh checkout with a new receipt directory")
    receipts.mkdir(parents=True)
    argv = ["uv", "sync", "--locked", "--python", "3.11", "--no-cache"]
    before = shutil.disk_usage(Path.cwd())._asdict()
    started_at, started = datetime.now(UTC).isoformat(), time.monotonic()
    with (receipts / "uv.stdout.log").open("wb") as out, (receipts / "uv.stderr.log").open("wb") as err:
        result = subprocess.run(argv, stdout=out, stderr=err, check=False)
    elapsed = time.monotonic() - started
    summary: dict[str, Any] = dict(
        schema_version=1, started_at=started_at, argv=argv,
        elapsed_seconds=elapsed, returncode=result.returncode,
        git_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        bootstrap_python=sys.version, uv_version=subprocess.check_output(["uv", "--version"], text=True).strip(),
        disk_before=before, disk_after=shutil.disk_usage(Path.cwd())._asdict(),
        cpu_count=os.cpu_count(), memory_bytes=os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES"),
    )
    if result.returncode == 0:
        probe = subprocess.check_output([
            ".venv/bin/python", "-c",
            "import _sha256, importlib.metadata, json, sys; "
            "print(json.dumps(dict(python=sys.version, executable=sys.executable, "
            "independent_sha256_module=_sha256.__name__, "
            "packages={name:importlib.metadata.version(name) for name in "
            "['torch','torchvision','opencv-python','numpy','pandas','pytorch-lightning']})))",
        ], text=True)
        summary["environment"] = json.loads(probe)
        summary["venv_usage"] = tree_usage(Path(".venv"))
    (receipts / "summary.json").write_text(json.dumps(summary, sort_keys=True, indent=2) + "\n")
    print(json.dumps(summary), flush=True)
    if result.returncode:
        raise RuntimeError("Locked environment creation failed; inspect the preserved uv logs")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipts", type=Path, required=True)
    main(parser.parse_args().receipts)
