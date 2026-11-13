"""Rebuild a same-host recovery root with hardlinked immutable inputs only."""

import argparse
import json
import os
import subprocess
from pathlib import Path

from rebuild_dataset import rebuild, write
from state import prepare


def link_tree(source, destination):
    destination.mkdir(parents=True)
    for path in source.rglob("*"):
        relative = path.relative_to(source)
        target = destination / relative
        if path.is_symlink():
            raise ValueError("Immutable input tree contains a symlink")
        if path.is_dir():
            target.mkdir(exist_ok=True)
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            os.link(path, target)


def main(old, root):
    if root.exists():
        raise ValueError("Recovery bootstrap requires a new output root")
    latest = sorted(
        old.glob("production-registration-a*.json"),
        key=lambda p: int(p.stem.rsplit("-a", 1)[1]),
    )
    if not latest:
        raise ValueError("No previous production registration exists")
    registration = json.loads(latest[-1].read_text())
    state = subprocess.check_output(
        [
            "systemctl",
            "--user",
            "show",
            registration["unit"],
            "--property=ActiveState",
            "--value",
        ],
        text=True,
    ).strip()
    if state in {"active", "activating", "deactivating", "reloading"}:
        raise ValueError("The old writer's service is still live")
    root.mkdir(parents=True)
    for name in ("original", "inputs"):
        link_tree(old / name, root / name)
    for name in (
        "storage-baseline.json",
        "inputs-verified.json",
        "environment-verified.json",
        "environment-freeze.txt",
        "environment-exact-runtime.txt",
        "environment-missing-dependencies.txt",
        "original-vm-pip-freeze.txt",
        "archive-acceptance.json",
        "msda-build-manifest.json",
        "msda-numerical-check.json",
        "full-shape-equivalence.json",
    ):
        (root / name).write_bytes((old / name).read_bytes())
    for name in ("env-original", "msda-build"):
        (root / name).symlink_to((old / name).resolve(), target_is_directory=True)
    rebuild(old / "dataset", root / "dataset", root / "dataset-verified.json")
    prepare(root)
    attempt = max(int(p.stem.rsplit("-a", 1)[1]) for p in latest) + 1
    write(
        root / "recovery-bootstrap.json",
        dict(
            complete=True,
            reused_from=str(old),
            next_queue_attempt=attempt,
            old_service=registration["unit"],
            old_service_state=state,
            new_rgb_payload_bytes=0,
            private_environment_reused=True,
            storage_baseline_preserved=True,
        ),
    )
    print(json.dumps(dict(root=str(root), next_queue_attempt=attempt)), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--from-root", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    main(args.from_root.resolve(), args.root.resolve())
