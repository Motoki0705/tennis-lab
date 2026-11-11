"""Fresh-VM recovery from mounted Drive. Does not allocate a VM or resume12/24 layers."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
import urllib.request
import zipfile
from pathlib import Path

WORKSPACE = Path("/content/tennis-lab")
DRIVE = Path("/content/drive/MyDrive/tennis_lab")
RELATIVE = Path("outputs/ball_detection/train/query_controls/s42-20261111-main")
CAPACITY = Path("outputs/ball_detection/train/capacity_scaling/s42-20261110-12k")
OLD_COMMIT = "270702138e34c580c8fad703157d051cebc519a0"
RCLONE_BINARY_SHA = "ed7ab1f17876bb006c3df3949a3044959671afd9112362dacaa5dd1647735bc1"


def read(path):
    return json.loads(path.read_text())


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024**2), b""):
            h.update(chunk)
    return h.hexdigest()


def restore(receipt, destination, selected=None):
    if (
        receipt.get("status") != "verified"
        or receipt.get("schema") != "verified_artifact_snapshot.v1"
    ):
        raise ValueError("A verified snapshot receipt is required")
    remote = receipt["remote"]
    if (
        not remote.startswith("mounted_drive:outputs/")
        or ".." in Path(remote.split(":", 1)[1]).parts
    ):
        raise ValueError("Snapshot is outside the experiment output namespace")
    source = DRIVE / remote.split(":", 1)[1]
    marker = source / "_SNAPSHOT.json"
    if digest(marker) != receipt["manifest_sha256"]:
        raise ValueError("Snapshot marker differs")
    manifest = read(marker)
    if manifest["generation"] != receipt["generation"]:
        raise ValueError("Snapshot generation differs")
    names = manifest["files"] if selected is None else selected
    if not set(names) <= manifest["files"].keys():
        raise ValueError("Required input metadata is absent")
    if destination.exists():
        raise ValueError(
            f"Refusing to overwrite an existing recovery target: {destination}"
        )
    destination.mkdir(parents=True)
    for name in names:
        path = Path(name)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError("Unsafe snapshot member")
        target = destination / path
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source / path, target)
        if (
            target.stat().st_size != manifest["files"][name]["size_bytes"]
            or digest(target) != manifest["files"][name]["sha256"]
        ):
            raise ValueError("Snapshot payload differs: " + name)
    print(
        json.dumps(
            dict(event="restored", destination=str(destination), files=len(names))
        ),
        flush=True,
    )
    return destination


def run(argv, cwd=None, **kwargs):
    print(
        json.dumps(dict(event="command", cwd=str(cwd), argv=[str(x) for x in argv])),
        flush=True,
    )
    return subprocess.run([str(x) for x in argv], cwd=cwd, check=True, **kwargs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", required=True)
    parser.add_argument("--attempt", type=int, default=2)
    parser.add_argument(
        "--launch",
        action="store_true",
        help="After restoring and verifying, explicitly register and start the serial queue.",
    )
    args = parser.parse_args()
    if args.attempt < 2:
        parser.error("Recovery requires a new attempt>=2")
    gpu = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True
    ).strip()
    if gpu:
        raise ValueError(
            "A GPU owner is still present. Inspect it before recovery; do not stop another experiment."
        )
    controls = WORKSPACE / ".claude/worktrees/ball-query-controls"
    legacy = WORKSPACE / ".claude/worktrees/ball-capacity-12k"
    if controls.exists() or legacy.exists():
        raise ValueError(
            "This bootstrap is for a fresh VM; inspect live/local state before same-VM recovery"
        )
    if not shutil.which("uv"):
        raise RuntimeError("Install uv before invoking this fresh-VM recovery command")
    # The entry script itself must match the immutable inputs marker before orchestration.
    receipt = read(DRIVE / RELATIVE / "inputs-receipt.json")
    source = DRIVE / receipt["remote"].split(":", 1)[1]
    marker = source / "_SNAPSHOT.json"
    if digest(marker) != receipt["manifest_sha256"]:
        raise ValueError("Input marker differs")
    if (
        digest(Path(__file__))
        != read(marker)["files"]["reproduction/bootstrap.py"]["sha256"]
    ):
        raise ValueError("This bootstrap is not the frozen input version")
    with tempfile.TemporaryDirectory(prefix="query-controls-recovery-") as scratch:
        inputs = restore(receipt, Path(scratch) / "inputs")
        if not WORKSPACE.exists():
            run(
                [
                    "git",
                    "clone",
                    "--filter=blob:none",
                    "https://github.com/Motoki0705/tennis-lab.git",
                    WORKSPACE,
                ]
            )
        run(["git", "bundle", "verify", inputs / "source.bundle"], cwd=WORKSPACE)
        branch = f"codex/query-controls-recovery-a{args.attempt}"
        run(
            ["git", "fetch", inputs / "source.bundle", f"HEAD:refs/heads/{branch}"],
            cwd=WORKSPACE,
        )
        run(["git", "worktree", "add", controls, branch], cwd=WORKSPACE)
        run(["git", "worktree", "add", "--detach", legacy, OLD_COMMIT], cwd=WORKSPACE)
        run(["uv", "sync", "--locked", "--python", "3.11.17"], cwd=legacy)
        (controls / ".venv").symlink_to(legacy / ".venv", target_is_directory=True)
        tools = Path("/tmp/ball-tools")
        tools.mkdir(exist_ok=True)
        binary = tools / "rclone"
        if not binary.exists():
            archive = tools / "rclone.zip"
            urllib.request.urlretrieve(
                "https://downloads.rclone.org/v1.75.2/rclone-v1.75.2-linux-amd64.zip",
                archive,
            )
            with zipfile.ZipFile(archive) as bundle:
                binary.write_bytes(bundle.read("rclone-v1.75.2-linux-amd64/rclone"))
            binary.chmod(0o755)
        if digest(binary) != RCLONE_BINARY_SHA:
            raise ValueError("The pinned rclone binary differs")
        os.environ["PATH"] = "/tmp/ball-tools:" + os.environ["PATH"]
        config = Path("/tmp/ball-drive-mount.conf")
        content = "[mounted_drive]\ntype = alias\nremote = /content/drive/MyDrive/tennis_lab\n"
        if config.exists() and config.read_text() != content:
            raise ValueError("An existing run-specific Drive configuration differs")
        config.write_text(content)
        config.chmod(0o600)
        root = WORKSPACE / RELATIVE
        root.mkdir(parents=True, exist_ok=True)
        for name in (
            "plan.json",
            "initialization-audit.json",
            "pretraining-audit.json",
            "scope.json",
        ):
            shutil.copyfile(inputs / name, root / name)
        (root / "inputs-receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
        old_receipt = read(DRIVE / CAPACITY / "inputs-receipt.json")
        old = restore(
            old_receipt,
            Path(scratch) / "old-metadata",
            selected=[
                "plan.json",
                "parents-verified.json",
                "requirements-comment.json",
                "prepared/audit.json",
                "prepared/experiments.json",
                "prepared/train_T32.json",
                "prepared/train_T64.json",
                "prepared/evaluation_T32.json",
                "prepared/evaluation_T64.json",
            ],
        )
        old_root = WORKSPACE / CAPACITY
        old_root.mkdir(parents=True, exist_ok=True)
        for name in ("plan.json", "parents-verified.json", "requirements-comment.json"):
            shutil.copyfile(old / name, old_root / name)
        prepared = (
            WORKSPACE
            / "outputs/ball_detection/precompute/sequential_length/observed-packs-20261025"
        )
        shutil.copytree(old / "prepared", prepared)
        for parent_receipt in (inputs / "parent-receipts").glob("*.json"):
            if parent_receipt.stem == "eight-layer-4000":
                continue  # restore.py selects the newest actual prerequisite, never this older baseline.
            saved = read(parent_receipt)
            relative = Path(saved["remote"].split(":", 1)[1]).parent
            restore(saved, WORKSPACE / relative)
        # Shared full-frame data; every byte is verified with two independent SHA implementations.
        run(
            [
                legacy / ".venv/bin/python",
                controls
                / "knowledge/runs/run-ball-sequential-length-20261025/download_inputs.py",
            ],
            cwd=controls,
        )
        here = controls / "knowledge/runs/run-ball-query-controls-20261111"
        run(
            [
                controls / ".venv/bin/python",
                here / "restore.py",
                "--include-prerequisite",
            ],
            cwd=controls,
        )
        # Revalidate the original dataset gate using the unchanged #1675 controller.
        setup = "import sys,json,shutil;from pathlib import Path;sys.path.insert(0,'knowledge/runs/run-ball-capacity-12k-20261110');import campaign;campaign.verify_plan_sources(campaign.read(campaign.ROOT/'plan.json'));campaign.validate_download();shutil.copyfile(campaign.ROOT/'data-verified.json',Path('/content/tennis-lab')/'outputs/ball_detection/train/query_controls/s42-20261111-main/data-verified.json')"
        run([legacy / ".venv/bin/python", "-c", setup], cwd=legacy)
        failure = DRIVE / RELATIVE / "FAILED.json"
        if failure.exists():
            shutil.copyfile(failure, root / "FAILED.json")
            raise ValueError(
                "Failure state restored. Diagnose it and explicitly archive the failure before re-admission."
            )
        if args.launch:
            run(
                [
                    controls / ".venv/bin/python",
                    here / "control.py",
                    "launch",
                    "--session",
                    args.session,
                    "--attempt",
                    str(args.attempt),
                ],
                cwd=controls,
            )
            run(
                [controls / ".venv/bin/python", here / "start_service.py"], cwd=controls
            )
        print(
            json.dumps(
                dict(
                    complete=True,
                    restored=True,
                    launched=args.launch,
                    held_models=[12, 24],
                    workspace=str(controls),
                )
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
