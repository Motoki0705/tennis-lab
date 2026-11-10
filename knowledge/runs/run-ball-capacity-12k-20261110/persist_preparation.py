"""Publish immutable source/config snapshots and reviewable startup evidence."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))

from campaign import (
    HERE,
    RELATIVE,
    ROOT,
    WORKSPACE,
    now,
    read,
    sha,
    verify_plan_sources,
    write,
)

from src.utils.artifact_store import RcloneArtifactStore, RcloneSnapshotStore


def publish_inputs():
    plan = read(ROOT / "plan.json")
    verify_plan_sources(plan)
    if (ROOT / "inputs-receipt.json").exists():
        raise ValueError(
            "Inputs are immutable; inspect the existing receipt instead of republishing"
        )
    subprocess.run(["git", "diff", "--exit-code", "HEAD", "--", str(HERE)], check=True)
    untracked = subprocess.check_output(
        ["git", "ls-files", "--others", "--exclude-standard", str(HERE)], text=True
    )
    if untracked.strip():
        raise ValueError(
            "Commit all experiment files before making the restoration bundle"
        )
    subprocess.run(
        ["git", "bundle", "create", str(ROOT / "source.bundle"), "HEAD"], check=True
    )
    subprocess.run(["git", "bundle", "verify", str(ROOT / "source.bundle")], check=True)
    subprocess.run(
        ["git", "-C", str(WORKSPACE), "diff", "--exit-code", "HEAD"], check=True
    )
    code = dict(
        schema="ball_capacity_source_provenance.v1",
        created_at=now(),
        commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        branch=subprocess.check_output(
            ["git", "branch", "--show-current"], text=True
        ).strip(),
        source_bundle_sha256=sha(ROOT / "source.bundle"),
        original_source_head="e99753cc7f2da87199965faa7cf60c64f8bd006e",
        original_inputs_marker="b6cab45141f1ab50019870fe5ab7ea977d57a2a0af20751b676fafd0e3667a96",
        fixed_source_files=len(plan["fixed_source_sha256"]),
        source_hashes_match_original=True,
        lock_sha256=plan["lock_sha256"],
        torch_version=plan["torch_version"],
        cwd=str(Path.cwd()),
        prepared_manifest_paths=plan["prepared"],
        main_worktree_modified=False,
    )
    write(ROOT / "source-provenance.json", code)
    include = [
        "plan.json",
        "parents-verified.json",
        "requirements-comment.json",
        "source.bundle",
        "source-provenance.json",
    ]
    for path in sorted(Path(plan["prepared"]).iterdir()):
        if path.is_file():
            destination = ROOT / "prepared" / path.name
            destination.parent.mkdir(exist_ok=True)
            shutil.copyfile(path, destination)
            if sha(path) != sha(destination):
                raise ValueError("Prepared manifest copy changed")
            include.append(destination.relative_to(ROOT).as_posix())
    for path in sorted((ROOT / "parent-receipts").glob("*.json")):
        include.append(path.relative_to(ROOT).as_posix())
    receipt = RcloneSnapshotStore(
        ROOT,
        remote="mounted_drive",
        remote_root=str(RELATIVE / "inputs"),
        config_path=Path(plan["rclone_config"]),
    ).publish("inputs-v1", include=include)
    write(ROOT / "inputs-receipt.json", receipt)
    store = RcloneArtifactStore(
        ROOT,
        remote="mounted_drive",
        remote_root=RELATIVE.as_posix(),
        config_path=Path(plan["rclone_config"]),
    )
    store.publish_file(ROOT / "inputs-receipt.json")
    print(receipt, flush=True)


def publish_documentation(label):
    if Path(label).name != label or label in {".", ".."}:
        raise ValueError("Use a single immutable documentation generation name")
    directory = ROOT / "documentation" / label
    directory.mkdir(parents=True, exist_ok=False)
    names = [
        "plan.json",
        "parents-verified.json",
        "requirements-comment.json",
        "source-provenance.json",
        "inputs-receipt.json",
        "data-verified.json",
        "queue-registration.json",
        "queue-service-request.json",
        "queue-service-state.json",
        "worker-progress.json",
        "startup-acceptance.json",
    ]
    for name in names:
        if (ROOT / name).exists():
            shutil.copyfile(ROOT / name, directory / name)
    for subdirectory in (
        "job-status",
        "stage-status",
        "parent-receipts",
        "notifications",
        "local-cache-receipts",
    ):
        if (ROOT / subdirectory).exists():
            shutil.copytree(ROOT / subdirectory, directory / subdirectory)
    for path in sorted(HERE.iterdir()):
        if path.is_file() and path.suffix in {".md", ".py", ".json"}:
            target = directory / "reproduction" / path.name
            target.parent.mkdir(exist_ok=True)
            shutil.copyfile(path, target)
    for name in ("startup.json", "progress.json"):
        source = (
            WORKSPACE
            / "outputs/ball_detection/precompute/sequential_length/drive-download-20261025"
            / name
        )
        if source.exists():
            shutil.copyfile(source, directory / ("download-" + name))
    for name in (
        "environment-sync.log",
        "cpu-regression.log",
        "campaign-tests.log",
        "startup-verification.log",
    ):
        if (ROOT / name).exists():
            shutil.copyfile(ROOT / name, directory / name)
    base = read(ROOT / "source-provenance.json")["commit"]
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    if head != base:
        subprocess.run(
            [
                "git",
                "bundle",
                "create",
                str(directory / "source-increment.bundle"),
                "HEAD",
                "^" + base,
            ],
            check=True,
        )
        subprocess.run(
            ["git", "bundle", "verify", str(directory / "source-increment.bundle")],
            check=True,
        )
    write(
        directory / "restoration.json",
        dict(
            schema="ball_capacity_restoration.v1",
            source_commit=head,
            full_bundle_commit=base,
            full_bundle=f"{RELATIVE}/inputs/inputs-v1/source.bundle",
            incremental_bundle="source-increment.bundle" if head != base else None,
            original_inputs=f"{RELATIVE.parent}/s42-20261103-main/inputs/inputs-v1",
            original_inputs_marker="b6cab45141f1ab50019870fe5ab7ea977d57a2a0af20751b676fafd0e3667a96",
            checkpoint_receipts=f"{RELATIVE}/persistence",
            auto_allocate_vm=False,
        ),
    )
    receipt = RcloneSnapshotStore(
        directory,
        remote="mounted_drive",
        remote_root=str(RELATIVE / "documentation"),
        config_path=Path("/tmp/ball-drive-mount.conf"),
    ).publish(label)
    path = ROOT / "documentation" / (label + "-receipt.json")
    write(path, receipt)
    RcloneArtifactStore(
        ROOT,
        remote="mounted_drive",
        remote_root=RELATIVE.as_posix(),
        config_path=Path("/tmp/ball-drive-mount.conf"),
    ).publish_file(path)
    print(receipt, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["inputs", "documentation"])
    parser.add_argument("--label")
    args = parser.parse_args()
    if args.action == "inputs":
        publish_inputs()
    else:
        if not args.label:
            parser.error("--label is required")
        publish_documentation(args.label)
