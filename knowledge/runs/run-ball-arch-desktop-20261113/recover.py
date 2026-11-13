"""Recover the newest verified desktop generations after explicit writer shutdown."""

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))
from desktop import REMOTE, STAGES, stage_name
from rebuild_dataset import sha, write
from state import migrate_resume, model_config


def completed_metadata(root, selected, target, status):
    """Keep durable completed stages without duplicating their tensor caches."""
    if status["event"] != "completed" or not status.get("checkpoint_state_equal"):
        raise ValueError("Completed recovery needs a full-state-equal durable receipt")
    target.mkdir(parents=True, exist_ok=False)
    retired = []
    for path in selected.iterdir():
        if path.suffix == ".pt":
            retired.append(dict(path=str((target/path.name).relative_to(root)), sha256=sha(path),
                                size_bytes=path.stat().st_size,
                                protected_generation=status["receipt"]["generation"],
                                remote=status["receipt"]["remote"], execution_commit=status["execution_commit"],
                                reason="completed stage; independently restored tensor remains in Drive"))
        elif path.is_file():
            shutil.copyfile(path, target/path.name)
    write(target/"retired-weights.json",dict(schema_version=1, files=retired))


def main(root, stopped):
    if not stopped:
        raise ValueError(
            "Inspect the service/queue/GPU and explicitly confirm the old writer stopped"
        )
    if not json.loads((root / "preparation.json").read_text())["complete"]:
        raise ValueError("Recreate and verify fixed inputs/environment before recovery")
    if (root / "production").exists():
        raise ValueError("Recovery requires a new local production directory")
    config = Path(os.environ["RCLONE_CONFIG"])
    from src.utils.artifact_store import RcloneSnapshotStore

    work = root / "recovery"
    work.mkdir(exist_ok=False)
    # A closed pointer subtree is independently downloaded and content-verified.
    cli = Path(".agents/skills/tennis-drive/scripts/drive.py")
    prefix = [
        str(Path(".venv/bin/python").absolute()),
        str(cli),
        "--rclone-config",
        str(config),
    ]
    command = [
        *prefix,
        "download-tree",
        REMOTE.removeprefix("tennis_lab/") + "/stage-status",
        str(work / "stage-status"),
        "--progress-file",
        str(work / "stage-status-progress.json"),
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    write(
        work / "pointer-download.json",
        dict(returncode=result.returncode, stdout=result.stdout, stderr=result.stderr),
    )
    result.check_returncode()
    receipt_path = work / "preflight-receipt.json"
    result = subprocess.run(
        [
            *prefix,
            "download",
            REMOTE.removeprefix("tennis_lab/") + "/preflight-receipt.json",
            str(receipt_path),
            "--verify",
        ],
        capture_output=True,
        text=True,
    )
    write(
        work / "preflight-download.json",
        dict(returncode=result.returncode, stdout=result.stdout, stderr=result.stderr),
    )
    result.check_returncode()
    receipt = json.loads(receipt_path.read_text())
    preflight = work / "preflight"
    store = RcloneSnapshotStore(
        preflight,
        remote="gdrive",
        remote_root=REMOTE + "/documentation/preflight",
        config_path=config,
    )
    store.restore(receipt["generation"], manifest_sha256=receipt["manifest_sha256"])
    policy = json.loads((preflight / "checkpoint-policy.json").read_text())
    if not policy["confirmed"]:
        raise ValueError("Original checkpoint policy was never confirmed")
    for name in (
        "checkpoint-policy.json",
        "probe-acceptance.json",
        "probe-matrix.json",
        "native-qualification.json",
        "archive-acceptance.json",
    ):
        target = root / name
        if target.exists() and json.loads(target.read_text()) != json.loads(
            (preflight / name).read_text()
        ):
            raise ValueError("Local recovery prerequisite differs: " + name)
        shutil.copyfile(preflight / name, target)
    plan = json.loads((root / "original/plan.json").read_text())
    pointers = {stage_name(c,n): json.loads(path.read_text()) for c,n in STAGES
                if (path := work/"stage-status"/stage_name(c,n)/"state.json").exists()}
    restored = []
    for case, number in STAGES:
        name = stage_name(case, number)
        pointer = work / "stage-status" / name / "state.json"
        if not pointer.exists():
            continue
        status = json.loads(pointer.read_text())
        if (
            status["readback"]["status"] != "verified"
            or status["receipt"]["status"] != "verified"
        ):
            raise ValueError("Newest pointer is not independently verified")
        if not status["update"]:
            # An initialized generation has no scientific state to resume.
            # The normal stage initializer will reproduce the pinned parent.
            if status["event"] != "initialized":
                raise ValueError("Zero-update pointer is not initialization")
            continue
        receipt = status["receipt"]
        selected = work / "selected" / name
        RcloneSnapshotStore(
            selected,
            remote="gdrive",
            remote_root=REMOTE + "/production/" + name,
            config_path=config,
        ).restore(receipt["generation"], manifest_sha256=receipt["manifest_sha256"])
        configuration = model_config(plan, case, policy["scope"])
        target = root / "production" / name
        child = pointers.get(stage_name("T2",1), {})
        needs_parent_state = case == "T2" and number == 0 and not child.get("update",0)
        completed = status["event"] == "completed"
        if completed and not needs_parent_state:
            completed_metadata(root,selected,target,status)
            digest=json.loads((target/"latest.json").read_text())["sha256"]
        else:
            checkpoint = migrate_resume(
                root, selected, target, configuration, expected_update=status["update"]
            )
            digest=sha(checkpoint)
            for source in selected.iterdir():
                if source.name.startswith("validation-") or source.name in {"COMPLETED.json", "STOPPED.json"}:
                    shutil.copyfile(source, target / source.name)
        write(root / "stage-status" / name / "state.json", status)
        restored.append(
            dict(
                stage=name,
                update=status["update"],
                completed=completed,
                source_receipt=receipt,
                adapted_checkpoint_sha256=digest,
                tensor_cache_retired=completed and not needs_parent_state,
            )
        )
        # The immutable generation is still on Drive. Only one downloaded
        # recovery cache is retained at a time to preserve physical D: headroom.
        shutil.rmtree(selected)
    if not restored:
        raise ValueError(
            "No verified desktop generation exists; do not silently fall back to the initial checkpoint"
        )
    write(
        root / "recovery-acceptance.json",
        dict(
            complete=True,
            stages=restored,
            policy=policy,
            old_writer_confirmed_stopped=True,
            scientific_state_exact=True,
            finished_at=time.time(),
        ),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--old-writer-stopped", action="store_true", required=True)
    args = parser.parse_args()
    main(args.root.resolve(), args.old_writer_stopped)
