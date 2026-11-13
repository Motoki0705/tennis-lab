"""Pinned serial desktop continuation: T1 tail -> T2 -> C1 -> C2 -> C3."""
from __future__ import annotations

import argparse
import fcntl
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
import uuid
from contextlib import nullcontext
from dataclasses import asdict, replace
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))
from audit import audit_resume, validate_evaluation, validate_log
from guard import configure, host_guard
from rebuild_dataset import sha, write
from state import migrate_t1, model_config, prepare, same

REMOTE = "tennis_lab/outputs/ball_detection/train/arch_efficiency/s42-20261111/desktop-attempts/20261113-a1"
STAGES = (("T1", 1), ("T2", 0), ("T2", 1), ("C1", 0), ("C2", 0), ("C3", 0))


def read(path):
    return json.loads(path.read_text())


def stage_name(case, number):
    return f"{case}-q{number*6000:05d}-q{(number+1)*6000:05d}" if case.startswith("T") else case + "-dpt-10000"


def budget(plan, case, number):
    from src.tasks.ball_detection.training.temporal.configuration import UpdateBudget
    raw = next(j for j in plan["jobs"] if j["case"] == case)["stages"][number]
    return UpdateBudget(**{k: raw[k] for k in UpdateBudget.__dataclass_fields__})


def publish_index(root, path, config):
    from src.utils.artifact_store import RcloneArtifactStore
    RcloneArtifactStore(root, remote="gdrive", remote_root=REMOTE, config_path=config).publish_file(path)
    # Mutable indexes also receive independent readback, with the same saved hash.
    with tempfile.TemporaryDirectory(dir=root, prefix="index-readback-") as temporary:
        check = Path(temporary) / path.name
        subprocess.run(["rclone", "--config", str(config), "copyto", "gdrive:" + REMOTE + "/" + path.relative_to(root).as_posix(), str(check)], check=True)
        if sha(check) != sha(path):
            raise ValueError("Index independent readback differs")


def publish_preflight(root, config):
    from src.utils.artifact_store import RcloneSnapshotStore
    source = Path(__file__).resolve().parent
    directory = root / "documentation/preflight"
    if directory.exists():
        receipt = read(root / "preflight-receipt.json")
        verification = read(root / "preflight-readback.json")
        if receipt["status"] != "verified" or verification["status"] != "verified":
            raise ValueError("Existing preflight was not independently verified")
        if read(directory / "checkpoint-policy.json") != read(root / "checkpoint-policy.json"):
            raise ValueError("Checkpointing policy changed after preflight publication")
        return
    directory.mkdir(parents=True)
    names = ("checkpoint-policy.json", "probe-acceptance.json", "probe-matrix.json",
             "native-qualification.json", "full-shape-equivalence.json", "archive-acceptance.json",
             "preparation.json", "inputs-verified.json", "dataset-verified.json", "environment-verified.json",
             "environment-freeze.txt", "environment-exact-runtime.txt", "environment-missing-dependencies.txt",
             "original-vm-pip-freeze.txt", "msda-build-manifest.json", "msda-numerical-check.json")
    for name in names:
        shutil.copyfile(root / name, directory / name)
    for name in ("recovery-bootstrap.json", "recovery-acceptance.json"):
        if (root/name).exists():
            shutil.copyfile(root/name,directory/name)
    for source_name, saved_name in (
        ("original-preparation/msda-compatibility.patch", "msda-compatibility.patch"),
        ("msda-build/invocation.json", "msda-build-invocation.json"),
        ("msda-build/resources.txt", "msda-build-resources.txt"),
        ("msda-build-sm120-unbounded/resources.txt", "msda-unbounded-resources.txt"),
        ("msda-build-manifest-unbounded.json", "msda-unbounded-build-manifest.json"),
        ("checkpoint-diagnostic.json", "checkpoint-diagnostic.json"),
    ):
        if (root/source_name).exists():
            shutil.copyfile(root/source_name,directory/saved_name)
    for path in source.iterdir():
        if path.is_file() and path.suffix in {".py", ".md", ".json"}:
            target = directory / "reproduction" / path.name
            target.parent.mkdir(exist_ok=True)
            shutil.copyfile(path, target)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    write(directory / "execution.json", dict(commit=commit, cwd=str(Path.cwd()), python=sys.executable,
          queue_id=os.environ["TENNIS_RUN_ID"], root=str(root), source_clean=True,
          remote_root=REMOTE, source_policy="pushed frozen checkout", timestamp=time.time()))
    store = RcloneSnapshotStore(directory, remote="gdrive", remote_root=REMOTE + "/documentation/preflight", config_path=config)
    generation = "preflight-" + uuid.uuid4().hex[:12]
    receipt = store.publish(generation)
    with tempfile.TemporaryDirectory(prefix="preflight-readback-", dir=root) as temporary:
        verification = RcloneSnapshotStore(Path(temporary)/"verified", remote="gdrive", remote_root=store.remote_root,
                                          config_path=config).restore(generation, manifest_sha256=receipt["manifest_sha256"])
    write(root / "preflight-receipt.json", receipt)
    write(root / "preflight-readback.json", verification)
    publish_index(root, root / "preflight-receipt.json", config)
    publish_index(root, root / "preflight-readback.json", config)


def stage_complete(root, case, number):
    pointer = root / "stage-status" / stage_name(case, number) / "state.json"
    if not pointer.exists():
        return False
    status = read(pointer)
    if status["event"] != "completed":
        return False
    if status["receipt"]["status"] != "verified" or status["readback"]["status"] != "verified":
        raise ValueError("Completion is not durably verified")
    output = root / "production" / stage_name(case, number)
    bud = budget(read(root / "original/plan.json"), case, number)
    marker = read(output / "COMPLETED.json")
    latest = read(output / "latest.json")
    if not marker["complete"] or latest["update"] != bud.updates or status["update"] != bud.updates:
        raise ValueError("Completion update differs from its scientific budget")
    checkpoint = output / latest["checkpoint"]
    if checkpoint.exists():
        if sha(checkpoint) != latest["sha256"]:
            raise ValueError("Completed local checkpoint is corrupt")
    else:
        retired = read(output / "retired-weights.json")
        if not any(item["path"] == str(checkpoint.relative_to(root)) and item["sha256"] == latest["sha256"]
                   and item["protected_generation"] == status["receipt"]["generation"] for item in retired["files"]):
            raise ValueError("Missing completed checkpoint has no verified retirement record")
    validate_log(output, bud)
    validate_evaluation(output / f"validation-update-{bud.updates:08d}.json",
                        root / "prepared" / f"evaluation_T{64 if case.startswith('T') else 32}.json")
    return True


def retire_weights(root, directory, *, include_latest):
    """Evict redundant local tensor caches; immutable Drive generations stay intact."""
    latest = read(directory / "latest.json")
    protected = {}
    for path in sorted((root / "persistence" / directory.name).glob("*.json")):
        status = read(path)
        if status["receipt"]["status"] != "verified" or status["readback"]["status"] != "verified":
            continue
        if not status.get("checkpoint_state_equal"):
            continue
        for field, digest in (("checkpoint_file", "checkpoint_sha256"), ("weights_file", "weights_sha256")):
            if field in status:
                key = (status[field], status[digest])
                if key not in protected or status["updated_at"] > protected[key]["updated_at"]:
                    protected[key] = status
    ledger_path = directory / "retired-weights.json"
    ledger = read(ledger_path) if ledger_path.exists() else dict(schema_version=1, files=[])
    entries = []
    for path in sorted(directory.glob("*-update-*.pt")):
        if not include_latest and path.name in {latest["checkpoint"], f"weights-update-{latest['update']:08d}.pt"}:
            continue
        digest = sha(path)
        proof = protected.get((path.name, digest))
        if proof is None:
            raise ValueError("Refusing to retire a tensor without its exact verified snapshot")
        entries.append(dict(path=str(path.relative_to(root)), sha256=digest, size_bytes=path.stat().st_size,
                            protected_generation=proof["receipt"]["generation"], remote=proof["receipt"]["remote"],
                            execution_commit=proof["execution_commit"], reason="verified redundant local cache; Drive generation retained"))
    if entries:
        ledger["files"].extend(entries)
        write(ledger_path, ledger)
        publish_index(root, ledger_path, Path(os.environ["RCLONE_CONFIG"]))
        for item in entries:
            (root / item["path"]).unlink()


def run_ordered(stages, complete, execute, save):
    for key in stages:
        if complete(key):
            save(key, "completed")
            continue
        save(key, "running")
        try:
            execute(key)
            if not complete(key):
                raise RuntimeError("Stage returned without durable completion")
        except BaseException:
            save(key, "failed")
            raise
        save(key, "completed")


def persistence(root, output, case, number, mode, config, actual_budget=None):
    import torch

    from src.utils.artifact_store import RcloneSnapshotStore
    plan = read(root / "original/plan.json")
    bud = actual_budget or budget(plan, case, number)

    def persist(directory, event):
        if directory != output:
            raise ValueError("Unexpected persistence path")
        rows = validate_log(directory, bud) if (directory / "train.jsonl").exists() else []
        latest_path = directory / "latest.json"
        update = read(latest_path)["update"] if latest_path.exists() else 0
        status = dict(schema_version=1, case=case, stage=stage_name(case, number), event=event,
                      update=update, rows=len(rows), updated_at=time.time(), mode=mode,
                      execution_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip())
        names = [name for name in ("run.json", "train_manifest.json", "evaluation_manifest.json", "train.jsonl",
                    "migration.json", "actual-resume.json", "COMPLETED.json", "STOPPED.json", "FAILED.json") if (directory / name).exists()]
        if update:
            latest = read(latest_path)
            status.update(checkpoint_file=latest["checkpoint"], checkpoint_sha256=latest["sha256"],
                          weights_file=f"weights-update-{update:08d}.pt",
                          weights_sha256=sha(directory / f"weights-update-{update:08d}.pt"))
            names += ["latest.json", latest["checkpoint"], f"weights-update-{update:08d}.pt"]
            report = directory / f"validation-update-{update:08d}.json"
            if report.exists():
                validate_evaluation(report, root / "prepared" / f"evaluation_T{64 if case.startswith('T') else 32}.json")
                names += [report.name, report.with_suffix(".frames.jsonl").name]
        if mode == "production":
            store = RcloneSnapshotStore(directory, remote="gdrive", remote_root=REMOTE + "/production/" + directory.name, config_path=config)
            generation = f"{event}-{update:08d}-{uuid.uuid4().hex[:8]}"
            receipt = store.publish(generation, include=names)
            with tempfile.TemporaryDirectory(prefix="checkpoint-readback-", dir=root) as temporary:
                restored = Path(temporary) / "verified"
                verification = RcloneSnapshotStore(restored, remote="gdrive", remote_root=store.remote_root,
                                    config_path=config).restore(generation, manifest_sha256=receipt["manifest_sha256"])
                if update:
                    expected = torch.load(directory / latest["checkpoint"], map_location="cpu", weights_only=True)
                    actual = torch.load(restored / latest["checkpoint"], map_location="cpu", weights_only=True)
                    if not same(expected, actual):
                        raise ValueError("Checkpoint independent full-state readback differs")
                    status["checkpoint_state_equal"] = True
                    del expected, actual
            status.update(receipt=receipt, readback=verification)
            receipt_path = root / "persistence" / directory.name / (generation + ".json")
            write(receipt_path, status)
            publish_index(root, receipt_path, config)
            pointer = root / "stage-status" / directory.name / "state.json"
            write(pointer, status)
            publish_index(root, pointer, config)
            if update:
                retire_weights(root, directory, include_latest=event == "completed" and (case, number) != ("T2", 0))
            if case == "T2" and number == 1 and update >= 500:
                parent = directory.parent / stage_name("T2", 0)
                retire_weights(root, parent, include_latest=True)
        write(root / (mode + "-status.json"), status)
        print(json.dumps(dict(notification="persistence", **status)), flush=True)
    return persist


def run_stage(root, mode, case, number, scope, stop=None, joint=False):
    import torch

    from src.tasks.ball_detection.data.augmentation.video import AugmentationConfig
    from src.tasks.ball_detection.training.runtime.execution import CoordinateRuntime
    from src.tasks.ball_detection.training.temporal.runner import train_temporal_model
    if not read(root / "preparation.json")["complete"]:
        raise ValueError("Verified preparation required")
    if mode == "production":
        policy = read(root / "checkpoint-policy.json")
        if not policy["confirmed"] or scope != policy["scope"]:
            raise ValueError("Production checkpointing policy is immutable")
    guard = configure(root)
    plan = read(root / "original/plan.json")
    config = model_config(plan, case, scope)
    bud = budget(plan, case, number)
    if joint:
        if mode != "probe":
            raise ValueError("Joint-memory diagnostic must never change production")
        bud = replace(bud, freeze_updates=0)
    output = root / ("production" if mode == "production" else "probes/" + scope) / stage_name(case, number)
    options = {}
    if case == "T1" and not output.exists():
        checkpoint = migrate_t1(root, output, config)
        options["resume"] = checkpoint
    elif (output / "latest.json").exists():
        options["resume"] = output / read(output / "latest.json")["checkpoint"]
    elif case.startswith("T"):
        if number:
            previous = output.parent / stage_name(case, 0)
            if not (previous / "COMPLETED.json").exists():
                raise ValueError("Previous T2 stage is incomplete")
            latest = read(previous / "latest.json")
            checkpoint, parent_update = previous / latest["checkpoint"], 6000
        else:
            checkpoint, parent_update = root / "prepared/c0-checkpoint.pt", 10000
        options.update(initial_artifact=checkpoint, initial_artifact_sha256=sha(checkpoint), parent_update=parent_update)
    if stop is not None:
        options["stop_after_update"] = stop
    origin_dir = root / "original/T1-q06000-q12000/checkpoint-00005500-cddcef20" if case.startswith("T") else root / "inputs/common-dpt-T32"
    origin = read(origin_dir / "run.json")["recipe"]
    metadata = dict(guard, model=asdict(config), budget=bud.payload(), diagnostic_joint_unfrozen=joint,
                    mode=mode, case=case, stage=number, pid=os.getpid(), started_at=time.time())
    path = root / (mode + "-environment-" + case + "-" + str(number) + "-" + scope + ".json")
    write(path, metadata)
    audit = audit_resume(options["resume"], output / "actual-resume.json") if "resume" in options else nullcontext()
    with audit, host_guard(path.with_name(path.stem + "-host.json")):
        result = train_temporal_model(
            architecture="query" if case.startswith("T") else "dpt", phase="finetune" if case.startswith("T") else "pretrain",
            manifest=root / "prepared" / f"train_T{64 if case.startswith('T') else 32}.json",
            evaluation_manifest=root / "prepared" / f"evaluation_T{64 if case.startswith('T') else 32}.json",
            model_config=config, output=output, budget=bud, runtime=CoordinateRuntime(**origin["runtime"]),
            device=torch.device("cuda"), augmentation=AugmentationConfig(**origin["augmentation"]), distance_px=20.,
            persist=persistence(root, output, case, number, mode, Path(os.environ["RCLONE_CONFIG"]), bud), **options)
    rows = validate_log(output, bud)
    relevant = [row for row in rows if row["update"] > (5500 if case == "T1" else 0)]
    summary = dict(metadata, result=result, status="passed", samples=relevant,
                   last_update_peak_allocated=torch.cuda.max_memory_allocated(), last_update_peak_reserved=torch.cuda.max_memory_reserved())
    write(path.with_name(path.stem + "-result.json"), summary)
    return result


def production(root):
    policy = read(root / "checkpoint-policy.json")
    if not policy["confirmed"] or not read(root / "probe-acceptance.json")["complete"]:
        raise ValueError("Confirmed common checkpoint policy and accepted probes required")
    if (root / "production-FAILED.json").exists():
        raise ValueError("Failed attempt requires explicit diagnosis; no automatic retry")
    if not read(root / "archive-acceptance.json")["complete"]:
        raise ValueError("Closed original Drive tree must be archived before adding desktop writes")
    with (root / "desktop-writer.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        publish_preflight(root, Path(os.environ["RCLONE_CONFIG"]))
        status = dict(schema_version=1, started_at=time.time(), stages={stage_name(c,n): "queued" for c,n in STAGES},
                      scope=policy["scope"], pid=os.getpid(), queue_id=os.environ["TENNIS_RUN_ID"])
        def save(key, value):
            case, number = key
            name = stage_name(case, number)
            status["stages"][name] = value
            status["updated_at"] = time.time()
            write(root / "serial-status.json", status)
            publish_index(root, root / "serial-status.json", Path(os.environ["RCLONE_CONFIG"]))

        def execute(key):
            case, number = key
            name = stage_name(case, number)
            command = [sys.executable, __file__, "stage", "--root", str(root), "--case", case, "--number", str(number), "--scope", policy["scope"], "--mode", "production"]
            logs = root / "logs"
            logs.mkdir(exist_ok=True)
            with (logs / (name + ".log")).open("ab", buffering=0) as stream:
                subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, check=True)
        run_ordered(STAGES, lambda key: stage_complete(root, *key), execute, save)
        write(root / "COMPLETED.json", dict(complete=True, status=status, finished_at=time.time()))
        publish_index(root, root / "COMPLETED.json", Path(os.environ["RCLONE_CONFIG"]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "stage", "production"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--mode", choices=("probe", "production"), default="probe")
    parser.add_argument("--case", choices=("T1", "T2", "C1", "C2", "C3"))
    parser.add_argument("--number", type=int, default=0, choices=(0,1))
    parser.add_argument("--scope", choices=("msda", "encoder_msda"), default="encoder_msda")
    parser.add_argument("--stop", type=int)
    parser.add_argument("--joint", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    try:
        if args.action == "prepare":
            prepare(root)
        elif args.action == "production":
            production(root)
        else:
            run_stage(root, args.mode, args.case, args.number, args.scope, args.stop, args.joint)
    except BaseException as error:
        import torch
        memory = dict(peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                      peak_reserved_bytes=torch.cuda.max_memory_reserved()) if torch.cuda.is_initialized() else {}
        write(root / (args.mode + "-FAILED-" + (args.case or args.action) + "-" + args.scope + ".json"),
              dict(error=str(error), traceback=traceback.format_exc(), timestamp=time.time(),
                   kind="oom" if isinstance(error, torch.OutOfMemoryError) else "failed", **memory))
        if args.action == "production":
            path = root / "production-FAILED.json"
            write(path, dict(error=str(error), traceback=traceback.format_exc(), timestamp=time.time()))
            try:
                publish_index(root, path, Path(os.environ["RCLONE_CONFIG"]))
            except BaseException:
                traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
