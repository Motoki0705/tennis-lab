"""Idempotently register a queue command, supervised by a systemd user unit."""
import argparse
import json
import os
import shlex
import subprocess
import time
from pathlib import Path

from rebuild_dataset import write


def launch(root, kind, session, attempt):
    root = root.resolve()
    queue = Path(subprocess.check_output(["git", "rev-parse", "--git-common-dir"], text=True).strip()).resolve().parent / ".training_queue"
    source = Path.cwd()
    script = Path(__file__).resolve().parent
    python = root / "env-original/bin/python"
    action = {
        "qualification": [str(script / "qualify.py"), "--root", str(root)],
        "probes": [str(script / "probe_matrix.py"), "--root", str(root)],
        "diagnostics": [str(script / "diagnose_checkpoint.py"), "--root", str(root)],
        "production": [str(script / "desktop.py"), "production", "--root", str(root)],
    }[kind]
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    if kind == "production":
        if subprocess.check_output(["git", "status", "--porcelain"], text=True).strip():
            raise ValueError("Production source must be a clean frozen checkout")
        subprocess.run(["git", "merge-base", "--is-ancestor", "HEAD", "origin/codex/ball-arch-desktop-20261113"], check=True)
        if not json.loads((root / "archive-acceptance.json").read_text())["complete"]:
            raise ValueError("Original Drive archive must finish before desktop writes")
    env = ["env", "TRAINING_QUEUE_DIR=" + str(queue), "RCLONE_CONFIG=" + os.environ["RCLONE_CONFIG"],
           "TENNIS_LAB_DINO_EXTENSION=" + str(root / "msda-build/lib/MultiScaleDeformableAttention.so"),
           "TORCHINDUCTOR_COMPILE_THREADS=1", "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True",
           "PYTHONDONTWRITEBYTECODE=1", "OMP_NUM_THREADS=2", "MKL_NUM_THREADS=2",
           "TORCHINDUCTOR_CACHE_DIR=" + str(root / "cache/inductor"), "TRITON_CACHE_DIR=" + str(root / "cache/triton"),
           "TENNIS_SOURCE_COMMIT=" + commit]
    command = shlex.join([*env, str(python), *action])
    cli = source / ".agents/skills/training-queue/scripts/training_queue.sh"
    request = ["bash", str(cli), "add", command, "--name", f"ball-arch-desktop-{kind}-a{attempt}",
               "--provider", "codex", "--session", session, "--issue", "1859", "--resource", "all",
               "--idempotency-key", f"i1859/desktop-20261113/{kind}-a{attempt}"]
    result = subprocess.run(request, capture_output=True, text=True,
                            env=dict(os.environ, TRAINING_QUEUE_DIR=str(queue)), check=True)
    if not result.stdout.startswith("queued: "):
        raise ValueError("Unexpected queue registration result")
    job = result.stdout.strip().removeprefix("queued: ").removesuffix(".job")
    unit = f"tennis-ball-arch-desktop-{kind}-20261113-a{attempt}"
    registration = root / (kind + f"-registration-a{attempt}.json")
    record = dict(schema_version=1, job=job, command=command, request=request, cwd=str(source),
                  source_commit=commit, unit=unit + ".service", session=session, resource="all", timestamp=time.time())
    if registration.exists():
        previous = json.loads(registration.read_text())
        if previous["job"] != job or previous["command"] != command:
            raise ValueError("Registration changed")
        print(json.dumps(previous), flush=True)
        return
    write(registration, record)
    # The systemd manager, rather than the agent runtime, owns every wrapper and
    # child. No long-lived descendant escapes the queue-created process group.
    service = ["systemd-run", "--user", "--unit=" + unit, "--property=WorkingDirectory=" + str(source),
               "--property=MemoryHigh=18G", "--property=MemoryMax=20G", "--property=MemorySwapMax=0",
               "--property=LimitNOFILE=1048576",
               "--setenv=TRAINING_QUEUE_DIR=" + str(queue), "--setenv=RCLONE_CONFIG=" + os.environ["RCLONE_CONFIG"],
               str(python), str(script / "service.py"), "--root", str(root),
               "--kind", "production" if kind == "production" else "qualification", "--queue", str(queue), "--job", job]
    notification = [str(python), str(script / "service.py"), "--root", str(root),
                    "--kind", "production" if kind == "production" else "qualification",
                    "--queue", str(queue), "--job", job, "--notify-only"]
    service.insert(3, "--property=ExecStopPost=" + shlex.join(notification))
    launch_result = subprocess.run(service, capture_output=True, text=True)
    write(root / (kind + f"-service-a{attempt}.json"), dict(command=service, returncode=launch_result.returncode,
          stdout=launch_result.stdout, stderr=launch_result.stderr))
    launch_result.check_returncode()
    print(json.dumps(record), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--kind", choices=("qualification", "probes", "production", "diagnostics"), required=True)
    parser.add_argument("--session", default=os.environ.get("CODEX_SESSION_ID"), required=not os.environ.get("CODEX_SESSION_ID"))
    parser.add_argument("--attempt", type=int, default=1)
    args = parser.parse_args()
    launch(args.root, args.kind, args.session, args.attempt)
