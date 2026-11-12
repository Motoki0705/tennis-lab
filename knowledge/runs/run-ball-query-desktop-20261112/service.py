"""Systemd-owned shared queue lifecycle with terminal events, without polling."""

import argparse
import json
import os
import select
import shlex
import subprocess
import sys
import time
from pathlib import Path


def notify(args, returncode=None):
    from desktop import REMOTE, write

    previous = args.root / "terminal-event-upload.json"
    if previous.exists():
        if json.loads(previous.read_text())["returncode"]:
            raise RuntimeError(
                "A previous terminal event upload failed; no automatic retry"
            )
        return
    registration = args.root / "production-registration.json"
    job = args.job or (
        json.loads(registration.read_text())["job"]
        if registration.exists()
        else "not-registered"
    )
    states = [
        state
        for state in ("done", "failed", "cancelled")
        if (args.queue / state / (job + ".job")).is_file()
    ]
    record = dict(
        schema_version=1,
        job=job,
        service_result=os.environ.get("SERVICE_RESULT"),
        queue_returncode=returncode,
        terminal_states=states,
        state=states[0]
        if len(states) == 1
        else "process_ended_without_terminal_receipt",
        timestamp=time.time(),
        periodic_polling=False,
    )
    target = args.root / "notifications" / (job + ".json")
    write(target, record)
    progress_file = args.root / "progress.json"
    if progress_file.exists():
        progress = json.loads(progress_file.read_text())
        progress.update(
            phase="production_terminal_event",
            production_terminal=record,
            updated_at=time.time(),
        )
        write(progress_file, progress)
    cli = args.checkout / ".agents/skills/tennis-drive/scripts/drive.py"
    command = [
        str(args.checkout / ".venv/bin/python"),
        str(cli),
        "--rclone-config",
        str(args.rclone_config),
        "upload",
        str(target),
        REMOTE.removeprefix("tennis_lab/") + "/notifications/" + target.name,
        "--verify",
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    write(
        args.root / "terminal-event-upload.json",
        dict(returncode=result.returncode, stdout=result.stdout, stderr=result.stderr),
    )
    if result.returncode:
        raise RuntimeError("Terminal event Drive persistence failed")


def wait_and_register(args):
    from desktop import write

    # One process-exit event, without periodic wakeups or follow-up messages.
    stat_path = Path(f"/proc/{args.wait_pid}/stat")
    if stat_path.exists():
        stat = stat_path.read_text()
        token = stat[stat.rindex(")") + 2 :].split()[19]
        if token != args.wait_start_token:
            raise ValueError("The owning runtime PID was reused")
        descriptor = os.pidfd_open(args.wait_pid)
        try:
            poll = select.poll()
            poll.register(descriptor, select.POLLIN)
            write(
                args.root / "deferred-launch.json",
                dict(
                    state="waiting_for_worker_runtime_exit",
                    pid=os.getpid(),
                    wait_pid=args.wait_pid,
                    wait_start_token=token,
                    old_probe_pgid=args.old_probe_pgid,
                    timestamp=time.time(),
                ),
            )
            poll.poll()
        finally:
            os.close(descriptor)
    try:
        os.killpg(args.old_probe_pgid, 0)
    except ProcessLookupError:
        pass
    else:
        raise RuntimeError("Previous probe PGID still exists; no lock bypass")
    if not json.loads((args.root / "probe-acceptance.json").read_text())["complete"]:
        raise ValueError("The exact-recipe probe did not pass")
    if not json.loads((args.root / "science-audit.json").read_text())["complete"]:
        raise ValueError("The fixed scientific conditions did not pass")
    command = shlex.join(
        [
            "env",
            "TRAINING_QUEUE_DIR=" + str(args.queue),
            "TORCHINDUCTOR_COMPILE_THREADS=1",
            "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True",
            str(args.root / "env/bin/python"),
            str(Path(__file__).with_name("desktop.py")),
            "production",
            "--root",
            str(args.root),
            "--source",
            str(args.source),
            "--rclone-config",
            str(args.rclone_config),
        ]
    )
    cli = args.checkout / ".agents/skills/training-queue/scripts/training_queue.sh"
    request = [
        "bash",
        str(cli),
        "add",
        command,
        "--name",
        "ball-query-length-desktop-production-a1",
        "--provider",
        "codex",
        "--session",
        args.session,
        "--issue",
        "1899",
        "--resource",
        "all",
        "--idempotency-key",
        "i1899/desktop-20261112/production-a1",
    ]
    result = subprocess.run(
        request,
        cwd=args.source,
        capture_output=True,
        text=True,
        env=dict(os.environ, TRAINING_QUEUE_DIR=str(args.queue)),
    )
    if result.returncode or not result.stdout.startswith("queued: "):
        write(
            args.root / "deferred-launch-FAILED.json",
            dict(
                returncode=result.returncode, stdout=result.stdout, stderr=result.stderr
            ),
        )
        raise RuntimeError("Production registration failed; no automatic retry")
    job = result.stdout.strip().removeprefix("queued: ").removesuffix(".job")
    write(
        args.root / "production-registration.json",
        dict(
            job=job,
            command=command,
            cwd=str(args.source),
            stdout=result.stdout,
            stderr=result.stderr,
            session=args.session,
            provider="codex",
            resource="all",
            timestamp=time.time(),
        ),
    )
    write(
        args.root / "deferred-launch.json",
        dict(state="registered", job=job, timestamp=time.time()),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("serve", "notify", "wait-and-serve"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--queue", type=Path, required=True)
    parser.add_argument("--checkout", type=Path, required=True)
    parser.add_argument("--job")
    parser.add_argument("--source", type=Path)
    parser.add_argument("--session")
    parser.add_argument("--wait-pid", type=int)
    parser.add_argument("--wait-start-token")
    parser.add_argument("--old-probe-pgid", type=int)
    parser.add_argument("--rclone-config", type=Path, required=True)
    args = parser.parse_args()
    if args.action == "notify":
        notify(args)
        return
    if args.action == "wait-and-serve":
        wait_and_register(args)
    cli = args.checkout / ".agents/skills/training-queue/scripts/training_queue.sh"
    result = subprocess.run(
        ["bash", str(cli), "serve", "--idle-timeout", "0"],
        env=dict(os.environ, TRAINING_QUEUE_DIR=str(args.queue)),
    )
    notify(args, result.returncode)
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
