"""Systemd owns the queue and records terminal state even on process failure."""
import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))
from rebuild_dataset import sha, write

REMOTE = "tennis_lab/outputs/ball_detection/train/arch_efficiency/s42-20261111/desktop-attempts/20261113-a1"


def notify(args, queue_returncode=None):
    terminal = [state for state in ("done", "failed", "cancelled")
                if (args.queue / state / (args.job + ".job")).is_file()]
    code = queue_returncode or 0
    if terminal != ["done"]:
        code = code or 1
    path = args.root / "notifications" / (args.job + ".json")
    if path.exists():
        record = json.loads(path.read_text())
        if record["job"] != args.job:
            raise ValueError("Terminal event identity differs")
    else:
        record = dict(schema_version=1, returncode=code, queue_returncode=queue_returncode,
                      terminal_states=terminal, state=terminal[0] if len(terminal)==1 else "process_ended_without_terminal_receipt",
                      finished_at=time.time(), kind=args.kind, job=args.job,
                      service_result=os.environ.get("SERVICE_RESULT"), pid=os.getpid(), periodic_progress_polling=False)
        write(path, record)
    if args.kind == "production":
        receipt_path = args.root / "notifications" / (args.job + "-upload.json")
        if receipt_path.exists():
            prior = json.loads(receipt_path.read_text())
            if not prior["verified"]:
                raise RuntimeError("Previous terminal event persistence failed; inspect without automatic retry")
            return record["returncode"]
        try:
            from src.utils.artifact_store import RcloneArtifactStore
            RcloneArtifactStore(args.root, remote="gdrive", remote_root=REMOTE,
                config_path=Path(os.environ["RCLONE_CONFIG"])).publish_file(path)
            remote = "gdrive:" + REMOTE + "/" + path.relative_to(args.root).as_posix()
            with tempfile.TemporaryDirectory(prefix="terminal-readback-",dir=args.root) as temporary:
                restored=Path(temporary)/path.name
                subprocess.run(["rclone","--config",os.environ["RCLONE_CONFIG"],"copyto",remote,str(restored)],check=True)
                if sha(restored)!=sha(path):
                    raise ValueError("Terminal event independent readback differs")
            write(receipt_path, dict(verified=True, remote=remote, sha256=sha(path)))
        except BaseException as error:
            write(receipt_path, dict(verified=False, error=str(error), local_event=str(path)))
            raise
    return record["returncode"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--kind", choices=("qualification", "production"), required=True)
    parser.add_argument("--queue", type=Path, required=True)
    parser.add_argument("--job", required=True)
    parser.add_argument("--notify-only", action="store_true")
    args = parser.parse_args()
    if args.notify_only:
        notify(args)
        return
    cli = Path.cwd() / ".agents/skills/training-queue/scripts/training_queue.sh"
    result = subprocess.run(["bash", str(cli), "serve", "--idle-timeout", "0"],
                            env=dict(os.environ, TRAINING_QUEUE_DIR=str(args.queue)))
    sys.exit(notify(args, result.returncode))


if __name__ == "__main__":
    main()
