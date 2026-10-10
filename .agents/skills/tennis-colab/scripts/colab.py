"""JSON CLI for owned tennis-lab Colab sessions and remote commands."""

from __future__ import annotations

import argparse
import base64
import configparser
import json
import os
import re
import subprocess
import sys
import uuid
from pathlib import Path
from typing import Any

from colab_core import (
    COLAB_BIN,
    STATE_ROOT,
    ColabError,
    Session,
    atomic_json,
    checked,
    github_url,
    now,
    validate_name,
)
from colab_remote import TERMINAL, process_token, relative


def _repo_source(repo: Path, ref: str) -> tuple[str, str]:
    url = (
        checked(["git", "-C", str(repo), "remote", "get-url", "origin"])
        .stdout.decode()
        .strip()
    )
    if url.startswith("git@github.com:"):
        url = "https://github.com/" + url.removeprefix("git@github.com:")
    elif url.startswith("ssh://git@github.com/"):
        url = "https://github.com/" + url.removeprefix("ssh://git@github.com/")
    github_url(url)
    commit = (
        checked(
            [
                "git",
                "-C",
                str(repo),
                "rev-parse",
                "--verify",
                "--end-of-options",
                ref + "^{commit}",
            ]
        )
        .stdout.decode()
        .strip()
    )
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise ColabError("Could not resolve an exact Git commit")
    return url, commit


def _drive_binding(value: str) -> dict[str, Any]:
    drive_cli = Path(__file__).resolve().parents[2] / "tennis-drive/scripts/drive.py"
    result = checked(
        [
            sys.executable,
            str(drive_cli),
            "--remote-root",
            value,
            "inspect",
            ".",
            "--metadata-only",
        ],
        timeout=600,
    )
    metadata = json.loads(result.stdout)
    if metadata.get("type") != "directory" or not metadata.get("id"):
        raise ColabError("Drive project root must resolve to a unique directory ID")
    remote, root = value.split(":", 1)
    return {
        "drive_remote": remote,
        "drive_root": relative(root),
        "drive_root_id": metadata["id"],
    }


def _credential_copy(remote: str, destination: Path) -> None:
    # Select exactly one remote, without displaying or logging the config dump.
    try:
        data = json.loads(checked(["rclone", "config", "dump"]).stdout)
        section = data[remote]
        if section.get("type") != "drive" or not all(
            isinstance(value, str) for value in section.values()
        ):
            raise ColabError("A direct Google Drive rclone remote is required")
    except Exception:
        raise ColabError(
            "Cannot prepare the selected rclone credentials; configuration output withheld"
        ) from None
    config = configparser.ConfigParser(interpolation=None)
    config[remote] = section
    descriptor = os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w") as stream:
        config.write(stream)


def start(args: argparse.Namespace, session: Session) -> dict[str, Any]:
    repo = Path(args.repo).resolve()
    url, commit = _repo_source(repo, args.ref)
    plan = {
        "session": session.name,
        "gpu": args.gpu,
        "repo_url": url,
        "commit": commit,
        "drive": args.drive_root,
        "repo_root": "/content/tennis-lab",
    }
    if args.dry_run:
        return {"status": "planned", **plan}
    if session.state_file.exists() or session.registry.exists():
        raise ColabError(
            "Session name has existing state; inspect it or choose a new name"
        )
    binding = _drive_binding(args.drive_root)
    version = checked([str(session.colab_bin), "version"], data=b"").stdout.decode()
    if "0.7.4" not in version:
        raise ColabError(
            "This skill is verified with Colab CLI 0.7.4; run install_cli.sh"
        )
    credential = session.directory / "rclone-upload.conf"
    _credential_copy(binding["drive_remote"], credential)
    checked(
        [
            "ssh-keygen",
            "-q",
            "-t",
            "ed25519",
            "-N",
            "",
            "-C",
            f"tennis-colab:{session.name}",
            "-f",
            str(session.identity),
        ]
    )
    session.save(
        phase="allocating",
        created_at=now(),
        jobs={},
        colab_bin=str(session.colab_bin),
        **plan,
        **binding,
    )
    try:
        session.cli("new", "-s", session.name, "--gpu", args.gpu, timeout=300)
        session.save(phase="allocated")
        scripts = Path(__file__).resolve().parent
        helpers = {
            "core": session.upload(
                scripts / "colab_core.py", session.remote_root + "/colab_core.py"
            ),
            "worker": session.upload(
                scripts / "colab_remote.py", session.remote_helper
            ),
        }
        session.upload(credential, session.remote_root + "/secrets/rclone.conf")
        cfg = {
            **plan,
            **binding,
            "rclone_config": session.remote_root + "/secrets/rclone.conf",
            "sync_interval_seconds": 60,
            "helpers": helpers,
        }
        cfg_file = session.directory / "remote-config.json"
        atomic_json(cfg_file, cfg)
        session.upload(cfg_file, session.remote_root + "/session.json")
        observed = session.rpc("bootstrap", timeout=1800)
        return session.save(phase="ready", observed=observed, helpers=helpers)
    except Exception as error:
        session.save(phase="setup_failed", error=str(error))
        raise
    finally:
        credential.unlink(missing_ok=True)


def submit(args: argparse.Namespace, session: Session) -> dict[str, Any]:
    argv = args.argv[1:] if args.argv and args.argv[0] == "--" else args.argv
    if not argv:
        raise ColabError("Supply a command argument array after --")
    job_id = validate_name(args.job_id or "j-" + uuid.uuid4().hex[:16])
    receipt = session.rpc(
        "prepare",
        job_id=job_id,
        argv=argv,
        cwd=args.cwd,
        persist=args.persist,
        runner_outputs=args.runner_output,
    )
    local = session.directory / "jobs" / job_id
    local.mkdir(parents=True, exist_ok=False)
    atomic_json(
        local / "request.json",
        {
            "argv": argv,
            "cwd": args.cwd,
            "persist": args.persist,
            "runner_outputs": args.runner_output,
            "job_id": job_id,
        },
    )
    with (local / "transport.log").open("ab", buffering=0) as log:
        process = subprocess.Popen(
            session.ssh_argv(["python3", session.remote_helper, "run", job_id]),
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=log,
            start_new_session=True,
        )
    state = session.state()
    jobs = dict(state.get("jobs", {}))
    jobs[job_id] = {
        "transport_pid": process.pid,
        "transport_token": process_token(process.pid),
        "submitted_at": now(),
        "local_dir": str(local),
    }
    session.save(jobs=jobs)
    return {
        "session": session.name,
        "job_id": job_id,
        "status": "submitted",
        "remote": receipt,
        "transport_log": str(local / "transport.log"),
    }


def status(args: argparse.Namespace, session: Session) -> dict[str, Any]:
    state = session.state()
    if state["phase"] == "stopped":
        return {
            "session": session.name,
            "phase": "stopped",
            "jobs": state.get("jobs", {}),
        }
    response = session.rpc("status", job_id=args.job_id)
    for job in response["jobs"]:
        local = state.get("jobs", {}).get(job["job_id"])
        if local:
            token = local.get("transport_token")
            alive = token is not None and process_token(local["transport_pid"]) == token
            job["transport_alive"] = alive
            if job["status"] == "pending" and not alive:
                job["status"] = "dispatch_failed"
                job["transport_log"] = str(Path(local["local_dir"]) / "transport.log")
    return {"phase": state["phase"], **response}


def recover_diff(session: Session, output: Path) -> dict[str, Any]:
    response = session.rpc("diff", include_untracked=True)
    output.mkdir(parents=True, exist_ok=False)
    (output / "changes.patch").write_bytes(
        base64.b64decode(response["patch"], validate=True)
    )
    for name, encoded in response["untracked"].items():
        path = output / "untracked" / relative(name)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(base64.b64decode(encoded, validate=True))
    receipt = {
        "commit": response["commit"],
        "excluded": response["excluded"],
        "untracked": list(response["untracked"]),
        "output": str(output),
    }
    atomic_json(output / "receipt.json", receipt)
    return receipt


def stop(args: argparse.Namespace, session: Session) -> dict[str, Any]:
    state = session.state()
    if state["phase"] == "stopped":
        return {"session": session.name, "status": "already_stopped"}
    response = session.rpc("status")
    unsaved = [
        job["job_id"]
        for job in response["jobs"]
        if job["status"] not in TERMINAL or not job.get("drive_saved_at")
    ]
    if unsaved:
        raise ColabError(
            f"Commands are active or lack a saved receipt: {unsaved}; cancel/save and inspect before stopping"
        )
    recovered = recover_diff(
        session, session.directory / "recovered" / uuid.uuid4().hex[:12]
    )
    if recovered["excluded"]:
        raise ColabError(
            "Some untracked files were excluded from recovery; inspect them before stopping"
        )
    session.close_transport()
    session.cli("stop", "-s", session.name)
    session.save(phase="stopped", stopped_at=now(), recovered_diff=recovered)
    return {"session": session.name, "status": "stopped", "recovered_diff": recovered}


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--state-dir", type=Path, default=STATE_ROOT)
    result.add_argument(
        "--colab-bin",
        type=Path,
        default=Path(os.environ.get("TENNIS_LAB_COLAB_BIN", str(COLAB_BIN))),
    )
    commands = result.add_subparsers(dest="command", required=True)
    doctor = commands.add_parser("doctor")
    doctor.add_argument("--remote", action="store_true")
    start_parser = commands.add_parser("start")
    start_parser.add_argument("--session", required=True)
    start_parser.add_argument("--repo", default=".")
    start_parser.add_argument("--ref", default="HEAD")
    start_parser.add_argument(
        "--gpu", choices=("L4", "T4", "A100", "H100", "G4"), default="L4"
    )
    start_parser.add_argument(
        "--drive-root",
        default=os.environ.get("TENNIS_LAB_DRIVE_REMOTE", "gdrive:tennis_lab"),
    )
    start_parser.add_argument("--dry-run", action="store_true")
    execute = commands.add_parser("exec")
    execute.add_argument("--session", required=True)
    execute.add_argument("--job-id")
    execute.add_argument("--cwd")
    execute.add_argument("--persist", action="append", default=[])
    execute.add_argument(
        "--runner-output",
        action="append",
        default=[],
        help="Output already persisted live by the training runner; copy only after the process stops.",
    )
    execute.add_argument("argv", nargs=argparse.REMAINDER)
    for name in (
        "status",
        "logs",
        "cancel",
        "save",
        "upload",
        "download",
        "diff",
        "stop",
        "setup",
    ):
        command = commands.add_parser(name)
        command.add_argument("--session", required=True)
        if name in {"status", "logs", "cancel", "save"}:
            command.add_argument("--job-id", required=name != "status")
        if name == "logs":
            command.add_argument("--lines", type=int, default=80)
        if name in {"upload", "download"}:
            command.add_argument("source")
            command.add_argument("destination")
        if name == "diff":
            command.add_argument("--output", type=Path, required=True)
    return result


def main() -> int:
    args = parser().parse_args()
    try:
        if args.command == "doctor":
            result: dict[str, Any] = {
                "version": checked([str(args.colab_bin), "version"], data=b"")
                .stdout.decode()
                .strip()
            }
            if args.remote:
                result["usage"] = (
                    checked(
                        [str(args.colab_bin), "--auth", "oauth2", "usage"], data=b""
                    )
                    .stdout.decode()
                    .strip()
                )
        else:
            session = Session(
                args.session, state_root=args.state_dir, colab_bin=args.colab_bin
            )
            with session.lock():
                if args.command == "start":
                    result = start(args, session)
                elif args.command == "exec":
                    result = submit(args, session)
                elif args.command == "status":
                    result = status(args, session)
                elif args.command in {"logs", "cancel", "save"}:
                    result = session.rpc(
                        args.command,
                        job_id=args.job_id,
                        lines=getattr(args, "lines", 80),
                        timeout=1800 if args.command == "save" else 180,
                    )
                elif args.command == "upload":
                    if (
                        Path(args.destination) == Path(session.remote_root)
                        or Path(session.remote_root) in Path(args.destination).parents
                    ):
                        raise ColabError(
                            "Generic upload cannot overwrite managed session files"
                        )
                    result = session.upload(Path(args.source), args.destination)
                elif args.command == "download":
                    result = session.download(args.source, Path(args.destination))
                elif args.command == "diff":
                    result = recover_diff(session, args.output)
                elif args.command == "setup":
                    result = session.rpc("bootstrap", timeout=1800)
                    session.save(phase="ready", observed=result)
                else:
                    result = stop(args, session)
        print(
            json.dumps(
                {"schema_version": 1, "ok": True, **result},
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0
    except (ColabError, OSError, ValueError) as error:
        print(
            json.dumps(
                {"schema_version": 1, "ok": False, "error": str(error)},
                ensure_ascii=False,
            )
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
