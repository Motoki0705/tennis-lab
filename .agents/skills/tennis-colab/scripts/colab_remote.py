"""Remote worker and JSON RPC deployed into one owned Colab session."""

from __future__ import annotations

import base64
import hashlib
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path, PurePosixPath
from typing import Any

from colab_core import (
    ColabError,
    atomic_json,
    checked,
    github_url,
    now,
    read_json,
    validate_name,
)

ROOT = Path(__file__).resolve().parent
TERMINAL = {"completed", "failed", "cancelled", "save_failed"}


def config() -> dict[str, Any]:
    return read_json(ROOT / "session.json")


def relative(value: str) -> str:
    path = PurePosixPath(value)
    if (
        not value
        or path.is_absolute()
        or any(part in {".", ".."} for part in value.split("/"))
        or ":" in value
        or "\\" in value
    ):
        raise ColabError("Expected a normalized, non-empty relative path")
    return path.as_posix()


def job_dir(job_id: str) -> Path:
    return ROOT / "jobs" / validate_name(job_id)


def process_token(pid: int) -> str | None:
    try:
        fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
    except FileNotFoundError:
        return None
    return None if fields[0] == "Z" else fields[19]


def safe_file(path: str) -> Path:
    resolved = Path(path).resolve()
    if resolved == ROOT / "secrets" or ROOT / "secrets" in resolved.parents:
        raise ColabError("Managed credentials cannot be downloaded or published")
    if not resolved.is_file():
        raise ColabError(f"Not a regular file: {path}")
    return resolved


def tail(path: Path, *, max_bytes: int = 64 * 1024, lines: int = 80) -> str:
    if not path.exists():
        return ""
    with path.open("rb") as stream:
        stream.seek(max(0, path.stat().st_size - max_bytes))
        return "\n".join(
            stream.read(max_bytes).decode(errors="replace").splitlines()[-lines:]
        )


def _persist_path(cfg: dict[str, Any], path: str) -> Path:
    fragment = relative(path)
    if any(
        part in {".git", ".venv", ".secrets"} for part in PurePosixPath(fragment).parts
    ):
        raise ColabError("Repository internals and credentials are not artifacts")
    repo = Path(cfg["repo_root"]).resolve()
    result = (repo / fragment).resolve()
    if repo not in result.parents:
        raise ColabError("Artifact path escapes the remote repository")
    return result


def bootstrap(request: dict[str, Any]) -> dict[str, Any]:
    cfg = config()
    repo = Path(cfg["repo_root"])
    github_url(cfg["repo_url"])
    if not re.fullmatch(r"[0-9a-f]{40}", cfg["commit"]):
        raise ColabError("An exact Git commit is required")
    gpu = (
        checked(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
            timeout=60,
        )
        .stdout.decode()
        .strip()
    )
    if cfg["gpu"].casefold() not in gpu.casefold():
        raise ColabError(
            f"Requested {cfg['gpu']}, observed {gpu}; no accelerator fallback"
        )
    if repo.exists():
        changes = checked(["git", "-C", str(repo), "status", "--porcelain"]).stdout
        if changes:
            raise ColabError(
                "Remote checkout has changes; recover/inspect them before setup"
            )
    else:
        checked(
            [
                "git",
                "clone",
                "--filter=blob:none",
                "--no-checkout",
                cfg["repo_url"],
                str(repo),
            ],
            timeout=600,
        )
        checked(["git", "-C", str(repo), "fetch", "origin", cfg["commit"]], timeout=600)
        checked(
            ["git", "-C", str(repo), "checkout", "--detach", cfg["commit"]], timeout=600
        )
    observed = (
        checked(["git", "-C", str(repo), "rev-parse", "HEAD"]).stdout.decode().strip()
    )
    if observed != cfg["commit"]:
        raise ColabError("Remote Git checkout differs from requested commit")
    if shutil.which("rclone") is None:
        checked(["apt-get", "update", "-qq"], timeout=600)
        checked(["apt-get", "install", "-y", "rclone"], timeout=600)
    if shutil.which("uv") is None:
        checked([sys.executable, "-m", "pip", "install", "uv==0.11.3"], timeout=600)
    secret = Path(cfg["rclone_config"])
    if not secret.is_file() or secret.stat().st_mode & 0o077:
        raise ColabError("Remote rclone credentials must be a private regular file")
    quota = checked(
        [
            "rclone",
            "--config",
            str(secret),
            "about",
            cfg["drive_remote"] + ":",
            "--json",
        ],
        timeout=180,
    )
    return {
        "gpu": gpu,
        "commit": observed,
        "python": sys.version,
        "quota": json.loads(quota.stdout),
    }


def jobs() -> list[dict[str, Any]]:
    result = []
    for state_path in sorted((ROOT / "jobs").glob("*/status.json")):
        state = read_json(state_path)
        if state["status"] in {"starting", "running", "finalizing"} and state.get(
            "worker_pid"
        ):
            token = state.get("worker_token")
            alive = token is not None and process_token(state["worker_pid"]) == token
            if not alive:
                state = {
                    **state,
                    "status": "unknown",
                    "reason": "worker exited without a durable completion receipt",
                }
        result.append(state)
    return result


def prepare(request: dict[str, Any]) -> dict[str, Any]:
    cfg = config()
    pending = [item for item in jobs() if item["status"] not in TERMINAL]
    if pending:
        raise ColabError(
            f"Inspect the existing command before starting another: {[item['job_id'] for item in pending]}"
        )
    directory = job_dir(request["job_id"])
    if directory.exists():
        raise ColabError("Job ID already exists; never resubmit it blindly")
    argv = request["argv"]
    if (
        not isinstance(argv, list)
        or not argv
        or not all(isinstance(item, str) for item in argv)
    ):
        raise ColabError("Command must be a non-empty argument array")
    cwd = Path(request.get("cwd") or cfg["repo_root"]).resolve()
    if not cwd.is_dir():
        raise ColabError(f"Working directory is absent: {cwd}")
    persist = request.get("persist", [])
    runner_outputs = request.get("runner_outputs", [])
    if not isinstance(persist, list) or not isinstance(runner_outputs, list):
        raise ColabError("Output declarations must be arrays of relative directories")
    outputs = [*persist, *runner_outputs]
    if not all(isinstance(item, str) for item in outputs):
        raise ColabError("Output directories must be strings")
    for index, item in enumerate(outputs):
        _persist_path(cfg, item)
        for other in outputs[:index]:
            if (
                item == other
                or PurePosixPath(item) in PurePosixPath(other).parents
                or PurePosixPath(other) in PurePosixPath(item).parents
            ):
                raise ColabError(
                    "Output directories overlap; choose exactly one persistence owner"
                )
    source_changes = diff({"include_untracked": True})
    directory.mkdir(parents=True)
    payload = {
        "schema_version": 1,
        "job_id": request["job_id"],
        "argv": argv,
        "cwd": str(cwd),
        "persist": persist,
        "runner_outputs": runner_outputs,
        "execution_backend": request.get("execution_backend", "unknown"),
        "created_at": now(),
        "source_commit": checked(["git", "-C", cfg["repo_root"], "rev-parse", "HEAD"])
        .stdout.decode()
        .strip(),
        "helpers": cfg.get("helpers", {}),
    }
    atomic_json(directory / "request.json", payload)
    atomic_json(directory / "source_changes.json", source_changes)
    state = {
        "schema_version": 1,
        "job_id": request["job_id"],
        "status": "pending",
        "updated_at": now(),
        "returncode": None,
    }
    atomic_json(directory / "status.json", state)
    return state


def _store(cfg: dict[str, Any], local: Path, remote_fragment: str) -> Any:
    # Reuse the same persistence contract as training; no model-library imports.
    sys.path.insert(0, cfg["repo_root"])
    from src.utils.artifact_store import RcloneArtifactStore

    return RcloneArtifactStore(
        local,
        remote=cfg["drive_remote"],
        remote_root=cfg["drive_root"] + "/" + remote_fragment,
        config_path=Path(cfg["rclone_config"]),
        timeout_seconds=600,
    )


def snapshot_log(source: Path, destination: Path) -> None:
    temporary = destination.with_suffix(".partial")
    with temporary.open("wb") as output:
        if source.exists():
            with source.open("rb") as stream:
                remaining = source.stat().st_size
                while remaining:
                    chunk = stream.read(min(1024 * 1024, remaining))
                    if not chunk:
                        break
                    output.write(chunk)
                    remaining -= len(chunk)
    os.replace(temporary, destination)


def sync_job(
    cfg: dict[str, Any],
    directory: Path,
    *,
    final: bool = False,
    completion: str | None = None,
) -> dict[str, Any]:
    request = read_json(directory / "request.json")
    # The runner owns live checkpoint writes. Salvage its files only after exit.
    fragments = list(request["persist"])
    if completion is not None:
        fragments.extend(request.get("runner_outputs", []))
    for fragment in fragments:
        path = _persist_path(cfg, fragment)
        if not path.exists():
            if final:
                raise ColabError(
                    f"Declared artifact directory was not produced: {fragment}"
                )
            continue
        if not path.is_dir():
            raise ColabError("Declare an artifact directory, not a single file")
        if any(item.is_symlink() for item in path.rglob("*")):
            raise ColabError(f"Artifact directory contains a symbolic link: {fragment}")
        _store(cfg, path, fragment).sync_tree()
    published = directory / "published"
    published.mkdir(exist_ok=True)
    shutil.copyfile(directory / "request.json", published / "request.json")
    shutil.copyfile(
        directory / "source_changes.json", published / "source_changes.json"
    )
    snapshot_log(directory / "stdout.log", published / "stdout.log")
    store = _store(
        cfg, published, f"operations/colab/{cfg['session']}/{directory.name}"
    )
    store.sync_tree()
    # Status is published last, only after the preceding artifacts succeeded.
    published_state = read_json(directory / "status.json")
    if completion is not None:
        published_state.update(
            status=completion, drive_saved_at=now(), finished_at=now()
        )
    atomic_json(published / "status.json", published_state)
    store.publish_file(published / "status.json")
    return published_state


def run_job(job_id: str) -> int:
    cfg = config()
    directory = job_dir(job_id)
    request = read_json(directory / "request.json")
    state = read_json(directory / "status.json")
    if state["status"] != "pending":
        raise ColabError("Only a pending job can be started")
    state.update(
        status="starting",
        worker_pid=os.getpid(),
        worker_token=process_token(os.getpid()),
        updated_at=now(),
    )
    atomic_json(directory / "status.json", state)
    stop = threading.Event()
    cancelled = threading.Event()
    sync_errors: list[str] = []
    sync_lock = threading.Lock()

    def interrupted(signum: int, frame: Any) -> None:
        cancelled.set()

    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(sig, interrupted)

    def persist_periodically() -> None:
        while not stop.wait(cfg.get("sync_interval_seconds", 60)):
            try:
                with sync_lock:
                    sync_job(cfg, directory)
            except Exception as error:
                sync_errors.append(str(error))
                cancelled.set()
                return

    process: subprocess.Popen[bytes] | None = None
    publisher: threading.Thread | None = None
    phase = "initial_save"
    try:
        sync_job(cfg, directory)  # Confirm Drive writes before starting the command.
        if cancelled.is_set() or (directory / "cancel").exists():
            state.update(status="finalizing", returncode=130, updated_at=now())
            atomic_json(directory / "status.json", state)
            saved = sync_job(cfg, directory, completion="cancelled")
            atomic_json(directory / "status.json", saved)
            return 130
        phase = "execute"
        with (directory / "stdout.log").open("ab", buffering=0) as output:
            environment = {
                **os.environ,
                "RCLONE_CONFIG": cfg["rclone_config"],
                "PYTHONUNBUFFERED": "1",
            }
            if request.get("runner_outputs"):
                environment["TENNIS_COLAB_ARTIFACT_ROOTS"] = json.dumps(
                    [
                        f"{cfg['drive_remote']}:{cfg['drive_root']}/{fragment}"
                        for fragment in request["runner_outputs"]
                    ]
                )
            process = subprocess.Popen(
                request["argv"],
                cwd=request["cwd"],
                env=environment,
                stdin=subprocess.DEVNULL,
                stdout=output,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            state.update(
                status="running",
                pid=process.pid,
                process_token=process_token(process.pid),
                started_at=now(),
            )
            atomic_json(directory / "status.json", state)
            publisher = threading.Thread(target=persist_periodically, daemon=True)
            publisher.start()
            while process.poll() is None:
                if cancelled.is_set() or (directory / "cancel").exists():
                    cancelled.set()
                    os.killpg(process.pid, signal.SIGTERM)
                    try:
                        process.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                    break
                state["updated_at"] = now()
                atomic_json(directory / "status.json", state)
                time.sleep(2)
            returncode = process.wait()
        state["returncode"] = returncode
        stop.set()
        if publisher:
            publisher.join(timeout=650)
            if publisher.is_alive():
                raise ColabError("Drive synchronization has not settled")
        if sync_errors:
            phase = "periodic_save"
            raise ColabError(sync_errors[0])
        phase = "final_save"
        state.update(status="finalizing", returncode=returncode, updated_at=now())
        atomic_json(directory / "status.json", state)
        with sync_lock:
            completion = (
                "cancelled"
                if cancelled.is_set()
                else "completed"
                if returncode == 0
                else "failed"
            )
            saved = sync_job(
                cfg, directory, final=returncode == 0, completion=completion
            )
            atomic_json(directory / "status.json", saved)
        return returncode if 0 <= returncode <= 125 else 1
    except Exception as error:
        if process and process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
        state.update(
            status="save_failed" if phase.endswith("save") else "failed",
            error=str(error),
            failure_stage=phase,
            updated_at=now(),
        )
        atomic_json(directory / "status.json", state)
        print(json.dumps({"ok": False, "error": str(error)}), file=sys.stderr)
        return 1
    finally:
        stop.set()


def diff(request: dict[str, Any]) -> dict[str, Any]:
    repo = Path(config()["repo_root"])
    patch = checked(["git", "-C", str(repo), "diff", "--binary", "HEAD"]).stdout
    if len(patch) > 16 * 1024 * 1024:
        raise ColabError("Diff is larger than 16 MiB; inspect the changed files first")
    files: dict[str, str] = {}
    excluded = []
    storage_excluded = []
    exclusion_reasons: dict[str, str] = {}
    if request.get("include_untracked"):
        names = checked(
            ["git", "-C", str(repo), "ls-files", "--others", "--exclude-standard", "-z"]
        ).stdout
        total = 0
        for raw in names.split(b"\0"):
            if not raw:
                continue
            name = raw.decode()
            path = repo / relative(name)
            if PurePosixPath(name).parts[0] in {"data", "ckpt", "outputs", ".cache", ".venv"}:
                excluded.append(name)
                storage_excluded.append(name)
                exclusion_reasons[name] = "project storage; managed through Drive"
                continue
            if path.is_symlink() or _credential_path(name):
                excluded.append(name)
                exclusion_reasons[name] = "symbolic link or credential path"
                continue
            content = path.read_bytes()
            total += len(content)
            if total > 16 * 1024 * 1024:
                raise ColabError(
                    "Untracked files exceed 16 MiB; retrieve selected files instead"
                )
            files[name] = base64.b64encode(content).decode()
    return {
        "commit": checked(["git", "-C", str(repo), "rev-parse", "HEAD"])
        .stdout.decode()
        .strip(),
        "patch": base64.b64encode(patch).decode(),
        "untracked": files,
        "excluded": excluded,
        "storage_excluded": storage_excluded,
        "exclusion_reasons": exclusion_reasons,
    }


def _credential_path(name: str) -> bool:
    """Exclude credential filenames without dropping code such as tokenizer.py."""
    for part in PurePosixPath(name.lower()).parts:
        if part == ".env" or part.startswith(".env."):
            return True
        if part in {".secrets", ".ssh", "secrets", "credentials", "id_rsa", "id_ed25519"}:
            return True
        if part.endswith((".pem", ".key", ".p12", ".pfx")):
            return True
        if not part.endswith((".py", ".sh", ".md")) and re.search(
            r"(^|[_.-])(credentials?|secrets?|tokens?)([_.-]|$)", part
        ):
            return True
    return False


def rpc(request: dict[str, Any]) -> dict[str, Any]:
    action = request["action"]
    if action == "bootstrap":
        return bootstrap(request)
    if action == "prepare":
        return prepare(request)
    if action == "status":
        cfg = config()
        result = jobs()
        if request.get("job_id"):
            result = [item for item in result if item["job_id"] == request["job_id"]]
            if not result:
                raise ColabError("Unknown job")
        return {
            "session": cfg["session"],
            "jobs": result,
            "repo_root": cfg["repo_root"],
        }
    if action == "logs":
        return {
            "text": tail(
                job_dir(request["job_id"]) / "stdout.log",
                lines=min(int(request.get("lines", 80)), 500),
            )
        }
    if action == "cancel":
        directory = job_dir(request["job_id"])
        state = read_json(directory / "status.json")
        requested = state["status"] not in TERMINAL
        if requested:
            (directory / "cancel").touch()
        if state["status"] == "pending":
            state.update(status="finalizing", returncode=130, updated_at=now())
            atomic_json(directory / "status.json", state)
            try:
                saved = sync_job(config(), directory, completion="cancelled")
                atomic_json(directory / "status.json", saved)
            except Exception as error:
                state.update(status="save_failed", error=str(error))
                atomic_json(directory / "status.json", state)
                raise
        return {"job_id": request["job_id"], "requested": requested}
    if action == "save":
        directory = job_dir(request["job_id"])
        state = read_json(directory / "status.json")
        if state["status"] not in TERMINAL:
            raise ColabError(
                "Cancel the active command and wait for its worker before explicit recovery save"
            )
        completion = state["status"]
        if completion == "save_failed":
            completion = "completed" if state.get("returncode") == 0 else "failed"
        state.update(status="finalizing", updated_at=now())
        atomic_json(directory / "status.json", state)
        try:
            saved = sync_job(
                config(),
                directory,
                final=state.get("returncode") == 0,
                completion=completion,
            )
            atomic_json(directory / "status.json", saved)
            return saved
        except Exception as error:
            state.update(status="save_failed", error=str(error))
            atomic_json(directory / "status.json", state)
            raise
    if action == "read_file":
        path = safe_file(request["path"])
        if path.stat().st_size > request["max_bytes"]:
            raise ColabError("File exceeds the small-file limit; use Drive transfer")
        content = path.read_bytes()
        return {
            "data": base64.b64encode(content).decode(),
            "sha256": hashlib.sha256(content).hexdigest(),
        }
    if action == "diff":
        return diff(request)
    raise ColabError(f"Unknown remote action: {action}")


def main() -> int:
    if len(sys.argv) == 3 and sys.argv[1] == "run":
        return run_job(sys.argv[2])
    try:
        response = rpc(json.load(sys.stdin))
        print(json.dumps({"ok": True, **response}, ensure_ascii=False))
        return 0
    except Exception as error:
        print(json.dumps({"ok": False, "error": str(error)}, ensure_ascii=False))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
