"""End-to-end contracts for the terminal-to-Colab workflow interface.

The command tests use a stateful fake ``colab`` executable.  They deliberately
exercise the public process boundary without requiring OAuth, a network, or a
GPU.  Small remote-runner tests cover the security-sensitive filesystem and
``rclone`` boundaries that cannot execute on the local host as real Colab code.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import shutil
import signal
import subprocess
import sys
import tarfile
import time
from pathlib import Path
from typing import Any

import pytest

from scripts.colab.workflow import remote_runner
from scripts.colab.workflow.common import WorkflowError
from scripts.colab.workflow.jobs import load_job
from scripts.colab.workflow.snapshot import create_snapshot

ROOT = Path(__file__).parents[3]
CLI_MODULE = "scripts.colab.workflow.cli"
RUN_ID = "test-run-0001"
BUILTIN_ACCELERATORS = {
    "ball_detection": "gpu",
    "ball_detection_staged": "gpu",
    "blcs": "gpu",
    "blcs_generate_dataset": "cpu",
    "blcs_tracking": "gpu",
    "court_detection": "gpu",
    "court_detection_materialize_targets": "cpu",
    "court_detection_mixed": "gpu",
    "plcs": "gpu",
    "plcs_generate_dataset": "cpu",
    "plcs_tracking": "gpu",
    "slcs": "gpu",
    "slcs_make_splits": "cpu",
    "slcs_precompute_dino_tokens": "gpu",
    "submodules_demo_gvhmr": "gpu",
    "synthetic_data_generation": "gpu",
    "tennis_scene": "gpu",
    "tennis_scene_generate_dataset": "gpu",
}


@pytest.fixture
def fake_colab(tmp_path: Path) -> dict[str, Path | dict[str, str]]:
    """Install a fake Colab CLI with remote storage and an invocation log."""

    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    remote_dir = tmp_path / "fake-remote"
    remote_dir.mkdir()
    log_path = tmp_path / "colab-invocations.jsonl"
    marker_path = tmp_path / "run-started"
    executable = bin_dir / "colab"
    executable.write_text(
        r"""#!/usr/bin/env python3
import ast
import hashlib
import json
import os
import re
import shutil
import sys
import time
from pathlib import Path

args = sys.argv[1:]
log_path = Path(os.environ["FAKE_COLAB_LOG"])
with log_path.open("a", encoding="utf-8") as stream:
    stream.write(json.dumps(args) + "\n")

remote_root = Path(os.environ["FAKE_COLAB_REMOTE"])

def remote_path(value):
    return remote_root / value.lstrip("/")

def fail(message, code=2):
    print(message, file=sys.stderr)
    raise SystemExit(code)

def exact_options(values, required, optional=()):
    parsed = {}
    index = 0
    allowed = set(required) | set(optional)
    while index < len(values):
        option = values[index]
        if option not in allowed:
            fail(f"unknown option: {option}")
        if option in parsed:
            fail(f"duplicate option: {option}")
        if index + 1 >= len(values):
            fail(f"missing value for {option}")
        parsed[option] = values[index + 1]
        index += 2
    missing = set(required) - set(parsed)
    if missing:
        fail(f"missing options: {sorted(missing)}")
    return parsed

def assignment_value(source, name):
    match = re.search(rf"(?m)^{name} = (.+)$", source)
    if match is None:
        fail(f"bootstrap is missing {name}", 92)
    return ast.literal_eval(match.group(1))

def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
        + b"\n"
    )

def source_provenance(request):
    source = request["source"]
    result = {
        "mode": source["mode"],
        "repo_sha": source["repo_sha"],
        "repo_tree": source["repo_tree"],
        "tracked_clean": source["tracked_clean"],
    }
    if source["mode"] == "snapshot":
        result.update(
            {
                "snapshot_digest": source["content_digest"],
                "snapshot_archive_sha256": source["archive_sha256"],
                "snapshot_git_commit": "a" * 40,
                "submodules": source["submodules"],
            }
        )
    return result

def completed_status(request, artifact, attempt):
    requested_gpu = request["runtime"]["gpu"]
    return {
        "schema_version": 1,
        "run_id": request["run_id"],
        "session": request["session"],
        "request_digest": request["request_digest"],
        "state": "completed",
        "step": "completed",
        "attempt": attempt,
        "updated_at": "2026-09-04T00:00:00+00:00",
        "error": None,
        "provenance": {
            "job_definition_digest": request["job"]["definition_digest"],
            "source": source_provenance(request),
            "resolved_argv": request["job"]["argv"],
            "input_manifest": [
                {**mapping, "entries": []} for mapping in request["job"]["inputs"]
            ],
            "output_manifest": [
                {"declared_path": path, "entries": []}
                for path in request["job"]["outputs"]
            ],
            "runtime": {
                "collected": True,
                "python": {"available": True},
                "platform": {"available": True},
                "gpu": {
                    "required": request["job"]["accelerator"] == "gpu",
                    "requested": requested_gpu,
                    "available": requested_gpu is not None,
                    "nvidia_smi": {},
                },
                "cuda": {"available": requested_gpu is not None},
                "torch": {"available": True},
                "uv": {"available": True},
            },
        },
        "artifacts": {
            "archive_path": (
                f"/content/tennis-lab-runs/{request['run_id']}"
                "/bundle/artifacts.tar.gz"
            ),
            "archive_sha256": hashlib.sha256(artifact).hexdigest(),
            "archive_size": len(artifact),
        },
        "drive": {
            "mode": request["drive"]["mode"],
            "final": (
                f"{request['drive']['root']}/colab-runs/{request['run_id']}"
            ),
        },
    }

def publish_rclone(request, workspace, status):
    root = (
        Path(os.environ["FAKE_RCLONE_REMOTE"])
        / request["drive"]["remote"]
        / request["drive"]["root"]
        / "colab-runs"
        / request["run_id"]
    )
    root.mkdir(parents=True, exist_ok=True)
    shutil.copy2(workspace / "request.json", root / "request.json")
    shutil.copy2(workspace / "bundle/artifacts.tar.gz", root / "artifacts.tar.gz")
    published_status = json.loads(json.dumps(status))
    if os.environ.get("FAKE_PUBLISHED_STATUS_MODE") == "running":
        published_status["state"] = "running"
        published_status["step"] = "publishing"
    write_json(root / "status.json", published_status)
    manifest = []
    for name in ("artifacts.tar.gz", "request.json", "status.json"):
        path = root / name
        manifest.append(
            {
                "path": name,
                "size": path.stat().st_size,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
    write_json(root / "manifest.json", manifest)

if args[:1] != ["--config"] or len(args) < 3:
    fail("usage: colab --config CONFIG COMMAND", 91)
operation = args[2]
operation_args = args[3:]

if operation == "version":
    if operation_args:
        fail(f"unknown option: {operation_args[0]}")
    print("Version: 0.6.0")
    raise SystemExit(0)

if operation == "exec" and operation_args == ["--help"]:
    print("Usage: colab exec -s SESSION -f FILE --timeout SECONDS")
    print("  -s, --session  -f, --file  --timeout")
    raise SystemExit(0)

if operation == "new" and operation_args == ["--help"]:
    print("Usage: colab new -s SESSION [--gpu GPU]")
    print("  -s, --session  --gpu")
    raise SystemExit(0)

if operation == "new":
    exact_options(operation_args, {"-s"}, {"--gpu"})
    raise SystemExit(0)

if operation == "upload":
    if len(operation_args) != 4 or operation_args[0] != "-s":
        fail("usage: colab upload -s SESSION SOURCE DESTINATION")
    source = Path(operation_args[2])
    destination = remote_path(operation_args[3])
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    if destination.name == "rclone.conf":
        Path(os.environ["FAKE_SECRET_MODE_LOG"]).write_text(
            oct(destination.stat().st_mode & 0o777), encoding="utf-8"
        )
    raise SystemExit(0)

if operation == "download":
    if len(operation_args) != 4 or operation_args[0] != "-s":
        fail("usage: colab download -s SESSION SOURCE DESTINATION")
    source = remote_path(operation_args[2])
    destination = Path(operation_args[3])
    if not source.is_file():
        raise SystemExit(7)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    raise SystemExit(0)

if operation == "drivemount":
    if (
        len(operation_args) != 3
        or operation_args[0] != "-s"
        or operation_args[2] != "/content/drive"
    ):
        fail("usage: colab drivemount -s SESSION /content/drive")
    raise SystemExit(0)

if operation == "status":
    if len(operation_args) != 2 or operation_args[0] != "-s":
        fail("usage: colab status -s SESSION")
    returncode = int(os.environ.get("FAKE_SESSION_STATUS_RC", "0"))
    print("RUNNING" if returncode == 0 else "session not found")
    raise SystemExit(returncode)

if operation == "stop":
    if len(operation_args) != 2 or operation_args[0] != "-s":
        fail("usage: colab stop -s SESSION")
    raise SystemExit(int(os.environ.get("FAKE_STOP_RC", "0")))

if operation != "exec":
    fail(f"unknown command: {operation}")

options = exact_options(operation_args, {"-s", "-f", "--timeout"})
if not options["--timeout"].isdigit():
    fail("--timeout must be an integer")
bootstrap = Path(options["-f"]).read_text(encoding="utf-8")
action = assignment_value(bootstrap, "TENNIS_COLAB_ACTION")
run_id = assignment_value(bootstrap, "TENNIS_COLAB_RUN_ID")
rclone_config_remote = assignment_value(
    bootstrap, "TENNIS_COLAB_RCLONE_CONFIG"
)
workspace = remote_path(f"/content/tennis-lab-runs/{run_id}")
workspace.mkdir(parents=True, exist_ok=True)
if action == "prepare":
    (workspace / ".secrets").mkdir(exist_ok=True)
    raise SystemExit(0)
if action == "cleanup-secret":
    if rclone_config_remote:
        remote_path(rclone_config_remote).unlink(missing_ok=True)
    raise SystemExit(int(os.environ.get("FAKE_CLEANUP_RC", "0")))
if action != "run":
    raise SystemExit(92)

mode = os.environ.get("FAKE_STATUS_MODE", "completed")
if mode == "interrupt":
    Path(os.environ["FAKE_RUN_MARKER"]).write_text("started", encoding="utf-8")
    time.sleep(30)

request = json.loads((workspace / "request.json").read_text(encoding="utf-8"))
status_path = workspace / "status.json"
if mode == "missing":
    status_path.unlink(missing_ok=True)
elif mode == "corrupt":
    status_path.write_text("{not-json", encoding="utf-8")
else:
    previous_attempt = 0
    if status_path.is_file():
        try:
            previous_attempt = int(json.loads(status_path.read_text())["attempt"])
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            pass
    artifact = b"fake verified artifact\n"
    artifact_path = workspace / "bundle/artifacts.tar.gz"
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.write_bytes(artifact)
    state = "failed" if mode == "failed" else "completed"
    status = completed_status(request, artifact, previous_attempt + 1)
    if state == "failed":
        status["state"] = "failed"
        status["step"] = "running_job"
        status["error"] = {
            "type": "RuntimeError",
            "message": "remote failure",
            "traceback": "",
        }
        status["artifacts"] = {}
    if mode == "digest-mismatch":
        status["request_digest"] = "0" * 64
    elif mode == "run-id-mismatch":
        status["run_id"] = "attacker-run"
    elif mode == "path-run-id":
        status["run_id"] = "../../escape"
    elif mode == "session-mismatch":
        status["session"] = "attacker-session"
    write_json(status_path, status)
    if state == "completed" and request["drive"]["mode"] == "rclone":
        publish_rclone(request, workspace, status)
raise SystemExit(int(os.environ.get("FAKE_EXEC_RC", "0")))
""",
        encoding="utf-8",
    )
    executable.chmod(0o755)

    rclone_log = tmp_path / "rclone-invocations.jsonl"
    rclone_remote = tmp_path / "fake-rclone"
    rclone_remote.mkdir()
    rclone = bin_dir / "rclone"
    rclone.write_text(
        r"""#!/usr/bin/env python3
import json
import os
import shutil
import sys
from pathlib import Path

args = sys.argv[1:]
with Path(os.environ["FAKE_RCLONE_LOG"]).open("a", encoding="utf-8") as stream:
    stream.write(json.dumps(args) + "\n")
if len(args) != 5 or args[0] != "--config" or args[2] != "copyto":
    print("unsupported fake rclone invocation", file=sys.stderr)
    raise SystemExit(2)
source = args[3]
if ":" not in source:
    print("source must be remote", file=sys.stderr)
    raise SystemExit(2)
remote, relative = source.split(":", 1)
source_path = Path(os.environ["FAKE_RCLONE_REMOTE"]) / remote / relative.lstrip("/")
destination = Path(args[4])
if not source_path.is_file():
    print(f"remote file does not exist: {source}", file=sys.stderr)
    raise SystemExit(7)
destination.parent.mkdir(parents=True, exist_ok=True)
shutil.copy2(source_path, destination)
""",
        encoding="utf-8",
    )
    rclone.chmod(0o755)
    secret_mode_log = tmp_path / "rclone-secret-mode"
    environment = {
        **os.environ,
        "PATH": f"{bin_dir}:{os.environ.get('PATH', '')}",
        "FAKE_COLAB_LOG": str(log_path),
        "FAKE_COLAB_REMOTE": str(remote_dir),
        "FAKE_RUN_MARKER": str(marker_path),
        "FAKE_RCLONE_LOG": str(rclone_log),
        "FAKE_RCLONE_REMOTE": str(rclone_remote),
        "FAKE_SECRET_MODE_LOG": str(secret_mode_log),
    }
    return {
        "bin": bin_dir,
        "remote": remote_dir,
        "rclone_remote": rclone_remote,
        "log": log_path,
        "rclone_log": rclone_log,
        "secret_mode_log": secret_mode_log,
        "marker": marker_path,
        "env": environment,
    }


def _run_cli(
    tmp_path: Path,
    *arguments: str,
    env: dict[str, str] | None = None,
    repo_root: Path | None = None,
) -> subprocess.CompletedProcess[str]:
    source_repo = repo_root or _clean_source_repo(tmp_path)
    state_dir = tmp_path / "state"
    return subprocess.run(
        [
            sys.executable,
            "-m",
            CLI_MODULE,
            "--repo-root",
            str(source_repo),
            "--state-dir",
            str(state_dir),
            "--jobs-dir",
            str(ROOT / "scripts/colab/workflows/jobs"),
            *arguments,
        ],
        cwd=ROOT,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )


def _clean_source_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "source-repo"
    if repo.exists():
        return repo
    repo.mkdir()
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    subprocess.run(
        ["git", "-C", str(repo), "config", "user.email", "test@example.com"],
        check=True,
    )
    subprocess.run(["git", "-C", str(repo), "config", "user.name", "Test"], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(repo),
            "remote",
            "add",
            "origin",
            "https://github.com/example/tennis-lab.git",
        ],
        check=True,
    )
    (repo / "pyproject.toml").write_text(
        "[project]\nname='fixture'\n", encoding="utf-8"
    )
    (repo / "tracked.txt").write_text("clean\n", encoding="utf-8")
    runner = repo / "scripts/colab/workflow/remote_runner.py"
    runner.parent.mkdir(parents=True)
    shutil.copy2(ROOT / "scripts/colab/workflow/remote_runner.py", runner)
    subprocess.run(["git", "-C", str(repo), "add", "."], check=True)
    subprocess.run(["git", "-C", str(repo), "commit", "-qm", "initial"], check=True)
    return repo


def _invocations(fake_colab: dict[str, Any]) -> list[list[str]]:
    path = Path(fake_colab["log"])
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines()]


def _rclone_invocations(fake_colab: dict[str, Any]) -> list[list[str]]:
    path = Path(fake_colab["rclone_log"])
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines()]


def _operation(call: list[str]) -> str:
    offset = 1 if call[0] == "colab" else 0
    assert call[offset] == "--config"
    return call[offset + 2]


def _action(call: list[str]) -> str | None:
    if _operation(call) == "exec" and "-f" in call:
        return Path(call[call.index("-f") + 1]).stem
    return None


def _rclone_config(tmp_path: Path) -> Path:
    path = tmp_path / "rclone.conf"
    path.write_text(
        "[gdrive]\ntype = drive\ntoken = ultra-secret-refresh-token\n",
        encoding="utf-8",
    )
    path.chmod(0o600)
    return path


def _valid_job(name: str = "sample") -> str:
    return f'''schema_version = 1
name = "{name}"
description = "Test job"
accelerator = "gpu"
timeout_seconds = 120
setup = ["base"]
protected_override_keys = ["paths.output_root", "run.gpus"]
outputs = ["outputs/model"]

[command]
module = "package.train"
args = ["paths.output_root=outputs"]

[[inputs]]
source = "data/input"
destination = "data/input"
writable = false
'''


def test_jobs_lists_every_validated_builtin_manifest(tmp_path: Path) -> None:
    result = _run_cli(tmp_path, "jobs", "--json")

    assert result.returncode == 0, result.stderr
    jobs = json.loads(result.stdout)
    catalog_names = sorted(path.stem for path in (ROOT / "scripts/colab/workflows/jobs").glob("*.toml"))
    assert [job["name"] for job in jobs] == catalog_names
    accelerators = {job["name"]: job["accelerator"] for job in jobs}
    assert {name: accelerators[name] for name in BUILTIN_ACCELERATORS} == BUILTIN_ACCELERATORS
    assert all(
        set(job)
        == {
            "name",
            "description",
            "accelerator",
            "timeout_seconds",
            "setup",
        }
        for job in jobs
    )
    assert all(job["setup"][0] == "base" for job in jobs)


@pytest.mark.parametrize(
    ("job_name", "accelerator"), tuple(BUILTIN_ACCELERATORS.items())
)
def test_every_builtin_manifest_loads_and_builds_a_strict_dry_run(
    tmp_path: Path,
    job_name: str,
    accelerator: str,
) -> None:
    manifest = ROOT / "scripts/colab/workflows/jobs" / f"{job_name}.toml"
    job = load_job(manifest)

    assert job.name == job_name
    assert job.accelerator == accelerator
    assert job.protected_override_keys
    assert all(isinstance(mapping.writable, bool) for mapping in job.inputs)

    result = _run_cli(
        tmp_path,
        "run",
        job_name,
        "--run-id",
        RUN_ID,
        "--dry-run",
    )

    assert result.returncode == 0, result.stderr
    plan = json.loads(result.stdout)
    request = plan["request"]
    assert request["job"]["name"] == job_name
    assert request["job"]["accelerator"] == accelerator
    assert request["job"]["protected_override_keys"] == list(
        job.protected_override_keys
    )
    assert request["job"]["inputs"] == [
        {
            "source": mapping.source,
            "destination": mapping.destination,
            "writable": mapping.writable,
        }
        for mapping in job.inputs
    ]
    new_command = plan["commands"][0]
    assert _operation(new_command) == "new"
    if accelerator == "gpu":
        assert new_command[-2:] == ["--gpu", "T4"]
        assert request["runtime"] == {
            "accelerator": "gpu",
            "gpu": "T4",
            "high_mem": False,
        }
    else:
        assert "--gpu" not in new_command
        assert request["runtime"] == {
            "accelerator": "cpu",
            "gpu": None,
            "high_mem": False,
        }


@pytest.mark.parametrize(
    ("old", "new"),
    (
        (
            'description = "Test job"',
            'description = "Test job"\nunknown_contract_key = true',
        ),
        ("schema_version = 1", "schema_version = 99"),
        ('setup = ["base"]', 'setup = ["cuda_ops"]'),
        ('source = "data/input"', 'source = "../outside"'),
    ),
)
def test_job_manifest_rejects_unknown_or_unsafe_contracts(
    tmp_path: Path,
    old: str,
    new: str,
) -> None:
    path = tmp_path / "sample.toml"
    path.write_text(_valid_job().replace(old, new), encoding="utf-8")

    with pytest.raises(WorkflowError):
        load_job(path)


@pytest.mark.parametrize(
    ("writable", "output", "accepted"),
    (
        (False, "outputs/model", True),
        (False, "data/input", False),
        (True, "outputs/model", False),
        (True, "data", False),
        (True, "data/input", True),
    ),
)
def test_job_manifest_allows_input_output_overlap_only_for_explicit_exact_writable(
    tmp_path: Path,
    writable: bool,
    output: str,
    accepted: bool,
) -> None:
    path = tmp_path / "sample.toml"
    contents = _valid_job().replace(
        'outputs = ["outputs/model"]', f'outputs = ["{output}"]'
    )
    contents = contents.replace(
        "writable = false", f"writable = {str(writable).lower()}"
    )
    path.write_text(contents, encoding="utf-8")

    if accepted:
        job = load_job(path)
        assert job.inputs[0].writable is writable
        assert job.outputs == (output,)
    else:
        with pytest.raises(WorkflowError):
            load_job(path)


@pytest.mark.parametrize(
    "arguments",
    (
        ("run", "does-not-exist", "--run-id", RUN_ID),
        ("run", "court_detection", "--run-id", RUN_ID, "--unknown-option"),
    ),
)
def test_unknown_job_or_option_fails_before_session_creation(
    tmp_path: Path,
    fake_colab: dict[str, Any],
    arguments: tuple[str, ...],
) -> None:
    result = _run_cli(tmp_path, *arguments, env=fake_colab["env"])

    assert result.returncode in {2, 3}
    assert _invocations(fake_colab) == []


def test_missing_colab_points_to_pinned_setup_before_session_creation(
    tmp_path: Path,
) -> None:
    empty_bin = tmp_path / "bin"
    empty_bin.mkdir()
    result = _run_cli(
        tmp_path,
        "run",
        "court_detection",
        "--run-id",
        RUN_ID,
        "--drive-mode",
        "mount",
        env={**os.environ, "PATH": str(empty_bin)},
    )

    assert result.returncode == 3
    assert "colab executable not found" in result.stderr
    assert "scripts/colab/README.md" in result.stderr
    assert "uv tool install google-colab-cli" not in result.stderr
    assert not (tmp_path / "state").exists()


@pytest.mark.parametrize("source", ("git", "snapshot"))
def test_dry_run_has_no_state_or_colab_side_effects(
    tmp_path: Path,
    fake_colab: dict[str, Any],
    source: str,
) -> None:
    result = _run_cli(
        tmp_path,
        "run",
        "court_detection",
        "--run-id",
        RUN_ID,
        "--source",
        source,
        "--dry-run",
        env=fake_colab["env"],
    )

    assert result.returncode == 0, result.stderr
    plan = json.loads(result.stdout)
    assert plan["request"]["source"]["mode"] == source
    assert not (tmp_path / "state").exists()
    assert _invocations(fake_colab) == []


@pytest.mark.parametrize("dirty_kind", ("modified", "deleted", "staged", "untracked"))
@pytest.mark.parametrize("dry_run", (False, True), ids=("execute", "dry-run"))
def test_git_source_rejects_local_changes_before_colab_access(
    tmp_path: Path,
    fake_colab: dict[str, Any],
    dirty_kind: str,
    dry_run: bool,
) -> None:
    repo = _clean_source_repo(tmp_path)
    tracked = repo / "tracked.txt"
    if dirty_kind == "modified":
        tracked.write_text("modified\n", encoding="utf-8")
    elif dirty_kind == "deleted":
        tracked.unlink()
    elif dirty_kind == "staged":
        tracked.write_text("staged\n", encoding="utf-8")
        subprocess.run(["git", "-C", str(repo), "add", "tracked.txt"], check=True)
    else:
        (repo / "untracked.txt").write_text("untracked\n", encoding="utf-8")

    arguments = [
        "run",
        "court_detection",
        "--run-id",
        RUN_ID,
        "--source",
        "git",
        "--drive-mode",
        "mount",
    ]
    if dry_run:
        arguments.append("--dry-run")
    result = _run_cli(
        tmp_path,
        *arguments,
        env=fake_colab["env"],
        repo_root=repo,
    )

    assert result.returncode == 3
    assert "working-tree changes" in result.stderr
    assert "--source snapshot" in result.stderr
    assert not (tmp_path / "state").exists()
    assert _invocations(fake_colab) == []


def test_snapshot_source_captures_local_changes_with_truthful_provenance(
    tmp_path: Path, fake_colab: dict[str, Any]
) -> None:
    repo = _clean_source_repo(tmp_path)
    (repo / "tracked.txt").write_text("modified\n", encoding="utf-8")
    (repo / "staged.txt").write_text("before\n", encoding="utf-8")
    (repo / "deleted.txt").write_text("delete me\n", encoding="utf-8")
    subprocess.run(
        ["git", "-C", str(repo), "add", "staged.txt", "deleted.txt"], check=True
    )
    subprocess.run(["git", "-C", str(repo), "commit", "-qm", "add files"], check=True)
    (repo / "staged.txt").write_text("staged\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(repo), "add", "staged.txt"], check=True)
    (repo / "deleted.txt").unlink()
    (repo / "untracked.txt").write_text("untracked\n", encoding="utf-8")

    result = _run_cli(
        tmp_path,
        "run",
        "court_detection",
        "--run-id",
        RUN_ID,
        "--source",
        "snapshot",
        "--dry-run",
        env=fake_colab["env"],
        repo_root=repo,
    )

    assert result.returncode == 0, result.stderr
    source = json.loads(result.stdout)["request"]["source"]
    assert source["tracked_clean"] is False
    assert source["working_tree_dirty"] is True
    assert {"tracked.txt", "staged.txt", "untracked.txt"} <= set(
        source["included_paths"]
    )
    assert source["deleted_paths"] == ["deleted.txt"]
    assert not (tmp_path / "state").exists()
    assert _invocations(fake_colab) == []


def test_dry_run_preserves_hydra_override_argv_boundaries(tmp_path: Path) -> None:
    overrides = [
        "trainer.note=value with spaces",
        "++model.layers=[64,128]",
        "paths.literal=$HOME/not-expanded",
    ]
    result = _run_cli(
        tmp_path,
        "run",
        "court_detection",
        "--run-id",
        RUN_ID,
        "--dry-run",
        "--",
        *overrides,
    )

    assert result.returncode == 0, result.stderr
    plan = json.loads(result.stdout)
    argv = plan["request"]["job"]["argv"]
    assert argv[-len(overrides) :] == overrides
    assert "-m" in argv
    assert argv[argv.index("-m") + 1] == "src.tasks.court_detection.scripts.train"


@pytest.mark.parametrize(
    "override",
    (
        "paths.output_root=attacker-output",
        "paths/output_root=attacker-output",
        "+data/source=real",
        "~run.gpus",
        "trainer.devices=0",
        "--config-name=attacker",
    ),
)
def test_managed_dot_and_slash_hydra_overrides_fail_before_colab_access(
    tmp_path: Path,
    fake_colab: dict[str, Any],
    override: str,
) -> None:
    result = _run_cli(
        tmp_path,
        "run",
        "court_detection",
        "--run-id",
        RUN_ID,
        "--drive-mode",
        "mount",
        "--",
        override,
        env=fake_colab["env"],
    )

    assert result.returncode == 3
    assert "managed" in result.stderr
    assert _invocations(fake_colab) == []


def test_cpu_auto_uses_no_gpu_provisioning_flag(tmp_path: Path) -> None:
    result = _run_cli(
        tmp_path,
        "run",
        "blcs_generate_dataset",
        "--run-id",
        RUN_ID,
        "--dry-run",
    )

    assert result.returncode == 0, result.stderr
    plan = json.loads(result.stdout)
    assert plan["request"]["job"]["accelerator"] == "cpu"
    assert plan["request"]["runtime"] == {
        "accelerator": "cpu",
        "gpu": None,
        "high_mem": False,
    }
    assert "--gpu" not in plan["commands"][0]


def test_cpu_job_can_explicitly_request_gpu_with_consistent_runtime_provenance(
    tmp_path: Path, fake_colab: dict[str, Any]
) -> None:
    result = _run_cli(
        tmp_path,
        "run",
        "blcs_generate_dataset",
        "--run-id",
        RUN_ID,
        "--gpu",
        "L4",
        "--drive-mode",
        "mount",
        env=fake_colab["env"],
    )

    assert result.returncode == 0, result.stderr
    status = json.loads(result.stdout)
    assert status["provenance"]["runtime"]["gpu"] == {
        "required": False,
        "requested": "L4",
        "available": True,
        "nvidia_smi": {},
    }
    new = next(
        call
        for call in _invocations(fake_colab)
        if _operation(call) == "new" and call[-1] != "--help"
    )
    assert new[-2:] == ["--gpu", "L4"]


def test_gpu_job_rejects_cpu_runtime_before_colab_access(
    tmp_path: Path, fake_colab: dict[str, Any]
) -> None:
    result = _run_cli(
        tmp_path,
        "run",
        "court_detection",
        "--run-id",
        RUN_ID,
        "--gpu",
        "cpu",
        "--drive-mode",
        "mount",
        env=fake_colab["env"],
    )

    assert result.returncode == 3
    assert "requires a GPU" in result.stderr
    assert _invocations(fake_colab) == []


def test_unsupported_high_memory_request_fails_capability_preflight_before_new(
    tmp_path: Path, fake_colab: dict[str, Any]
) -> None:
    result = _run_cli(
        tmp_path,
        "run",
        "blcs_generate_dataset",
        "--run-id",
        RUN_ID,
        "--drive-mode",
        "mount",
        "--high-mem",
        env=fake_colab["env"],
    )

    assert result.returncode == 3, result.stderr
    calls = _invocations(fake_colab)
    assert [_operation(call) for call in calls] == ["version", "exec", "new"]
    assert calls[-1][-1] == "--help"
    assert not any(_operation(call) == "new" and call[-1] != "--help" for call in calls)


@pytest.mark.parametrize("drive_mode", ("mount", "rclone"))
def test_dry_run_constructs_drive_and_secret_safe_commands(
    tmp_path: Path,
    drive_mode: str,
) -> None:
    result = _run_cli(
        tmp_path,
        "run",
        "court_detection",
        "--run-id",
        RUN_ID,
        "--drive-mode",
        drive_mode,
        "--drive-root",
        "team/training",
        "--dry-run",
    )

    assert result.returncode == 0, result.stderr
    plan = json.loads(result.stdout)
    commands = plan["commands"]
    if drive_mode == "mount":
        assert any(
            command[-4:]
            == ["drivemount", "-s", f"tennis-{RUN_ID[-13:]}", "/content/drive"]
            for command in commands
        )
        assert all(
            "TENNIS_COLAB_RCLONE_CONFIG" not in " ".join(command)
            for command in commands
        )
    else:
        uploads = [command for command in commands if "upload" in command]
        assert any("<rclone-config>" in command for command in uploads)
        assert "ultra-secret" not in result.stdout


def test_official_colab_cli_uses_versioned_file_exec_without_env_options(
    tmp_path: Path, fake_colab: dict[str, Any]
) -> None:
    result = _run_cli(
        tmp_path,
        "run",
        "court_detection",
        "--run-id",
        RUN_ID,
        "--drive-mode",
        "mount",
        env=fake_colab["env"],
    )

    assert result.returncode == 0, result.stderr
    calls = _invocations(fake_colab)
    assert [_operation(call) for call in calls[:3]] == ["version", "exec", "new"]
    assert calls[0][-1] == "version"
    assert calls[1][-1] == calls[2][-1] == "--help"
    exec_calls = [
        call for call in calls if _operation(call) == "exec" and call[-1] != "--help"
    ]
    assert {_action(call) for call in exec_calls} == {"prepare", "run"}
    assert all("-f" in call and "--timeout" in call for call in exec_calls)
    assert all("--env" not in call for call in exec_calls)


def test_fake_colab_matches_official_0_6_help_and_rejects_unknown_options(
    tmp_path: Path, fake_colab: dict[str, Any]
) -> None:
    prefix = [
        str(Path(fake_colab["bin"]) / "colab"),
        "--config",
        str(tmp_path / "sessions.json"),
    ]
    version = subprocess.run(
        [*prefix, "version"],
        check=False,
        capture_output=True,
        text=True,
        env=fake_colab["env"],
    )
    exec_help = subprocess.run(
        [*prefix, "exec", "--help"],
        check=False,
        capture_output=True,
        text=True,
        env=fake_colab["env"],
    )
    unsupported_env = subprocess.run(
        [
            *prefix,
            "exec",
            "-s",
            "session",
            "-f",
            str(tmp_path / "bootstrap.py"),
            "--timeout",
            "60",
            "--env",
            "KEY=value",
        ],
        check=False,
        capture_output=True,
        text=True,
        env=fake_colab["env"],
    )
    unknown_new = subprocess.run(
        [*prefix, "new", "-s", "session", "--unknown"],
        check=False,
        capture_output=True,
        text=True,
        env=fake_colab["env"],
    )

    assert version.returncode == 0
    assert version.stdout.strip() == "Version: 0.6.0"
    assert exec_help.returncode == 0
    assert all(flag in exec_help.stdout for flag in ("-s", "-f", "--timeout"))
    assert unsupported_env.returncode == 2
    assert "unknown option: --env" in unsupported_env.stderr
    assert unknown_new.returncode == 2
    assert "unknown option: --unknown" in unknown_new.stderr


def test_mount_mode_never_uploads_or_logs_an_rclone_secret(
    tmp_path: Path, fake_colab: dict[str, Any]
) -> None:
    config = _rclone_config(tmp_path)
    result = _run_cli(
        tmp_path,
        "run",
        "court_detection",
        "--run-id",
        RUN_ID,
        "--drive-mode",
        "mount",
        "--rclone-config",
        str(config),
        env=fake_colab["env"],
    )

    assert result.returncode == 0, result.stderr
    calls = _invocations(fake_colab)
    uploads = [call for call in calls if _operation(call) == "upload"]
    assert uploads
    assert all("rclone.conf" not in " ".join(call) for call in uploads)
    logged = Path(fake_colab["log"]).read_text(encoding="utf-8")
    assert "ultra-secret-refresh-token" not in logged
    assert not Path(fake_colab["secret_mode_log"]).exists()
    assert _rclone_invocations(fake_colab) == []


def test_snapshot_records_modified_untracked_deleted_and_excluded_secrets(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    subprocess.run(
        ["git", "-C", str(repo), "config", "user.email", "test@example.com"],
        check=True,
    )
    subprocess.run(["git", "-C", str(repo), "config", "user.name", "Test"], check=True)
    (repo / "modified.txt").write_text("before\n", encoding="utf-8")
    (repo / "deleted.txt").write_text("delete me\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(repo), "add", "."], check=True)
    subprocess.run(["git", "-C", str(repo), "commit", "-qm", "initial"], check=True)
    (repo / "modified.txt").write_text("after\n", encoding="utf-8")
    (repo / "deleted.txt").unlink()
    (repo / "untracked.txt").write_text("new\n", encoding="utf-8")
    (repo / ".env").write_text("TOKEN=do-not-copy\n", encoding="utf-8")
    (repo / ".ENV.PRODUCTION").write_text(
        "TOKEN=case-insensitive-secret\n", encoding="utf-8"
    )
    (repo / "RCLONE.CONF").write_text(
        "token=case-insensitive-rclone\n", encoding="utf-8"
    )
    secret_dir = repo / "SeCrEtS"
    secret_dir.mkdir()
    (secret_dir / "payload.txt").write_text(
        "case-insensitive-secret-directory\n", encoding="utf-8"
    )

    snapshot = create_snapshot(repo, tmp_path / "snapshot.tar.gz")

    assert snapshot.included_paths == ("modified.txt", "untracked.txt")
    assert snapshot.deleted_paths == ("deleted.txt",)
    assert snapshot.excluded_secret_paths == (
        ".ENV.PRODUCTION",
        ".env",
        "RCLONE.CONF",
        "SeCrEtS/payload.txt",
    )
    with tarfile.open(snapshot.archive_path, "r:gz") as archive:
        assert sorted(archive.getnames()) == ["modified.txt", "untracked.txt"]


@pytest.mark.parametrize("member_name", ("../escape", "/absolute", "a/../../b"))
def test_snapshot_extraction_rejects_escaping_tar_members(
    tmp_path: Path, member_name: str
) -> None:
    archive_path = tmp_path / "unsafe.tar.gz"
    payload = b"escape"
    with tarfile.open(archive_path, "w:gz") as archive:
        member = tarfile.TarInfo(member_name)
        member.size = len(payload)
        archive.addfile(member, io.BytesIO(payload))

    with pytest.raises(remote_runner.RemoteWorkflowError):
        remote_runner._safe_extract(archive_path, tmp_path / "extract")

    assert not (tmp_path / "escape").exists()


def test_snapshot_rejects_dirty_submodule_worktree(tmp_path: Path) -> None:
    submodule = tmp_path / "submodule-source"
    submodule.mkdir()
    subprocess.run(["git", "init", "-q", str(submodule)], check=True)
    subprocess.run(
        ["git", "-C", str(submodule), "config", "user.email", "test@example.com"],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(submodule), "config", "user.name", "Test"],
        check=True,
    )
    (submodule / "tracked.txt").write_text("clean\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(submodule), "add", "."], check=True)
    subprocess.run(
        ["git", "-C", str(submodule), "commit", "-qm", "submodule"], check=True
    )

    repo = tmp_path / "repo-with-submodule"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    subprocess.run(
        ["git", "-C", str(repo), "config", "user.email", "test@example.com"],
        check=True,
    )
    subprocess.run(["git", "-C", str(repo), "config", "user.name", "Test"], check=True)
    (repo / "tracked.txt").write_text("parent\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(repo), "add", "."], check=True)
    subprocess.run(["git", "-C", str(repo), "commit", "-qm", "parent"], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(repo),
            "-c",
            "protocol.file.allow=always",
            "submodule",
            "add",
            "-q",
            str(submodule),
            "deps/example",
        ],
        check=True,
    )
    subprocess.run(["git", "-C", str(repo), "add", "."], check=True)
    subprocess.run(
        ["git", "-C", str(repo), "commit", "-qm", "add submodule"], check=True
    )
    (repo / "deps/example/tracked.txt").write_text("dirty\n", encoding="utf-8")

    with pytest.raises(WorkflowError, match="submodule checkout has tracked"):
        create_snapshot(repo, tmp_path / "snapshot.tar.gz")


def test_snapshot_dry_run_accepts_an_unpushed_main_repository_commit(
    tmp_path: Path, fake_colab: dict[str, Any]
) -> None:
    repo = tmp_path / "unpushed-repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    subprocess.run(
        ["git", "-C", str(repo), "config", "user.email", "test@example.com"],
        check=True,
    )
    subprocess.run(["git", "-C", str(repo), "config", "user.name", "Test"], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(repo),
            "remote",
            "add",
            "origin",
            "https://github.com/example/unpushed.git",
        ],
        check=True,
    )
    (repo / "pyproject.toml").write_text(
        "[project]\nname='fixture'\n", encoding="utf-8"
    )
    (repo / "unpushed.txt").write_text("only local\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(repo), "add", "."], check=True)
    subprocess.run(["git", "-C", str(repo), "commit", "-qm", "unpushed"], check=True)
    head = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            CLI_MODULE,
            "--repo-root",
            str(repo),
            "--state-dir",
            str(tmp_path / "state"),
            "--jobs-dir",
            str(ROOT / "scripts/colab/workflows/jobs"),
            "run",
            "blcs_generate_dataset",
            "--run-id",
            RUN_ID,
            "--source",
            "snapshot",
            "--dry-run",
        ],
        cwd=ROOT,
        env=fake_colab["env"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    source = json.loads(result.stdout)["request"]["source"]
    assert source["mode"] == "snapshot"
    assert source["repo_sha"] == head
    assert "unpushed.txt" in source["included_paths"]
    assert _invocations(fake_colab) == []


def test_remote_rclone_input_uses_config_and_exact_remote_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[list[str]] = []

    def fake_run(argv: list[str], **_kwargs: Any) -> subprocess.CompletedProcess[str]:
        calls.append(argv)
        stdout = json.dumps({"IsDir": False}) if "lsjson" in argv else ""
        return subprocess.CompletedProcess(argv, 0, stdout=stdout, stderr="")

    monkeypatch.setattr(remote_runner, "_run", fake_run)
    config = tmp_path / "rclone.conf"
    destination = tmp_path / "repo/data/input.bin"
    request = {"drive": {"mode": "rclone", "root": "team/root", "remote": "drive"}}

    remote_runner._copy_rclone_input(request, "datasets/input.bin", destination, config)

    remote = "drive:team/root/datasets/input.bin"
    assert calls == [
        ["rclone", "--config", str(config), "lsjson", "--stat", remote],
        ["rclone", "--config", str(config), "copyto", remote, str(destination)],
    ]


def test_resume_restages_writable_inputs_and_rejects_canonical_manifest_changes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    drive_root = tmp_path / "drive"
    canonical = drive_root / "dataset"
    canonical.mkdir(parents=True)
    (canonical / "sample.txt").write_text("version-one\n", encoding="utf-8")
    repo = tmp_path / "repo"
    repo.mkdir()
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    request = {
        "request_digest": "a" * 64,
        "drive": {"mode": "mount", "root": "team", "remote": None},
        "job": {
            "inputs": [
                {
                    "source": "dataset",
                    "destination": "data/dataset",
                    "writable": True,
                }
            ]
        },
    }
    monkeypatch.setattr(remote_runner, "_mount_drive_root", lambda _request: drive_root)

    first = remote_runner._stage_inputs(request, repo, None, reset_existing=False)
    preserved = remote_runner._preserve_input_manifest(workspace, request, first, None)
    assert preserved == first

    staged = repo / "data/dataset"
    (staged / "sample.txt").write_text("failed-attempt-change\n", encoding="utf-8")
    (staged / "partial-output.txt").write_text("discard me\n", encoding="utf-8")
    restarted = remote_runner._stage_inputs(request, repo, None, reset_existing=True)
    assert (staged / "sample.txt").read_text(encoding="utf-8") == "version-one\n"
    assert not (staged / "partial-output.txt").exists()
    assert restarted == first
    assert (
        remote_runner._preserve_input_manifest(
            workspace,
            request,
            restarted,
            {"provenance": {"input_manifest": first}},
        )
        == first
    )

    (canonical / "sample.txt").write_text("version-two\n", encoding="utf-8")
    changed = remote_runner._stage_inputs(request, repo, None, reset_existing=True)
    with pytest.raises(
        remote_runner.RemoteWorkflowError,
        match="restaged inputs differ from the immutable first-attempt manifest",
    ):
        remote_runner._preserve_input_manifest(
            workspace,
            request,
            changed,
            {"provenance": {"input_manifest": first}},
        )


def test_remote_runner_publishes_completed_bundle_before_marking_vm_completed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dry_run = _run_cli(
        tmp_path,
        "run",
        "court_detection",
        "--run-id",
        RUN_ID,
        "--drive-mode",
        "mount",
        "--dry-run",
    )
    assert dry_run.returncode == 0, dry_run.stderr
    request = json.loads(dry_run.stdout)["request"]
    workspace = tmp_path / "remote-workspace"
    workspace.mkdir()
    (workspace / "request.json").write_text(json.dumps(request), encoding="utf-8")
    source = request["source"]
    source_provenance = {
        "mode": source["mode"],
        "repo_sha": source["repo_sha"],
        "repo_tree": source["repo_tree"],
        "tracked_clean": source["tracked_clean"],
    }
    runtime = {
        "collected": True,
        "python": {"available": True},
        "platform": {"available": True},
        "gpu": {
            "required": True,
            "requested": "T4",
            "available": True,
            "nvidia_smi": {},
        },
        "cuda": {"available": True},
        "torch": {"available": True},
        "uv": {"available": True},
    }
    input_manifest = [
        {**mapping, "entries": []} for mapping in request["job"]["inputs"]
    ]
    output_manifest = [
        {"declared_path": output, "entries": []} for output in request["job"]["outputs"]
    ]
    observed: dict[str, str] = {}

    def fake_collect(
        _request: dict[str, Any], _repo: Path, bundle: Path
    ) -> tuple[list[dict[str, Any]], Path]:
        bundle.mkdir()
        artifact = bundle / "artifacts.tar.gz"
        artifact.write_bytes(b"verified artifact")
        return output_manifest, artifact

    def fake_publish(
        _request: dict[str, Any], bundle: Path, _attempt: int, _config: Path | None
    ) -> str:
        vm_status = json.loads((workspace / "status.json").read_text(encoding="utf-8"))
        bundle_status = json.loads((bundle / "status.json").read_text(encoding="utf-8"))
        observed["vm_state"] = vm_status["state"]
        observed["vm_step"] = vm_status["step"]
        observed["bundle_state"] = bundle_status["state"]
        return f"tennis_lab/colab-runs/{RUN_ID}"

    monkeypatch.setattr(remote_runner, "_workspace", lambda: workspace)
    monkeypatch.setattr(remote_runner, "_recover_published_status", lambda *_args: None)
    monkeypatch.setattr(
        remote_runner,
        "_prepare_source",
        lambda _request, _workspace: source_provenance,
    )
    monkeypatch.setattr(remote_runner, "_run_hook", lambda *_args: None)
    monkeypatch.setattr(remote_runner, "_runtime_provenance", lambda *_args: runtime)
    monkeypatch.setattr(
        remote_runner, "_stage_inputs", lambda *_args, **_kwargs: input_manifest
    )
    monkeypatch.setattr(
        remote_runner,
        "_run",
        lambda argv, **_kwargs: subprocess.CompletedProcess(argv, 0),
    )
    monkeypatch.setattr(remote_runner, "_collect_outputs", fake_collect)
    monkeypatch.setattr(remote_runner, "_publish", fake_publish)
    monkeypatch.setattr(remote_runner, "TENNIS_COLAB_RUN_ID", RUN_ID, raising=False)

    assert remote_runner._execute_action() == 0
    assert observed == {
        "vm_state": "running",
        "vm_step": "publishing",
        "bundle_state": "completed",
    }
    final_status = json.loads((workspace / "status.json").read_text(encoding="utf-8"))
    assert final_status["state"] == final_status["step"] == "completed"


def test_artifact_manifest_archive_and_rclone_staged_publish(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = tmp_path / "repo"
    output = repo / "outputs/model"
    output.mkdir(parents=True)
    (output / "metrics.json").write_text('{"loss": 0.1}\n', encoding="utf-8")
    bundle = tmp_path / "bundle"
    request = {
        "run_id": RUN_ID,
        "drive": {"mode": "rclone", "root": "team/root", "remote": "drive"},
        "job": {"outputs": ["outputs/model"]},
    }

    manifest, archive_path = remote_runner._collect_outputs(request, repo, bundle)

    assert manifest[0]["declared_path"] == "outputs/model"
    entry = manifest[0]["entries"][0]
    assert entry["path"] == "outputs/model/metrics.json"
    assert (
        entry["sha256"]
        == hashlib.sha256((output / "metrics.json").read_bytes()).hexdigest()
    )
    with tarfile.open(archive_path, "r:gz") as archive:
        assert "outputs/model/metrics.json" in archive.getnames()

    existence = iter((False, False, False))
    commands: list[list[str]] = []
    monkeypatch.setattr(
        remote_runner, "_rclone_exists", lambda _path, _config: next(existence)
    )
    monkeypatch.setattr(
        remote_runner,
        "_run",
        lambda argv, **_kwargs: commands.append(argv),
    )
    published = remote_runner._publish_rclone(
        request, bundle, attempt=2, config=tmp_path / "rclone.conf"
    )

    assert published == f"team/root/colab-runs/{RUN_ID}"
    assert [command[3] for command in commands] == ["copy", "check", "moveto"]
    staging = commands[0][-1]
    assert f".{RUN_ID}.uploading-2-" in staging
    assert commands[-1][-2:] == [
        staging,
        f"drive:team/root/colab-runs/{RUN_ID}",
    ]


@pytest.mark.parametrize(
    "mode",
    (
        "missing",
        "corrupt",
        "digest-mismatch",
        "run-id-mismatch",
        "path-run-id",
        "session-mismatch",
    ),
)
def test_run_rejects_missing_corrupt_or_wrong_identity_remote_status(
    tmp_path: Path,
    fake_colab: dict[str, Any],
    mode: str,
) -> None:
    environment = {**fake_colab["env"], "FAKE_STATUS_MODE": mode}
    result = _run_cli(
        tmp_path,
        "run",
        "court_detection",
        "--run-id",
        RUN_ID,
        "--drive-mode",
        "mount",
        env=environment,
    )

    assert result.returncode == 3
    calls = _invocations(fake_colab)
    assert not any(_operation(call) == "stop" for call in calls)
    metadata = json.loads(
        (tmp_path / "state" / RUN_ID / "local.json").read_text(encoding="utf-8")
    )
    assert metadata["session_state"] == "retained-status-unknown"
    assert not (tmp_path / "escape").exists()
    assert sorted(path.name for path in (tmp_path / "state").iterdir()) == [RUN_ID]


def test_exec_zero_with_failed_remote_status_is_runtime_failure_and_cleans_up(
    tmp_path: Path, fake_colab: dict[str, Any]
) -> None:
    config = _rclone_config(tmp_path)
    environment = {**fake_colab["env"], "FAKE_STATUS_MODE": "failed"}

    result = _run_cli(
        tmp_path,
        "run",
        "court_detection",
        "--run-id",
        RUN_ID,
        "--rclone-config",
        str(config),
        env=environment,
    )

    assert result.returncode == 1, result.stderr
    status = json.loads(result.stdout)
    assert status["state"] == "failed"
    calls = _invocations(fake_colab)
    assert any(_action(call) == "cleanup-secret" for call in calls)
    assert any(_operation(call) == "stop" for call in calls)
    logged = "\n".join(json.dumps(call) for call in calls)
    assert "ultra-secret-refresh-token" not in logged
    assert "ultra-secret-refresh-token" not in result.stdout + result.stderr
    assert Path(fake_colab["secret_mode_log"]).read_text(encoding="utf-8") == "0o600"
    remote_secret = (
        Path(fake_colab["remote"])
        / f"content/tennis-lab-runs/{RUN_ID}/.secrets/rclone.conf"
    )
    assert not remote_secret.exists()
    assert config.is_file()


def test_success_status_optional_download_and_idempotent_download_and_stop(
    tmp_path: Path, fake_colab: dict[str, Any]
) -> None:
    download_dir = tmp_path / "downloads"
    environment = {**fake_colab["env"], "FAKE_STATUS_MODE": "completed"}
    result = _run_cli(
        tmp_path,
        "run",
        "court_detection",
        "--run-id",
        RUN_ID,
        "--drive-mode",
        "mount",
        "--download-to",
        str(download_dir),
        env=environment,
    )

    assert result.returncode == 0, result.stderr
    status_start = result.stdout.index("{")
    assert json.loads(result.stdout[status_start:])["state"] == "completed"
    artifact = download_dir / f"{RUN_ID}-artifacts.tar.gz"
    assert artifact.read_bytes() == b"fake verified artifact\n"

    second_download = _run_cli(
        tmp_path,
        "download",
        RUN_ID,
        "--to",
        str(download_dir),
        env=environment,
    )
    first_stop = _run_cli(tmp_path, "stop", RUN_ID, env=environment)
    second_stop = _run_cli(tmp_path, "stop", RUN_ID, env=environment)

    assert second_download.returncode == 0, second_download.stderr
    assert "artifact already verified" in second_download.stdout
    assert first_stop.returncode == second_stop.returncode == 0


def test_success_without_download_only_fetches_status(
    tmp_path: Path, fake_colab: dict[str, Any]
) -> None:
    result = _run_cli(
        tmp_path,
        "run",
        "court_detection",
        "--run-id",
        RUN_ID,
        "--drive-mode",
        "mount",
        env=fake_colab["env"],
    )

    assert result.returncode == 0, result.stderr
    downloads = [
        call for call in _invocations(fake_colab) if _operation(call) == "download"
    ]
    assert len(downloads) == 1
    assert downloads[0][-2].endswith("/status.json")


def test_rclone_status_and_download_work_directly_after_session_stop(
    tmp_path: Path, fake_colab: dict[str, Any]
) -> None:
    config = _rclone_config(tmp_path)
    run = _run_cli(
        tmp_path,
        "run",
        "court_detection",
        "--run-id",
        RUN_ID,
        "--rclone-config",
        str(config),
        env=fake_colab["env"],
    )
    assert run.returncode == 0, run.stderr
    calls_after_run = len(_invocations(fake_colab))

    status_path = tmp_path / "state" / RUN_ID / "status.json"
    cached = json.loads(status_path.read_text(encoding="utf-8"))
    cached.update(
        {
            "state": "failed",
            "step": "running_job",
            "error": {
                "type": "RuntimeError",
                "message": "stale local cache",
                "traceback": "",
            },
            "artifacts": {},
        }
    )
    status_path.write_text(json.dumps(cached), encoding="utf-8")

    status = _run_cli(
        tmp_path,
        "status",
        RUN_ID,
        "--rclone-config",
        str(config),
        env=fake_colab["env"],
    )
    assert status.returncode == 0, status.stderr
    assert json.loads(status.stdout)["status"]["state"] == "completed"
    assert len(_invocations(fake_colab)) == calls_after_run
    assert len(_rclone_invocations(fake_colab)) == 3

    download_dir = tmp_path / "downloads"
    download = _run_cli(
        tmp_path,
        "download",
        RUN_ID,
        "--to",
        str(download_dir),
        "--rclone-config",
        str(config),
        env=fake_colab["env"],
    )
    assert download.returncode == 0, download.stderr
    assert (
        download_dir / f"{RUN_ID}-artifacts.tar.gz"
    ).read_bytes() == b"fake verified artifact\n"
    assert len(_invocations(fake_colab)) == calls_after_run
    assert len(_rclone_invocations(fake_colab)) == 7


def test_rclone_download_rejects_publication_without_completed_status(
    tmp_path: Path, fake_colab: dict[str, Any]
) -> None:
    config = _rclone_config(tmp_path)
    environment = {
        **fake_colab["env"],
        "FAKE_PUBLISHED_STATUS_MODE": "running",
    }
    run = _run_cli(
        tmp_path,
        "run",
        "court_detection",
        "--run-id",
        RUN_ID,
        "--rclone-config",
        str(config),
        env=environment,
    )
    assert run.returncode == 0, run.stderr

    download_dir = tmp_path / "downloads"
    download = _run_cli(
        tmp_path,
        "download",
        RUN_ID,
        "--to",
        str(download_dir),
        "--rclone-config",
        str(config),
        env=environment,
    )

    assert download.returncode == 3
    assert "not completed" in download.stderr
    assert not (download_dir / f"{RUN_ID}-artifacts.tar.gz").exists()


def test_keep_on_failure_retains_session_but_always_removes_rclone_secret(
    tmp_path: Path, fake_colab: dict[str, Any]
) -> None:
    config = _rclone_config(tmp_path)
    environment = {**fake_colab["env"], "FAKE_STATUS_MODE": "failed"}
    result = _run_cli(
        tmp_path,
        "run",
        "court_detection",
        "--run-id",
        RUN_ID,
        "--rclone-config",
        str(config),
        "--keep-on-failure",
        env=environment,
    )

    assert result.returncode == 1
    calls = _invocations(fake_colab)
    assert any(_action(call) == "cleanup-secret" for call in calls)
    assert not any(_operation(call) == "stop" for call in calls)
    metadata = json.loads(
        (tmp_path / "state" / RUN_ID / "local.json").read_text(encoding="utf-8")
    )
    assert metadata["session_state"] == "retained-after-failure"


def test_resume_reuses_request_digest_increments_attempt_and_stops_on_success(
    tmp_path: Path, fake_colab: dict[str, Any]
) -> None:
    config = _rclone_config(tmp_path)
    failed_environment = {**fake_colab["env"], "FAKE_STATUS_MODE": "failed"}
    first = _run_cli(
        tmp_path,
        "run",
        "court_detection",
        "--run-id",
        RUN_ID,
        "--rclone-config",
        str(config),
        "--keep-on-failure",
        env=failed_environment,
    )
    assert first.returncode == 1
    request = json.loads(
        (tmp_path / "state" / RUN_ID / "request.json").read_text(encoding="utf-8")
    )

    resumed_environment = {**fake_colab["env"], "FAKE_STATUS_MODE": "completed"}
    resumed = _run_cli(
        tmp_path,
        "resume",
        RUN_ID,
        "--rclone-config",
        str(config),
        env=resumed_environment,
    )

    assert resumed.returncode == 0, resumed.stderr
    status = json.loads(resumed.stdout)
    assert status["request_digest"] == request["request_digest"]
    assert status["attempt"] == 2
    calls = _invocations(fake_colab)
    assert sum(_action(call) == "run" for call in calls) == 2
    assert _operation(calls[-1]) == "stop"


def test_resume_rejects_tampered_local_request_before_colab_access(
    tmp_path: Path, fake_colab: dict[str, Any]
) -> None:
    config = _rclone_config(tmp_path)
    environment = {**fake_colab["env"], "FAKE_STATUS_MODE": "failed"}
    first = _run_cli(
        tmp_path,
        "run",
        "court_detection",
        "--run-id",
        RUN_ID,
        "--rclone-config",
        str(config),
        "--keep-on-failure",
        env=environment,
    )
    assert first.returncode == 1
    before = len(_invocations(fake_colab))
    request_path = tmp_path / "state" / RUN_ID / "request.json"
    request = json.loads(request_path.read_text(encoding="utf-8"))
    request["job"]["argv"].append("malicious=true")
    request_path.write_text(json.dumps(request), encoding="utf-8")

    resumed = _run_cli(
        tmp_path,
        "resume",
        RUN_ID,
        "--rclone-config",
        str(config),
        env=environment,
    )

    assert resumed.returncode == 3
    assert "does not match its digest" in resumed.stderr
    assert len(_invocations(fake_colab)) == before


def test_sigterm_returns_130_and_stops_provisioned_session(
    tmp_path: Path, fake_colab: dict[str, Any]
) -> None:
    environment = {**fake_colab["env"], "FAKE_STATUS_MODE": "interrupt"}
    repo = _clean_source_repo(tmp_path)
    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            CLI_MODULE,
            "--repo-root",
            str(repo),
            "--state-dir",
            str(tmp_path / "state"),
            "--jobs-dir",
            str(ROOT / "scripts/colab/workflows/jobs"),
            "run",
            "court_detection",
            "--run-id",
            RUN_ID,
            "--drive-mode",
            "mount",
        ],
        cwd=ROOT,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    marker = Path(fake_colab["marker"])
    deadline = time.monotonic() + 10
    while (
        not marker.exists() and process.poll() is None and time.monotonic() < deadline
    ):
        time.sleep(0.02)
    assert marker.exists(), process.communicate(timeout=2)

    process.send_signal(signal.SIGTERM)
    stdout, stderr = process.communicate(timeout=10)

    assert process.returncode == 130, (stdout, stderr)
    assert "received signal SIGTERM" in stderr
    assert any(_operation(call) == "stop" for call in _invocations(fake_colab))
