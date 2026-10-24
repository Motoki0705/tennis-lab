"""Pinned CPU-test requests, disposable clones and ledger-backed run evidence."""

from __future__ import annotations

import fcntl
import hashlib
import os
import re
import shutil
import subprocess
import uuid
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from jsonschema import Draft7Validator  # type: ignore[import-untyped]

from .ledger import Ledger
from .repro_isolation import LinuxRunner, TestJob, TestRunner, clean_environment
from .retrieval import (
    SHA,
    STRING,
    TEXT,
    RetrievalError,
    Retrieved,
    Tool,
    encode,
    strict_json,
    text_page,
)

RUN_ID = re.compile(r"[0-9a-f]{32}\Z")
NODE_ID = re.compile(
    r"tests/(?:[A-Za-z0-9_.-]+/)*[A-Za-z0-9_.-]+\.py"
    r"(?:::[A-Za-z_][A-Za-z0-9_]*){0,2}\Z"
)
WRAPPER = ".agents/ops/pytest_locked.sh"
RUN_SCHEMA = Tool(
    "cpu_tests",
    "Run only host-approved existing CPU tests at the fixed base SHA; no commands or options.",
    lambda: Retrieved(None, []),
    {
        "finding_id": {**STRING, "minLength": 1, "maxLength": 128},
        "nodeids": {
            "type": "array",
            "minItems": 1,
            "maxItems": 32,
            "uniqueItems": True,
            "items": {"type": "string", "minLength": 1, "maxLength": 512},
        },
    },
    ("finding_id", "nodeids"),
).schema


def now() -> str:
    return datetime.now(UTC).isoformat()


def digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def recorded_bytes(path: Path) -> bytes:
    try:
        return path.read_bytes()
    except OSError as exc:
        raise RetrievalError(
            "MISSING_RECORD", "recorded result or log is unavailable"
        ) from exc


def write_record(path: Path, value: Any) -> str:
    """Immutable, private records; an interrupted write is not a valid result."""
    raw = encode(value) + b"\n"
    temporary = path.with_suffix(".pending")
    fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "wb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    # link refuses overwriting either an earlier result or a symlink.
    os.link(temporary, path)
    temporary.unlink()
    directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)
    return digest(raw)


def git(repo: Path, *args: str) -> bytes:
    env = {
        **clean_environment(),
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_CONFIG_GLOBAL": os.devnull,
        "GIT_NO_REPLACE_OBJECTS": "1",
        "GIT_OPTIONAL_LOCKS": "0",
        "GIT_TERMINAL_PROMPT": "0",
    }
    result = subprocess.run(
        [
            "/usr/bin/git",
            "--no-replace-objects",
            "-c", "core.hooksPath=/dev/null",
            "-c", "core.fsmonitor=false",
            "-c", "core.untrackedCache=false",
            "-c", "protocol.allow=never",
            "-c", "protocol.file.allow=always",
            *args,
        ],
        cwd=repo,
        env=env,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        timeout=45,
        check=False,
    )  # fmt: skip
    if result.returncode:
        raise RetrievalError(
            "CHECKOUT_ERROR",
            "local git operation failed",
            exit_code=result.returncode,
            stderr=result.stderr.decode(errors="replace")[:1000],
        )
    return result.stdout


def node_path(nodeid: str) -> str:
    if not NODE_ID.fullmatch(nodeid):
        raise ValueError(
            "only tests/*.py with optional class/function selectors are allowed"
        )
    path = nodeid.split("::", 1)[0]
    if any(p in {"", ".", "..", ".git"} for p in path.split("/")):
        raise ValueError("test selector contains a path escape")
    name = Path(path).name
    if not (name.startswith("test_") or name.endswith("_test.py")):
        raise ValueError("expected an existing pytest test file")
    return path


class Reproduction:
    def __init__(
        self,
        repo: Path,
        base_sha: str,
        slot: str,
        state_dir: Path,
        ledger_path: Path,
        cpu_nodeids: list[str],
        *,
        pytest_lock: Path | None = None,
        runner: TestRunner | None = None,
    ) -> None:
        self.repo = repo.resolve(strict=True)
        if not SHA.fullmatch(base_sha):
            raise ValueError("base_sha must be a full immutable commit SHA")
        self.base_sha, self.slot = base_sha, slot
        self.ledger_path = ledger_path.resolve(strict=True)
        self._book()
        if git(self.repo, "cat-file", "-t", base_sha).strip() != b"commit":
            raise ValueError("base_sha is not a commit")
        if (
            not isinstance(cpu_nodeids, list)
            or not cpu_nodeids
            or len(cpu_nodeids) > 256
            or any(not isinstance(n, str) or len(n) > 512 for n in cpu_nodeids)
            or len(set(cpu_nodeids)) != len(cpu_nodeids)
        ):
            raise ValueError(
                "a nonempty, unique host-approved CPU-test allowlist is required"
            )
        self.cpu_nodeids = frozenset(cpu_nodeids)
        for nodeid in cpu_nodeids:
            self._regular_blob(node_path(nodeid))
        self.wrapper = self._regular_blob(WRAPPER)
        trusted_wrapper = Path(__file__).resolve().parents[2] / "pytest_locked.sh"
        if self.wrapper != trusted_wrapper.read_bytes():
            raise ValueError(
                "pinned pytest lock wrapper differs from the deployed wrapper"
            )
        self.state_dir = state_dir.resolve()
        if self.state_dir.is_relative_to(self.repo):
            raise ValueError("reproduction records must be outside the source checkout")
        self.state_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
        if self.state_dir.stat().st_mode & 0o077:
            raise ValueError("reproduction records require a private directory")
        default_state = Path(
            os.environ.get("XDG_STATE_HOME", str(Path.home() / ".local/state"))
        )
        self.pytest_lock = pytest_lock or Path(
            os.environ.get(
                "PYTEST_LOCK", str(default_state / "tennis-lab-agents/pytest.lock")
            )
        )
        if not self.pytest_lock.is_absolute():
            raise ValueError("pytest lock must be absolute")
        self.execution_lock = self.pytest_lock.with_name("audit-reproduction.lock")
        self.runner = runner if runner is not None else LinuxRunner()

    def _book(self) -> Ledger:
        return Ledger.load(self.ledger_path, slot=self.slot, base_sha=self.base_sha)

    def _regular_blob(self, path: str) -> bytes:
        # ls-tree must match the literal path and a regular blob, never a
        # directory, symlink or submodule boundary. No worktree files are read.
        rows = git(self.repo, "ls-tree", "-z", self.base_sha, "--", path).split(b"\0")
        if len(rows) != 2 or not rows[0]:
            raise ValueError(f"not a committed regular file: {path}")
        info, actual = rows[0].split(b"\t", 1)
        mode, kind, object_id = info.split()
        if (
            actual != path.encode()
            or mode not in {b"100644", b"100755"}
            or kind != b"blob"
        ):
            raise ValueError(f"symlink, gitlink or non-file selector: {path}")
        return git(self.repo, "cat-file", "blob", object_id.decode())

    def status(self) -> Retrieved:
        return Retrieved(
            {
                **self.runner.inspect(),
                "base_sha": self.base_sha,
                "cpu_nodeids": sorted(self.cpu_nodeids),
                "policy": "existing host-approved CPU tests only; no arbitrary code/shell/network/GPU",
            },
            [{"kind": "isolation-preflight", "base_sha": self.base_sha}],
        )

    def tools(self) -> list[Tool]:
        tools = [
            Tool(
                "cpu_test_status",
                "Inspect isolation blockers and the fixed CPU-test allowlist.",
                self.status,
                {},
            ),
            Tool(
                "cpu_test_log",
                "Read a bounded, digest-checked recorded CPU-test log.",
                self.log,
                {"run_id": {"type": "string", "pattern": "^[0-9a-f]{32}$"}, **TEXT},
                ("run_id",),
            ),
        ]
        if self.runner.inspect().get("enabled") is True:
            tools.append(
                Tool(
                    "cpu_tests",
                    "Request an allowlisted existing CPU test at the fixed SHA.",
                    lambda **args: self.request(args),
                    RUN_SCHEMA["properties"],
                    tuple(RUN_SCHEMA["required"]),
                )
            )
        return tools

    def request(self, arguments: Any) -> Retrieved:
        run_id = uuid.uuid4().hex
        directory = self.state_dir / run_id
        directory.mkdir(mode=0o700)
        started = now()
        write_record(
            directory / "request.json",
            {
                "schema_version": 1,
                "run_id": run_id,
                "slot": self.slot,
                "base_sha": self.base_sha,
                "at": started,
                "arguments": arguments,
            },
        )
        result: dict[str, Any] = {
            "schema_version": 1,
            "run_id": run_id,
            "slot": self.slot,
            "base_sha": self.base_sha,
            "started_at": started,
            "disposition": "needs-reproduction",
            "status": "rejected",
            "checkout": None,
            "cleanup": "not-created",
            "process": None,
            "isolation": self.runner.inspect(),
            "log_sha256": None,
        }
        if isinstance(arguments, dict) and isinstance(arguments.get("finding_id"), str):
            result["finding_id"] = arguments["finding_id"]
        error: RetrievalError | None = None
        try:
            if list(Draft7Validator(RUN_SCHEMA).iter_errors(arguments)):
                raise RetrievalError(
                    "FORBIDDEN_REQUEST",
                    "only finding_id and approved nodeids are accepted",
                )
            self._validate(arguments)
            result["finding_id"] = arguments["finding_id"]
            result["nodeids"] = arguments["nodeids"]
            if result["isolation"].get("enabled") is not True:
                raise RetrievalError(
                    "ISOLATION_UNAVAILABLE",
                    "CPU-test tool is disabled; see isolation blockers",
                )
            self._execute(directory, arguments["nodeids"], result)
        except (
            OSError,
            ValueError,
            RuntimeError,
            subprocess.SubprocessError,
            RetrievalError,
        ) as exc:
            error = (
                exc
                if isinstance(exc, RetrievalError)
                else RetrievalError("REPRODUCTION_ERROR", str(exc))
            )
            result["error"] = {
                "code": error.code,
                "message": str(error),
                "details": error.details,
            }
            result["status"] = "error" if result["checkout"] else "rejected"
            result["disposition"] = "needs-reproduction"
        result["finished_at"] = now()
        manifest = directory / "result.json"
        manifest_hash = write_record(manifest, result)
        evidence_id = "cpu-test-" + run_id
        reference = manifest.as_uri() + "#sha256=" + manifest_hash
        try:
            book = self._book()
            book.record_evidence(
                {
                    "id": evidence_id,
                    "source_kind": "isolated-cpu-test",
                    "reference": reference,
                    "acquired_at": result["finished_at"],
                    "target_version": self.base_sha,
                    "pr_head": None,
                },
                at=result["finished_at"],
            )
            finding_id = result.get("finding_id")
            if finding_id in book.snapshot["findings"]:
                # Preserve the investigator's finding status. Neither a refused
                # request nor a failing test proves the suspected issue.
                finding = book.snapshot["findings"][finding_id]
                finding["evidence"].append(evidence_id)
                book.record_finding(finding, at=result["finished_at"])
        except (OSError, ValueError, KeyError) as exc:
            raise RetrievalError(
                "RECORDING_FAILED",
                "run manifest preserved; ledger update incomplete",
                run_id=run_id,
                reference=reference,
                reason=str(exc),
            ) from exc
        source = {
            "kind": "isolated-cpu-test",
            "base_sha": self.base_sha,
            "run_id": run_id,
            "evidence_id": evidence_id,
            "reference": reference,
            "disposition": result["disposition"],
            "cleanup": result["cleanup"],
        }
        if error:
            raise RetrievalError(
                error.code, str(error), sources=[source], **error.details
            )
        return Retrieved(
            {**result, "evidence_id": evidence_id, "reference": reference}, [source]
        )

    def _validate(self, arguments: dict[str, Any]) -> None:
        for nodeid in arguments["nodeids"]:
            try:
                node_path(nodeid)
            except ValueError as exc:
                raise RetrievalError("FORBIDDEN_REQUEST", str(exc)) from exc
            if nodeid not in self.cpu_nodeids:
                raise RetrievalError(
                    "FORBIDDEN_REQUEST",
                    "test is not in the host-approved CPU allowlist",
                )
        if arguments["finding_id"] not in self._book().snapshot["findings"]:
            raise RetrievalError(
                "UNKNOWN_FINDING", "finding_id must already exist in the pinned ledger"
            )

    def _source_state(self) -> dict[str, str]:
        return {
            "head": git(self.repo, "rev-parse", "HEAD").decode().strip(),
            "status_sha256": digest(
                git(
                    self.repo, "status", "--porcelain=v1", "-z", "--untracked-files=all"
                )
            ),
        }

    def _execute(
        self, directory: Path, nodeids: list[str], result: dict[str, Any]
    ) -> None:
        self.execution_lock.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        fd = os.open(self.execution_lock, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
        try:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise RetrievalError(
                    "REPRODUCTION_BUSY", "another isolated CPU-test request is active"
                ) from exc
            self._checkout_run(directory, nodeids, result)
        finally:
            os.close(fd)

    def _checkout_run(
        self, directory: Path, nodeids: list[str], result: dict[str, Any]
    ) -> None:
        before = self._source_state()
        result["source_before"] = before
        checkout = directory / "checkout"
        log = directory / "pytest.log"
        result["checkout"] = str(checkout)
        try:
            # Independent objects and no registered source worktree. Test code
            # never gets alternates/hardlinks pointing into the source checkout.
            git(
                self.repo,
                "clone",
                "--no-hardlinks",
                "--no-checkout",
                "--no-recurse-submodules",
                "--",
                str(self.repo),
                str(checkout),
            )
            git(checkout, "checkout", "--detach", self.base_sha)
            if git(checkout, "rev-parse", "HEAD").decode().strip() != self.base_sha:
                raise RetrievalError(
                    "CHECKOUT_MISMATCH",
                    "disposable clone HEAD differs from the fixed SHA",
                )
            job = TestJob(checkout, tuple(nodeids), self.pytest_lock, log)
            result["argv"] = job.argv
            result["environment"] = job.env
            result["status"] = "running"
            process = self.runner.run(job)
            result["process"] = asdict(process)
            result["log_sha256"] = digest(log.read_bytes())
            result["status"] = (
                "passed"
                if process.reason == "exited" and process.exit_code == 0
                else "failed"
            )
            result["disposition"] = (
                "test-result" if result["status"] == "passed" else "needs-reproduction"
            )
        finally:
            if log.exists():
                log.chmod(0o600)
                result["log_sha256"] = digest(log.read_bytes())
            if checkout.exists():
                try:
                    head = git(checkout, "rev-parse", "HEAD").decode().strip()
                    dirty = git(
                        checkout,
                        "status",
                        "--porcelain=v1",
                        "-z",
                        "--untracked-files=all",
                        "--ignored",
                    )
                    if head != self.base_sha or dirty:
                        result["cleanup"] = "retained-dirty-or-head-changed"
                    else:
                        shutil.rmtree(checkout)
                        result["cleanup"] = "removed-clean"
                except (OSError, RetrievalError, subprocess.SubprocessError) as exc:
                    result["cleanup"] = "retained-unverified"
                    raise RetrievalError(
                        "CLEANUP_UNVERIFIED",
                        "checkout retained; cleanup could not be verified",
                        reason=str(exc),
                    ) from exc
            after = self._source_state()
            result["source_after"] = after
            if before != after:
                raise RetrievalError(
                    "SOURCE_CHANGED", "source HEAD/status changed during the request"
                )
        if result["cleanup"] != "removed-clean":
            raise RetrievalError(
                "CHECKOUT_DIRTY", "checkout retained; manual inspection required"
            )

    def log(self, run_id: str, **ranges: Any) -> Retrieved:
        if not RUN_ID.fullmatch(run_id):
            raise RetrievalError("INVALID_ARGUMENT", "invalid run_id")
        directory = self.state_dir / run_id
        if directory.is_symlink() or not directory.is_dir():
            raise RetrievalError("MISSING_RECORD", "run record is absent")
        raw = recorded_bytes(directory / "result.json")
        result = strict_json(raw)
        if (result["slot"], result["base_sha"], result["run_id"]) != (
            self.slot,
            self.base_sha,
            run_id,
        ):
            raise RetrievalError(
                "RECORD_MISMATCH", "run identity differs from the pinned context"
            )
        reference = (directory / "result.json").as_uri() + "#sha256=" + digest(raw)
        evidence = self._book().snapshot["evidence"].get("cpu-test-" + run_id)
        if evidence is None or evidence["reference"] != reference:
            raise RetrievalError(
                "RECORD_MISMATCH", "manifest differs from immutable ledger evidence"
            )
        log = directory / "pytest.log"
        data = recorded_bytes(log) if result["log_sha256"] else b""
        if result["log_sha256"] is not None and digest(data) != result["log_sha256"]:
            raise RetrievalError("RECORD_MISMATCH", "test log content changed")
        page, pagination = text_page(data.decode(errors="replace"), **ranges)
        if pagination["next"]:
            pagination["next"]["run_id"] = run_id
        return Retrieved(
            {
                "status": result["status"],
                "disposition": result["disposition"],
                "log": page,
            },
            [
                {
                    "kind": "isolated-cpu-test",
                    "reference": reference,
                    "base_sha": self.base_sha,
                }
            ],
            pagination,
        )
