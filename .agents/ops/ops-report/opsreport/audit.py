"""Pinned round checkpoints and verification before ledger publication."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from . import handoff, native, shell
from .ledger import Ledger
from .retrieval import RetrievalError, strict_json

VERSION = 2
GOAL = (
    "Tennis-Lab全体の潜在的な問題を発見し、根拠と反証を確認して既存の改善経路へ戻す。"
)


def timestamp() -> str:
    return datetime.now(UTC).isoformat()


@contextmanager
def checkout(repo: Path, base: str, path: Path) -> Iterator[Path]:
    """Never force-remove a checkout changed by a faulty runner."""
    path.parent.mkdir(parents=True, exist_ok=True)
    shell.git(["worktree", "add", "--detach", str(path), base], cwd=repo)
    try:
        yield path
    finally:
        if shell.git(["rev-parse", "HEAD"], cwd=path).strip() != base:
            raise ValueError(f"audit checkout HEAD changed: {path}")
        shell.git(["worktree", "remove", str(path)], cwd=repo)


def worktree_path(repo: Path, run_id: str) -> Path:
    common = Path(
        shell.git(
            ["rev-parse", "--path-format=absolute", "--git-common-dir"], cwd=repo
        ).strip()
    ).resolve()
    if common.name != ".git":
        raise ValueError("audit requires a non-bare repository with .git")
    parent = common.parent / ".claude/worktrees"
    if parent.resolve() != parent:
        raise ValueError("audit worktree parent must not be a symlink")
    return parent / f"ops-report-audit-{run_id}"


def input_digest(collected: dict[str, Any]) -> str:
    frozen = {k: v for k, v in collected.items() if k != "audit_input_sha256"}
    return hashlib.sha256(
        json.dumps(frozen, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()


def load_checkpoint(path: Path, repo: Path) -> dict[str, Any]:
    try:
        data = strict_json(path.read_bytes())
        if (
            isinstance(data, dict)
            and type(data.get("version")) is int
            and data["version"] == 1
        ):
            raise ValueError(
                "checkpoint version 1 requires explicit migration by #1458: "
                f"migrate-checkpoint --slot {path.stem} --report-issue NUMBER; "
                "do not replace the round or its issue"
            )
        if (
            not isinstance(data, dict)
            or set(data) != {"version", "collected", "ledger", "attempts"}
            or type(data["version"]) is not int
            or data["version"] != VERSION
        ):
            raise ValueError("invalid audit checkpoint schema/version")
        c = data["collected"]
        if input_digest(c) != c["audit_input_sha256"]:
            raise ValueError("audit checkpoint input hash mismatch")
        base = c["meta"]["base_sha"]
        if (
            shell.git(["rev-parse", "--verify", f"{base}^{{commit}}"], cwd=repo).strip()
            != base
        ):
            raise ValueError("pinned base commit is missing")
        ledger_path = Path(data["ledger"])
        if ledger_path != path.with_suffix(".ledger.json").absolute():
            raise ValueError("checkpoint ledger path mismatch")
        book = Ledger.load(ledger_path, slot=path.stem, base_sha=base)
        from . import migration

        if "migration" in c:
            migration.verify_checkpoint(path, data, book)
        elif migration.SOURCE_ID in book.snapshot["evidence"]:
            raise ValueError("legacy provenance requires explicit migration input")
        if (
            not isinstance(data["attempts"], list)
            or len(set(data["attempts"])) != len(data["attempts"])
            or any(not Path(p).is_absolute() for p in data["attempts"])
        ):
            raise ValueError("invalid or duplicate native attempt paths")
        return data
    except (ValueError, KeyError, TypeError, OSError) as exc:
        raise ValueError(
            f"corrupt audit checkpoint at {path}: {exc}; repair explicitly"
        ) from exc


def ensure_quiescent(attempts: list[str]) -> None:
    """Resume only after the existing runner has confirmed process cleanup."""
    from opsproc import process_identity

    for directory in attempts:
        value = strict_json((Path(directory) / "result.json").read_bytes())
        for name in ("runner", "process"):
            identity = value.get(name)
            if identity is not None and process_identity(identity["pid"]) == identity:
                raise ValueError(
                    "previous audit runner/CLI is still active; use the existing runner's status/stop before resuming"
                )
        if value.get("cleanup_complete") is not True:
            raise ValueError(
                "previous audit cleanup is unconfirmed; reconcile the existing runner before resuming"
            )


def interrupt_tasks(book: Ledger, reason: str) -> None:
    for task in book.snapshot["tasks"].values():
        if task["status"] == "running":
            book.record_task(
                {
                    **task,
                    "status": "interrupted",
                    "finished_at": timestamp(),
                    "conclusion": reason,
                    "unconfirmed": [
                        {
                            "scope": " / ".join(task["scope"]),
                            "reason": reason,
                            "resume": "同じタスクIDの新しいnative子で再開",
                        }
                    ],
                },
                at=timestamp(),
            )


def capture_unreceived(book: Ledger, runtime_dir: Path) -> None:
    """Keep a spawn interrupted before MCP binding traceable, without declaring success."""
    eligible = {
        key: row
        for key, row in book.snapshot["tasks"].items()
        if row["status"] in {"planned", "failed", "interrupted"}
    }
    if not eligible:
        return
    try:
        observed = native.inspect_run(runtime_dir)
    except native.NativeEvidenceError:
        return  # verify() reports the missing evidence; it is never accepted here.
    history = strict_json(book.path.read_bytes())["events"]
    bound = {e["data"]["agent_id"] for e in history if e["kind"] == "task.recorded"}
    for child in observed.agents:
        key = child.task_name.removeprefix("/root/")
        if child.agent_id not in bound and key in eligible:
            book.record_task(
                {
                    **eligible[key],
                    "status": "running",
                    "agent_id": child.agent_id,
                    "started_at": child.started_at,
                    "finished_at": None,
                    "conclusion": "",
                    "evidence": [],
                    "next_questions": [],
                    "unconfirmed": [],
                },
                at=timestamp(),
            )


def _failed_before_session(path: Path) -> bool:
    value = strict_json((path / "result.json").read_bytes())
    return bool(
        type(value.get("schema_version")) is int
        and value["schema_version"] == 1
        and value.get("contract", {}).get("mode") == "read-only"
        and value.get("contract", {}).get("settings", {}).get("agent") == "codex"
        and value.get("result") == "failed"
        and value.get("cleanup_complete") is True
        and value.get("runtime", {}).get("session_id") is None
        and value.get("turn_end_reason")
        in {"startup_failed", "cli_error", "usage_limit"}
    )


def receipts(path: Path, *, slot: str, base_sha: str) -> dict[str, dict[str, Any]]:
    """Recheck stored content hashes; never replace missing receipts with live data."""
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    try:
        fcntl.flock(fd, fcntl.LOCK_SH)
        with os.fdopen(os.dup(fd), "rb") as reader:
            rows = reader.read()
    finally:
        os.close(fd)
    if not rows.endswith(b"\n"):
        raise ValueError("retrieval journal is truncated")
    records = [strict_json(line) for line in rows.splitlines()]
    context = records[0]
    if (
        context.get("event") != "context"
        or context.get("round_id") != slot
        or context.get("base_sha") != base_sha
    ):
        raise ValueError("retrieval journal round/base mismatch")
    requests: dict[str, dict[str, Any]] = {}
    responses = {}
    for row in records[1:]:
        key = row["receipt"]
        if row["event"] == "requested":
            if key in requests:
                raise ValueError("duplicate retrieval request")
            requests[key] = row
        elif row["event"] == "received":
            response = row["response"]
            from .retrieval import encode

            if (
                key not in requests
                or key in responses
                or response["receipt"] != key
                or response["round_id"] != slot
                or response["base_sha"] != base_sha
                or hashlib.sha256(encode(response)).hexdigest()
                != row["response_sha256"]
            ):
                raise ValueError("retrieval receipt/hash mismatch")
            responses[key] = {"request": requests[key], "response": response}
        else:
            raise ValueError("unknown retrieval journal event")
    return responses


def verify(
    book: Ledger, attempts: list[str], journal: Path, max_threads: int
) -> tuple[list[str], set[str]]:
    """Cross-check every current success and retain older attempts as evidence."""
    errors = []
    from . import migration

    imported_tasks: set[str] = set()
    imported_evidence: set[str] = set()
    try:
        imported_tasks, imported_evidence = migration.verify_import(book)
    except (ValueError, KeyError, TypeError, OSError) as exc:
        errors.append(f"旧checkpoint移行根拠の照合: {exc}")
    observed: dict[str, native.NativeAgent] = {}
    history = strict_json(book.path.read_bytes())["events"]
    owners: dict[str, str] = {}
    for event in history:
        if event["kind"] == "task.recorded" and event["data"]["agent_id"]:
            row = event["data"]
            key = row["agent_id"]
            if key in owners and owners[key] != row["id"]:
                errors.append("native ID is assigned to multiple ledger tasks")
            owners[key] = row["id"]
    for path in attempts:
        try:
            run = native.inspect_run(Path(path))
            run_agents = {}
            for child in run.agents:
                if child.agent_id in observed or owners.get(
                    child.agent_id
                ) != child.task_name.removeprefix("/root/"):
                    raise native.NativeEvidenceError(
                        "unassigned/duplicate native launch or task name mismatch"
                    )
                run_agents[child.agent_id] = child
            intervals = [(datetime.fromisoformat(c.started_at), 1) for c in run.agents]
            intervals += [
                (datetime.fromisoformat(c.completed_at), -1)
                for c in run.agents
                if c.completed_at
            ]
            active = 0
            for _, delta in sorted(intervals):
                active += delta
                if active > max_threads:
                    raise native.NativeEvidenceError(
                        "native concurrency exceeds OPS_REPORT_MAX_THREADS"
                    )
            observed.update(run_agents)
        except native.NativeEvidenceError as exc:
            # A prior startup failure remains recorded as a failed attempt, but
            # does not invalidate a later run that actually completed the work.
            if path == attempts[-1] or not _failed_before_session(Path(path)):
                errors.append(str(exc))
    tasks = book.snapshot["tasks"]
    invalid_tasks = set()
    for key, row in tasks.items():
        if key in imported_tasks:
            continue
        current_child = observed.get(row["agent_id"])
        if row["status"] == "completed" and (
            current_child is None or current_child.status != "completed"
        ):
            errors.append(
                f"{key}: completed task has no completed native execution evidence"
            )
            invalid_tasks.add(key)
    if not observed and not imported_tasks:
        errors.append("native起動0件: 実行証拠を確認できない")
    invalid_evidence = set()
    try:
        acquired = receipts(
            journal, slot=book.snapshot["slot"], base_sha=book.snapshot["base_sha"]
        )
        for key, row in book.snapshot["evidence"].items():
            if key in imported_evidence:
                continue
            if row["source_kind"] == "isolated-cpu-test":
                try:
                    usable = verify_cpu_evidence(row, journal, acquired)
                    affected = {
                        f["id"]
                        for f in book.snapshot["findings"].values()
                        if f["status"] in {"confirmed", "refuted"}
                        and key in [*f["evidence"], *f["counterevidence"]]
                    }
                    if not usable and affected:
                        raise ValueError("CPU test remains needs-reproduction")
                except (
                    OSError,
                    ValueError,
                    KeyError,
                    TypeError,
                    RetrievalError,
                ) as exc:
                    invalid_evidence.add(key)
                    errors.append(f"{key}: 隔離テストの根拠が未確認/不一致: {exc}")
                continue
            receipt = row["reference"].rsplit(" / receipt=", 1)[-1]
            found = acquired.get(receipt)
            if (
                found is None
                or not found["response"]["ok"]
                or evidence_record(key, found) != row
            ):
                invalid_evidence.add(key)
                errors.append(
                    f"{key}: 根拠の取得記録が欠落/不一致。原本receiptを復元して同じ回で再開する"
                )
    except (OSError, ValueError, KeyError, TypeError) as exc:
        invalid_evidence.update(book.snapshot["evidence"])
        errors.append(f"取得記録の照合: {exc}")
    invalid_findings = {
        key
        for key, f in book.snapshot["findings"].items()
        if f["task_id"] in invalid_tasks
        or invalid_evidence.intersection([*f["evidence"], *f["counterevidence"]])
    }
    return errors, invalid_findings


def verify_cpu_evidence(
    row: dict[str, Any], journal: Path, acquired: dict[str, dict[str, Any]]
) -> bool:
    """Verify the shared executor's manifest/log without running a test again."""
    from lib.review_tools.reproduction import RUN_ID, digest, recorded_bytes

    run_id = row["id"].removeprefix("cpu-test-")
    if not row["id"].startswith("cpu-test-") or not RUN_ID.fullmatch(run_id):
        raise ValueError("invalid CPU evidence ID")
    directory = journal.parent / "reproduction" / run_id
    manifest = directory / "result.json"
    if directory.is_symlink() or manifest.is_symlink():
        raise ValueError("CPU record must not be a symlink")
    raw = recorded_bytes(manifest)
    result = strict_json(raw)
    reference = manifest.as_uri() + "#sha256=" + digest(raw)
    expected = {
        "id": "cpu-test-" + run_id,
        "source_kind": "isolated-cpu-test",
        "reference": reference,
        "acquired_at": result["finished_at"],
        "target_version": result["base_sha"],
        "pr_head": None,
    }
    linked = [
        acquisition
        for acquisition in acquired.values()
        if acquisition["request"]["tool"] == "cpu_tests"
        and any(
            s.get("reference") == reference for s in acquisition["response"]["sources"]
        )
    ]
    if (
        row != expected
        or len(linked) != 1
        or result["run_id"] != run_id
        or result["slot"] != linked[0]["response"]["round_id"]
        or result["base_sha"] != linked[0]["response"]["base_sha"]
    ):
        raise ValueError("CPU manifest/ledger/receipt identity mismatch")
    if (
        result["log_sha256"] is not None
        and digest(recorded_bytes(directory / "pytest.log")) != result["log_sha256"]
    ):
        raise ValueError("CPU test log content changed")
    usable = bool(result["disposition"] == "test-result")
    if usable and (
        not linked[0]["response"]["ok"]
        or result["status"] != "passed"
        or result["cleanup"] != "removed-clean"
        or result["source_before"] != result["source_after"]
        or result["process"]["reason"] != "exited"
        or result["process"]["exit_code"] != 0
        or result["log_sha256"] is None
    ):
        raise ValueError("CPU test completion/cleanup is unverified")
    return usable


def evidence_record(key: str, acquisition: dict[str, Any]) -> dict[str, Any]:
    response = acquisition["response"]
    sources = response["sources"]
    versions = {s["sha"] for s in sources if s.get("kind") == "git"}
    heads = {s["pr_head_sha"] for s in sources if s.get("pr_head_sha")}
    if len(versions) > 1 or len(heads) > 1:
        raise ValueError("a ledger evidence record must identify a single version")
    args = acquisition["request"]["arguments"]
    return {
        "id": key,
        "source_kind": "code"
        if versions == {response["base_sha"]}
        else "record"
        if versions
        else "github",
        "reference": " / ".join(s["url"] for s in sources)
        + " / range="
        + json.dumps(args, ensure_ascii=False, sort_keys=True)
        + " / receipt="
        + response["receipt"],
        "acquired_at": response["retrieved_at"],
        "target_version": next(
            iter(versions),
            next(
                (s["head_sha"] for s in sources if s.get("head_sha")),
                response["base_sha"],
            ),
        ),
        "pr_head": next(iter(heads), None),
    }


def finish(
    book: Ledger,
    ending: dict[str, Any] | None,
    errors: list[str],
    invalid_findings: set[str],
) -> bool:
    """Only the host ends the round, after execution and source verification."""
    for key in invalid_findings:
        finding = book.snapshot["findings"][key]
        if finding["status"] in {"confirmed", "refuted"}:
            book.record_finding({**finding, "status": "unverified"}, at=timestamp())
    interrupt_tasks(book, "親終了時に子の結果が未受領（親中断または子の失敗）")
    if ending is None:
        errors.append("親の終了理由が未記録")
        ending = {
            "status": "incomplete",
            "reason": "親の調査が中断した",
            "checked_scope": [],
            "excluded_scope": [],
            "unverified_risks": [],
            "further_investigation": [],
        }
    if errors:
        ending = {
            **ending,
            "status": "incomplete",
            "reason": ending["reason"] + " / " + "; ".join(errors),
            "unverified_risks": [
                *ending["unverified_risks"],
                *[
                    {
                        "scope": "実行・取得根拠の照合",
                        "reason": e,
                        "resume": "原本記録を確認し report --if-due で同じissueを再開",
                    }
                    for e in errors
                ],
            ],
        }
    handoff.record_proposal_handoffs(book, at=timestamp())
    book.end(ending, at=timestamp())
    return bool(ending["status"] == "completed")
