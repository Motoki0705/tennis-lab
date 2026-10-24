"""Explicit, recoverable v1 checkpoint migration; never invoked as a fallback."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import re
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any

from . import audit, collect, legacy_checkpoint, proposals, schedule, shell
from .ledger import Ledger
from .retrieval import strict_json

SOURCE_ID = "legacy_checkpoint"


def archive_path(checkpoint: Path) -> Path:
    return checkpoint.with_suffix(".migration")


def digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def encode(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    ).encode()


def read(path: Path) -> bytes:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"missing/non-regular migration file: {path}")
    return path.read_bytes()


def _sync_directory(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def atomic_bytes(path: Path, raw: bytes) -> None:
    """Publish complete files, including the checkpoint commit point, durably."""
    if path.is_symlink():
        raise ValueError(f"refusing a symlink: {path}")
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        try:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
            temporary.replace(path)
            _sync_directory(path.parent)
        finally:
            temporary.unlink(missing_ok=True)


def _issue(repo: Path, slot: str, number: int) -> dict[str, Any]:
    matches = [
        r for r in collect.collect_report_history(repo)["reports"] if r["slot"] == slot
    ]
    if len(matches) != 1 or matches[0]["number"] != number:
        raise ValueError(
            "report issue is missing, duplicated or belongs to another slot"
        )
    row = matches[0]
    if row["state"] != "OPEN" or row["status"] != "incomplete":
        raise ValueError("migration requires the original open, incomplete report")
    _, parsed = proposals.parse_report(row["body"])
    region = (
        row["body"]
        .split(proposals.PROPOSALS_BEGIN)[1]
        .split(proposals.PROPOSALS_END)[0]
    )
    if any(
        p.child or p.head_checked or any(v for _, v in p.options) for p in parsed
    ) or re.search(r"(?m)^\s*[-*+] \[[xX]\] ", region):
        raise ValueError("report has selections/children; reconcile them explicitly")
    return {k: row[k] for k in ("number", "url", "author", "title", "body")}


def _check_issue(row: dict[str, Any], collected: dict[str, Any], slot: str) -> None:
    meta = proposals.parse_report_meta(row["body"])
    c = collected["meta"]
    if (
        meta.slot != slot
        or meta.base != c["base_sha"]
        or meta.audit != collected["audit_input_sha256"]
        or meta.since != datetime.fromisoformat(c["period_start"])
        or meta.until != datetime.fromisoformat(c["period_end"])
    ):
        raise ValueError("report metadata and pinned checkpoint input disagree")


def _collected(
    source: dict[str, Any], sha: str, issue: dict[str, Any], repository: str
) -> dict[str, Any]:
    old = source["collected"]
    # v1 had no repository field and wrapped history in {reports: [...]}. This
    # one-time conversion is recorded with BOTH digests, never hidden in run().
    value = {
        "meta": {**old["meta"], "repository": repository},
        "ops_memory": copy.deepcopy(old["ops_memory"]),
        "previous_reports": copy.deepcopy(old["previous_reports"]["reports"]),
        "migration": {
            "checkpoint_sha256": sha,
            "input_sha256": old["audit_input_sha256"],
            "report_issue": issue["number"],
        },
    }
    value["audit_input_sha256"] = audit.input_digest(value)
    return value


def records(
    source: dict[str, Any], sha: str, at: str
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    """Map validated v1 results without inventing v2 acquisition/native receipts."""
    c = source["collected"]
    base = c["meta"]["base_sha"]
    evidence: dict[str, Any] = {}
    tasks: dict[str, Any] = {}
    findings = []

    def add(key: str, kind: str, reference: str, acquired: str) -> str:
        evidence[key] = {
            "id": key,
            "source_kind": kind,
            "reference": reference,
            "acquired_at": acquired,
            "target_version": base,
            "pr_head": None,
        }
        return key

    add(
        SOURCE_ID,
        "legacy-checkpoint",
        f"checkpoint-v1 sha256={sha} input_sha256={c['audit_input_sha256']}",
        at,
    )
    assignments = {a["id"]: a for a in c["audit_plan"]["assignments"]}
    for area, row in sorted(source["results"].items()):
        if row["status"] != "completed":
            continue
        key = f"legacy_{area}"
        origin = add(
            key + "_origin",
            "legacy-assignment",
            f"checkpoint-v1 sha256={sha} / results/{area}",
            row["finished_at"],
        )

        def observations(
            field: str,
            texts: list[str],
            *,
            key: str = key,
            area: str = area,
            finished_at: str = row["finished_at"],
        ) -> list[str]:
            return [
                add(
                    f"{key}_{field}_{i}",
                    "legacy-observation",
                    f"{text} / checkpoint-v1 sha256={sha} / results/{area}/{field}/{i}",
                    finished_at,
                )
                for i, text in enumerate(texts)
            ]

        refs = observations("evidence", row["evidence"])
        counters = observations("counterevidence", row["counterevidence"])
        assignment = assignments[area]
        tasks[key] = {
            "id": key,
            "question": assignment["purpose"],
            "scope": [
                *assignment["targets"],
                *["input:" + s for s in assignment["inputs"]],
            ],
            "origin": {"kind": "legacy", "id": origin},
            "agent_id": row["agent_id"],
            "status": "completed",
            "started_at": row["started_at"],
            "finished_at": row["finished_at"],
            "conclusion": row["conclusion"],
            "evidence": [origin, *refs, *counters],
            "next_questions": [],
            "unconfirmed": [],
        }
        refs = observations("findings", [f["evidence"] for f in row["findings"]])
        for finding, ref in zip(row["findings"], refs, strict=True):
            findings.append(
                {
                    "id": f"{key}_{finding['topic']}",
                    "task_id": key,
                    "kind": "legacy-observation",
                    "status": "unverified",
                    "conclusion": f"{finding['position']}: {finding['conclusion']}",
                    "evidence": [ref],
                    "counterevidence": counters,
                    "details": {},
                }
            )
    return evidence, tasks, findings


def _build_ledger(
    path: Path, source: dict[str, Any], sha: str, slot: str, number: int, at: str
) -> Ledger:
    evidence, tasks, findings = records(source, sha, at)
    start = min(
        [
            at,
            source["collected"]["meta"]["generated_at"],
            *[t["started_at"] for t in tasks.values()],
        ],
        key=datetime.fromisoformat,
    )
    book = Ledger.create(
        path,
        slot=slot,
        base_sha=source["collected"]["meta"]["base_sha"],
        goal=audit.GOAL,
        report_issue=number,
        at=start,
    )
    for row in evidence.values():
        book.record_evidence(row, at=at)
    for row in tasks.values():
        planned = {
            **row,
            "status": "planned",
            "agent_id": None,
            "started_at": None,
            "finished_at": None,
            "conclusion": "",
            "evidence": [],
        }
        book.record_task(planned, at=at)
        book.record_task(
            {
                **planned,
                "status": "running",
                "agent_id": row["agent_id"],
                "started_at": row["started_at"],
            },
            at=at,
        )
        book.record_task(row, at=at)
    for row in findings:
        book.record_finding(row, at=at)
    book.end(
        {
            "status": "incomplete",
            "reason": "固定担当checkpointから明示移行。旧契約で検証済みの担当結果を保存し、残る探索と発見の統合を同じ回で再開する。",
            "checked_scope": [
                legacy_checkpoint.AREAS[k]
                for k, v in source["results"].items()
                if v["status"] == "completed"
            ],
            "excluded_scope": [],
            "unverified_risks": [
                {
                    "scope": "旧回の未確認範囲と発見の統合",
                    "reason": "移行は全体調査の完了を証明しない",
                    "resume": "report --if-due で同じslot・基準版・issueを再開",
                }
            ],
            "further_investigation": [
                "旧担当の起点と根拠を確認し、目的から追加の問いを選ぶ"
            ],
        },
        at=at,
    )
    return book


def source_original(book: Ledger) -> dict[str, Any]:
    """Original checkpoint for bounded MCP reads and host verification."""
    archive = book.path.with_name(
        book.path.name.removesuffix(".ledger.json") + ".migration"
    )
    if archive.is_symlink():
        raise ValueError("migration archive must not be a symlink")
    raw = read(archive / "source.json")
    source = legacy_checkpoint.load(raw)
    expected = records(
        source, digest(raw), book.snapshot["evidence"][SOURCE_ID]["acquired_at"]
    )[0][SOURCE_ID]
    if book.snapshot["evidence"][SOURCE_ID] != expected:
        raise ValueError("legacy checkpoint provenance/hash mismatch")
    return source


def verify_import(book: Ledger) -> tuple[set[str], set[str]]:
    """Accept only the exact archived task/evidence records, not a source-kind bypass."""
    snapshot = book.snapshot
    if SOURCE_ID not in snapshot["evidence"]:
        if any(
            r["source_kind"].startswith("legacy-")
            for r in snapshot["evidence"].values()
        ) or any(t["origin"]["kind"] == "legacy" for t in snapshot["tasks"].values()):
            raise ValueError("legacy records have no migration provenance")
        return set(), set()
    source = source_original(book)
    archive = book.path.with_name(
        book.path.name.removesuffix(".ledger.json") + ".migration"
    )
    evidence, tasks, _ = records(
        source,
        digest(read(archive / "source.json")),
        snapshot["evidence"][SOURCE_ID]["acquired_at"],
    )
    if any(snapshot["evidence"].get(k) != v for k, v in evidence.items()) or any(
        snapshot["tasks"].get(k) != v for k, v in tasks.items()
    ):
        raise ValueError("imported legacy task/evidence changed or disappeared")
    if {
        k
        for k, v in snapshot["evidence"].items()
        if v["source_kind"].startswith("legacy-")
    } != evidence.keys() or {
        k for k, v in snapshot["tasks"].items() if v["origin"]["kind"] == "legacy"
    } != tasks.keys():
        raise ValueError("unregistered legacy task/evidence")
    return set(tasks), set(evidence)


def verify_checkpoint(checkpoint: Path, data: dict[str, Any], book: Ledger) -> None:
    """A migrated checkpoint must still match its prepared conversion and prefix."""
    archive = archive_path(checkpoint)
    prepared = strict_json(read(archive / "checkpoint.json"))
    if (
        data["collected"] != prepared["collected"]
        or data["ledger"] != prepared["ledger"]
    ):
        raise ValueError("migrated checkpoint differs from prepared input")
    source = source_original(book)
    original_issue = strict_json(read(archive / "issue-before.json"))
    _check_issue(original_issue, source["collected"], checkpoint.stem)
    expected = {
        "version": audit.VERSION,
        "collected": _collected(
            source,
            digest(read(archive / "source.json")),
            original_issue,
            data["collected"]["meta"]["repository"],
        ),
        "ledger": str(checkpoint.with_suffix(".ledger.json").absolute()),
        "attempts": [],
    }
    if prepared != expected:
        raise ValueError("prepared migration checkpoint does not match its source")
    migration = data["collected"]["migration"]
    if (
        migration["input_sha256"] != source["collected"]["audit_input_sha256"]
        or migration["checkpoint_sha256"] != digest(read(archive / "source.json"))
        or book.snapshot["run"]["report_issue"] != migration["report_issue"]
    ):
        raise ValueError("migration source/issue/digest mismatch")
    initial = Ledger.load(
        archive / "ledger.json",
        slot=checkpoint.stem,
        base_sha=book.snapshot["base_sha"],
    )
    if initial.snapshot["run"]["status"] != "incomplete":
        raise ValueError("prepared migration ledger must be incomplete")
    prefix = strict_json(read(archive / "ledger.json"))
    current = strict_json(read(book.path))
    if current["events"][: len(prefix["events"])] != prefix["events"]:
        raise ValueError("migration ledger prefix changed")
    verify_import(book)


def migrate(
    *, repo: Path, checkpoint: Path, number: int, run_dir: Path, dry_run: bool
) -> dict[str, Any]:
    """Caller holds schedule.lock then the shared issue lock for the transaction."""
    checkpoint = checkpoint.absolute()
    slot = checkpoint.stem
    if (
        not re.fullmatch(r"[0-9]{8}", slot)
        or datetime.strptime(slot, "%Y%m%d").strftime("%Y%m%d") != slot
    ):
        raise ValueError("slot must be YYYYMMDD")
    if type(number) is not int or number <= 0:
        raise ValueError("report issue must be a positive integer")
    sched = schedule.Schedule.load(checkpoint.parent.parent / "schedule.json")
    when = schedule.aware_time(f"{slot}T08:00:00+09:00")
    if when < sched.anchor or (when - sched.anchor) % schedule.PERIOD:
        raise ValueError("legacy slot is outside the original three-day schedule")
    raw = read(checkpoint)
    current = strict_json(raw)
    issue = _issue(repo, slot, number)
    archive = archive_path(checkpoint)
    if current.get("version") == 2:
        saved = audit.load_checkpoint(checkpoint, repo)
        if saved["collected"].get("migration", {}).get("report_issue") != number:
            raise ValueError("checkpoint was not migrated from this report")
        _check_issue(issue, saved["collected"], slot)
        original_issue = strict_json(read(archive / "issue-before.json"))
        if any(issue[k] != original_issue[k] for k in ("number", "url", "author")):
            raise ValueError("migrated report identity changed")
        return {"status": "already_migrated", "slot": slot, "report_issue": number}
    round_dir = checkpoint.parent / slot
    if round_dir.is_symlink() or (
        round_dir.exists() and (not round_dir.is_dir() or any(round_dir.iterdir()))
    ):
        raise ValueError("legacy round already has unexpected v2 acquisition state")
    source = legacy_checkpoint.load(raw, repo)
    sha = digest(raw)
    repository = shell.gh_json(["repo", "view", "--json", "nameWithOwner"], cwd=repo)[
        "nameWithOwner"
    ]
    if issue["url"] != f"https://github.com/{repository}/issues/{number}":
        raise ValueError("report URL does not match the repository/issue")
    if (
        "repository" in source["collected"]["meta"]
        and source["collected"]["meta"]["repository"] != repository
    ):
        raise ValueError("legacy repository mismatch")
    if archive.exists() or archive.is_symlink():
        if archive.is_symlink() or read(archive / "source.json") != raw:
            raise ValueError(
                "existing migration archive differs from the original checkpoint"
            )
        old_issue = strict_json(read(archive / "issue-before.json"))
        new_issue = strict_json(read(archive / "issue-after.json"))
        prepared = strict_json(read(archive / "checkpoint.json"))
        if old_issue["number"] != number or issue not in (old_issue, new_issue):
            raise ValueError(
                "issue changed since migration preparation; reconcile it explicitly"
            )
        _check_issue(old_issue, source["collected"], slot)
        if prepared != {
            "version": audit.VERSION,
            "collected": _collected(source, sha, old_issue, repository),
            "ledger": str(checkpoint.with_suffix(".ledger.json")),
            "attempts": [],
        }:
            raise ValueError("prepared migration input mismatch")
        _check_issue(new_issue, prepared["collected"], slot)
    else:
        _check_issue(issue, source["collected"], slot)
        collected = _collected(source, sha, issue, repository)
        prepared = {
            "version": 2,
            "collected": collected,
            "ledger": str(checkpoint.with_suffix(".ledger.json")),
            "attempts": [],
        }
        first, separator, rest = issue["body"].partition("\n")
        if any(
            f" {key}=" in first for key in ("checkpoint", "legacy_audit", "migration")
        ):
            raise ValueError(
                "report already has migration metadata without its archive"
            )
        first = first.replace(
            f"audit={source['collected']['audit_input_sha256']}",
            f"audit={collected['audit_input_sha256']}",
        )
        first = (
            first.removesuffix(" -->")
            + f" checkpoint=2 legacy_audit={source['collected']['audit_input_sha256']} migration={sha} -->"
        )
        old_issue, new_issue = issue, {**issue, "body": first + separator + rest}
        with tempfile.TemporaryDirectory(
            dir=run_dir if dry_run else checkpoint.parent, prefix=".migration-"
        ) as temporary:
            stage = Path(temporary) / "prepared"
            stage.mkdir(mode=0o700)
            for name, data in {
                "source.json": raw,
                "issue-before.json": encode(old_issue),
                "issue-after.json": encode(new_issue),
                "checkpoint.json": encode(prepared),
            }.items():
                atomic_bytes(stage / name, data)
            _build_ledger(
                stage / "ledger.json", source, sha, slot, number, audit.timestamp()
            )
            _sync_directory(stage)
            if not dry_run:
                stage.rename(archive)
                _sync_directory(archive.parent)
    result = {
        "status": "preview" if dry_run else "migrated",
        "slot": slot,
        "report_issue": number,
        "base_sha": prepared["collected"]["meta"]["base_sha"],
        "legacy_input_sha256": source["collected"]["audit_input_sha256"],
        "input_sha256": prepared["collected"]["audit_input_sha256"],
        "source_sha256": sha,
        "archive": str(archive),
        "completed_tasks": [
            f"legacy_{k}"
            for k, v in sorted(source["results"].items())
            if v["status"] == "completed"
        ],
        "resume": "report --if-due",
    }
    if dry_run:
        return result
    ledger_path = Path(prepared["ledger"])
    if ledger_path != checkpoint.with_suffix(".ledger.json"):
        raise ValueError("prepared ledger path mismatch")
    candidate = read(archive / "ledger.json")
    if ledger_path.exists() or ledger_path.is_symlink():
        if read(ledger_path) != candidate:
            raise ValueError("destination ledger already exists with different content")
    else:
        atomic_bytes(ledger_path, candidate)
    book = Ledger.load(ledger_path, slot=slot, base_sha=result["base_sha"])
    verify_checkpoint(checkpoint, prepared, book)
    body_path = run_dir / "migration-report.md"
    body_path.write_text(new_issue["body"], encoding="utf-8")
    fresh = _issue(repo, slot, number)
    if fresh not in (old_issue, new_issue):
        raise ValueError("issue changed before migration publication")
    if fresh != new_issue:
        shell.gh(
            ["issue", "edit", str(number), "--body-file", str(body_path)], cwd=repo
        )
    if _issue(repo, slot, number) != new_issue:
        raise ValueError("migration issue update is unconfirmed or changed")
    if read(checkpoint) != raw or read(ledger_path) != candidate:
        raise ValueError("checkpoint/ledger changed before migration commit")
    round_dir.mkdir(mode=0o700, exist_ok=True)
    _sync_directory(round_dir.parent)
    atomic_bytes(checkpoint, encode(prepared))
    return result
