"""Frozen v1 validation, used only by explicit migration and its provenance check.

This is not an execution path: no agent, collection, publication or v1 resume.
"""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any

from . import audit, memory, shell
from .retrieval import strict_json

AREAS = {
    "code": "コード・モデル・データの流れ",
    "platform": "開発・実験基盤・CI・依存・掃除",
    "governance": "文書・AI運用・共有/専用memory",
    "direction": "issue・PR・knowledge・研究の次の一手",
}
_KEY = re.compile(r"[a-z][a-z0-9-]*")


def _owner(path: str) -> str:
    if path.startswith("knowledge/"):
        return "direction"
    if path.startswith((".agents/", ".claude/", ".codex/", ".github/agents/")):
        return "governance"
    if path.startswith(("src/", "configs/", "config/")):
        return "code"
    if path.endswith(".md"):
        return "governance"
    return "platform"


def plan(repo: Path, base: str) -> dict[str, Any]:
    entries = []
    gitlinks = []
    for row in shell.git(["ls-tree", "-r", "-z", base], cwd=repo).split("\0"):
        if not row:
            continue
        metadata, path = row.split("\t", 1)
        mode, kind, sha = metadata.split()
        entries.append(path)
        if mode == "160000" and kind == "commit":
            gitlinks.append({"path": path, "sha": sha})
    targets = sorted(entries)
    digest = hashlib.sha256(
        json.dumps([base, targets], ensure_ascii=False).encode()
    ).hexdigest()
    return {
        "base_sha": base,
        "inventory_sha256": digest,
        "gitlinks": gitlinks,
        "assignments": [
            {
                "id": key,
                "purpose": label + "の根拠付き改善候補を調べる",
                "targets": [p for p in targets if _owner(p) == key],
                "inputs": {
                    "code": ["git", "github"],
                    "platform": ["github.ci_runs_in_period", "cleanup", "workspace"],
                    "governance": ["ai_env", "memory", "ops_memory"],
                    "direction": [
                        "github",
                        "knowledge",
                        "previous_reports",
                        "ops_memory.decisions",
                    ],
                }[key],
                "excludes": [other for other in AREAS if other != key],
                "end_condition": "担当対象と入力を精査し、結論・根拠・反証・未確認範囲・改善候補を返す。未確認が残ればcompletedにしない。",
            }
            for key, label in AREAS.items()
        ],
    }


def _text(row: dict[str, Any], key: str) -> str:
    value = row[key]
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"legacy {key}: non-empty text required")
    return value


def _strings(value: Any, name: str, *, empty: bool = True) -> list[str]:
    if (
        not isinstance(value, list)
        or (not value and not empty)
        or not all(isinstance(v, str) and v.strip() for v in value)
    ):
        raise ValueError(f"{name} must be a list of non-empty strings")
    if len(value) != len(set(value)):
        raise ValueError(f"{name} contains duplicates")
    return value


def _result(row: Any, key: str, base: str) -> dict[str, Any]:
    if not isinstance(row, dict) or set(row) != {
        "id",
        "agent_id",
        "agent_type",
        "base_sha",
        "started_at",
        "finished_at",
        "status",
        "conclusion",
        "evidence",
        "counterevidence",
        "unconfirmed",
        "findings",
    }:
        raise ValueError(f"{key}: invalid scout result fields")
    if row["id"] != key or row["base_sha"] != base or row["agent_type"] != "scout":
        raise ValueError(f"{key}: assignment/base/role mismatch")
    _text(row, "agent_id")
    start, end = (
        datetime.fromisoformat(_text(row, field))
        for field in ("started_at", "finished_at")
    )
    if start.tzinfo is None or end.tzinfo is None or end <= start:
        raise ValueError(f"{key}: invalid scout time interval")
    if row["status"] not in {"completed", "failed", "interrupted"}:
        raise ValueError(f"{key}: invalid status")
    _text(row, "conclusion")
    _strings(row["evidence"], "evidence", empty=row["status"] != "completed")
    _strings(row["counterevidence"], "counterevidence")
    gaps = row["unconfirmed"]
    if not isinstance(gaps, list):
        raise ValueError(f"{key}: unconfirmed must be a list")
    for gap in gaps:
        if not isinstance(gap, dict) or set(gap) != {"scope", "reason", "resume"}:
            raise ValueError(f"{key}: unconfirmed requires scope, reason and resume")
        for field in ("scope", "reason", "resume"):
            _text(gap, field)
    if row["status"] == "completed" and row["unconfirmed"]:
        raise ValueError(f"{key}: completed scout has unconfirmed scope")
    if row["status"] != "completed" and not row["unconfirmed"]:
        raise ValueError(f"{key}: failure must explain scope, reason and resumption")
    if not isinstance(row["findings"], list):
        raise ValueError(f"{key}: findings must be a list")
    seen = set()
    for finding in row["findings"]:
        if not isinstance(finding, dict) or set(finding) != {
            "topic",
            "position",
            "conclusion",
            "evidence",
        }:
            raise ValueError(f"{key}: invalid finding")
        topic = _text(finding, "topic")
        if not _KEY.fullmatch(topic) or topic in seen:
            raise ValueError(f"{key}: invalid/duplicate finding topic")
        seen.add(topic)
        if finding["position"] not in {"support", "oppose"}:
            raise ValueError(f"{key}: invalid finding position")
        _text(finding, "conclusion")
        _text(finding, "evidence")
    return row


def load(raw: bytes, repo: Path | None = None) -> dict[str, Any]:
    data = strict_json(raw)
    if (
        not isinstance(data, dict)
        or set(data) != {"version", "collected", "results"}
        or type(data["version"]) is not int
        or data["version"] != 1
    ):
        raise ValueError("invalid legacy checkpoint schema/version")
    c, results = data["collected"], data["results"]
    frozen = {k: v for k, v in c.items() if k != "audit_runtime"}
    if audit.input_digest(frozen) != c["audit_input_sha256"]:
        raise ValueError("legacy input hash mismatch")
    meta = c["meta"]
    base = meta["base_sha"]
    if not isinstance(base, str) or not re.fullmatch(
        r"[0-9a-f]{40}|[0-9a-f]{64}", base
    ):
        raise ValueError("legacy base must be a full commit SHA")
    times = [datetime.fromisoformat(meta[k]) for k in ("period_start", "period_end")]
    if any(t.tzinfo is None for t in times) or times[0] > times[1]:
        raise ValueError("legacy collection boundary is invalid")
    generated = datetime.fromisoformat(_text(meta, "generated_at"))
    if generated.tzinfo is None:
        raise ValueError("legacy generated_at needs a timezone")
    _text(meta, "base_ref")
    if meta["base_fetched_from"] is not None:
        _text(meta, "base_fetched_from")
    if repo is not None:
        if (
            shell.git(["rev-parse", "--verify", f"{base}^{{commit}}"], cwd=repo).strip()
            != base
        ):
            raise ValueError("legacy pinned commit is missing")
        if c["audit_plan"] != plan(repo, base):
            raise ValueError("legacy plan/base inventory mismatch")
    dedicated = c["ops_memory"]
    if dedicated["status"] not in {"ok", "not_installed"}:
        raise ValueError("legacy memory status is invalid")
    if (
        not isinstance(dedicated["files"], dict)
        or not isinstance(dedicated["decisions"], list)
        or not isinstance(dedicated["persistence"], list)
    ):
        raise ValueError(
            "legacy memory/adoption/persistence input is missing or invalid"
        )
    if (
        repo is not None
        and "base_sha" in dedicated
        and memory.read_tree(repo, dedicated["base_sha"]) != dedicated["files"]
    ):
        raise ValueError("legacy memory input does not match its pinned tree")
    if not isinstance(c["previous_reports"]["reports"], list):
        raise ValueError("legacy report history is invalid")
    if not isinstance(results, dict) or not results.keys() <= AREAS.keys():
        raise ValueError("invalid legacy checkpoint assignments")
    ids = set()
    for key, row in results.items():
        _result(row, key, base)
        if row["agent_id"] in ids:
            raise ValueError("duplicate legacy agent ID")
        ids.add(row["agent_id"])
    return data
