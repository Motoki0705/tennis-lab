"""Pinned audit plans and validation of native scout results and integration.

The parent CLI is the only publisher. Untrusted or absent scout results reduce
coverage; they never make a scheduled run successful. A retry retains validated
results and asks only the outstanding scouts (or just the parent integration).
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from . import agent, shell

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


@contextmanager
def checkout(repo: Path, base: str, path: Path) -> Iterator[Path]:
    """Disposable detached checkout; every native scout inherits this same cwd.

    The sandbox enforces read-only access. Never reset or forcibly remove a tree
    changed by a faulty runner; leave it for explicit recovery instead.
    """
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
    frozen = {
        k: v
        for k, v in collected.items()
        if k not in {"audit_input_sha256", "audit_runtime"}
    }
    return hashlib.sha256(
        json.dumps(frozen, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()


def load_checkpoint(path: Path, repo: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    try:
        data = json.loads(
            path.read_text(encoding="utf-8"), object_pairs_hook=agent._unique_keys
        )
        if (
            not isinstance(data, dict)
            or set(data) != {"version", "collected", "results"}
            or type(data["version"]) is not int
            or data["version"] != 1
        ):
            raise ValueError("invalid audit checkpoint schema/version")
        collected, results = data["collected"], data["results"]
        if input_digest(collected) != collected["audit_input_sha256"]:
            raise ValueError("audit checkpoint input hash mismatch")
        if collected["audit_plan"] != plan(repo, collected["meta"]["base_sha"]):
            raise ValueError("audit checkpoint plan/base mismatch")
        if not isinstance(results, dict) or not results.keys() <= AREAS.keys():
            raise ValueError("invalid audit checkpoint assignments")
        for key, row in results.items():
            _result(row, key, collected["meta"]["base_sha"])
        return collected, results
    except (ValueError, KeyError, TypeError, AttributeError) as exc:
        raise ValueError(
            f"corrupt audit checkpoint at {path}: {exc}; repair explicitly"
        ) from exc


def _text(row: dict[str, Any], key: str) -> str:
    return agent._req_str(row, key, "audit")


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


def _overlap(results: dict[str, dict[str, Any]]) -> bool:
    intervals = [
        (
            datetime.fromisoformat(row["started_at"]),
            datetime.fromisoformat(row["finished_at"]),
        )
        for row in results.values()
        if row["status"] == "completed"
    ]
    return any(
        max(a[0], b[0]) < min(a[1], b[1])
        for i, a in enumerate(intervals)
        for b in intervals[i + 1 :]
    )


def requested(results: dict[str, dict[str, Any]]) -> list[str]:
    keys = [
        key
        for key in AREAS
        if key not in results or results[key]["status"] != "completed"
    ]
    if not _overlap(results):
        # A serial or single-result attempt needs two scouts again to establish
        # concurrency. Otherwise a retry could remain incomplete forever.
        keys += [key for key in AREAS if key not in keys][: max(0, 2 - len(keys))]
    return keys


@dataclass
class Audit:
    results: dict[str, dict[str, Any]]
    errors: list[str]
    decisions: list[dict[str, Any]]
    complete: bool
    analysis: agent.AgentReport

    def record(self) -> dict[str, Any]:
        return {k: v for k, v in asdict(self).items() if k != "analysis"}


def integrate(
    text: str,
    plan: dict[str, Any],
    stale: set[int],
    *,
    previous: dict[str, Any] | None = None,
    failure: str | None = None,
) -> Audit:
    """Validate each result independently and keep only traceable proposals.

    A malformed parent integration suppresses its proposals/memory, while valid
    scout conclusions remain publishable. Duplicate assignment returns invalidate
    that assignment, rather than arbitrarily choosing a winner.
    """
    results = dict(previous or {})
    for key, row in results.items():
        if key not in AREAS:
            raise ValueError("checkpoint contains an unknown assignment")
        _result(row, key, plan["base_sha"])
    pending = set(requested(results))
    errors = []
    try:
        if failure:
            raise ValueError(failure)
        data = agent._extract_json(text)
        if not isinstance(data, dict) or set(data) != {
            "summary",
            "proposals",
            "stale_pr_notes",
            "memory_update",
            "assignments",
            "decisions",
        }:
            raise ValueError("invalid parent JSON fields")
        rows = data["assignments"]
        if not isinstance(rows, list):
            raise ValueError("assignments must be a list")
    except ValueError as exc:
        data, rows = {}, []
        errors.append(f"親の実行/返却: {exc}")
    counts = Counter(
        row.get("id")
        for row in rows
        if isinstance(row, dict) and isinstance(row.get("id"), str)
    )
    if any(
        not isinstance(row, dict)
        or not isinstance(row.get("id"), str)
        or row["id"] not in pending
        for row in rows
    ):
        errors.append("担当外または不正な返却（再利用済み担当の再返却を含む）")
    for key in sorted(pending):
        results.pop(key, None)
        if counts[key] != 1:
            errors.append(f"{key}: 返却{counts[key]}件（欠落/重複）")
            continue
        try:
            results[key] = _result(
                next(
                    row
                    for row in rows
                    if isinstance(row, dict) and row.get("id") == key
                ),
                key,
                plan["base_sha"],
            )
        except (ValueError, TypeError) as exc:
            errors.append(f"{key}: 不正な返却: {exc}")
    identities = [row["agent_id"] for row in results.values()]
    if len(identities) != len(set(identities)):
        repeated = {value for value, count in Counter(identities).items() if count > 1}
        for key in list(results):
            if results[key]["agent_id"] in repeated:
                errors.append(f"{key}: native agent_idの重複")
                del results[key]
    done = {key: row for key, row in results.items() if row["status"] == "completed"}
    for key in AREAS.keys() - done.keys():
        if key in results:
            errors.append(
                f"{key}: {results[key]['status']}: "
                + "; ".join(
                    f"{gap['scope']}: {gap['reason']} / 再開: {gap['resume']}"
                    for gap in results[key]["unconfirmed"]
                )
            )
    if not _overlap(done):
        errors.append("2つ以上のnative scoutの実行時間が重なる記録がない")
    decisions: list[dict[str, Any]] = []
    analysis = agent.AgentReport(
        "確認済みの担当結果を以下に掲載。親の統合は未完了。", [], {}, None
    )
    try:
        findings = {
            f"{key}:{f['topic']}": f
            for key, row in done.items()
            for f in row["findings"]
        }
        raw_decisions = data.get("decisions")
        if not isinstance(raw_decisions, list):
            raise ValueError("decisions must be a list")
        seen: set[str] = set()
        publishable = []
        for decision in raw_decisions:
            if not isinstance(decision, dict) or set(decision) != {
                "sources",
                "outcome",
                "reason",
                "evidence",
            }:
                raise ValueError("invalid integration decision fields")
            sources = _strings(decision["sources"], "decision.sources", empty=False)
            if not set(sources) <= findings.keys() or seen.intersection(sources):
                raise ValueError(
                    "integration refers to missing/invalid/duplicate findings"
                )
            seen.update(sources)
            group = [findings[s] for s in sources]
            topic = group[0]["topic"]
            if any(f["topic"] != topic for f in group) or set(sources) != {
                s for s, f in findings.items() if f["topic"] == topic
            }:
                raise ValueError("same topic must be integrated together exactly once")
            outcome = decision["outcome"]
            _text(decision, "reason")
            _text(decision, "evidence")
            if outcome not in {
                "propose",
                "duplicate",
                "resolved",
                "existing_issue",
                "deferred",
                "conflict",
            }:
                raise ValueError("invalid integration outcome")
            opposed = len({f["position"] for f in group}) > 1
            if opposed and outcome not in {"resolved", "conflict"}:
                raise ValueError(
                    "contradictory findings require resolved evidence or conflict"
                )
            if outcome in {"resolved", "conflict"} and not opposed:
                raise ValueError("resolved/conflict must refer to opposing findings")
            if outcome == "duplicate" and len(group) < 2:
                raise ValueError("duplicate requires multiple findings")
            if outcome == "conflict":
                errors.append(
                    f"未解決の矛盾 {topic}: {decision['reason']} / 再開: 親が根拠を照合して再統合"
                )
            if outcome in {"propose", "duplicate", "resolved"}:
                if len(group) > 1 and not opposed and outcome != "duplicate":
                    raise ValueError(
                        "duplicate findings require one duplicate decision"
                    )
                publishable.append(frozenset(sources))
            decisions.append(decision)
        if seen != findings.keys():
            raise ValueError("integration is missing scout findings")
        props = data["proposals"]
        if not isinstance(props, list):
            raise ValueError("proposals must be a list")
        links = [
            frozenset(_strings(p.get("sources"), "proposal.sources", empty=False))
            for p in props
            if isinstance(p, dict)
        ]
        if len(links) != len(props) or Counter(links) != Counter(publishable):
            raise ValueError("proposals must match publishable decisions exactly once")
        titles = [" ".join(_text(p, "title").split()).casefold() for p in props]
        if len(titles) != len(set(titles)):
            raise ValueError("duplicate proposal titles must be integrated")
        notes = data["stale_pr_notes"]
        if not isinstance(notes, dict) or any(not k.isdigit() for k in notes):
            raise ValueError("invalid stale_pr_notes")
        note_keys = {int(k) for k in notes}
        if not note_keys <= stale:
            raise ValueError("unexpected stale PR notes")
        if note_keys != stale:
            errors.append(f"放置PRの未確認: {sorted(stale - note_keys)}")
        aggregate = {
            k: data[k]
            for k in ("summary", "proposals", "stale_pr_notes", "memory_update")
        }
        analysis = agent.parse_agent_output(json.dumps(aggregate), note_keys)
        # This source link is script-written and survives Markdown neutralization.
        analysis = agent.AgentReport(
            analysis.summary,
            [
                agent.Proposal(
                    **{
                        **asdict(prop),
                        "evidence": prop.evidence
                        + " / 担当: "
                        + ", ".join(raw["sources"])
                        + f" / base={plan['base_sha']}",
                    }
                )
                for prop, raw in zip(analysis.proposals, props, strict=True)
            ],
            analysis.stale_pr_notes,
            analysis.memory_update,
        )
    except (ValueError, TypeError, KeyError) as exc:
        errors.append(f"親の統合: {exc}")
    complete = len(done) == len(AREAS) and not errors
    if not complete:
        # Defer *all* memory writes until complete, including summary/date updates.
        analysis = agent.AgentReport(
            analysis.summary, analysis.proposals, analysis.stale_pr_notes, None
        )
    return Audit(results, errors, decisions, complete, analysis)
