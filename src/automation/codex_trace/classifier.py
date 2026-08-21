"""Deterministic baseline semantic clustering for Codex inference steps."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ClusterRule:
    """Weighted lexical and tool-action signals for one semantic cluster."""

    keywords: tuple[str, ...]
    tool_kinds: tuple[str, ...] = ()


@dataclass(frozen=True)
class Classification:
    """Normalized cluster probabilities and the evidence used to derive them."""

    probabilities: dict[str, float]
    evidence_mode: str

    @property
    def primary_cluster(self) -> str:
        """Return the most probable cluster with deterministic tie-breaking."""

        return min(
            self.probabilities,
            key=lambda cluster: (-self.probabilities[cluster], cluster),
        )

    @property
    def primary_probability(self) -> float:
        """Return the primary cluster probability."""

        return self.probabilities[self.primary_cluster]


DEFAULT_RULES: dict[str, ClusterRule] = {
    "task_understanding": ClusterRule(
        keywords=(
            "requirement",
            "request",
            "acceptance",
            "scope",
            "constraint",
            "要件",
            "依頼",
            "制約",
        )
    ),
    "planning": ClusterRule(
        keywords=(
            "plan",
            "approach",
            "strategy",
            "design",
            "architecture",
            "decide",
            "計画",
            "設計",
            "方針",
        )
    ),
    "codebase_exploration": ClusterRule(
        keywords=(
            "inspect",
            "search",
            "read",
            "locate",
            "find",
            "source",
            "schema",
            "readme",
            "調査",
            "探索",
            "確認",
        ),
        tool_kinds=("exec_command", "web", "mcp"),
    ),
    "implementation": ClusterRule(
        keywords=(
            "implement",
            "edit",
            "patch",
            "create",
            "refactor",
            "update",
            "write",
            "実装",
            "編集",
            "追加",
        ),
        tool_kinds=("apply_patch", "image_generation"),
    ),
    "verification": ClusterRule(
        keywords=(
            "test",
            "pytest",
            "lint",
            "mypy",
            "verify",
            "validate",
            "check",
            "テスト",
            "検証",
        )
    ),
    "debugging": ClusterRule(
        keywords=(
            "error",
            "failed",
            "failure",
            "bug",
            "debug",
            "diagnose",
            "traceback",
            "fix",
            "エラー",
            "失敗",
            "修正",
        )
    ),
    "coordination": ClusterRule(
        keywords=(
            "agent",
            "delegate",
            "wait",
            "message",
            "worktree",
            "エージェント",
            "委譲",
            "待機",
        ),
        tool_kinds=(
            "spawn_agent",
            "assign_agent_task",
            "send_message",
            "wait_agent",
            "close_agent",
        ),
    ),
    "reporting": ClusterRule(
        keywords=(
            "summarize",
            "summary",
            "report",
            "explain",
            "final",
            "報告",
            "要約",
            "説明",
        )
    ),
}


class SemanticClassifier:
    """Classify inference-level reasoning with visible text and action signals."""

    def __init__(self, rules: dict[str, ClusterRule] | None = None) -> None:
        selected = DEFAULT_RULES if rules is None else rules
        if not selected:
            raise ValueError("at least one semantic cluster rule is required")
        if "other" in selected:
            raise ValueError("'other' is reserved for unmatched evidence")
        self._rules = dict(selected)

    @classmethod
    def from_json(cls, path: Path) -> SemanticClassifier:
        """Load an explicit cluster taxonomy from a JSON file."""

        raw: Any = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict) or not isinstance(raw.get("clusters"), dict):
            raise ValueError("cluster rules must contain an object-valued 'clusters'")
        rules: dict[str, ClusterRule] = {}
        for name, value in raw["clusters"].items():
            if not isinstance(name, str) or not isinstance(value, dict):
                raise ValueError("each cluster must map a string name to an object")
            keywords = _string_list(value.get("keywords", []), f"{name}.keywords")
            tool_kinds = _string_list(value.get("tool_kinds", []), f"{name}.tool_kinds")
            if not keywords and not tool_kinds:
                raise ValueError(f"cluster {name!r} has no classification signals")
            rules[name] = ClusterRule(keywords=keywords, tool_kinds=tool_kinds)
        return cls(rules)

    @property
    def cluster_names(self) -> tuple[str, ...]:
        """Return configured cluster names in deterministic order."""

        return tuple(sorted((*self._rules, "other")))

    def classify(
        self,
        text: str,
        *,
        tool_kinds: tuple[str, ...] = (),
        action_text: str = "",
    ) -> Classification:
        """Return a probability distribution without pretending to be a model label."""

        searchable = f"{text}\n{action_text}".casefold()
        normalized_tools = {kind.casefold() for kind in tool_kinds}
        scores: dict[str, float] = {}
        for name, rule in self._rules.items():
            keyword_hits = sum(
                1.0 for keyword in rule.keywords if keyword.casefold() in searchable
            )
            tool_hits = sum(
                3.0 for kind in rule.tool_kinds if kind.casefold() in normalized_tools
            )
            score = keyword_hits + tool_hits
            if score > 0:
                scores[name] = score

        if not scores:
            evidence_mode = "content_unmatched" if text.strip() else "unclassified"
            return Classification({"other": 1.0}, evidence_mode)

        total = sum(scores.values())
        probabilities = {name: score / total for name, score in sorted(scores.items())}
        evidence_mode = "reasoning_text" if text.strip() else "response_action"
        return Classification(probabilities, evidence_mode)


def _string_list(value: Any, label: str) -> tuple[str, ...]:
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise ValueError(f"{label} must be an array of strings")
    return tuple(value)
