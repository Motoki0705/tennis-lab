"""Sync label definitions or strip bot-added area labels from open issues.

Both commands are dry-run by default. Only --apply writes to GitHub.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal
from urllib.parse import quote

import yaml

MANIFEST = Path(__file__).resolve().parents[1] / ".github/labels.yml"
AXES = ("hierarchy", "work-type", "origin", "area", "status")
LABELER_BOT = "github-actions[bot]"


@dataclass(frozen=True)
class LabelSpec:
    name: str
    color: str
    description: str
    axis: str


@dataclass(frozen=True)
class RepoLabel:
    name: str
    color: str
    description: str


@dataclass(frozen=True)
class Action:
    kind: Literal["create", "update", "delete"]
    name: str
    spec: LabelSpec | None = None

    def describe(self) -> str:
        style = (
            f" color={self.spec.color} description={self.spec.description!r}"
            if self.spec
            else ""
        )
        return f"{self.kind:<7} {self.name!r}{style}"


@dataclass(frozen=True)
class StripDecision:
    issue: int
    label: str
    remove: bool
    reason: str

    def describe(self) -> str:
        return (
            f"#{self.issue} {'remove' if self.remove else 'keep'} "
            f"{self.label!r}: {self.reason}"
        )


class ManifestError(ValueError):
    pass


def load_manifest(path: Path = MANIFEST) -> list[LabelSpec]:
    """Keep the axis as data; label names do not acquire an axis prefix."""
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or set(raw) != set(AXES):
        raise ManifestError(f"manifest must contain exactly these axes: {AXES}")
    specs: list[LabelSpec] = []
    seen: set[str] = set()
    for axis in AXES:
        group = raw[axis]
        if not isinstance(group, list):
            raise ManifestError(f"{axis}: expected a list of labels")
        for entry in group:
            if not isinstance(entry, dict) or set(entry) != {
                "name",
                "color",
                "description",
            }:
                raise ManifestError(f"{axis}: expected name/color/description")
            name, color, description = (
                entry["name"],
                entry["color"],
                entry["description"],
            )
            if not isinstance(name, str) or not name.strip():
                raise ManifestError(f"{axis}: name must be a nonempty string")
            if name.casefold() in seen:
                raise ManifestError(f"{name!r} appears more than once in the manifest")
            if not isinstance(color, str) or not re.fullmatch(r"[0-9a-fA-F]{6}", color):
                raise ManifestError(f"{name!r}: color must be 6 hex digits")
            if not isinstance(description, str):
                raise ManifestError(f"{name!r}: description must be a string")
            seen.add(name.casefold())
            specs.append(LabelSpec(name, color.upper(), description, axis))
    return specs


def plan(
    specs: Sequence[LabelSpec], existing: Sequence[RepoLabel], *, prune: bool
) -> tuple[list[Action], list[str]]:
    """Return changes and, without --prune, unmanaged labels to leave in place."""
    by_name = {label.name: label for label in existing}
    if len(by_name) != len(existing):
        raise ValueError("duplicate label names in GitHub response")
    actions: list[Action] = []
    for spec in specs:
        current = by_name.get(spec.name)
        if current is None:
            actions.append(Action("create", spec.name, spec))
        elif (current.color.upper(), current.description) != (
            spec.color.upper(),
            spec.description,
        ):
            actions.append(Action("update", spec.name, spec))
    unmanaged = sorted(by_name.keys() - {spec.name for spec in specs})
    if prune:
        actions.extend(Action("delete", name) for name in unmanaged)
        return actions, []
    return actions, unmanaged


def plan_strip(
    issue: int,
    labels: set[str],
    events: Sequence[dict[str, Any]],
    area_labels: set[str],
) -> list[StripDecision]:
    """Only the last labeled event for a currently attached area label counts."""
    decisions: list[StripDecision] = []
    for name in sorted(labels & area_labels):
        labeled = [
            event
            for event in events
            if event.get("event") == "labeled"
            and (event.get("label") or {}).get("name") == name
        ]
        # REST timestamps are UTC; IDs break ties within the same second.
        if not labeled or any(
            type(event.get("id")) is not int
            or not isinstance(event.get("created_at"), str)
            for event in labeled
        ):
            decisions.append(
                StripDecision(
                    issue, name, False, "unknown last labeled event; leave in place"
                )
            )
            continue
        last = max(labeled, key=lambda event: (event["created_at"], event["id"]))
        actor = (last.get("actor") or {}).get("login")
        if not isinstance(actor, str) or not actor:
            reason = f"unknown actor of last labeled event {last['id']}; leave in place"
        else:
            reason = f"last labeled by {actor} (event {last['id']})"
        decisions.append(StripDecision(issue, name, actor == LABELER_BOT, reason))
    return decisions


def _gh(*args: str) -> str:
    return subprocess.run(
        ["gh", *args], check=True, capture_output=True, text=True
    ).stdout


def _object(value: object, context: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{context}: expected a JSON object")
    return value


def _rows(repo: str, endpoint: str) -> list[dict[str, Any]]:
    pages = json.loads(
        _gh(
            "api",
            "--method",
            "GET",
            "--paginate",
            "--slurp",
            f"repos/{repo}/{endpoint}",
        )
    )
    if not isinstance(pages, list) or any(not isinstance(p, list) for p in pages):
        raise ValueError(f"{endpoint}: expected paginated JSON arrays")
    return [_object(row, endpoint) for page in pages for row in page]


def _issue(repo: str, number: int) -> dict[str, Any]:
    endpoint = f"repos/{repo}/issues/{number}"
    return _object(json.loads(_gh("api", "--method", "GET", endpoint)), endpoint)


def _label_names(issue: dict[str, Any]) -> set[str]:
    names = {label["name"] for label in issue["labels"]}
    if any(not isinstance(name, str) for name in names):
        raise ValueError("issue labels must have string names")
    return names


def fetch_strip_plan(
    repo: str, area_labels: set[str]
) -> tuple[int, list[StripDecision]]:
    rows = _rows(repo, "issues?state=open&per_page=100")
    issues = [
        row for row in rows if "pull_request" not in row and row["state"] == "open"
    ]
    decisions = []
    for issue in issues:
        labels = _label_names(issue)
        if labels & area_labels:
            number = issue["number"]
            events = _rows(repo, f"issues/{number}/events?per_page=100")
            decisions.extend(plan_strip(number, labels, events, area_labels))
    return len(issues), decisions


def apply_action(repo: str, action: Action) -> None:
    if action.kind == "delete":
        _gh("label", "delete", action.name, "--repo", repo, "--yes")
        return
    if action.spec is None:
        raise ValueError(f"{action.kind} requires a label spec")
    verb = "create" if action.kind == "create" else "edit"
    _gh(
        "label",
        verb,
        action.name,
        "--repo",
        repo,
        "--color",
        action.spec.color,
        "--description",
        action.spec.description,
    )


def apply_strip(repo: str, decisions: Sequence[StripDecision]) -> int:
    """Recheck each issue and its full event history before removing any labels."""
    removed = 0
    for number in sorted({d.issue for d in decisions if d.remove}):
        requested = {d.label for d in decisions if d.issue == number and d.remove}
        current = _issue(repo, number)
        if "pull_request" in current or current["state"] != "open":
            print(f"skip #{number}: no longer an open issue")
            continue
        present = _label_names(current)
        events = _rows(repo, f"issues/{number}/events?per_page=100")
        for name in sorted(requested - present):
            print(f"skip #{number} {name!r}: label no longer present")
        for decision in plan_strip(number, present, events, requested):
            if not decision.remove:
                print("skip " + decision.describe())
                continue
            _gh(
                "api",
                "--method",
                "DELETE",
                f"repos/{repo}/issues/{number}/labels/{quote(decision.label, safe='')}",
            )
            print("removed " + decision.describe())
            removed += 1
    return removed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for command in ("sync", "strip-auto-labels"):
        sub = commands.add_parser(command)
        sub.add_argument("--repo", help="owner/repo (default: gh repo view)")
        sub.add_argument("--apply", action="store_true", help="execute the plan")
        if command == "sync":
            sub.add_argument(
                "--prune", action="store_true", help="delete unmanaged labels"
            )
    args = parser.parse_args(argv)
    try:
        specs = load_manifest()
        repo = (
            args.repo
            or _gh(
                "repo", "view", "--json", "nameWithOwner", "--jq", ".nameWithOwner"
            ).strip()
        )
        if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", repo):
            raise ValueError("repo must be owner/repo")
        prefix = "[plan] " if args.apply else "[dry-run] "
        if args.command == "sync":
            labels = [
                RepoLabel(row["name"], row["color"], row["description"] or "")
                for row in _rows(repo, "labels?per_page=100")
            ]
            actions, unmanaged = plan(specs, labels, prune=args.prune)
            for action in actions:
                print(prefix + action.describe())
            for name in unmanaged:
                print(f"unmanaged {name!r} (left in place; --prune deletes it)")
            if args.apply:
                for action in actions:
                    apply_action(repo, action)
            print(
                "sync: "
                + ", ".join(
                    f"{kind}={sum(a.kind == kind for a in actions)}"
                    for kind in ("create", "update", "delete")
                )
            )
        else:
            area_labels = {spec.name for spec in specs if spec.axis == "area"}
            count, decisions = fetch_strip_plan(repo, area_labels)
            for decision in decisions:
                print(prefix + decision.describe())
            targets = [d for d in decisions if d.remove]
            print(
                f"strip: open issues={count}, remove={len(targets)} labels "
                f"on {len({d.issue for d in targets})} issues, "
                f"keep={len(decisions) - len(targets)} area labels"
            )
            if args.apply:
                print(f"strip: removed={apply_strip(repo, decisions)} labels")
    except subprocess.CalledProcessError as exc:
        print(
            f"error: gh exited {exc.returncode}: {exc.stderr.strip()}", file=sys.stderr
        )
        return 1
    except (ValueError, OSError, KeyError, TypeError, yaml.YAMLError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
