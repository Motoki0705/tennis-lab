"""Sync the repository's GitHub labels with ``.github/labels.yml``.

Dry-run by default: prints the plan and changes nothing. ``--apply`` executes it.
A label still under one of its ``aliases`` is renamed (issues/PRs keep it); labels
missing from the manifest are only reported unless ``--prune`` deletes them.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

MANIFEST = Path(__file__).resolve().parents[1] / ".github" / "labels.yml"
_COLOR_RE = re.compile(r"[0-9A-F]{6}")


@dataclass(frozen=True)
class LabelSpec:
    name: str
    color: str
    description: str
    aliases: tuple[str, ...] = ()


@dataclass(frozen=True)
class RepoLabel:
    name: str
    color: str
    description: str


@dataclass(frozen=True)
class Action:
    kind: str  # "create" | "rename" | "update" | "delete"
    name: str
    spec: LabelSpec | None = None
    old_name: str | None = None

    def describe(self) -> str:
        if self.kind == "rename":
            return f"rename  {self.old_name!r} -> {self.name!r}"
        return f"{self.kind:<7} {self.name!r}"


class ManifestError(ValueError):
    pass


def load_manifest(path: Path = MANIFEST) -> list[LabelSpec]:
    """Parse and validate the manifest; every name is ``<axis>: <value>``."""
    raw: dict[str, Any] = yaml.safe_load(path.read_text(encoding="utf-8"))
    specs: list[LabelSpec] = []
    for axis, group in raw.items():
        for entry in group["labels"]:
            name = entry["name"]
            if not name.startswith(f"{axis}: "):
                raise ManifestError(f"{name!r} must start with {axis + ': '!r}")
            color = str(entry.get("color", group["color"]))
            if not _COLOR_RE.fullmatch(color):
                raise ManifestError(f"{name!r}: color must be 6 upper-case hex digits")
            specs.append(
                LabelSpec(
                    name=name,
                    color=color,
                    description=entry["description"],
                    aliases=tuple(entry.get("aliases", ())),
                )
            )
    seen: set[str] = set()
    for spec in specs:
        for label in (spec.name, *spec.aliases):
            if label in seen:
                raise ManifestError(f"{label!r} appears more than once in the manifest")
            seen.add(label)
    return specs


def plan(
    specs: list[LabelSpec], existing: list[RepoLabel], *, prune: bool
) -> tuple[list[Action], list[str]]:
    """Return the actions and the unmanaged labels left in place (when not pruning)."""
    by_name = {label.name: label for label in existing}
    actions: list[Action] = []
    claimed: set[str] = set()
    for spec in specs:
        present_aliases = [a for a in spec.aliases if a in by_name]
        if spec.name in by_name and present_aliases:
            raise ManifestError(
                f"both {spec.name!r} and its alias {present_aliases} exist;"
                " merge them by hand first"
            )
        if len(present_aliases) > 1:
            raise ManifestError(
                f"several aliases of {spec.name!r} exist: {present_aliases}"
            )
        if present_aliases:
            claimed.add(present_aliases[0])
            actions.append(Action("rename", spec.name, spec, present_aliases[0]))
        elif spec.name in by_name:
            claimed.add(spec.name)
            current = by_name[spec.name]
            if (current.color.upper(), current.description) != (
                spec.color,
                spec.description,
            ):
                actions.append(Action("update", spec.name, spec))
        else:
            actions.append(Action("create", spec.name, spec))
    unmanaged = sorted(name for name in by_name if name not in claimed)
    if prune:
        actions.extend(Action("delete", name) for name in unmanaged)
        return actions, []
    return actions, unmanaged


def _gh(*args: str) -> str:
    return subprocess.run(
        ["gh", *args], check=True, capture_output=True, text=True
    ).stdout


def fetch_labels() -> list[RepoLabel]:
    rows = json.loads(
        _gh("label", "list", "--limit", "1000", "--json", "name,color,description")
    )
    return [RepoLabel(r["name"], r["color"], r["description"] or "") for r in rows]


def apply(action: Action) -> None:
    spec = action.spec
    if action.kind == "delete":
        _gh("label", "delete", action.name, "--yes")
        return
    assert spec is not None
    style = ["--color", spec.color, "--description", spec.description]
    if action.kind == "create":
        _gh("label", "create", spec.name, *style)
    elif action.kind == "rename":
        assert action.old_name is not None
        _gh("label", "edit", action.old_name, "--name", spec.name, *style)
    elif action.kind == "update":
        _gh("label", "edit", spec.name, *style)
    else:
        raise ValueError(f"unknown action kind: {action.kind}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true", help="execute the plan")
    parser.add_argument(
        "--prune", action="store_true", help="delete labels missing from the manifest"
    )
    args = parser.parse_args(argv)
    try:
        actions, unmanaged = plan(load_manifest(), fetch_labels(), prune=args.prune)
    except ManifestError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    for action in actions:
        print(("" if args.apply else "[dry-run] ") + action.describe())
        if args.apply:
            apply(action)
    for name in unmanaged:
        print(f"unmanaged {name!r} (not in the manifest; --prune deletes it)")
    if not actions and not unmanaged:
        print("labels are in sync with the manifest")
    return 0


if __name__ == "__main__":
    sys.exit(main())
