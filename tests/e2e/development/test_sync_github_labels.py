"""``scripts/sync_github_labels.py``: manifest validation, planning and the gh calls."""

from __future__ import annotations

import importlib
import json
import os
import stat
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "scripts" / "sync_github_labels.py"
sys.path.insert(0, str(SCRIPT.parent))
sync: Any = importlib.import_module("sync_github_labels")
Action, LabelSpec, RepoLabel = sync.Action, sync.LabelSpec, sync.RepoLabel
ManifestError, load_manifest, plan = sync.ManifestError, sync.load_manifest, sync.plan

BUG = LabelSpec("type: bug", "D73A4A", "不具合の修正", ("bug",))
CI = LabelSpec("area: ci", "FEF2C0", "CI", ("ci",))


def test_repository_manifest_is_valid() -> None:
    specs = load_manifest()
    assert {spec.name.split(": ")[0] for spec in specs} == {
        "type",
        "area",
        "status",
        "by",
    }
    # Colors are shared per axis, except type which tells kinds apart.
    for axis in ("area", "status", "by"):
        assert len({s.color for s in specs if s.name.startswith(axis)}) == 1, axis


@pytest.mark.parametrize(
    ("yaml_text", "message"),
    [
        (
            'type:\n  color: "D73A4A"\n  labels:\n'
            '    - {name: "bug", description: "x"}\n',
            "must start with 'type: '",
        ),
        (
            'type:\n  color: "d73a4a"\n  labels:\n'
            '    - {name: "type: bug", description: "x"}\n',
            "6 upper-case hex",
        ),
        (
            'type:\n  color: "D73A4A"\n  labels:\n'
            '    - {name: "type: bug", description: "x", aliases: ["bug"]}\n'
            '    - {name: "type: fix", description: "x", aliases: ["bug"]}\n',
            "'bug' appears more than once",
        ),
    ],
)
def test_invalid_manifest_is_rejected(
    tmp_path: Path, yaml_text: str, message: str
) -> None:
    path = tmp_path / "labels.yml"
    path.write_text(yaml_text, encoding="utf-8")
    with pytest.raises(ManifestError, match=message):
        load_manifest(path)


def test_plan_renames_aliases_creates_missing_and_updates_style() -> None:
    actions, unmanaged = plan(
        [BUG, CI, LabelSpec("type: followup", "C2C7CC", "追補")],
        [
            RepoLabel("bug", "d73a4a", "Something isn't working"),
            RepoLabel("area: ci", "fef2c0", "old description"),
            RepoLabel("wontfix", "ffffff", ""),
        ],
        prune=False,
    )
    assert actions == [
        Action("rename", "type: bug", BUG, "bug"),
        Action("update", "area: ci", CI),
        Action("create", "type: followup", LabelSpec("type: followup", "C2C7CC", "追補")),
    ]
    assert unmanaged == ["wontfix"]


def test_plan_is_empty_when_in_sync_and_compares_color_case_insensitively() -> None:
    actions, unmanaged = plan(
        [BUG], [RepoLabel("type: bug", "d73a4a", "不具合の修正")], prune=True
    )
    assert actions == []
    assert unmanaged == []


def test_plan_prune_deletes_unmanaged_labels() -> None:
    actions, unmanaged = plan(
        [BUG], [RepoLabel("bug", "d73a4a", ""), RepoLabel("python", "000000", "")],
        prune=True,
    )
    assert actions == [
        Action("rename", "type: bug", BUG, "bug"),
        Action("delete", "python"),
    ]
    assert unmanaged == []


def test_plan_refuses_when_name_and_alias_both_exist() -> None:
    with pytest.raises(ManifestError, match="merge them by hand"):
        plan(
            [BUG],
            [RepoLabel("bug", "d73a4a", ""), RepoLabel("type: bug", "d73a4a", "")],
            prune=False,
        )


def test_plan_refuses_when_several_aliases_exist() -> None:
    spec = LabelSpec("type: bug", "D73A4A", "x", ("bug", "Bug report"))
    with pytest.raises(ManifestError, match="several aliases"):
        plan(
            [spec],
            [RepoLabel("bug", "d73a4a", ""), RepoLabel("Bug report", "d73a4a", "")],
            prune=False,
        )


FAKE_GH = r"""#!/usr/bin/env python3
import json, os, sys
args = sys.argv[1:]
with open(os.environ["FAKE_GH_CALLS"], "a", encoding="utf-8") as fh:
    fh.write(json.dumps(args) + "\n")
if args[:2] == ["label", "list"]:
    print(json.dumps([
        {"name": "bug", "color": "d73a4a", "description": "Something isn't working"},
        {"name": "wontfix", "color": "ffffff", "description": ""},
    ]))
"""


def _run(tmp_path: Path, *args: str) -> tuple[subprocess.CompletedProcess[str], list]:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(exist_ok=True)
    gh = bin_dir / "gh"
    gh.write_text(FAKE_GH, encoding="utf-8")
    gh.chmod(gh.stat().st_mode | stat.S_IEXEC)
    calls = tmp_path / "calls.jsonl"
    calls.unlink(missing_ok=True)
    result = subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        capture_output=True,
        text=True,
        env={
            **os.environ,
            "PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}",
            "FAKE_GH_CALLS": str(calls),
        },
        check=False,
    )
    lines = calls.read_text(encoding="utf-8").splitlines()
    return result, [json.loads(line) for line in lines]


def test_cli_dry_run_only_lists_labels(tmp_path: Path) -> None:
    result, calls = _run(tmp_path)

    assert result.returncode == 0, result.stderr
    assert "[dry-run] rename  'bug' -> 'type: bug'" in result.stdout
    assert "[dry-run] create  'type: followup'" in result.stdout
    assert "unmanaged 'wontfix'" in result.stdout
    assert [call[:2] for call in calls] == [["label", "list"]]


def test_cli_apply_prune_issues_gh_edits(tmp_path: Path) -> None:
    result, calls = _run(tmp_path, "--apply", "--prune")

    assert result.returncode == 0, result.stderr
    bug = next(s for s in load_manifest() if s.name == "type: bug")
    assert [
        "label", "edit", "bug", "--name", "type: bug",
        "--color", bug.color, "--description", bug.description,
    ] in calls  # fmt: skip
    assert ["label", "delete", "wontfix", "--yes"] in calls
    created = {call[2] for call in calls if call[:2] == ["label", "create"]}
    assert created == {s.name for s in load_manifest()} - {"type: bug"}
