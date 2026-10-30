"""Label manifest, pure plans and CLI behavior against a PATH-isolated fake gh."""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pytest
import yaml

from scripts import sync_github_labels as sync

ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "scripts/sync_github_labels.py"
REPO = "fixture/labels"
DELETED = {
    "duplicate",
    "good first issue",
    "help wanted",
    "invalid",
    "question",
    "wontfix",
    "ready",
    "agent-review:changes-requested",
}
BUG = sync.LabelSpec("bug", "D73A4A", "bug description", "work-type")
CI = sync.LabelSpec("ci", "5319E7", "CI description", "area")


def test_repository_manifest_is_valid_and_keeps_existing_names() -> None:
    specs = sync.load_manifest()
    names = {spec.name for spec in specs}
    assert len(specs) == len(names) == 26
    assert not names & DELETED
    assert {spec.axis for spec in specs} == set(sync.AXES)
    assert all(re.fullmatch(r"[0-9A-Fa-f]{6}", spec.color) for spec in specs)
    assert {
        "bug",
        "enhancement",
        "research",
        "documentation",
        "followup",
        "ai-proposed",
        "ops-report",
        "agent:claude",
        "agent:codex",
        "python",
        "knowledge",
        "needs-human",
        "ci",
        "tests",
        "dependencies",
        "third-party",
        "module: base",
        "module: ball-detection",
        "module: court-detection",
        "module: blcs",
        "module: plcs",
        "module: submodules",
        "module: synthetic-data",
        "module: tennis-scene",
        "issue:parent",
        "issue:child",
    } == names
    for name, color in (("issue:parent", "F97316"), ("issue:child", "FED7AA")):
        assert next(s.color for s in specs if s.name == name) == color
        assert sum(s.color.upper() == color for s in specs) == 1
    followup = next(s for s in specs if s.name == "followup")
    assert sum(s.color == followup.color for s in specs) == 1


def test_all_automation_and_template_label_references_exist() -> None:
    names = {spec.name for spec in sync.load_manifest()}
    github = ROOT / ".github"
    references = set(yaml.safe_load((github / "issue-labeler.yml").read_text()))
    references.update(yaml.safe_load((github / "labeler.yml").read_text()))
    for template in (github / "ISSUE_TEMPLATE").glob("*.md"):
        metadata = yaml.safe_load(template.read_text(encoding="utf-8").split("---")[1])
        references.update(metadata.get("labels", []))
    dependabot = yaml.safe_load((github / "dependabot.yml").read_text())
    for update in dependabot["updates"]:
        references.update(update["labels"])
    assert references <= names, references - names


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"color": "12345"}, "6 hex digits"),
        ({"color": "GG0000"}, "6 hex digits"),
        ({"color": 123456}, "6 hex digits"),
        ({"description": None}, "description must be a string"),
        ({"name": " "}, "nonempty string"),
        ({"aliases": ["old"]}, "expected name/color/description"),
    ],
)
def test_invalid_manifest_entries_are_rejected(
    tmp_path: Path, change: dict[str, Any], message: str
) -> None:
    groups: dict[str, list[dict[str, Any]]] = {axis: [] for axis in sync.AXES}
    groups["work-type"] = [
        {"name": "bug", "color": "abcdef", "description": "", **change}
    ]
    path = tmp_path / "labels.yml"
    path.write_text(yaml.safe_dump(groups), encoding="utf-8")
    with pytest.raises(sync.ManifestError, match=message):
        sync.load_manifest(path)


@pytest.mark.parametrize("duplicate", ["bug", "BUG"])
def test_duplicate_names_across_axes_are_rejected(
    tmp_path: Path, duplicate: str
) -> None:
    groups: dict[str, list[dict[str, Any]]] = {axis: [] for axis in sync.AXES}
    groups["work-type"] = [{"name": "bug", "color": "aabbcc", "description": ""}]
    groups["area"] = [{"name": duplicate, "color": "000000", "description": ""}]
    path = tmp_path / "labels.yml"
    path.write_text(yaml.safe_dump(groups), encoding="utf-8")
    with pytest.raises(sync.ManifestError, match="appears more than once"):
        sync.load_manifest(path)


@pytest.mark.parametrize("content", ["null", "[]", "area: []", "hierarchy: {}"])
def test_invalid_manifest_shape_is_rejected(tmp_path: Path, content: str) -> None:
    path = tmp_path / "labels.yml"
    path.write_text(content, encoding="utf-8")
    with pytest.raises(sync.ManifestError):
        sync.load_manifest(path)


def test_plan_creates_updates_and_preserves_unmanaged_labels_without_prune() -> None:
    existing = [
        sync.RepoLabel("bug", "d73a4a", BUG.description),
        sync.RepoLabel("ci", "5319e7", "old description"),
        sync.RepoLabel("wontfix", "ffffff", ""),
    ]
    specs = [BUG, CI, sync.LabelSpec("followup", "C2C7CC", "追補", "work-type")]
    before = list(existing)
    actions, unmanaged = sync.plan(specs, existing, prune=False)
    assert actions == [
        sync.Action("update", "ci", CI),
        sync.Action("create", "followup", specs[2]),
    ]
    assert unmanaged == ["wontfix"]
    assert existing == before


def test_plan_detects_color_changes_and_prunes_only_unmanaged_names() -> None:
    actions, unmanaged = sync.plan(
        [BUG],
        [
            sync.RepoLabel("bug", "000000", BUG.description),
            sync.RepoLabel("wontfix", "ffffff", ""),
        ],
        prune=True,
    )
    assert actions == [
        sync.Action("update", "bug", BUG),
        sync.Action("delete", "wontfix"),
    ]
    assert unmanaged == []


def test_plan_compares_color_case_insensitively_and_is_idempotent() -> None:
    assert sync.plan(
        [BUG], [sync.RepoLabel("bug", BUG.color.lower(), BUG.description)], prune=True
    ) == ([], [])
    assert sync.plan([], [], prune=True) == ([], [])


def event(
    ident: int, name: str, actor: str | None, kind: str = "labeled"
) -> dict[str, Any]:
    return {
        "id": ident,
        "event": kind,
        "label": {"name": name},
        "actor": {"login": actor} if actor is not None else None,
        "created_at": f"2026-10-01T00:{ident // 60:02}:{ident % 60:02}Z",
    }


def test_strip_plan_uses_last_labeled_actor_and_only_current_area_labels() -> None:
    events = [
        event(20, "ci", "human"),
        event(10, "ci", sync.LABELER_BOT),
        event(5, "tests", "agent-account"),
        event(30, "tests", sync.LABELER_BOT),
        event(40, "bug", sync.LABELER_BOT),
        event(50, "dependencies", sync.LABELER_BOT),  # currently absent
        event(60, "tests", "human", "unlabeled"),  # not a labeled event
    ]
    decisions = sync.plan_strip(
        1, {"ci", "tests", "bug"}, events, {"ci", "tests", "dependencies"}
    )
    assert [(d.label, d.remove) for d in decisions] == [("ci", False), ("tests", True)]
    assert "human" in decisions[0].reason
    assert "event 30" in decisions[1].reason


@pytest.mark.parametrize(
    "events",
    [
        [],
        [event(1, "ci", None)],
        [event(1, "ci", sync.LABELER_BOT), event(2, "ci", None)],
        [
            {
                "event": "labeled",
                "label": {"name": "ci"},
                "actor": {"login": sync.LABELER_BOT},
            }
        ],
    ],
)
def test_strip_plan_preserves_unknown_provenance(events: list[dict[str, Any]]) -> None:
    (decision,) = sync.plan_strip(1, {"ci"}, events, {"ci"})
    assert not decision.remove
    assert "unknown" in decision.reason


@pytest.mark.parametrize(
    "actor", ["human", "codex-account", "dependabot[bot]", "github-actions"]
)
def test_strip_plan_preserves_other_actors(actor: str) -> None:
    (decision,) = sync.plan_strip(1, {"ci"}, [event(1, "ci", actor)], {"ci"})
    assert not decision.remove
    assert actor in decision.reason


FAKE_GH = r"""
import json, os, sys
from pathlib import Path
args = sys.argv[1:]
calls_path = Path(os.environ["FAKE_GH_CALLS"])
previous = [json.loads(line) for line in calls_path.read_text().splitlines()] if calls_path.exists() else []
with calls_path.open("a") as log:
    log.write(json.dumps(args) + "\n")
routes = json.loads(Path(os.environ["FAKE_GH_ROUTES"]).read_text())
if args[:2] == ["repo", "view"]:
    assert args == ["repo", "view", "--json", "nameWithOwner", "--jq", ".nameWithOwner"]
    print("fixture/labels")
elif args[0] == "api":
    method = args[args.index("--method") + 1]
    endpoint = args[-1]
    assert endpoint.startswith("repos/fixture/labels/")
    if method == "DELETE":
        assert "/issues/" in endpoint and "/labels/" in endpoint
        sys.exit(0)
    assert method == "GET"
    if "per_page=" in endpoint:
        assert "--paginate" in args and "--slurp" in args
    route = routes[endpoint]
    if isinstance(route, dict) and "error" in route:
        print(route["error"], file=sys.stderr)
        sys.exit(1)
    if isinstance(route, dict) and "sequence" in route:
        index = sum(call == args for call in previous)
        route = route["sequence"][index]
    if isinstance(route, dict) and "raw" in route:
        print(route["raw"])
    else:
        print(json.dumps(route))
elif args[:2] in (["label", "create"], ["label", "edit"], ["label", "delete"]):
    assert args[args.index("--repo") + 1] == "fixture/labels"
else:
    raise AssertionError(f"unsupported fake gh invocation: {args}")
"""


@dataclass
class GHEnv:
    env: dict[str, str]
    routes_path: Path
    calls_path: Path

    def run(self, *args: str) -> subprocess.CompletedProcess[str]:
        self.calls_path.unlink(missing_ok=True)
        return subprocess.run(
            [sys.executable, str(SCRIPT), *args],
            env=self.env,
            capture_output=True,
            text=True,
            check=False,
        )

    def routes(self, value: dict[str, Any]) -> None:
        self.routes_path.write_text(json.dumps(value), encoding="utf-8")

    def calls(self) -> list[list[str]]:
        return [json.loads(line) for line in self.calls_path.read_text().splitlines()]

    def writes(self) -> list[list[str]]:
        return [
            call
            for call in self.calls()
            if (
                call[:2]
                in (["label", "create"], ["label", "edit"], ["label", "delete"])
                or "DELETE" in call
            )
        ]


@pytest.fixture
def gh_env(tmp_path: Path) -> GHEnv:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    gh = bin_dir / "gh"
    gh.write_text(f"#!{sys.executable}\n" + FAKE_GH, encoding="utf-8")
    gh.chmod(0o755)
    env = {
        **os.environ,
        "PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}",
        "FAKE_GH_CALLS": str(tmp_path / "calls.jsonl"),
        "FAKE_GH_ROUTES": str(tmp_path / "routes.json"),
    }
    return GHEnv(env, tmp_path / "routes.json", tmp_path / "calls.jsonl")


def sync_routes() -> dict[str, Any]:
    existing = []
    for spec in sync.load_manifest():
        if spec.name == "followup":
            continue
        row = asdict(spec)
        row.pop("axis")
        row["color"] = row["color"].lower()
        if spec.name == "issue:parent":
            row["color"] = "0e8a16"
        elif spec.name == "issue:child":
            row["color"] = "bfd4f2"
        elif spec.name == "ai-proposed":
            row["description"] = "old origin description"
        existing.append(row)
    existing.extend(
        {"name": name, "color": "000000", "description": ""} for name in sorted(DELETED)
    )
    return {f"repos/{REPO}/labels?per_page=100": [existing[:10], existing[10:]]}


def test_sync_cli_dry_run_only_reads_and_leaves_unmanaged_without_prune(
    gh_env: GHEnv,
) -> None:
    gh_env.routes(sync_routes())
    result = gh_env.run("sync")  # also exercise repo discovery
    assert result.returncode == 0, result.stderr
    assert "[dry-run] create  'followup'" in result.stdout
    assert "sync: create=1, update=3, delete=0" in result.stdout
    assert "unmanaged 'wontfix'" in result.stdout
    assert gh_env.writes() == []
    assert gh_env.calls()[0][:2] == ["repo", "view"]
    assert len(gh_env.calls()) == 2


def test_sync_cli_prune_is_still_dry_run_by_default(gh_env: GHEnv) -> None:
    gh_env.routes(sync_routes())
    result = gh_env.run("sync", "--repo", REPO, "--prune")
    assert result.returncode == 0, result.stderr
    assert "sync: create=1, update=3, delete=8" in result.stdout
    assert "[dry-run] delete  'ready'" in result.stdout
    assert gh_env.writes() == []
    assert len(gh_env.calls()) == 1


@pytest.mark.parametrize("prune", [False, True])
def test_sync_cli_apply_passes_manifest_styles_and_repo_to_gh(
    gh_env: GHEnv, prune: bool
) -> None:
    gh_env.routes(sync_routes())
    result = gh_env.run(
        "sync", "--repo", REPO, "--apply", *(["--prune"] if prune else [])
    )
    assert result.returncode == 0, result.stderr
    for name in ("followup", "issue:parent", "issue:child", "ai-proposed"):
        spec = next(s for s in sync.load_manifest() if s.name == name)
        verb = "create" if name == "followup" else "edit"
        assert [
            "label",
            verb,
            name,
            "--repo",
            REPO,
            "--color",
            spec.color,
            "--description",
            spec.description,
        ] in gh_env.writes()
    assert {call[2] for call in gh_env.writes() if call[1] == "delete"} == (
        DELETED if prune else set()
    )
    assert len(gh_env.writes()) == (12 if prune else 4)


def issue(number: int, *labels: str, **extra: Any) -> dict[str, Any]:
    return {
        "number": number,
        "state": "open",
        "labels": [{"name": name} for name in labels],
        **extra,
    }


def strip_routes() -> dict[str, Any]:
    first = issue(
        1, "ci", "tests", "module: blcs", "bug", "knowledge", "python", "third-party"
    )
    second = issue(2, "module: base", "dependencies")
    third = issue(3, "ci")
    events = [
        event(1, "ci", sync.LABELER_BOT),
        event(2, "tests", sync.LABELER_BOT),
        event(3, "module: blcs", sync.LABELER_BOT),
    ]
    events += [{"id": i, "event": "commented"} for i in range(4, 101)]
    tail = [
        event(101, "tests", "human"),
        event(102, "module: blcs", "codex-account"),
        event(103, "python", None),
        event(104, "third-party", "dependabot[bot]"),
    ]
    return {
        f"repos/{REPO}/issues?state=open&per_page=100": [
            [first],
            [
                second,
                third,
                issue(4, "ci", pull_request={}),
                issue(5, "ci", state="closed"),
            ],
        ],
        f"repos/{REPO}/issues/1/events?per_page=100": [events, tail],
        f"repos/{REPO}/issues/2/events?per_page=100": [
            [
                event(1, "module: base", sync.LABELER_BOT),
                event(2, "dependencies", sync.LABELER_BOT),
            ]
        ],
        f"repos/{REPO}/issues/3/events?per_page=100": [[]],
        f"repos/{REPO}/issues/1": first,
        f"repos/{REPO}/issues/2": second,
    }


def test_strip_cli_dry_run_reports_bot_human_agent_and_unknown_on_all_pages(
    gh_env: GHEnv,
) -> None:
    gh_env.routes(strip_routes())
    result = gh_env.run("strip-auto-labels", "--repo", REPO)
    assert result.returncode == 0, result.stderr
    assert "#1 remove 'ci': last labeled by github-actions[bot]" in result.stdout
    assert "#1 keep 'tests': last labeled by human" in result.stdout
    assert "#1 keep 'module: blcs': last labeled by codex-account" in result.stdout
    assert "#1 keep 'knowledge': unknown last labeled event" in result.stdout
    assert "#1 keep 'python': unknown actor" in result.stdout
    assert "#1 keep 'third-party': last labeled by dependabot[bot]" in result.stdout
    assert (
        "strip: open issues=3, remove=3 labels on 2 issues, keep=6 area labels"
        in result.stdout
    )
    assert "#4" not in result.stdout and "#5" not in result.stdout
    assert "'bug'" not in result.stdout
    assert gh_env.writes() == []
    assert len(gh_env.calls()) == 4


def test_strip_cli_apply_rechecks_and_only_deletes_bot_area_labels(
    gh_env: GHEnv,
) -> None:
    gh_env.routes(strip_routes())
    result = gh_env.run("strip-auto-labels", "--repo", REPO, "--apply")
    assert result.returncode == 0, result.stderr
    assert gh_env.writes() == [
        ["api", "--method", "DELETE", f"repos/{REPO}/issues/1/labels/ci"],
        ["api", "--method", "DELETE", f"repos/{REPO}/issues/2/labels/dependencies"],
        ["api", "--method", "DELETE", f"repos/{REPO}/issues/2/labels/module%3A%20base"],
    ]
    assert "strip: removed=3 labels" in result.stdout
    assert (
        sum(call[-1].endswith("/1/events?per_page=100") for call in gh_env.calls()) == 2
    )


def test_strip_apply_preserves_a_label_reapplied_by_human_since_plan(
    gh_env: GHEnv,
) -> None:
    routes = strip_routes()
    endpoint = f"repos/{REPO}/issues/1/events?per_page=100"
    initial = routes[endpoint]
    routes[endpoint] = {"sequence": [initial, [*initial, [event(200, "ci", "human")]]]}
    gh_env.routes(routes)
    result = gh_env.run("strip-auto-labels", "--repo", REPO, "--apply")
    assert result.returncode == 0, result.stderr
    assert "skip #1 keep 'ci': last labeled by human" in result.stdout
    assert all("/issues/1/labels/" not in call[-1] for call in gh_env.writes())
    assert len(gh_env.writes()) == 2


def test_strip_apply_skips_an_issue_closed_since_plan(gh_env: GHEnv) -> None:
    routes = strip_routes()
    routes[f"repos/{REPO}/issues/1"] = {
        **routes[f"repos/{REPO}/issues/1"],
        "state": "closed",
    }
    gh_env.routes(routes)
    result = gh_env.run("strip-auto-labels", "--repo", REPO, "--apply")
    assert result.returncode == 0, result.stderr
    assert "skip #1: no longer an open issue" in result.stdout
    assert len(gh_env.writes()) == 2


@pytest.mark.parametrize("command", ["sync", "strip-auto-labels"])
@pytest.mark.parametrize(
    "failure", [{"error": "HTTP 403 forbidden"}, {"raw": "not JSON"}, {}]
)
def test_failed_or_malformed_reads_never_become_an_empty_plan_or_write(
    gh_env: GHEnv, command: str, failure: dict[str, Any]
) -> None:
    routes = sync_routes() if command == "sync" else strip_routes()
    endpoint = (
        f"repos/{REPO}/labels?per_page=100"
        if command == "sync"
        else f"repos/{REPO}/issues/2/events?per_page=100"
    )
    routes[endpoint] = failure
    gh_env.routes(routes)
    result = gh_env.run(command, "--repo", REPO, "--apply")
    assert result.returncode == 1
    assert "error:" in result.stderr
    assert gh_env.writes() == []
    assert "sync:" not in result.stdout and "strip:" not in result.stdout
