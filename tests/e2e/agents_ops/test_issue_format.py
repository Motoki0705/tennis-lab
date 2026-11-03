"""The shared issue contract, lifecycle changes and read-only CLI (fake gh only)."""

from __future__ import annotations

import importlib.util
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[3]
OPS = ROOT / ".agents/ops"
TEMPLATES = ROOT / ".github/ISSUE_TEMPLATE"
PARENT_SECTIONS = {
    "目的・背景": "親が合意した開発ループの進捗を集約する。",
    "全体のスコープ": "対象: 書式と委任。対象外: 学習モデル。",
    "全体方針・合意事項": "2026-10-18 人間と範囲・完了条件を合意済み。",
    "全体の完了条件": "共通書式と委任が連携し、異常系も検証できる。",
    "進捗概要": "#11 進行中、#12 完了。",
    "依存関係・進め方": "#11 と #12 は独立して実行できる。",
}
CHILD_SECTIONS = {
    "このタスクの目的": "親の目的に沿って共通書式を実装する。",
    "具体的な変更内容・対象範囲": ".agents/ops/issues/ とテンプレートを変更する。",
    "対象外": "レビュー実行と GitHub への書き込み。",
    "依存関係": "先行タスクなし。下流: 親セッションと triage。",
    "作業状況・停止事項": "進行中。停止事項なし。",
    "Acceptance checklist": (
        "- [ ] 親子関係の矛盾を検出できる\n"
        "  文脈: 一覧と本文で同じ親を指す必要がある。\n"
        "  下流への影響: 担当親の誤認を防ぐ。\n"
        "  検証方法・証拠: fake gh で親1件・子2件の関係を照合する。"
    ),
    "成果物・関連PR": "未作成。",
    "次の改善候補（該当する場合）": "なし。",
}


@pytest.fixture(scope="module")
def fmt() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "shared_issue_contract", OPS / "issues/__init__.py"
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def family(fmt: ModuleType) -> dict[int, Any]:
    parent = fmt.ParentRef("開発ループ", 10)
    return {
        10: fmt.Issue(
            10,
            fmt.parent_title(parent.identifier, "実装と改善を循環させる"),
            fmt.render_parent_body(
                PARENT_SECTIONS,
                [
                    fmt.ChildTask(11, "共通書式を実装する"),
                    fmt.ChildTask(12, "委任を整える", "completed"),
                ],
            ),
            frozenset({fmt.PARENT_LABEL, "enhancement"}),
            children=(11, 12),
        ),
        11: fmt.Issue(
            11,
            fmt.child_title(parent, "共通書式を実装する"),
            fmt.render_child_body(10, CHILD_SECTIONS),
            frozenset({fmt.CHILD_LABEL, "enhancement"}),
            parent=10,
        ),
        12: fmt.Issue(
            12,
            fmt.child_title(parent, "委任を整える"),
            fmt.render_child_body(10, CHILD_SECTIONS),
            frozenset({fmt.CHILD_LABEL, "enhancement"}),
            parent=10,
            state="closed",
            state_reason="completed",
        ),
    }


def codes(fmt: ModuleType, snapshots: dict[int, Any]) -> set[str]:
    return {problem.code for problem in fmt.check_consistency(snapshots)}


def test_parent_two_children_and_standalone_are_distinct(fmt: ModuleType) -> None:
    snapshots = family(fmt)
    snapshots[50] = fmt.Issue(
        50, "[バグ] 単独のバグ", "既存の単独テンプレート本文", frozenset({"bug"})
    )
    assert fmt.check_consistency(snapshots) == ()
    assert fmt.parse_title(snapshots[11].title).parent == fmt.ParentRef(
        "開発ループ", 10
    )
    assert fmt.parent_reference(snapshots[11].body) == 10


@pytest.mark.parametrize(
    "title",
    [
        "[開発ループ #0] 実装",
        "[#10] 実装",
        "[10] 実装",
        "[開発ループ] ",
        "[開発ループ #10] [余分] [さらに] 実装",
        " [開発ループ #10] 実装",
        "[開発ループ #10] 実装\n別行",
    ],
)
def test_ambiguous_or_incomplete_titles_are_rejected(
    fmt: ModuleType, title: str
) -> None:
    with pytest.raises(ValueError):
        fmt.parse_title(title)


@pytest.mark.parametrize(
    ("number", "change", "code"),
    [
        (10, {"labels": frozenset({"enhancement"})}, "label.parent"),
        (11, {"labels": frozenset({"bug"})}, "label.child"),
        (11, {"parent": None}, "relationship.parent"),
        (11, {"title": "[別の親 #10] 共通書式を実装する"}, "title.prefix"),
        (11, {"title": "[開発ループ #20] 共通書式を実装する"}, "title.parent"),
        (11, {"title": "[開発ループ #10] タスクを改名した"}, "checklist.title"),
        (10, {"children": (11,)}, "checklist.members"),
        (10, {"children": (11, 12, 99)}, "relationship.children"),
        (10, {"children": (11, 12, 12)}, "relationship.children"),
        (12, {"state": "open"}, "checklist.state"),
        (12, {"state_reason": None}, "state"),
        (12, {"state": "unknown"}, "state"),
    ],
)
def test_inconsistent_snapshots_fail_closed(
    fmt: ModuleType, number: int, change: dict[str, Any], code: str
) -> None:
    snapshots = family(fmt)
    snapshots[number] = replace(snapshots[number], **change)
    assert code in codes(fmt, snapshots)


@pytest.mark.parametrize(
    "heading",
    [
        "目的・背景",
        "全体のスコープ",
        "全体の完了条件",
        "子タスクのチェックリスト",
        "進捗概要",
    ],
)
def test_missing_parent_sections_are_detected(fmt: ModuleType, heading: str) -> None:
    snapshots = family(fmt)
    body = snapshots[10].body
    snapshots[10] = replace(
        snapshots[10],
        body=re.sub(rf"## {heading}\n.*?(?=\n## |\Z)", "", body, flags=re.DOTALL),
    )
    assert any(heading in p.message for p in fmt.check_consistency(snapshots))


def test_templates_and_examples_do_not_satisfy_missing_requirements(
    fmt: ModuleType,
) -> None:
    for wrapper in ("<!--\n{}\n-->", "```markdown\n{}\n```", "~~~md\n{}\n~~~"):
        assert fmt.validate_body(
            wrapper.format(family(fmt)[10].body), is_parent=True, is_child=False
        )
    body = family(fmt)[11].body
    assert fmt.validate_body(
        body.replace(
            "## 依存関係\n\n先行タスクなし。下流: 親セッションと triage。",
            "## 依存関係\n\n<!-- 未記入 -->",
        ),
        is_parent=False,
        is_child=True,
    )
    assert fmt.validate_body(
        body + "\n## 依存関係\n\n別の値", is_parent=False, is_child=True
    )
    assert fmt.validate_body(body + "\n```\nunclosed", is_parent=False, is_child=True)


def test_child_requires_a_unique_parent_and_actionable_acceptance(
    fmt: ModuleType,
) -> None:
    body = family(fmt)[11].body
    for damaged in (
        body.replace("親issue: #10", "親issue: #20"),
        body.replace("親issue: #10", "親issue: #10\n親issue: #10"),
        body.replace("  検証方法・証拠: fake gh で親1件・子2件の関係を照合する。", ""),
        body.replace("  文脈:", "\n  文脈:"),
    ):
        snapshots = family(fmt)
        snapshots[11] = replace(snapshots[11], body=damaged)
        assert fmt.check_consistency(snapshots)
    with pytest.raises(ValueError, match="missing or empty section"):
        fmt.render_child_body(10, {**CHILD_SECTIONS, "依存関係": "-"})


def test_no_fictitious_child_before_decomposition(fmt: ModuleType) -> None:
    pending = fmt.Issue(
        10,
        fmt.parent_title("開発ループ", "要件を確認する"),
        fmt.render_parent_body(PARENT_SECTIONS),
        frozenset({fmt.PARENT_LABEL}),
    )
    assert fmt.NO_CHILDREN in pending.body
    assert fmt.check_consistency({10: pending}) == ()
    for text in (
        "- [ ] #<番号> 未作成",
        "- [ ] #11 タスク\n- [ ] #11 タスク",
        "- [x] #11 タスク（取消）",
    ):
        with pytest.raises(ValueError):
            fmt.parse_child_tasks(text)


def test_missing_labels_do_not_hide_orphaned_child_information(fmt: ModuleType) -> None:
    orphan = replace(family(fmt)[11], parent=None, labels=frozenset({"enhancement"}))
    assert {"relationship.parent", "label.child"} <= codes(fmt, {11: orphan})
    root = replace(family(fmt)[10], children=(), labels=frozenset())
    assert "label.parent" in codes(fmt, {10: root})


def test_generator_rejects_embedded_sections_instead_of_accepting_missing_keys(
    fmt: ModuleType,
) -> None:
    with pytest.raises(ValueError, match="embed other"):
        fmt.render_child_body(
            10,
            {
                "このタスクの目的": "\n\n".join(
                    f"## {k}\n\n{v}" for k, v in CHILD_SECTIONS.items()
                )
            },
        )


def test_rename_reparent_and_interrupted_retry(fmt: ModuleType) -> None:
    snapshots = family(fmt)
    snapshots[10] = replace(
        snapshots[10], title=fmt.parent_title("運用改善", "実装と改善を循環させる")
    )
    assert "title.prefix" in codes(fmt, snapshots)
    for n in (11, 12):
        snapshots[n] = replace(
            snapshots[n],
            title=fmt.child_title(
                fmt.ParentRef("運用改善", 10),
                fmt.parse_title(snapshots[n].title).summary,
            ),
        )
    assert fmt.check_consistency(snapshots) == ()
    snapshots[20] = fmt.Issue(
        20,
        fmt.parent_title("別テーマ", "別の目的を達成する"),
        fmt.render_parent_body(PARENT_SECTIONS),
        frozenset({fmt.PARENT_LABEL}),
    )
    snapshots[11] = replace(snapshots[11], parent=20)
    assert fmt.check_consistency(snapshots)  # An interrupted move is visible.
    snapshots[11] = replace(
        snapshots[11],
        title=fmt.child_title(fmt.ParentRef("別テーマ", 20), "共通書式を実装する"),
        body=fmt.render_child_body(20, CHILD_SECTIONS),
    )
    snapshots[10] = replace(
        snapshots[10],
        children=(12,),
        body=fmt.render_parent_body(
            PARENT_SECTIONS, [fmt.ChildTask(12, "委任を整える", "completed")]
        ),
    )
    snapshots[20] = replace(
        snapshots[20],
        children=(11,),
        body=fmt.render_parent_body(
            PARENT_SECTIONS, [fmt.ChildTask(11, "共通書式を実装する")]
        ),
    )
    assert fmt.check_consistency(snapshots) == fmt.check_consistency(snapshots) == ()


def test_split_cancel_and_complete_do_not_conflate_cancel_with_success(
    fmt: ModuleType,
) -> None:
    snapshots = family(fmt)
    snapshots[11] = replace(
        snapshots[11],
        state="closed",
        state_reason="not_planned",
        body=fmt.render_child_body(
            10,
            {
                **CHILD_SECTIONS,
                "作業状況・停止事項": "取消: #13 へ分割。要件合意は維持し、残作業を引き継ぐ。",
            },
        ),
    )
    snapshots[13] = replace(
        snapshots[12],
        number=13,
        title=fmt.child_title(fmt.ParentRef("開発ループ", 10), "分割後の実装を終える"),
    )
    snapshots[10] = replace(
        snapshots[10],
        children=(11, 12, 13),
        body=fmt.render_parent_body(
            PARENT_SECTIONS,
            [
                fmt.ChildTask(11, "共通書式を実装する", "cancelled"),
                fmt.ChildTask(12, "委任を整える", "completed"),
                fmt.ChildTask(13, "分割後の実装を終える", "completed"),
            ],
        ),
    )
    assert "- [ ] #11 共通書式を実装する（取消）" in snapshots[10].body
    assert fmt.check_consistency(snapshots) == ()
    snapshots[11] = replace(snapshots[11], state="open", state_reason="reopened")
    assert "checklist.state" in codes(fmt, snapshots)


def test_nested_parent_uses_both_roles_without_losing_direct_parent(
    fmt: ModuleType,
) -> None:
    snapshots = family(fmt)
    snapshots[11] = replace(
        snapshots[11],
        title=fmt.parent_title(
            "書式", "共通書式を実装する", parent=fmt.ParentRef("開発ループ", 10)
        ),
        labels=frozenset({fmt.PARENT_LABEL, fmt.CHILD_LABEL}),
        children=(13,),
        body=fmt.render_parent_body(
            {**PARENT_SECTIONS, **CHILD_SECTIONS},
            [fmt.ChildTask(13, "本文を検査する")],
            parent_number=10,
        ),
    )
    snapshots[13] = replace(
        snapshots[12],
        number=13,
        title=fmt.child_title(fmt.ParentRef("書式", 11), "本文を検査する"),
        body=fmt.render_child_body(11, CHILD_SECTIONS),
        parent=11,
        state="open",
        state_reason=None,
    )
    assert snapshots[11].title == "[開発ループ #10] [書式] 共通書式を実装する"
    assert fmt.check_consistency(snapshots) == ()
    snapshots[10] = replace(snapshots[10], parent=13)
    assert "relationship.cycle" in codes(fmt, snapshots)


def test_partial_stop_and_resume_preserve_independent_progress(fmt: ModuleType) -> None:
    snapshots = family(fmt)
    blocked = {
        **CHILD_SECTIONS,
        "作業状況・停止事項": "停止。理由・確認済みの事実: API の権限不足。影響する依存作業: #13。再開条件: 読取権限の復旧。次の対応: 親が権限を確認する。",
    }
    snapshots[11] = replace(snapshots[11], body=fmt.render_child_body(10, blocked))
    snapshots[13] = replace(
        snapshots[11],
        number=13,
        title=fmt.child_title(fmt.ParentRef("開発ループ", 10), "依存作業を進める"),
        body=fmt.render_child_body(
            10, {**CHILD_SECTIONS, "依存関係": "#11。停止中のため保留。"}
        ),
    )
    tasks = [
        fmt.ChildTask(11, "共通書式を実装する"),
        fmt.ChildTask(12, "委任を整える", "completed"),
        fmt.ChildTask(13, "依存作業を進める"),
    ]
    snapshots[10] = replace(
        snapshots[10],
        children=(11, 12, 13),
        body=fmt.render_parent_body(
            {
                **PARENT_SECTIONS,
                "進捗概要": "#11 停止、#13 保留。独立した #12 は継続して完了。",
            },
            tasks,
        ),
    )
    assert fmt.check_consistency(snapshots) == ()
    assert "API の権限不足" not in snapshots[10].body
    snapshots[11] = replace(
        snapshots[11],
        body=fmt.render_child_body(
            10,
            {
                **CHILD_SECTIONS,
                "作業状況・停止事項": "再開。読取権限の復旧を確認済み。次: 検査を続ける。",
            },
        ),
    )
    snapshots[10] = replace(
        snapshots[10],
        body=fmt.render_parent_body(
            {**PARENT_SECTIONS, "進捗概要": "#11 再開、#13 保留解除、#12 完了。"}, tasks
        ),
    )
    assert fmt.check_consistency(snapshots) == ()


def test_proposal_identity_survives_title_and_parent_changes(fmt: ModuleType) -> None:
    source = fmt.ProposalSource("R-20261018T080000-02", 90, "継続 / 調査")
    body = fmt.render_child_body(10, CHILD_SECTIONS, source=source)
    moved = fmt.render_child_body(20, CHILD_SECTIONS, source=source)
    assert fmt.parse_proposal_source(body) == fmt.parse_proposal_source(moved) == source
    assert fmt.index_proposals([(11, body)]) == {(90, source.proposal_id): 11}
    with pytest.raises(ValueError, match="duplicate proposal"):
        fmt.index_proposals([(11, body), (99, moved)])
    changed = fmt.render_child_body(
        10, CHILD_SECTIONS, source=replace(source, option="取消")
    )
    assert fmt.parse_proposal_source(changed) != source
    for bad in (
        "\n" + body,
        body + body.splitlines()[0],
        body.replace("%20", "%2G"),
        body.replace("report=#90", "report=#0"),
    ):
        with pytest.raises(ValueError):
            fmt.parse_proposal_source(bad)


def test_quoted_metadata_and_html_examples_are_not_issue_data(fmt: ModuleType) -> None:
    example = "```markdown\n<!-- ops-report-proposal: R-1 report=#90 option= -->\n```"
    assert fmt.parse_proposal_source("## 書式例\n\n" + example) is None
    assert fmt.index_proposals([(10, example)]) == {}
    assert fmt.parse_sections("## HTMLの例\n\n```html\n<!--\n```\n説明。") == {
        "HTMLの例": "説明。"
    }
    assert fmt.parse_sections("<!--\n```\n## 隠した見出し\n-->\n## 本文\n\n説明。") == {
        "本文": "説明。"
    }
    marker = fmt.proposal_marker(fmt.ProposalSource("R-1", 90))
    with pytest.raises(ValueError):
        fmt.parse_proposal_source("  " + marker)


FAKE_GH = r"""
import json, os, sys
args = sys.argv[1:]
with open(os.environ["ISSUE_FAKE_CALLS"], "a", encoding="utf-8") as f:
    f.write(json.dumps(args) + "\n")
if args == ["repo", "view", "--json", "nameWithOwner"]:
    print(json.dumps({"nameWithOwner": "o/r"}))
    sys.exit(0)
if args[:1] != ["api"] or "--method" not in args or args[args.index("--method") + 1] != "GET":
    raise RuntimeError("only explicit GETs are allowed: " + repr(args))
with open(os.environ["ISSUE_FAKE_ROUTES"], encoding="utf-8") as f:
    routes = json.load(f)
if args[-1] not in routes:
    raise RuntimeError("unconfigured endpoint: " + args[-1])
route = routes[args[-1]]
status = route.get("status", 200)
if not route.get("omit_headers"):
    print("HTTP/2.0 " + str(status) + " test\r\nContent-Type: application/json\r\n\r")
print(route["raw"] if "raw" in route else json.dumps(route["body"]))
if status != 200:
    print("gh: error (HTTP " + str(status) + ")", file=sys.stderr)
    sys.exit(1)
"""


@dataclass
class CliEnv:
    root: Path
    env: dict[str, str]
    routes: dict[str, Any]

    def run(self, *args: str, spec: object = None) -> subprocess.CompletedProcess[str]:
        (self.root / "routes.json").write_text(
            json.dumps(self.routes), encoding="utf-8"
        )
        return subprocess.run(
            [sys.executable, "-m", "issues", *args],
            env=self.env,
            cwd=self.root,
            input=json.dumps(spec),
            capture_output=True,
            text=True,
            check=False,
            timeout=60,
        )

    def calls(self) -> list[list[str]]:
        path = self.root / "calls.jsonl"
        return (
            [json.loads(line) for line in path.read_text().splitlines()]
            if path.exists()
            else []
        )


def routes_for(snapshots: dict[int, Any]) -> dict[str, Any]:
    routes: dict[str, Any] = {}
    raw = {
        n: {
            "number": n,
            "repository_url": "https://api.github.com/repos/o/r",
            "title": issue.title,
            "body": issue.body,
            "labels": [{"name": label} for label in sorted(issue.labels)],
            "state": issue.state,
            "state_reason": issue.state_reason,
        }
        for n, issue in snapshots.items()
    }
    for n, issue in snapshots.items():
        base = f"repos/o/r/issues/{n}"
        routes[base] = {"body": raw[n]}
        routes[f"{base}/parent"] = (
            {"body": raw[issue.parent]}
            if issue.parent is not None
            else {"status": 404, "body": {"message": "No parent issue found"}}
        )
        for index in range(len(issue.children) // 100 + 1):
            rows = [raw[k] for k in issue.children[index * 100 : (index + 1) * 100]]
            routes[f"{base}/sub_issues?per_page=100&page={index + 1}"] = {"body": rows}
    return routes


@pytest.fixture
def issue_cli(tmp_path: Path, fmt: ModuleType) -> CliEnv:
    fake = tmp_path / "bin"
    fake.mkdir()
    gh = fake / "gh"
    gh.write_text(f"#!{sys.executable}\n{FAKE_GH}", encoding="utf-8")
    gh.chmod(0o755)
    env = {
        **os.environ,
        "PATH": f"{fake}:{os.environ['PATH']}",
        "PYTHONPATH": str(OPS),
        "ISSUE_FAKE_ROUTES": str(tmp_path / "routes.json"),
        "ISSUE_FAKE_CALLS": str(tmp_path / "calls.jsonl"),
    }
    return CliEnv(tmp_path, env, routes_for(family(fmt)))


def test_cli_starting_from_a_child_checks_parent_and_siblings_without_writes(
    issue_cli: CliEnv,
) -> None:
    result = issue_cli.run("check", "--issue", "11")
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == {
        "ok": True,
        "issues": [10, 11, 12],
        "problems": [],
    }
    assert all(
        call[:3] == ["repo", "view", "--json"]
        or call[call.index("--method") + 1] == "GET"
        for call in issue_cli.calls()
    )


def test_cli_migration_checks_only_the_selected_family(
    fmt: ModuleType, issue_cli: CliEnv
) -> None:
    before = family(fmt)
    before[11] = replace(before[11], title="[R-2026W42-01] 旧書式")
    issue_cli.routes = routes_for(before)
    result = issue_cli.run("check", "--repo", "o/r", "--issue", "10")
    assert result.returncode == 1
    assert not json.loads(result.stdout)["ok"]
    issue_cli.routes = routes_for(family(fmt))
    fixed = issue_cli.run("check", "--repo", "o/r", "--issue", "10")
    assert fixed.returncode == 0, fixed.stderr
    assert all("issues/50" not in call[-1] for call in issue_cli.calls())


@pytest.mark.parametrize(
    ("endpoint", "response"),
    [
        ("10", {"status": 404, "body": {"message": "Not Found"}}),
        ("10/parent", {"status": 403, "body": {"message": "Forbidden"}}),
        ("10/parent", {"status": 500, "body": {"message": "failure"}}),
        (
            "10/parent",
            {"status": 404, "body": {"message": "not a documented missing parent"}},
        ),
        ("10/parent", {"body": None}),
        ("10/parent", {"raw": "not-json"}),
        ("10/parent", {"omit_headers": True, "body": {"message": "No parent issue found"}}),
        # GitHub answers a missing parent with "No parent issue found"; a bare
        # "Not Found" means the issue itself is gone and must not read as no parent.
        ("10/parent", {"status": 404, "body": {"message": "Not Found"}}),
        (
            "10/sub_issues?per_page=100&page=1",
            {"status": 404, "body": {"message": "Not Found"}},
        ),
        ("10/sub_issues?per_page=100&page=1", {"body": {"unexpected": []}}),
        (
            "10/sub_issues?per_page=100&page=1",
            {
                "body": [
                    {
                        "number": 11,
                        "repository_url": "https://api.github.com/repos/other/r",
                    }
                ]
            },
        ),
    ],
)
def test_cli_api_failures_never_become_a_successful_partial_check(
    issue_cli: CliEnv, endpoint: str, response: dict[str, Any]
) -> None:
    issue_cli.routes[f"repos/o/r/issues/{endpoint}"] = response
    result = issue_cli.run("check", "--issue", "10", "--repo", "o/r")
    assert result.returncode == 2
    assert result.stdout == ""
    assert "issue-format:" in result.stderr


def test_cli_follows_all_subissue_pages(fmt: ModuleType, issue_cli: CliEnv) -> None:
    snapshots = family(fmt)
    template = snapshots[11]
    snapshots = {10: snapshots[10]}
    tasks = []
    for number in range(11, 112):
        title = f"作業 {number}"
        snapshots[number] = replace(
            template,
            number=number,
            title=fmt.child_title(fmt.ParentRef("開発ループ", 10), title),
        )
        tasks.append(fmt.ChildTask(number, title))
    snapshots[10] = replace(
        snapshots[10],
        children=tuple(range(11, 112)),
        body=fmt.render_parent_body(PARENT_SECTIONS, tasks),
    )
    issue_cli.routes = routes_for(snapshots)
    result = issue_cli.run("check", "--issue", "10", "--repo", "o/r")
    assert result.returncode == 0, result.stderr
    assert len(json.loads(result.stdout)["issues"]) == 102
    assert any(
        call[-1] == "repos/o/r/issues/10/sub_issues?per_page=100&page=2"
        for call in issue_cli.calls()
    )


def child_spec() -> dict[str, Any]:
    return {
        "role": "child",
        "summary": "共通書式を実装する",
        "parent": {"identifier": "開発ループ", "number": 10},
        "sections": CHILD_SECTIONS,
        "labels": ["enhancement", "ci", "module: base"],
        "source": {
            "proposal_id": "R-20261018-01",
            "report_number": 90,
            "option": "継続",
        },
    }


def test_render_is_deterministic_and_always_a_local_dry_run(
    issue_cli: CliEnv, fmt: ModuleType
) -> None:
    first = issue_cli.run("render", "--spec", "-", spec=child_spec())
    second = issue_cli.run("render", "--dry-run", "--spec", "-", spec=child_spec())
    assert first.returncode == second.returncode == 0, first.stderr
    assert first.stdout == second.stdout
    output = json.loads(first.stdout)
    assert output["dry_run"] is True
    assert output["title"] == "[開発ループ #10] 共通書式を実装する"
    assert set(output["labels"]) == {fmt.CHILD_LABEL, "enhancement", "ci", "module: base"}
    assert fmt.parse_proposal_source(output["body"]).proposal_id == "R-20261018-01"
    assert issue_cli.calls() == []
    assert (
        issue_cli.run("render", "--apply", "--spec", "-", spec=child_spec()).returncode
        == 2
    )


@pytest.mark.parametrize(
    "change",
    [
        {"parent": None},
        {"parent": {"identifier": "開発ループ", "number": True}},
        {"children": []},
        {"labels": ["issue:parent"]},
        {"unknown": "not silently ignored"},
        {"sections": {"このタスクの目的": "目的だけ"}},
    ],
)
def test_render_invalid_input_fails_without_gh(
    issue_cli: CliEnv, change: dict[str, Any]
) -> None:
    result = issue_cli.run("render", "--spec", "-", spec={**child_spec(), **change})
    assert result.returncode == 2
    assert result.stdout == ""
    assert issue_cli.calls() == []


def test_render_parent_preserves_agreement_and_closed_acceptance(
    issue_cli: CliEnv, fmt: ModuleType
) -> None:
    sections = {
        **PARENT_SECTIONS,
        **CHILD_SECTIONS,
        "合意記録": "以前の合意を削除しない。",
        "Acceptance checklist": CHILD_SECTIONS["Acceptance checklist"].replace(
            "- [ ]", "- [x]"
        ),
    }
    spec = {
        "role": "parent",
        "identifier": "書式",
        "summary": "共通書式を実装する",
        "parent": {"identifier": "開発ループ", "number": 10},
        "sections": sections,
        "children": [{"number": 13, "title": "本文を検査する", "status": "cancelled"}],
        "labels": ["research"],
    }
    result = issue_cli.run("render", "--spec", "-", spec=spec)
    assert result.returncode == 0, result.stderr
    output = json.loads(result.stdout)
    assert set(output["labels"]) == {fmt.PARENT_LABEL, fmt.CHILD_LABEL, "research"}
    assert "以前の合意を削除しない。" in output["body"]
    assert "- [x] 親子関係の矛盾を検出できる" in output["body"]
    assert "- [ ] #13 本文を検査する（取消）" in output["body"]
    assert issue_cli.calls() == []


@pytest.mark.parametrize(
    ("name", "required", "labels"),
    [
        ("06_parent.md", "PARENT_REQUIRED", {"issue:parent"}),
        ("07_child.md", "CHILD_REQUIRED", {"issue:child"}),
        ("04_research_theme.md", "PARENT_REQUIRED", {"research", "issue:parent"}),
        ("05_research_cycle.md", "CHILD_REQUIRED", {"research", "issue:child"}),
    ],
)
def test_manual_templates_share_the_contract(
    fmt: ModuleType, name: str, required: str, labels: set[str]
) -> None:
    text = (TEMPLATES / name).read_text(encoding="utf-8")
    metadata = yaml.safe_load(text.split("---", 2)[1])
    assert set(metadata["labels"]) == labels
    assert set(getattr(fmt, required)) <= set(fmt.parse_sections(text))
    assert ".agents/ops/issues/README.md" in text


def test_issue_docs_relative_links_resolve() -> None:
    docs = [OPS / "issues/README.md", *TEMPLATES.glob("*.md")]
    for doc in docs:
        for target in re.findall(
            r"\[[^]]+\]\(([^)]+)\)", doc.read_text(encoding="utf-8")
        ):
            if "://" not in target:
                assert (doc.parent / target.split("#")[0]).exists(), (doc, target)
