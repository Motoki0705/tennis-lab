import fnmatch
import re
from pathlib import Path
from typing import Any, cast

import pytest
import yaml

ROOT = Path(__file__).parents[3]
GITHUB = ROOT / ".github"


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as file:
        return cast(dict[str, Any], yaml.safe_load(file))


def _pr_labels_for(*changed_files: str) -> set[str]:
    config = _load_yaml(GITHUB / "labeler.yml")
    matched: set[str] = set()
    for label, rules in config.items():
        patterns = rules[0]["changed-files"][0]["any-glob-to-any-file"]
        if isinstance(patterns, str):
            patterns = [patterns]
        if any(
            fnmatch.fnmatch(changed_file, pattern)
            for changed_file in changed_files
            for pattern in patterns
        ):
            matched.add(label)
    return matched


def _issue_labels_for(
    title: str, body: str, *, inputs: dict[str, Any] | None = None
) -> set[str]:
    # github/issue-labeler main.ts builds this string from its numeric inputs.
    # Each label requires all its expressions to match the assembled content.
    if inputs is None:
        workflow = _load_yaml(GITHUB / "workflows/issue-labeler.yml")
        inputs = workflow["jobs"]["label"]["steps"][0]["with"]
    content = f"{title}\n\n" if int(inputs["include-title"]) == 1 else ""
    if int(inputs["include-body"]) == 1:
        content += body
    config = _load_yaml(GITHUB / "issue-labeler.yml")
    matched: set[str] = set()
    for label, expressions in config.items():
        for expression in expressions:
            regex = re.fullmatch(r"/(.*)/([a-z]*)", expression)
            assert regex is not None, f"invalid issue-labeler regex: {expression}"
            flags = re.IGNORECASE if "i" in regex.group(2) else 0
            if not re.search(regex.group(1), content, flags=flags):
                break
        else:
            matched.add(label)
    return matched


def test_dependabot_covers_project_dependencies_and_submodules() -> None:
    config = _load_yaml(GITHUB / "dependabot.yml")
    updates = {update["package-ecosystem"]: update for update in config["updates"]}

    assert config["version"] == 2
    assert set(updates) == {"github-actions", "gitsubmodule", "uv"}
    assert all(update["directory"] == "/" for update in updates.values())
    assert updates["gitsubmodule"]["schedule"]["interval"] == "daily"
    assert updates["gitsubmodule"]["open-pull-requests-limit"] == 4
    assert updates["gitsubmodule"]["labels"] == ["dependencies", "third-party"]

    gitmodules = (ROOT / ".gitmodules").read_text(encoding="utf-8")
    assert gitmodules.count("path = third_party/") == 4
    assert "git@github.com:" not in gitmodules


def test_pull_request_labeler_matches_representative_files() -> None:
    assert _pr_labels_for("third_party/GVHMR") == {"dependencies", "third-party"}
    assert _pr_labels_for("src/tasks/blcs/models/model.py") == {"module: blcs"}
    assert _pr_labels_for("docs/setup.md") == {"documentation"}
    assert _pr_labels_for("tests/unit/utils/test_io.py") == {"tests"}
    assert _pr_labels_for(".github/workflows/ci.yml") == {"ci"}


def test_issue_labeler_only_configures_work_types_and_scans_title() -> None:
    assert set(_load_yaml(GITHUB / "issue-labeler.yml")) == {
        "bug",
        "enhancement",
        "research",
        "documentation",
        "followup",
    }
    workflow = _load_yaml(GITHUB / "workflows/issue-labeler.yml")
    inputs = workflow["jobs"]["label"]["steps"][0]["with"]
    assert inputs["include-body"] == 0
    assert inputs["include-title"] == 1


@pytest.mark.parametrize(
    ("title", "expected"),
    [
        ("[バグ] GVHMR のテストが失敗する", {"bug"}),
        ("[修正] CIの検査を直す", {"bug"}),
        ("[bug] DINO", {"bug"}),
        ("[BUG] 大文字も扱う", {"bug"}),
        ("[機能] BLCSを改善する", {"enhancement"}),
        ("[機能・xx] PLCSを改善する", {"enhancement"}),
        ("[feature] コート検出", {"enhancement"}),
        ("[Feature・xx] 共通基盤", {"enhancement"}),
        ("[調査] ボール検出", {"research"}),
        ("[研究テーマ] 合成データ", {"research"}),
        ("[研究サイクル] src/submodules", {"research"}),
        ("[research] src/tasks/blcs", {"research"}),
        ("[ドキュメント] GitHub Actions", {"documentation"}),
        ("[docs] third_party/GVHMR", {"documentation"}),
        ("[doc] 依存関係", {"documentation"}),
        ("[追補] pytest", {"followup"}),
        ("[識別名] 親の目的を達成する", set()),
        ("[識別名 #123] 子のCIを修正する", set()),
        ("[識別名 #123] [別名] 子が親も兼ねる", set()),
        ("[bug #123] 作業種別と同名の親の子", set()),
        ("[識別名 #123] [バグ] タイトル途中のprefix", set()),
        ("タイトル途中の CI と pytest", set()),
        ("prefixなし src/tasks/blcs のテスト", set()),
        ("[質問] 手順を確認する", set()),
        ("[機能改善] prefixに一致しない", set()),
        ("運用レポート 2026-10-30", set()),
    ],
)
def test_issue_body_and_area_words_do_not_add_labels(
    title: str, expected: set[str]
) -> None:
    body = (
        "[バグ] [機能] [調査] [docs] [追補]\n"
        "## 依存関係\nテスト CI pytest src/tasks/blcs src/tasks/plcs\n"
        "src/tasks/base src/tasks/ball_detection src/tasks/court_detection\n"
        "src/tennis_scene src/synthetic_data_generation third_party/GVHMR DINO\n"
    )
    assert _issue_labels_for(title, body) == expected


@pytest.mark.parametrize(
    ("title_input", "body_input", "expected"),
    [(1, 0, {"bug"}), (0, 1, {"enhancement"}), (0, 0, set()), (1, 1, {"bug"})],
)
def test_issue_labeler_model_obeys_inputs(
    title_input: int, body_input: int, expected: set[str]
) -> None:
    assert (
        _issue_labels_for(
            "[バグ] title",
            "[機能] body",
            inputs={"include-title": title_input, "include-body": body_input},
        )
        == expected
    )


def test_labeler_workflows_support_events_and_manual_verification() -> None:
    pull_request_workflow = (GITHUB / "workflows/pull-request-labeler.yml").read_text(
        encoding="utf-8"
    )
    issue_workflow = (GITHUB / "workflows/issue-labeler.yml").read_text(
        encoding="utf-8"
    )

    assert "pull_request_target:" in pull_request_workflow
    assert "workflow_dispatch:" in pull_request_workflow
    assert "pr_number:" in pull_request_workflow
    assert "issues:" in issue_workflow
    assert "workflow_dispatch:" in issue_workflow
    assert "issue_number:" in issue_workflow
