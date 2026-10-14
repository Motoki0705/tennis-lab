import fnmatch
import importlib
import re
import sys
from pathlib import Path
from typing import Any, cast

import pytest
import yaml

ROOT = Path(__file__).parents[3]
GITHUB = ROOT / ".github"

sys.path.insert(0, str(ROOT / "scripts"))
load_manifest: Any = importlib.import_module("sync_github_labels").load_manifest


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


def _issue_labels_for(content: str) -> set[str]:
    config = _load_yaml(GITHUB / "issue-labeler.yml")
    matched: set[str] = set()
    for label, expressions in config.items():
        for expression in expressions:
            regex = re.fullmatch(r"/(.*)/([a-z]*)", expression)
            assert regex is not None, f"invalid issue-labeler regex: {expression}"
            flags = re.IGNORECASE if "i" in regex.group(2) else 0
            if re.search(regex.group(1), content, flags=flags):
                matched.add(label)
                break
    return matched


def test_dependabot_covers_project_dependencies_and_submodules() -> None:
    config = _load_yaml(GITHUB / "dependabot.yml")
    updates = {
        update["package-ecosystem"]: update for update in config["updates"]
    }

    assert config["version"] == 2
    assert set(updates) == {"github-actions", "gitsubmodule", "uv"}
    assert all(update["directory"] == "/" for update in updates.values())
    assert updates["gitsubmodule"]["schedule"]["interval"] == "daily"
    assert updates["gitsubmodule"]["open-pull-requests-limit"] == 4
    assert updates["gitsubmodule"]["labels"] == ["area: dependencies", "area: third-party"]

    gitmodules = (ROOT / ".gitmodules").read_text(encoding="utf-8")
    assert gitmodules.count("path = third_party/") == 4
    assert "git@github.com:" not in gitmodules


def test_pull_request_labeler_matches_representative_files() -> None:
    assert _pr_labels_for("third_party/GVHMR") == {
        "area: dependencies",
        "area: third-party",
    }
    assert _pr_labels_for("src/tasks/blcs/models/model.py") == {"area: blcs"}
    assert _pr_labels_for("docs/setup.md") == {"area: docs"}
    assert _pr_labels_for("tests/unit/utils/test_io.py") == {"area: tests"}
    assert _pr_labels_for(".github/workflows/ci.yml") == {"area: ci"}
    assert _pr_labels_for("knowledge/summary.md") == {"area: docs", "area: knowledge"}


@pytest.mark.parametrize(
    ("title", "expected"),
    [
        ("[バグ] GVHMR のテストが失敗する", {"type: bug"}),
        ("[Bug] crash", {"type: bug"}),
        ("[機能] BLCSの損失を再設計する", {"type: feature"}),
        ("[機能・整理] 共通基盤の再編", {"type: feature"}),
        ("[Feature] add exporter", {"type: feature"}),
        ("[調査] ARISでSfMを比較する", {"type: research"}),
        ("[研究テーマ] 欠損補完", {"type: research"}),
        ("[研究サイクル] 欠損補完 #2: 融合", {"type: research"}),
        ("[追補] PR #1093 のレビュー軽微指摘", {"type: followup"}),
        ("BLCS の [バグ] を直す", set()),
        ("[R-2026W42-02] big.py を分割する", set()),
    ],
)
def test_issue_labeler_sets_type_from_title_prefix(
    title: str, expected: set[str]
) -> None:
    assert _issue_labels_for(title) == expected


def test_issue_labeler_ignores_body() -> None:
    workflow = yaml.safe_load(
        (GITHUB / "workflows/issue-labeler.yml").read_text(encoding="utf-8")
    )
    [step] = workflow["jobs"]["label"]["steps"]
    assert step["with"]["include-title"] == 1
    assert step["with"]["include-body"] == 0
    config = _load_yaml(GITHUB / "issue-labeler.yml")
    assert all(label.startswith("type: ") for label in config)


def _referenced_labels() -> dict[str, set[str]]:
    """Every label name a config file or ops constant hands to GitHub."""
    refs: dict[str, set[str]] = {
        "labeler.yml": set(_load_yaml(GITHUB / "labeler.yml")),
        "issue-labeler.yml": set(_load_yaml(GITHUB / "issue-labeler.yml")),
        "dependabot.yml": {
            label
            for update in _load_yaml(GITHUB / "dependabot.yml")["updates"]
            for label in update.get("labels", [])
        },
    }
    for template in sorted((GITHUB / "ISSUE_TEMPLATE").glob("*.md")):
        front = template.read_text(encoding="utf-8").split("---")[1]
        refs[template.name] = set(yaml.safe_load(front).get("labels", []))
    sys.path.insert(0, str(ROOT / ".agents/ops/review"))
    sys.path.insert(0, str(ROOT / ".agents/ops/weekly-report"))
    try:
        policy = importlib.import_module("review_policy")
        collect = importlib.import_module("opsreport.collect")
    finally:
        sys.path.remove(str(ROOT / ".agents/ops/review"))
        sys.path.remove(str(ROOT / ".agents/ops/weekly-report"))
    refs["review_policy.py"] = {
        policy.NEEDS_HUMAN_LABEL,
        policy.CHANGES_REQUESTED_LABEL,
        *policy.FOLLOWUP_LABELS,
        *policy.AGENT_LABELS,
    }
    refs["opsreport/collect.py"] = {collect.REPORT_LABEL, collect.AI_PROPOSED_LABEL}
    return refs


def test_every_referenced_label_is_in_the_manifest() -> None:
    names = {spec.name for spec in load_manifest()}
    for source, labels in _referenced_labels().items():
        assert labels, source
        assert labels <= names, f"{source}: {sorted(labels - names)}"


def test_labeler_workflows_support_events_and_manual_verification() -> None:
    pull_request_workflow = (
        GITHUB / "workflows/pull-request-labeler.yml"
    ).read_text(encoding="utf-8")
    issue_workflow = (GITHUB / "workflows/issue-labeler.yml").read_text(
        encoding="utf-8"
    )

    assert "pull_request_target:" in pull_request_workflow
    assert "workflow_dispatch:" in pull_request_workflow
    assert "pr_number:" in pull_request_workflow
    assert "issues:" in issue_workflow
    assert "workflow_dispatch:" in issue_workflow
    assert "issue_number:" in issue_workflow
