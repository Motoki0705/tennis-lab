import fnmatch
import re
from pathlib import Path
from typing import Any, cast

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


def test_issue_labeler_matches_type_area_and_module_conditions() -> None:
    labels = _issue_labels_for(
        "[バグ] GVHMR のテストが失敗する\n\n"
        "third_party/GVHMR と pytest の互換性を確認する"
    )

    assert labels == {"bug", "module: submodules", "tests", "third-party"}

    assert _issue_labels_for("src/base の共通機能を改善する") == {"module: base"}


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
