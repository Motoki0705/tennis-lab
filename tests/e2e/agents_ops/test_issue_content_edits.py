"""Issue revision contracts and the real GraphQL client through an offline gh."""

from __future__ import annotations

import copy
import importlib
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

OPS = Path(__file__).resolve().parents[3] / ".agents/ops"
CREATED = "2026-10-31T00:00:00Z"
EDITED = "2026-10-31T00:00:01Z"


@pytest.fixture
def api(monkeypatch: pytest.MonkeyPatch) -> Any:
    monkeypatch.syspath_prepend(str(OPS))
    return importlib.import_module("issues.content_edits")


def edit(number: int, body: str) -> dict[str, Any]:
    return {"id": f"edit-{number}", "diff": body, "editedAt": EDITED, "deletedAt": None}


def page(
    nodes: list[dict[str, Any]],
    *,
    total: int | None = None,
    next_cursor: str | None = None,
) -> dict[str, Any]:
    return {
        "repository": {
            "issue": {
                "number": 1,
                "body": "latest",
                "createdAt": CREATED,
                "includesCreatedEdit": bool(nodes),
                "userContentEdits": {
                    "totalCount": len(nodes) if total is None else total,
                    "nodes": nodes,
                    "pageInfo": {
                        "hasNextPage": next_cursor is not None,
                        "endCursor": next_cursor,
                    },
                },
            }
        }
    }


def reader(pages: dict[str | None, dict[str, Any]]) -> Any:
    return SimpleNamespace(
        repo="o/r",
        graphql=lambda query, variables: copy.deepcopy(pages[variables["after"]]),
    )


def test_history_pages_preserve_complete_revision_order(api: Any) -> None:
    nodes = [edit(n, "latest" if n == 101 else f"body-{n}") for n in range(101, 0, -1)]
    gh = reader(
        {
            None: page(nodes[:100], total=101, next_cursor="next"),
            "next": page(nodes[100:], total=101),
        }
    )
    result = api.read_body_history(gh, 1)
    assert result.checkpoint.edit_count == 101
    assert result.checkpoint.edit_id == "edit-101"
    assert result.edits[-1].body == "body-1"


@pytest.mark.parametrize(
    "bad",
    [
        "redacted",
        "duplicate",
        "incomplete",
        "latest_diff",
        "missing_nodes",
        "cursor",
        "changing_page",
    ],
)
def test_invalid_history_cannot_be_used_as_concurrency_evidence(
    api: Any, bad: str
) -> None:
    value = page([edit(1, "latest")])
    issue = value["repository"]["issue"]
    connection = issue["userContentEdits"]
    pages: dict[str | None, dict[str, Any]] = {None: value}
    if bad == "redacted":
        connection["nodes"][0].update(deletedAt=EDITED, diff=None)
    elif bad == "duplicate":
        connection["nodes"] *= 2
        connection["totalCount"] = 2
    elif bad == "incomplete":
        connection["totalCount"] = 2
    elif bad == "latest_diff":
        connection["nodes"][0]["diff"] = "not the current body"
    elif bad == "missing_nodes":
        connection["nodes"] = None
    elif bad == "cursor":
        connection["pageInfo"] = {"hasNextPage": True, "endCursor": None}
    else:
        connection["pageInfo"] = {"hasNextPage": True, "endCursor": "next"}
        connection["totalCount"] = 2
        pages["next"] = page([edit(0, "old")], total=2)
        pages["next"]["repository"]["issue"]["body"] = "changed during pagination"
    with pytest.raises(ValueError):
        api.read_body_history(reader(pages), 1)


@pytest.mark.parametrize(
    "scenario",
    ["single", "before", "after", "duplicate_own", "missing_anchor", "missing_own"],
)
def test_patch_identity_uses_revision_ids_and_content_with_equal_timestamps(
    api: Any, scenario: str
) -> None:
    before = api.Checkpoint("old", "base", 1, CREATED)
    base = api.Edit("base", "old", EDITED)
    own = api.Edit("own", "our checklist", EDITED)
    foreign = api.Edit("foreign", "Gemini rewrite", EDITED)
    revisions: tuple[Any, ...] = (own, base)
    if scenario == "before":
        revisions = (own, foreign, base)
    elif scenario == "after":
        revisions = (foreign, own, base)
    elif scenario == "duplicate_own":
        revisions = (api.Edit("other-own", own.body, EDITED), own, base)
    elif scenario == "missing_anchor":
        revisions = (own, api.Edit("different-base", "old", EDITED))
    elif scenario == "missing_own":
        revisions = (foreign, base)
    after = api.BodyHistory(
        api.Checkpoint(revisions[0].body, revisions[0].id, len(revisions), CREATED),
        revisions,
    )
    if scenario in {"duplicate_own", "missing_anchor", "missing_own"}:
        with pytest.raises(ValueError):
            api.intervening_edits(before, after, own.body)
    else:
        assert [e.id for e in api.intervening_edits(before, after, own.body)] == (
            [] if scenario == "single" else ["foreign"]
        )


def test_first_patch_materialized_creation_revision_is_not_an_interruption(
    api: Any, ops_env: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    for key, value in ops_env.env.items():
        monkeypatch.setenv(key, value)
    gh = importlib.import_module("parent.github").GitHub("o/r")
    before = api.read_body_history(gh, 50)
    gh.write("PATCH", "repos/o/r/issues/50", {"body": "first body edit"})
    after = api.read_body_history(gh, 50)
    assert before.checkpoint.edit_count == 0 and after.checkpoint.edit_count == 2
    assert api.intervening_edits(before.checkpoint, after, "first body edit") == ()
    calls = ops_env.calls()
    assert len([c for c in calls if c[:2] == ["api", "graphql"]]) == 2


@pytest.mark.parametrize("failure", ["transport", "partial_graphql"])
def test_graphql_failure_is_never_partial_success(
    api: Any, monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    client = importlib.import_module("parent.github").GitHub("o/r")
    module = importlib.import_module("parent.github")

    def run(args: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        request = json.loads(kwargs["input"])
        assert "userContentEdits" in request["query"]
        assert request["variables"]["number"] == 1
        return subprocess.CompletedProcess(
            args, 1 if failure == "transport" else 0,
            json.dumps({"data": page([]), "errors": [{"message": "unavailable history"}]}),
            "transport failure",
        )

    monkeypatch.setattr(module.subprocess, "run", run)
    with pytest.raises(ValueError, match="graphql"):
        api.read_body_history(client, 1)
