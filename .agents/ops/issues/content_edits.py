"""Read issue body revisions and identify edits around a single PATCH."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

QUERY = """
query IssueBodyEdits($owner: String!, $repo: String!, $number: Int!, $after: String) {
  repository(owner: $owner, name: $repo) {
    issue(number: $number) {
      number body createdAt includesCreatedEdit
      userContentEdits(first: 100, after: $after) {
        totalCount nodes { id diff editedAt deletedAt }
        pageInfo { hasNextPage endCursor }
      }
    }
  }
}
"""


class GraphQLReader(Protocol):
    repo: str

    def graphql(self, query: str, variables: dict[str, Any]) -> dict[str, Any]: ...


@dataclass(frozen=True)
class Edit:
    id: str
    body: str
    edited_at: str


@dataclass(frozen=True)
class Checkpoint:
    body: str
    edit_id: str | None
    edit_count: int
    created_at: str


@dataclass(frozen=True)
class BodyHistory:
    checkpoint: Checkpoint
    edits: tuple[Edit, ...]


def _object(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("invalid issue edit history object")
    return value


def _text(value: Any) -> str:
    if not isinstance(value, str):
        raise ValueError("issue edit history text is missing or redacted")
    return value


def read_body_history(gh: GraphQLReader, number: int) -> BodyHistory:
    """Read every page newest first; partial, changing and redacted history fails."""
    owner, repo = gh.repo.split("/", 1)
    after: str | None = None
    cursors: set[str] = set()
    edits: list[Edit] = []
    ids: set[str] = set()
    initial: tuple[str, str, int, bool] | None = None
    while True:  # Pagination, never a polling loop.
        data = gh.graphql(
            QUERY, {"owner": owner, "repo": repo, "number": number, "after": after}
        )
        issue = _object(_object(data.get("repository")).get("issue"))
        connection = _object(issue.get("userContentEdits"))
        count, includes_created = (
            connection.get("totalCount"),
            issue.get("includesCreatedEdit"),
        )
        if (
            issue.get("number") != number
            or type(count) is not int
            or count < 0
            or type(includes_created) is not bool
        ):
            raise ValueError("invalid issue edit history identity/count")
        current = (
            _text(issue.get("body")),
            _text(issue.get("createdAt")),
            count,
            includes_created,
        )
        if initial is not None and current != initial:
            raise ValueError("issue edit history changed during pagination")
        initial = current
        nodes = connection.get("nodes")
        if not isinstance(nodes, list) or len(nodes) > 100:
            raise ValueError("invalid issue edit history page")
        for raw in nodes:
            node = _object(raw)
            if not {"id", "diff", "editedAt", "deletedAt"} <= node.keys():
                raise ValueError("incomplete issue edit history revision")
            edit_id = _text(node.get("id"))
            if not edit_id or edit_id in ids or node.get("deletedAt") is not None:
                raise ValueError("duplicate or redacted issue edit history")
            ids.add(edit_id)
            edits.append(
                Edit(edit_id, _text(node.get("diff")), _text(node.get("editedAt")))
            )
        page = _object(connection.get("pageInfo"))
        if type(page.get("hasNextPage")) is not bool:
            raise ValueError("invalid issue edit history pagination")
        if not page["hasNextPage"]:
            break
        after = _text(page.get("endCursor"))
        if not nodes or not after or after in cursors:
            raise ValueError("invalid/repeated issue edit history cursor")
        cursors.add(after)
    assert initial is not None
    body, created_at, count, _ = initial
    if len(edits) != count or (edits and edits[0].body != body):
        raise ValueError(
            "incomplete issue edit history or latest diff is not the full body"
        )
    return BodyHistory(
        Checkpoint(body, edits[0].id if edits else None, count, created_at),
        tuple(edits),
    )


def intervening_edits(
    before: Checkpoint, after: BodyHistory, written_body: str
) -> tuple[Edit, ...]:
    """Identify our unique body revision by content, never actor or second timestamps."""
    if before.edit_count:
        offset = len(after.edits) - before.edit_count
        if (
            offset < 0
            or offset >= len(after.edits)
            or after.edits[offset].id != before.edit_id
            or after.edits[offset].body != before.body
        ):
            raise ValueError("issue edit history lost its pre-PATCH checkpoint")
        recent = after.edits[:offset]
    else:
        recent = after.edits
        # GitHub materializes the creation revision when the first edit is made.
        if (
            recent
            and recent[-1].body == before.body
            and recent[-1].edited_at == before.created_at
        ):
            recent = recent[:-1]
    own = [edit for edit in recent if edit.body == written_body]
    if len(own) != 1:
        raise ValueError(
            "cannot uniquely identify the checklist PATCH in issue edit history"
        )
    return tuple(edit for edit in recent if edit.id != own[0].id)
