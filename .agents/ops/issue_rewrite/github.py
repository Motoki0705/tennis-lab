"""GitHub snapshot, existing execution-record parsing, and edit-history checks."""

from __future__ import annotations

import json
import subprocess
from typing import Any

from parent.github import GitHub as ParentGitHub
from parent.judgement_comments import PREFIX, owned_comment, read_json
from parent.policy import events, nonempty, object_value

HISTORY_QUERY = """
query($owner: String!, $name: String!, $number: Int!, $after: String) {
  viewer { login }
  repository(owner: $owner, name: $name) {
    issue(number: $number) {
      body
      createdAt
      author { login }
      includesCreatedEdit
      userContentEdits(first: 100, after: $after) {
        nodes { id editedAt deletedAt editor { login } }
        totalCount
        pageInfo { hasNextPage endCursor }
      }
    }
  }
}
"""


def has_execution_record(
    comments: list[dict[str, Any]], publisher: str, repo: str, number: int
) -> bool:
    history = events(comments, publisher, number)
    if any(
        event["event"] in {"agreed", "proposal_agreed", "execution_basis"}
        for event in history
    ):
        return True
    for comment in comments:
        if owned_comment(comment, publisher, repo, number):
            first = comment["body"].splitlines()[0]
            data = read_json(first[len(PREFIX) : -4])
            if data["detail"]["disposition"] == "startable":
                return True
    return False


class GitHub(ParentGitHub):
    def __init__(self, repo: str, *, enabled_since: str | None) -> None:
        super().__init__(repo)
        # Workflow vars are available without granting Variables API permission.
        # No permission-error fallback to an environment value is performed.
        self.enabled_since = enabled_since

    def request(self, method: str, endpoint: str, payload: dict[str, Any]) -> Any:
        result = subprocess.run(
            ["gh", "api", "--method", method, endpoint, "--input", "-"],
            input=json.dumps(payload, ensure_ascii=False, allow_nan=False),
            text=True,
            capture_output=True,
            timeout=60,
            check=False,
        )
        if result.returncode:
            # Raw stderr may include body fragments or credentials.
            raise ValueError("github_request_failed")
        return read_json(result.stdout) if result.stdout.strip() else None

    def history(self, number: int) -> dict[str, Any]:
        owner, name = self.repo.split("/")
        after = None
        seen_cursors: set[str] = set()
        edits: dict[str, Any] = {}
        pinned: tuple[Any, ...] | None = None
        while True:
            response = object_value(
                self.request(
                    "POST",
                    "graphql",
                    {
                        "query": HISTORY_QUERY,
                        "variables": {
                            "owner": owner,
                            "name": name,
                            "number": number,
                            "after": after,
                        },
                    },
                )
            )
            if response.get("errors"):
                raise ValueError("edit_history_graphql_error")
            data = object_value(response["data"])
            viewer = nonempty(object_value(data["viewer"])["login"], "viewer")
            issue = object_value(object_value(data["repository"])["issue"])
            body = issue["body"]
            created_at = nonempty(issue["createdAt"], "issue creation time")
            author = nonempty(object_value(issue["author"])["login"], "issue author")
            includes_created = issue["includesCreatedEdit"]
            connection = object_value(issue["userContentEdits"])
            total = connection["totalCount"]
            if (
                not isinstance(body, str)
                or type(includes_created) is not bool
                or type(total) is not int
                or total < 0
            ):
                raise ValueError("invalid_edit_history")
            current = (body, viewer, total, created_at, author, includes_created)
            if pinned is not None and current != pinned:
                raise ValueError("edit_history_changed_during_pagination")
            pinned = current
            nodes = connection["nodes"]
            if not isinstance(nodes, list) or len(nodes) > 100:
                raise ValueError("invalid_edit_history_nodes")
            for node in nodes:
                item = object_value(node)
                identity = nonempty(item["id"], "edit id")
                if identity in edits:
                    raise ValueError("duplicate_edit_history")
                nonempty(item["editedAt"], "edit time")
                if item["deletedAt"] is not None:
                    nonempty(item["deletedAt"], "deleted time")
                editor = item["editor"]
                if editor is not None:
                    nonempty(object_value(editor)["login"], "editor")
                edits[identity] = {
                    key: item[key] for key in ("id", "editedAt", "deletedAt", "editor")
                }
            page = object_value(connection["pageInfo"])
            if type(page["hasNextPage"]) is not bool:
                raise ValueError("invalid_edit_history_page")
            if not page["hasNextPage"]:
                if len(edits) != total:
                    raise ValueError("incomplete_edit_history")
                return {
                    "body": body,
                    "viewer": viewer,
                    "edits": edits,
                    "created_at": created_at,
                    "author": author,
                    "includes_created_edit": includes_created,
                }
            after = nonempty(page["endCursor"], "history cursor")
            if after in seen_cursors or not nodes:
                raise ValueError("invalid_edit_history_cursor")
            seen_cursors.add(after)


def uninterrupted(
    before: dict[str, Any], after: dict[str, Any], candidate: str
) -> bool:
    """Exactly our one new edit; same-publisher ABA edits also count as interference."""
    if after["body"] != candidate or any(
        before[key] != after[key] for key in ("viewer", "created_at", "author")
    ):
        return False
    old, new = before["edits"], after["edits"]
    if any(new.get(identity) != edit for identity, edit in old.items()):
        return False
    added = new.keys() - old.keys()
    if not before["includes_created_edit"] and after["includes_created_edit"]:
        # GitHub materializes the original creation version on the first edit.
        # Accept only this explicit transition with a unique creation record.
        creation = {
            identity
            for identity in added
            if new[identity]["editedAt"] == before["created_at"]
            and new[identity]["editor"] == {"login": before["author"]}
            and new[identity]["deletedAt"] is None
        }
        if old or len(creation) != 1:
            return False
        added -= creation
    elif before["includes_created_edit"] != after["includes_created_edit"]:
        return False
    if len(added) != 1:
        return False
    edit = new[next(iter(added))]
    return edit["deletedAt"] is None and edit["editor"] == {"login": after["viewer"]}
