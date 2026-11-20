"""Single-request GitHub transport and pull request checks for memory publication."""

from __future__ import annotations

import json
import re
import subprocess
from dataclasses import dataclass
from uuid import uuid4

from .git import is_sha
from .layout import is_shared_path
from .operations import load_json


class RefUpdateRejected(ValueError):
    """The server definitely refused the transaction, with neither ref changed."""


def _object(value: object, field: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ValueError(f"GitHub response {field} must be an object")
    return value


def _ref_update_rejected(value: object) -> bool:
    if not isinstance(value, dict) or value.get("data") != {"updateRefs": None}:
        return False
    errors = value.get("errors")
    return (
        isinstance(errors, list)
        and bool(errors)
        and all(
            isinstance(error, dict)
            and error.get("path") == ["updateRefs"]
            and error.get("type") in ("STALE_DATA", "UNPROCESSABLE")
            and isinstance(error.get("message"), str)
            for error in errors
        )
    )


@dataclass(frozen=True)
class Pull:
    changed_files: int
    url: str
    repository_id: str


class GitHub:
    def __init__(self, repo: str) -> None:
        self.repo = repo
        self.pulls = f"repos/{repo}/pulls"

    def request(
        self, method: str, endpoint: str, data: dict[str, object] | None = None
    ) -> object:
        command = [
            "gh",
            "api",
            "--hostname",
            "github.com",
            "--include",
            "--method",
            method,
            "--header",
            "Accept: application/vnd.github+json",
            "--header",
            "X-GitHub-Api-Version: 2022-11-28",
            endpoint,
        ]
        if data is not None:
            command.extend(("--input", "-"))
        context = f"GitHub {method} {endpoint}"
        try:
            response = subprocess.run(
                command,
                input=json.dumps(data).encode("utf-8") if data is not None else None,
                capture_output=True,
                check=False,
                timeout=60,
            )
        except subprocess.TimeoutExpired:
            raise ValueError(f"{context}: response timed out") from None
        headers, separator, body = response.stdout.replace(b"\r\n", b"\n").partition(
            b"\n\n"
        )
        match = re.match(rb"HTTP/\d+(?:\.\d+)? (\d{3})(?:[ \n]|$)", headers)
        if not match or not separator:
            raise ValueError(f"{context}: missing HTTP status or body separator")
        status = int(match[1])
        graphql = endpoint == "graphql"
        if status != (201 if method == "POST" and not graphql else 200):
            if graphql and status in (405, 409):
                try:
                    refusal = _object(load_json(body), "ref update refusal")
                except (ValueError, RecursionError):
                    raise ValueError(
                        f"{context}: ambiguous HTTP {status} response"
                    ) from None
                if response.returncode != 1 or not isinstance(
                    refusal.get("message"), str
                ):
                    raise ValueError(f"{context}: ambiguous HTTP {status} response")
                raise RefUpdateRejected(f"{context}: HTTP {status}")
            raise ValueError(f"{context}: HTTP {status}")
        try:
            value = load_json(body)
        except (ValueError, RecursionError):
            raise ValueError(f"{context}: invalid JSON response") from None
        if graphql and response.returncode in (0, 1) and _ref_update_rejected(value):
            raise RefUpdateRejected("GitHub refused the atomic ref update")
        if response.returncode:
            raise ValueError(f"{context}: gh failed after HTTP {status}")
        return value

    def create(self, branch: str, body: str) -> int:
        response = _object(
            self.request(
                "POST",
                self.pulls,
                {
                    "base": "main",
                    "head": branch,
                    "title": "共有memoryを更新する",
                    "body": body,
                    "draft": False,
                    "maintainer_can_modify": False,
                },
            ),
            "pull request",
        )
        number = response.get("number")
        if type(number) is not int or number < 1:
            raise ValueError("GitHub did not confirm a pull request number")
        return number

    def update_body(self, number: int, body: str) -> None:
        response = _object(
            self.request("PATCH", f"{self.pulls}/{number}", {"body": body}),
            "pull request",
        )
        if type(response.get("number")) is not int or response["number"] != number:
            raise ValueError("GitHub returned the wrong pull request after body update")

    def pull(
        self, number: int, branch: str, sha: str, body: str, *, merged: bool = False
    ) -> Pull:
        response = _object(
            self.request("GET", f"{self.pulls}/{number}"), "pull request"
        )
        base = _object(response.get("base"), "base")
        head = _object(response.get("head"), "head")
        count, url = response.get("changed_files"), response.get("html_url")
        if (
            type(response.get("number")) is not int
            or response["number"] != number
            or response.get("state") != ("closed" if merged else "open")
            or response.get("merged") is not merged
            or response.get("draft") is not False
            or response.get("body") != body
            or base.get("ref") != "main"
            or head.get("ref") != branch
            or head.get("sha") != sha
            or type(count) is not int
            or count < 0
            or not isinstance(url, str)
            or url.casefold()
            != f"https://github.com/{self.repo}/pull/{number}".casefold()
        ):
            raise ValueError("GitHub pull request identity or state changed")
        base_repo = _object(base.get("repo"), "base repository")
        head_repo = _object(head.get("repo"), "head repository")
        for repo in (base_repo, head_repo):
            name = repo.get("full_name")
            if not isinstance(name, str) or name.casefold() != self.repo.casefold():
                raise ValueError("GitHub pull request uses an unexpected repository")
        repository_id = base_repo.get("node_id")
        if (
            not isinstance(repository_id, str)
            or not repository_id
            or repository_id != head_repo.get("node_id")
        ):
            raise ValueError("GitHub pull request repository node IDs do not agree")
        return Pull(count, url, repository_id)

    def main(self) -> str:
        # A PR's base.sha can stay pinned to an older commit after main advances.
        response = _object(
            self.request("GET", f"repos/{self.repo}/git/ref/heads/main"), "main ref"
        )
        commit = _object(response.get("object"), "main ref object")
        sha = commit.get("sha")
        if (
            response.get("ref") != "refs/heads/main"
            or commit.get("type") != "commit"
            or not isinstance(sha, str)
            or not is_sha(sha)
        ):
            raise ValueError("GitHub returned an invalid main ref")
        return sha

    def verify_files(self, number: int, expected: dict[str, str]) -> None:
        remaining = expected.copy()
        for start in range(0, len(expected), 100):
            endpoint = f"{self.pulls}/{number}/files?per_page=100"
            if start:
                endpoint += f"&page={start // 100 + 1}"
            items = self.request("GET", endpoint)
            if not isinstance(items, list) or len(items) != min(100, len(remaining)):
                raise ValueError(
                    "GitHub pull request file count differs from committed diff"
                )
            for item in items:
                row = _object(item, "file")
                path, status = row.get("filename"), row.get("status")
                if (
                    not isinstance(path, str)
                    or not is_shared_path(path)
                    or "previous_filename" in row
                    or path not in remaining
                    or remaining.pop(path) != status
                ):
                    raise ValueError(
                        "GitHub pull request files differ from committed diff"
                    )

    def advance_main(
        self, repository_id: str, branch: str, base: str, head: str
    ) -> None:
        """Check both server refs in the same transaction as the fast-forward."""
        mutation_id = uuid4().hex
        response = _object(
            self.request(
                "POST",
                "graphql",
                {
                    "query": "mutation($input: UpdateRefsInput!) { updateRefs(input: $input) { clientMutationId } }",
                    "variables": {
                        "input": {
                            "repositoryId": repository_id,
                            "clientMutationId": mutation_id,
                            "refUpdates": [
                                {
                                    "name": "refs/heads/main",
                                    "beforeOid": base,
                                    "afterOid": head,
                                    "force": False,
                                },
                                {
                                    "name": f"refs/heads/{branch}",
                                    "beforeOid": head,
                                    "afterOid": head,
                                    "force": False,
                                },
                            ],
                        }
                    },
                },
            ),
            "ref update result",
        )
        if "errors" in response or response.get("data") != {
            "updateRefs": {"clientMutationId": mutation_id}
        }:
            raise ValueError("GitHub response did not confirm the atomic ref update")
