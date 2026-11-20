"""Single-request GitHub transport and pull request checks for memory publication."""

from __future__ import annotations

import json
import re
import subprocess
from dataclasses import dataclass

from .git import is_sha
from .operations import load_json
from .schema import is_shared_path


class GitHubHTTPError(ValueError):
    def __init__(self, context: str, status: int) -> None:
        self.status = status
        super().__init__(f"{context}: HTTP {status}")


def _object(value: object, field: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ValueError(f"GitHub response {field} must be an object")
    return value


@dataclass(frozen=True)
class Pull:
    changed_files: int
    url: str


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
        if status != (201 if method == "POST" else 200):
            if method == "PUT" and status in (405, 409):
                try:
                    refusal = _object(load_json(body), "merge refusal")
                except (ValueError, RecursionError):
                    raise ValueError(
                        f"{context}: ambiguous HTTP {status} response"
                    ) from None
                if response.returncode != 1 or not isinstance(
                    refusal.get("message"), str
                ):
                    raise ValueError(f"{context}: ambiguous HTTP {status} response")
            raise GitHubHTTPError(context, status)
        if response.returncode:
            raise ValueError(f"{context}: gh failed after HTTP {status}")
        try:
            return load_json(body)
        except (ValueError, RecursionError):
            raise ValueError(f"{context}: invalid JSON response") from None

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

    def pull(self, number: int, branch: str, sha: str, body: str) -> Pull:
        response = _object(
            self.request("GET", f"{self.pulls}/{number}"), "pull request"
        )
        base = _object(response.get("base"), "base")
        head = _object(response.get("head"), "head")
        count, url = response.get("changed_files"), response.get("html_url")
        if (
            type(response.get("number")) is not int
            or response["number"] != number
            or response.get("state") != "open"
            or response.get("merged") is not False
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
        for ref in (base, head):
            name = _object(ref.get("repo"), "branch repository").get("full_name")
            if not isinstance(name, str) or name.casefold() != self.repo.casefold():
                raise ValueError("GitHub pull request uses an unexpected repository")
        return Pull(count, url)

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

    def merge(self, number: int, head: str) -> str:
        response = _object(
            self.request(
                "PUT",
                f"{self.pulls}/{number}/merge",
                {"merge_method": "merge", "sha": head},
            ),
            "merge result",
        )
        sha = response.get("sha")
        if (
            response.get("merged") is not True
            or not isinstance(sha, str)
            or not is_sha(sha)
        ):
            raise ValueError("GitHub response did not confirm a merge SHA")
        return sha
