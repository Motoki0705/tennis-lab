"""Use the existing issue transport; only successful-job log reads are added."""

from __future__ import annotations

import subprocess
from typing import Any

from issues import index_ci_failures, parse_ci_failure_source
from parent.github import GitHub as IssueGitHub
from parent.policy import closing_issues


class GitHub(IssueGitHub):
    def inventory(self) -> tuple[dict[str, int], list[dict[str, Any]]]:
        issues = self.issue_rows()
        numbers = [self.identity(i) for i in issues]
        if len(numbers) != len(set(numbers)):
            raise ValueError("duplicate issues across GitHub pages")
        for issue in issues:
            if issue.get("state") not in {"open", "closed"} or not isinstance(
                issue.get("body"), (str, type(None))
            ):
                raise ValueError("invalid issue state/body")
        index = index_ci_failures((i["number"], i.get("body") or "") for i in issues)
        pulls = self.pulls()
        seen = set()
        for pr in pulls:
            n = pr.get("number")
            if type(n) is not int or n < 1 or n in seen:
                raise ValueError("invalid/duplicate PR number")
            seen.add(n)
            if (
                pr.get("state") not in {"open", "closed"}
                or not isinstance(pr.get("body"), (str, type(None)))
                or "merged_at" not in pr
            ):
                raise ValueError("invalid PR state/body/merge data")
            pr["signature"] = parse_ci_failure_source(pr.get("body") or "")
        return index, pulls

    def related(
        self, pulls: list[dict[str, Any]], signature: str, issue: int | None
    ) -> list[dict[str, Any]]:
        return [
            p
            for p in pulls
            if p["signature"] == signature
            or (
                issue is not None
                and issue in closing_issues(p.get("body") or "", self.repo)
            )
        ]

    def job_log(self, job: int) -> str:
        endpoint = f"repos/{self.repo}/actions/jobs/{job}/logs"
        result = subprocess.run(
            ["gh", "api", "--method", "GET", endpoint],
            text=True,
            capture_output=True,
            timeout=60,
            check=False,
        )
        if result.returncode:
            raise ValueError(f"gh GET {endpoint} failed: {result.stderr.strip()}")
        return result.stdout

    def comment_once(self, number: int, marker: str, body: str, publisher: str) -> None:
        matches = [
            c
            for c in self.comments(number)
            if (
                c.get("user", {}).get("login") == publisher
                and (c.get("body") or "").splitlines()[:1] == [marker]
            )
        ]
        if len(matches) > 1 or (matches and matches[0]["body"] != body):
            raise ValueError("CI failure comment provenance/content conflict")
        if not matches:
            self.write(
                "POST", f"repos/{self.repo}/issues/{number}/comments", {"body": body}
            )
