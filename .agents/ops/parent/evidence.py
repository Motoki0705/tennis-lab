"""One GitHub evidence issue per parent; uncertain POSTs are reconciled, not retried."""

from __future__ import annotations

import json
import subprocess
from contextlib import ExitStack, nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from issues.evidence import EVIDENCE_LABEL, EVIDENCE_PREFIX, is_evidence_issue
from worker_logs.journal import atomic, locked

from .judgement_comments import read_json
from .policy import digest, nonempty, number, object_value

if TYPE_CHECKING:
    from .github import GitHub


@dataclass(frozen=True)
class EvidenceIssue:
    repo: str
    parent: int
    number: int
    url: str
    publisher: str
    state: str


def issue_body(repo: str, parent: int) -> str:
    marker = json.dumps(
        {"parent": number(parent), "repo": repo.casefold()}, sort_keys=True
    )
    return (
        EVIDENCE_PREFIX + marker + " -->\n\n"
        f"親 #{parent} の機械処理用の証拠を保存します。作業Issueではありません。\n"
        "open/closedのどちらでも作業候補・進捗集計から除外します。\n"
    )


class EvidenceIssues:
    def __init__(self, gh: GitHub, directory: Path) -> None:
        self.gh, self.directory = gh, directory

    def path(self, parent: int) -> Path:
        return (
            self.directory / f"{digest([self.gh.repo.casefold(), number(parent)])}.json"
        )

    def _saved(self, parent: int, login: str) -> dict[str, Any] | None:
        path = self.path(parent)
        if path.is_symlink():
            raise ValueError("evidence creation journal is a symlink")
        if not path.exists():
            return None
        row = object_value(read_json(path.read_text(encoding="utf-8")))
        if (
            set(row) != {"version", "repo", "parent", "publisher", "state", "number"}
            or type(row["version"]) is not int
            or row["version"] != 1
            or row["repo"] != self.gh.repo.casefold()
            or type(row["parent"]) is not int
            or row["parent"] != parent
            or row["publisher"] != login
            or row["state"] not in {"sent", "bound"}
            or (row["state"] == "sent" and row["number"] is not None)
        ):
            raise ValueError("invalid evidence creation journal")
        if row["state"] == "bound":
            number(row["number"])
        return row

    def _binding(self, raw: dict[str, Any], login: str) -> EvidenceIssue:
        n = self.gh.identity(raw)
        body = raw.get("body") or ""
        line = body.split("\n", 1)[0].rstrip("\r")
        if not line.startswith(EVIDENCE_PREFIX) or not line.endswith(" -->"):
            raise ValueError(f"evidence issue #{n}: missing or unknown binding marker")
        data = object_value(read_json(line[len(EVIDENCE_PREFIX) : -4]))
        if (
            set(data) != {"repo", "parent"}
            or data["repo"] != self.gh.repo.casefold()
            or {x["name"] for x in raw.get("labels", [])} != {EVIDENCE_LABEL}
            or object_value(raw.get("user")).get("login") != login
            or raw.get("state") not in {"open", "closed"}
        ):
            raise ValueError(
                f"evidence issue #{n}: unverifiable binding/label/publisher"
            )
        parent = number(data["parent"])
        if parent == n:
            raise ValueError("evidence issue cannot be its own parent")
        url = f"https://github.com/{self.gh.repo}/issues/{n}"
        if raw.get("html_url", url).casefold() != url.casefold():
            raise ValueError("evidence issue URL mismatch")
        return EvidenceIssue(
            self.gh.repo.casefold(), parent, n, url, login, raw["state"]
        )

    def find(self, parent: int, login: str) -> EvidenceIssue | None:
        """Scan all states without the search index. Never create or repair on read."""
        number(parent)
        saved = self._saved(parent, login)
        matches = []
        for raw in self.gh.evidence_rows():
            if "pull_request" not in raw and is_evidence_issue(raw):
                binding = self._binding(raw, login)
                if binding.parent == parent:
                    matches.append(binding)
        if len(matches) > 1:
            raise ValueError(
                f"duplicate evidence issues for parent #{parent}; inspect explicitly"
            )
        if saved and saved["state"] == "bound":
            # Do not trust an index, a local number, or a missing old-success copy.
            raw = object_value(
                self.gh.get(f"repos/{self.gh.repo}/issues/{saved['number']}")
            )
            pinned = self._binding(raw, login)
            if (
                pinned.number != saved["number"]
                or pinned.parent != parent
                or matches != [pinned]
            ):
                raise ValueError("evidence issue binding changed or disappeared")
        if saved and not matches:
            raise ValueError(
                "evidence issue creation uncertain; reconcile GitHub before any new POST"
            )
        return matches[0] if matches else None

    def ensure(self, parent: int, login: str) -> EvidenceIssue:
        """One durable POST intent, serialized across local processes/worktrees.

        A crash before sending may require manual reconciliation. GitHub has no
        create-issue idempotency token; never assume an absent listing proves a
        timed-out POST was not accepted. Keep the journal when recovering state.
        """
        path = self.path(parent)
        with ExitStack() as locks, self.gh.fresh_reads():
            # Never wait for the creation lock while holding the registry:
            # its owner may be reacquiring that registry after GitHub I/O.
            with self.gh._store.unlocked_read() if self.gh._store else nullcontext():
                locks.enter_context(locked(path))
            self.gh.issue(parent)  # A container can never itself receive evidence.
            found = self.find(parent, login)
            sent = {
                "version": 1,
                "repo": self.gh.repo.casefold(),
                "parent": parent,
                "publisher": nonempty(login, "publisher"),
                "state": "sent",
                "number": None,
            }
            if found is None:
                label = object_value(
                    self.gh.get(f"repos/{self.gh.repo}/labels/{EVIDENCE_LABEL}")
                )
                if label.get("name") != EVIDENCE_LABEL:
                    raise ValueError(
                        "parent-evidence label must be provisioned from .github/labels.yml"
                    )
                atomic(path, sent)
                try:
                    response = self.gh.write(
                        "POST",
                        f"repos/{self.gh.repo}/issues",
                        {
                            "title": f"Parent evidence #{parent}",
                            "body": issue_body(self.gh.repo, parent),
                            "labels": [EVIDENCE_LABEL],
                        },
                    )
                    found = self._binding(response, login)
                    if found.parent != parent:
                        raise ValueError("evidence creation returned another parent")
                except (ValueError, OSError, subprocess.TimeoutExpired) as exc:
                    raise ValueError(
                        "evidence issue creation uncertain; retry only to reconcile"
                    ) from exc
                # Inspect authoritative GitHub state, including duplicates, before binding.
                if self.find(parent, login) != found:
                    raise ValueError("evidence creation could not be verified")
            atomic(path, {**sent, "state": "bound", "number": found.number})
            return found

    def describe(
        self, parent: int, login: str, *, create: bool = False
    ) -> dict[str, Any] | None:
        issue = self.ensure(parent, login) if create else self.find(parent, login)
        return asdict(issue) if issue else None
