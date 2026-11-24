"""Verified GitHub evidence issues and their lifecycle."""

from __future__ import annotations

import json
import subprocess
from collections import defaultdict
from collections.abc import Iterable
from contextlib import ExitStack
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from issues.evidence import EVIDENCE_LABEL, EVIDENCE_PREFIX, is_evidence_issue
from worker_logs.journal import atomic, locked

from .github import GitHubWriteError
from .judgement_comments import read_json
from .policy import digest, nonempty, number, object_value
from .state import RegistryConflict

if TYPE_CHECKING:
    from .github import GitHub
    from .state import Store


@dataclass(frozen=True)
class EvidenceIssue:
    repo: str
    parent: int
    number: int
    url: str
    publisher: str
    # The listing and the direct read can straddle a close. State is not
    # part of the binding identity.
    state: str = field(compare=False)


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
        ):
            raise ValueError("invalid evidence creation journal")
        if row["state"] == "bound" or row["number"] is not None:
            number(row["number"])
        return row

    def _marker_parent(self, raw: dict[str, Any], issue: int) -> int:
        body = raw.get("body")
        line = body.split("\n", 1)[0].rstrip("\r") if isinstance(body, str) else ""
        if not line.startswith(EVIDENCE_PREFIX) or not line.endswith(" -->"):
            raise ValueError(
                f"evidence issue #{issue}: missing or unknown binding marker"
            )
        data = object_value(read_json(line[len(EVIDENCE_PREFIX) : -4]))
        if set(data) != {"repo", "parent"} or data["repo"] != self.gh.repo.casefold():
            raise ValueError(f"evidence issue #{issue}: unverifiable binding marker")
        parent = number(data["parent"])
        if parent == issue:
            raise ValueError("evidence issue cannot be its own parent")
        return parent

    def _binding(self, raw: dict[str, Any], login: str) -> EvidenceIssue:
        n = self.gh.identity(raw)
        parent = self._marker_parent(raw, n)
        if (
            {x["name"] for x in raw.get("labels", [])} != {EVIDENCE_LABEL}
            or object_value(raw.get("user")).get("login") != login
            or raw.get("state") not in {"open", "closed"}
        ):
            raise ValueError(
                f"evidence issue #{n}: unverifiable binding/label/publisher"
            )
        url = f"https://github.com/{self.gh.repo}/issues/{n}"
        html_url = raw.get("html_url", url)
        if not isinstance(html_url, str) or html_url.casefold() != url.casefold():
            raise ValueError("evidence issue URL mismatch")
        return EvidenceIssue(
            self.gh.repo.casefold(), parent, n, url, login, raw["state"]
        )

    def _read_binding(self, issue: int, login: str) -> EvidenceIssue:
        raw = object_value(self.gh.get(f"repos/{self.gh.repo}/issues/{issue}"))
        return self._binding(raw, login)

    def find_bound(
        self, parent: int, login: str, *, required: bool = True
    ) -> EvidenceIssue | None:
        """Read only a saved relation; an absent optional binding is explicit."""
        saved = self._saved(number(parent), login)
        if saved is None and not required:
            return None
        if saved is None or saved["state"] != "bound":
            raise ValueError(
                "automatic progression requires a saved evidence issue binding; use evidence --create first"
            )
        pinned = self._read_binding(saved["number"], login)
        if pinned.parent != parent or pinned.number != saved["number"]:
            raise ValueError("saved evidence issue binding changed")
        return pinned

    def _listed(self, parent: int, login: str) -> EvidenceIssue | None:
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
        return matches[0] if matches else None

    def find(self, parent: int, login: str) -> EvidenceIssue | None:
        """Inspect labelled issues in all states; never create or repair on read."""
        number(parent)
        if self.gh.targeted_history:
            return self.find_bound(parent, login)
        saved = self._saved(parent, login)
        listed = self._listed(parent, login)
        if saved and saved["number"] is not None:
            # A validated direct read covers listing lag. Any visible competitor
            # or disagreement between the two representations still fails closed.
            pinned = self._read_binding(saved["number"], login)
            if (
                pinned.number != saved["number"]
                or pinned.parent != parent
                or listed not in (None, pinned)
            ):
                raise ValueError("evidence issue binding changed or disappeared")
            return pinned
        if saved and listed is None:
            raise ValueError(
                "evidence issue creation uncertain; reconcile GitHub before any new POST"
            )
        return listed

    def needs_binding(self, parent: int, login: str) -> bool:
        """Keep first-time binding and lost-creation recovery under ensure's lock."""
        saved = self._saved(parent, login)
        return saved is None or saved["state"] == "sent"

    def ensure(
        self, parent: int, login: str, *, store: Store | None = None
    ) -> EvidenceIssue:
        """One durable POST intent, serialized across local processes/worktrees.

        A crash before sending may require manual reconciliation. GitHub has no
        create-issue idempotency token; never assume an absent listing proves a
        timed-out POST was not accepted. Keep the journal when recovering state.
        """
        path = self.path(parent)
        if store is not None and store is not self.gh._store:
            raise ValueError("evidence client belongs to another registry transaction")
        with ExitStack() as locks, self.gh.fresh_reads():
            # Never wait for the creation lock while holding the registry:
            # its owner may be reacquiring that registry after GitHub I/O.
            with self.gh.io_scope():
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
                except RegistryConflict:
                    raise
                except (ValueError, OSError, subprocess.TimeoutExpired) as exc:
                    if isinstance(exc, GitHubWriteError) and exc.creation_rejected:
                        path.unlink()
                        raise
                    raise ValueError(
                        "evidence issue creation uncertain; retry only to reconcile"
                    ) from exc
                # Retain the POST identity if detail or duplicate checks are
                # interrupted while GitHub's labelled listing is incomplete.
                atomic(path, {**sent, "number": found.number})
                if self._read_binding(found.number, login) != found:
                    raise ValueError("evidence creation could not be verified")
                if self._listed(parent, login) not in (None, found):
                    raise ValueError(
                        f"duplicate evidence issues for parent #{parent}; inspect explicitly"
                    )
            atomic(path, {**sent, "state": "bound", "number": found.number})
            return found

    def describe(
        self, parent: int, login: str, *, create: bool = False
    ) -> dict[str, Any] | None:
        issue = self.ensure(parent, login) if create else self.find(parent, login)
        return asdict(issue) if issue else None

    def reconcile_closed(
        self, *, parent: int | None, apply: bool
    ) -> list[dict[str, Any]]:
        """Close open evidence for closed parents using GitHub alone."""
        repo = object_value(self.gh.get(f"repos/{self.gh.repo}"))
        # The Actions token is an installation token. The trusted publisher
        # for this repository's evidence is the repository owner.
        login = nonempty(object_value(repo.get("owner")).get("login"), "repo owner")
        unverified: dict[int | None, dict[str, Any]] = {}
        results: list[dict[str, Any]] = []
        with self.gh.fresh_reads():
            labelled = self._closing_index(
                self.gh.evidence_rows(), login, parent, unverified
            )
            inventory: dict[int, dict[int, EvidenceIssue]] = {}
            if parent is None:
                inventory = self._closing_index(
                    (
                        raw
                        for raw in self.gh.rows(
                            f"repos/{self.gh.repo}/issues?state=all"
                        )
                        if any(
                            label["name"] == EVIDENCE_LABEL for label in raw["labels"]
                        )
                    ),
                    login,
                    None,
                    unverified,
                )
                corroborating = {issue: set(rows) for issue, rows in inventory.items()}
                source = "repository_evidence"
            else:
                corroborating = {parent: self._timeline_evidence_numbers(parent, login)}
                source = "timeline_evidence"

            for current in sorted(labelled.keys() | corroborating.keys()):
                expected = labelled.get(current, {})
                observed = corroborating.get(current, set())
                if expected.keys() != observed:
                    results.append(
                        {
                            "parent": current,
                            "status": "binding_listing_mismatch",
                            "labelled_evidence": sorted(expected),
                            source: sorted(observed),
                        }
                    )
                    continue
                if not expected:
                    results.append({"parent": current, "status": "no_binding"})
                    continue
                for binding in expected.values():
                    if (
                        parent is None
                        and binding.state
                        == inventory[current][binding.number].state
                        == "closed"
                    ):
                        continue
                    target = {"parent": binding.parent, "evidence": binding.number}
                    if len(expected) != 1:
                        results.append({**target, "status": "duplicate_binding"})
                        continue
                    try:
                        result = self._close_binding(binding, apply=apply)
                    except (ValueError, KeyError, TypeError, OSError) as exc:
                        result = {"status": "error", "reason": str(exc)}
                    results.append({**target, **result})
        return [*unverified.values(), *results]

    def _timeline_evidence_numbers(self, parent: int, login: str) -> set[int]:
        """Observe the backlinks generated by issue_body, even while labels lag."""
        endpoint = f"repos/{self.gh.repo}/issues/{parent}/timeline"
        repository_url = f"https://api.github.com/repos/{self.gh.repo}".casefold()
        candidates = set()
        for event in self.gh.rows(endpoint):
            if event.get("event") != "cross-referenced":
                continue
            raw = object_value(object_value(event.get("source")).get("issue"))
            url = raw.get("repository_url")
            if (
                "pull_request" in raw
                or not isinstance(url, str)
                or url.casefold() != repository_url
                or not is_evidence_issue(raw)
                or object_value(raw.get("user")).get("login") != login
            ):
                continue
            issue = self.gh.identity(raw)
            if self._marker_parent(raw, issue) == parent:
                candidates.add(issue)
        return candidates

    def _closing_index(
        self,
        rows: Iterable[dict[str, Any]],
        login: str,
        parent: int | None,
        unverified: dict[int | None, dict[str, Any]],
    ) -> dict[int, dict[int, EvidenceIssue]]:
        index: dict[int, dict[int, EvidenceIssue]] = defaultdict(dict)
        for raw in rows:
            try:
                binding = self._binding(raw, login)
            except (ValueError, KeyError, TypeError) as exc:
                n = raw.get("number")
                issue = n if type(n) is int else None
                unverified[issue] = {
                    "evidence": issue,
                    "status": "unverified_binding",
                    "reason": str(exc),
                }
                continue
            if parent is None or binding.parent == parent:
                index[binding.parent][binding.number] = binding
        return index

    def _close_binding(self, binding: EvidenceIssue, *, apply: bool) -> dict[str, Any]:
        pinned = self._read_binding(binding.number, binding.publisher)
        if pinned != binding:
            raise ValueError("evidence issue binding changed since listing")
        if pinned.state == "closed":
            return {"status": "already_closed"}
        parent = self.gh.issue(binding.parent)
        if parent.get("state") == "open":
            return {"status": "parent_open"}
        if parent.get("state") != "closed":
            raise ValueError("invalid parent state")
        if apply:
            response = self.gh.write(
                "PATCH",
                f"repos/{self.gh.repo}/issues/{pinned.number}",
                {"state": "closed"},
            )
            closed = self._binding(response, pinned.publisher)
            if closed != pinned or closed.state != "closed":
                raise ValueError("evidence issue closure could not be verified")
        return {
            "status": "closed" if apply else "would_close",
            "parent_closed_at": parent.get("closed_at"),
            "publisher": pinned.publisher,
        }
