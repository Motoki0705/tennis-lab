"""Explicit reinspection of public PRs whose original private journal is lost.

This authority is narrower than normal capture: public bytes are verified, but
the original instruction/receipt provenance cannot be confirmed or recreated.
"""

from __future__ import annotations

import base64
import hashlib
import re
import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

from . import journal, schema
from .journal import PublicationError, require
from .ownership import require_owner
from .publication import (
    GitHub,
    GitHubHTTPError,
    MergeRejected,
    Publisher,
    canonical_remote,
)
from .reconciliation import continue_workers
from .safety import Guard
from .sources import location

BRANCH = re.compile(r"worker-log/(d-[a-f0-9]{32})/([0-9]{4,})")
TRANSPORT_ERRORS = {
    "github_failed",
    "github_invalid_response",
    "github_pagination",
    "git_failed",
}
PRIVATE_ERRORS = {
    "private_storage_symlink",
    "journal_unavailable",
    "journal_invalid",
    "journal_identity_changed",
    "source_changed",
    "publication_failed",
}


def safe_private(path: Path) -> None:
    require(
        not any(p.is_symlink() for p in (path, *path.parents)),
        "private_storage_symlink",
    )


class OrphanRecovery:
    def __init__(
        self, repo: str, *, state_dir: Path, root: Path, github: GitHub | None = None
    ) -> None:
        self.repo, self.state_dir, self.root = repo, state_dir, root
        self.github = github or GitHub(repo)
        self.trees: dict[str, list[dict[str, Any]]] = {}
        self.inspection_heads: dict[int, str] = {}

    def request(self, suffix: str) -> Any:
        return self.github.request(f"repos/{self.repo}/{suffix}")

    def tree(self, revision: str) -> list[dict[str, Any]]:
        require(schema.digest(revision, 40), "unsafe_reference")
        if revision not in self.trees:
            value = self.request(f"git/trees/{revision}")
            require(
                isinstance(value, dict)
                and value.get("truncated") is False
                and isinstance(value.get("tree"), list),
                "github_invalid_response",
            )
            rows = value["tree"]
            require(
                all(
                    isinstance(r, dict) and isinstance(r.get("path"), str) for r in rows
                ),
                "github_invalid_response",
            )
            require(
                len({r["path"] for r in rows}) == len(rows), "github_invalid_response"
            )
            self.trees[revision] = rows
        return self.trees[revision]

    def entry(self, tree: str, path: str) -> dict[str, Any] | None:
        parts = path.split("/")
        for index, part in enumerate(parts):
            row = next((r for r in self.tree(tree) if r["path"] == part), None)
            if row is None:
                return None
            if index == len(parts) - 1:
                return row
            require(
                row.get("type") == "tree" and row.get("mode") == "040000",
                "unsafe_file_type",
            )
            tree = row["sha"]
        return None

    def main(self) -> str:
        value = self.request("git/ref/heads/main")
        revision = (
            value.get("object", {}).get("sha") if isinstance(value, dict) else None
        )
        require(schema.digest(revision, 40), "github_invalid_response")
        return str(revision)

    def commit(self, revision: str) -> dict[str, Any]:
        value = self.request(f"git/commits/{revision}")
        require(
            isinstance(value, dict) and value.get("sha") == revision,
            "github_invalid_response",
        )
        require(
            schema.digest(value.get("tree", {}).get("sha"), 40),
            "github_invalid_response",
        )
        return dict(value)

    def inspect(
        self, number: int, *, confirmed: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        repository = self.github.repository()
        origin = (
            Publisher(self.root, Path("unused"))
            .git("remote", "get-url", "origin")
            .decode()
            .strip()
        )
        require(
            canonical_remote(origin)
            in {
                canonical_remote(u)
                for u in (repository.get("clone_url"), repository.get("ssh_url"))
                if isinstance(u, str)
            },
            "repository_changed",
        )
        pr = self.github.get(number)
        branch = pr.get("head", {}).get("ref", "")
        match = BRANCH.fullmatch(branch) if isinstance(branch, str) else None
        if match is None:
            raise PublicationError("publication_identity_changed")
        key, sequence = match[1], int(match[2])
        require(
            sequence > 0 and match[2] == f"{sequence:04d}",
            "publication_identity_changed",
        )
        head = pr.get("head", {}).get("sha")
        require(schema.digest(head, 40), "unsafe_reference")
        require(
            pr.get("head", {}).get("repo", {}).get("full_name", "").casefold()
            == self.repo.casefold()
            and pr.get("base", {}).get("repo", {}).get("full_name", "").casefold()
            == self.repo.casefold(),
            "repository_changed",
        )
        self.inspection_heads[number] = head
        # The exact one-commit PR and its sole parent pin the public input.
        commits = self.request(f"pulls/{number}/commits?per_page=100")
        require(
            isinstance(commits, list)
            and len(commits) == 1
            and commits[0].get("sha") == head
            and pr.get("commits") == 1,
            "unexpected_commit_parent",
        )
        commit = self.commit(head)
        parents = commit.get("parents")
        require(
            isinstance(parents, list)
            and len(parents) == 1
            and schema.digest(parents[0].get("sha"), 40),
            "unexpected_commit_parent",
        )
        parents = cast(list[dict[str, Any]], parents)
        base = parents[0]["sha"]
        changes = self.request(f"compare/{base}...{head}?per_page=100")
        require(
            isinstance(changes, dict)
            and changes.get("ahead_by") == 1
            and changes.get("total_commits") == 1,
            "unexpected_commit_parent",
        )
        files_list = changes.get("files")
        require(isinstance(files_list, list) and len(files_list) == 3, "unsafe_diff")
        files_list = cast(list[dict[str, Any]], files_list)
        names = [r.get("filename") for r in files_list if isinstance(r, dict)]
        require(
            len(names) == 3 and all(isinstance(n, str) for n in names), "unsafe_diff"
        )
        directory = str(names[0]).rsplit("/", 1)[0]
        scope = re.fullmatch(
            rf"\.agents/logs/parents/parent-([1-9][0-9]*)/child-([1-9][0-9]*)/{key}/"
            rf"{sequence:04d}-(initial|resume)",
            directory,
        )
        if scope is None:
            raise PublicationError("publication_identity_changed")
        original = location(
            self.state_dir, self.repo, int(scope[1]), int(scope[2]), key[2:], sequence
        )
        safe_private(original)
        if original.exists() or original.is_symlink():
            return {
                "source": {
                    "parent": int(scope[1]),
                    "issue": int(scope[2]),
                    "public_key": key[2:],
                    "sequence": sequence,
                },
                "journal_present": True,
            }
        expected = {directory + "/" + n for n in schema.FILENAMES}
        schema.pr_additions(files_list, expected)
        files = {}
        for name in sorted(schema.FILENAMES):
            entry = self.entry(commit["tree"]["sha"], directory + "/" + name)
            if entry is None:
                raise PublicationError("unsafe_file_type")
            require(
                entry is not None
                and entry.get("mode") == "100644"
                and entry.get("type") == "blob",
                "unsafe_file_type",
            )
            blob = self.request(f"git/blobs/{entry['sha']}")
            require(
                isinstance(blob, dict)
                and blob.get("encoding") == "base64"
                and blob.get("sha") == entry["sha"]
                and isinstance(blob.get("content"), str),
                "github_invalid_response",
            )
            try:
                raw = base64.b64decode("".join(blob["content"].split()), validate=True)
            except ValueError:
                raise PublicationError("github_invalid_response") from None
            require(blob.get("size") == len(raw), "github_invalid_response")
            require(
                hashlib.sha1(
                    b"blob " + str(len(raw)).encode() + b"\0" + raw
                ).hexdigest()
                == entry["sha"],
                "committed_content_changed",
            )
            files[name] = raw
        meta = schema.validate(files, Guard())
        require(
            meta["repo"].casefold() == self.repo.casefold()
            and meta["delegation_key"] == key
            and meta["turn"]["sequence"] == sequence
            and schema.path(meta) == directory,
            "publication_identity_changed",
        )
        source = {
            "repo": self.repo,
            "parent": meta["parent"],
            "issue": meta["issue"],
            "public_key": key[2:],
            "sequence": sequence,
        }
        expected_body = (
            f"<!-- worker-log:v1 key={key} turn={sequence} head={head} -->\n\n"
            f"Refs #{meta['parent']}, #{meta['issue']}\n\n"
            "検査済みworkerログ3ファイルの追加。専用入口が固定HEADとmainへの保存を照合する。\n"
        )
        Guard().check([expected_body, pr.get("body"), branch, directory])
        row = {
            "source": source,
            "head": head,
            "base": base,
            "branch": branch,
            "directory": directory,
            "pr_body": expected_body,
            "pr_number": number,
            "files": {n: raw.decode("utf-8") for n, raw in files.items()},
        }
        schema.identity(pr, row, self.repo)
        require(pr.get("changed_files") == 3, "unsafe_diff")
        schema.pr_additions(self.github.files(number), expected)
        path = location(
            self.state_dir, self.repo, meta["parent"], meta["issue"], key[2:], sequence
        )
        if path.exists() or path.is_symlink():
            row["journal_present"] = True
            return row
        main = self.main()
        ancestry = self.request(f"compare/{base}...{main}?per_page=1")
        require(
            isinstance(ancestry, dict)
            and ancestry.get("merge_base_commit", {}).get("sha") == base,
            "base_not_ancestor",
        )
        main_commit = self.commit(main)
        present = self.entry(main_commit["tree"]["sha"], directory)
        if confirmed and pr.get("merged") is True:
            require(
                confirmed.get("head") == head
                and confirmed.get("files") == row["files"],
                "prepared_content_changed",
            )
            self.confirm(row, pr)
        else:
            require(present is None, "record_already_exists")
            require(
                pr.get("state") == "open" and pr.get("merged") is False,
                "closed_unmerged",
            )
        return row

    def confirm(self, row: dict[str, Any], pr: dict[str, Any]) -> None:
        schema.identity(pr, row, self.repo)
        require(
            pr.get("merged") is True
            and pr.get("state") == "closed"
            and schema.digest(pr.get("merge_commit_sha"), 40),
            "merge_unconfirmed",
        )
        main = self.main()
        for revision in (row["head"], pr["merge_commit_sha"]):
            compare = self.request(f"compare/{revision}...{main}?per_page=1")
            require(
                compare.get("merge_base_commit", {}).get("sha") == revision,
                "main_content_changed",
            )
        tree = self.commit(main)["tree"]["sha"]
        for name, text in row["files"].items():
            entry = self.entry(tree, row["directory"] + "/" + name)
            if entry is None:
                raise PublicationError("main_content_changed")
            require(
                entry is not None
                and entry.get("mode") == "100644"
                and entry.get("type") == "blob",
                "main_content_changed",
            )
            blob = self.request(f"git/blobs/{entry['sha']}")
            require(
                blob.get("encoding") == "base64"
                and base64.b64decode("".join(blob["content"].split()), validate=True)
                == text.encode(),
                "main_content_changed",
            )
        row.update(
            state="merged",
            error=None,
            saved_to_main=True,
            merge_commit=pr["merge_commit_sha"],
            main_observed=main,
        )

    def reject(
        self, number: int, head: str, code: str, audit: dict[str, Any], path: Path
    ) -> None:
        fresh = self.github.get(number)
        require(fresh.get("head", {}).get("sha") == head, "pr_identity_changed")
        require(fresh.get("merged") is False, "merge_unconfirmed")
        body = (
            f"<!-- orphan-worker-log:v1 pr={number} head={head} -->\n\n"
            "元journalが存在しないworkerログPRの再検査で公開を拒否しました。\n"
            f"理由: `{code}`。元journalとの照合はできません。\n"
        )
        comments = self.request(f"issues/{number}/comments?per_page=100")
        require(isinstance(comments, list) and len(comments) < 100, "github_pagination")
        existing = [c for c in comments if c.get("body") == body]
        require(len(existing) <= 1, "multiple_log_prs")
        if not existing:
            require(not audit.get("comment_attempted"), "pr_creation_uncertain")
            audit["comment_attempted"] = True
            journal.atomic(path, audit)
            self.github.request(
                f"repos/{self.repo}/issues/{number}/comments", "POST", {"body": body}
            )
        self.github.request(
            f"repos/{self.repo}/pulls/{number}", "PATCH", {"state": "closed"}
        )
        closed = self.github.get(number)
        require(
            closed.get("state") == "closed"
            and closed.get("merged") is False
            and closed.get("head", {}).get("sha") == head,
            "closed_unmerged",
        )
        audit.update(state="closed", error=code, saved_to_main=False)

    def recover(
        self,
        number: int,
        *,
        dry_run: bool,
        owner_check: Callable[[], bool] | None = None,
    ) -> dict[str, Any]:
        self.inspection_heads.pop(number, None)
        path = (
            self.state_dir
            / "orphan-worker-prs"
            / schema.sha(self.repo.casefold().encode())
            / f"{number}.json"
        )
        safe_private(path)
        audit = (
            journal.decode(path.read_bytes())
            if path.exists()
            else {
                "version": 1,
                "repo": self.repo,
                "pr_number": number,
                "original_journal_available": False,
                "state": "uninspected",
                "error": None,
                "saved_to_main": False,
            }
        )
        require(
            audit.get("version") == 1
            and audit.get("repo") == self.repo
            and audit.get("pr_number") == number
            and audit.get("original_journal_available") is False,
            "journal_identity_changed",
        )
        validated = False
        try:
            row = self.inspect(
                number, confirmed=audit if audit.get("merge_attempted") else None
            )
            validated = True
            if not dry_run:
                require_owner(owner_check)
            if row.get("journal_present"):
                s = row["source"]
                logs = continue_workers(
                    self.repo,
                    s["parent"],
                    state_dir=self.state_dir,
                    dry_run=dry_run,
                    only={(s["issue"], "d-" + s["public_key"], s["sequence"])},
                    owner_check=owner_check,
                )
                return {
                    "pr_number": number,
                    "state": "journal_present",
                    "action": "normal_journal",
                    "logs": logs,
                    "original_journal_available": True,
                    "error": next((v["error"] for v in logs if v.get("error")), None),
                    "dry_run": dry_run,
                }
            if row.get("saved_to_main"):
                audit.update(row)
            elif dry_run:
                return {
                    "pr_number": number,
                    "head": row["head"],
                    "parent": row["source"]["parent"],
                    "issue": row["source"]["issue"],
                    "state": "eligible",
                    "action": "merge",
                    "original_journal_available": False,
                    "error": None,
                    "dry_run": True,
                }
            else:
                require(not audit.get("merge_attempted"), "merge_response_uncertain")
                audit.update(row)
                # The original turn lock excludes a concurrently restored journal.
                s = row["source"]
                original = location(
                    self.state_dir,
                    self.repo,
                    s["parent"],
                    s["issue"],
                    s["public_key"],
                    s["sequence"],
                )
                safe_private(original)
                with journal.locked(original):
                    require(not original.exists(), "source_changed")
                    fresh = self.github.get(number)
                    schema.identity(fresh, row, self.repo)
                    require(fresh.get("state") == "open", "closed_unmerged")
                    require(fresh.get("mergeable") is not False, "merge_conflict")
                    audit.update(merge_attempted=True, state="pending")
                    audit.pop("merge_http_status", None)
                    journal.atomic(path, audit)
                    self.github.merge(number, row["head"])
                    audit["merge_http_status"] = 200
                    self.confirm(audit, self.github.get(number))
        except (
            ValueError,
            OSError,
            KeyError,
            TypeError,
            subprocess.SubprocessError,
        ) as exc:
            code = (
                str(exc) if isinstance(exc, PublicationError) else "publication_failed"
            )
            uncertain = code in TRANSPORT_ERRORS or isinstance(
                exc, subprocess.SubprocessError
            )
            if dry_run:
                return {
                    "pr_number": number,
                    "state": "pending"
                    if uncertain
                    else "blocked"
                    if code in PRIVATE_ERRORS
                    else "rejected",
                    "action": "retry_inspection"
                    if uncertain or code in PRIVATE_ERRORS
                    else "close"
                    if number in self.inspection_heads
                    else "out_of_scope",
                    "original_journal_available": False,
                    "error": code,
                    "dry_run": True,
                }
            audit.update(error=code, state="blocked", saved_to_main=False)
            if isinstance(exc, GitHubHTTPError):
                audit["merge_http_status"] = exc.status
            if audit.get("merge_attempted") and (
                uncertain or code == "merge_response_uncertain"
            ):
                audit.update(state="pending", error="merge_response_uncertain")
            if isinstance(exc, MergeRejected):
                audit.update(merge_attempted=False, merge_http_status=exc.status)
            if (
                not validated
                and not uncertain
                and code not in PRIVATE_ERRORS
                and number in self.inspection_heads
                and not audit.get("merge_attempted")
                and (owner_check is None or owner_check())
            ):
                head = self.inspection_heads[number]
                self.reject(number, head, code, audit, path)
        if not dry_run:
            journal.atomic(path, audit)
        return {
            "dry_run": dry_run,
            **{
                k: audit.get(k)
                for k in (
                    "pr_number",
                    "state",
                    "error",
                    "saved_to_main",
                    "original_journal_available",
                    "merge_http_status",
                )
            },
        }


def recover_orphans(
    repo: str,
    *,
    state_dir: Path,
    root: Path,
    numbers: list[int] | None,
    dry_run: bool,
    owner_check: Callable[[], bool] | None = None,
) -> list[dict[str, Any]]:
    from contextlib import nullcontext

    recovery = OrphanRecovery(repo, state_dir=state_dir, root=root)
    if numbers is None:
        numbers = []
        page = 1
        while True:
            rows = recovery.request(f"pulls?state=open&per_page=100&page={page}")
            require(isinstance(rows, list) and len(rows) <= 100, "github_pagination")
            numbers.extend(
                r["number"]
                for r in rows
                if BRANCH.fullmatch(r.get("head", {}).get("ref", ""))
            )
            if len(rows) < 100:
                break
            page += 1
    require(all(schema.integer(n) for n in numbers), "invalid_log_reference")
    values = []
    for number in sorted(set(numbers)):
        path = (
            state_dir
            / "orphan-worker-prs"
            / schema.sha(repo.casefold().encode())
            / f"{number}.json"
        )
        try:
            if not dry_run:
                require_owner(owner_check)
            safe_private(path)
            with nullcontext() if dry_run else journal.locked(path):
                value = recovery.recover(
                    number, dry_run=dry_run, owner_check=owner_check
                )
                values.append(value)
                if value.get("error") == "log_owner_mismatch":
                    break
        except (
            ValueError,
            OSError,
            KeyError,
            TypeError,
            subprocess.SubprocessError,
        ) as exc:
            values.append(
                {
                    "pr_number": number,
                    "state": "blocked",
                    "saved_to_main": False,
                    "original_journal_available": False,
                    "error": str(exc)
                    if isinstance(exc, PublicationError)
                    else "publication_failed",
                    "dry_run": dry_run,
                }
            )
            if isinstance(exc, PublicationError) and str(exc) == "log_owner_mismatch":
                break
    return values
