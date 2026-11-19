"""A dedicated addition-only PR entry, independent of the normal CI/review gate."""

from __future__ import annotations

import json
import re
import subprocess
from contextlib import nullcontext
from pathlib import Path
from typing import Any

from . import journal, schema
from .journal import PublicationError, require
from .sources import guard_for, render


class GitHubHTTPError(PublicationError):
    """A definite GitHub response, distinct from a lost transport response."""

    def __init__(self, status: int, detail: str) -> None:
        super().__init__("github_failed", detail)
        self.status = status


class MergeRejected(PublicationError):
    def __init__(self, status: int, code: str, detail: str | None = None) -> None:
        super().__init__(code, detail)
        self.status = status


def canonical_remote(value: str) -> str:
    if value.casefold().startswith(("https://github.com/", "git@github.com:")):
        return value.casefold().rstrip("/").removesuffix(".git")
    return value


class GitHub:
    def __init__(self, repo: str) -> None:
        self.repo = repo

    def request(
        self, endpoint: str, method: str = "GET", payload: dict[str, Any] | None = None
    ) -> Any:
        command = [
            "gh",
            "api",
            "--hostname",
            "github.com",
            "--method",
            method,
            endpoint,
        ]
        merge_request = method == "PUT" and endpoint.endswith("/merge")
        if merge_request:
            command.append("--include")
        if payload is not None:
            command += ["--input", "-"]
        result = subprocess.run(
            command,
            input=json.dumps(payload).encode() if payload is not None else None,
            capture_output=True,
            check=False,
            timeout=60,
        )
        raw = result.stdout
        response_status = None
        if merge_request:
            # gh --include emits HTTP status/headers followed by a blank line.
            match = re.match(rb"HTTP/\S+\s+(\d{3})[^\r\n]*\r?\n", raw)
            if match:
                status = int(match[1])
                response_status = status
                parts = re.split(rb"\r?\n\r?\n", raw, maxsplit=1)
                require(len(parts) == 2, "github_invalid_response")
                raw = parts[1]
                if status != 200:
                    raise GitHubHTTPError(
                        status, raw.decode("utf-8", errors="replace")[:65536]
                    )
            elif not result.returncode:
                raise PublicationError("github_invalid_response")
        if result.returncode:
            if response_status is not None:
                raise GitHubHTTPError(
                    response_status, "gh failed after the HTTP response"
                )
            raise PublicationError(
                "github_failed", result.stderr.decode("utf-8", errors="replace")[:65536]
            )
        try:
            return json.loads(raw, object_pairs_hook=journal.unique)
        except (ValueError, UnicodeError):
            if response_status is not None:
                raise GitHubHTTPError(
                    response_status, "merge response is not valid JSON"
                ) from None
            raise PublicationError("github_invalid_response") from None

    def repository(self) -> dict[str, Any]:
        value = self.request(f"repos/{self.repo}")
        require(
            isinstance(value, dict)
            and value.get("full_name", "").casefold() == self.repo.casefold(),
            "repository_changed",
        )
        return dict(value)

    def find(self, branch: str) -> list[dict[str, Any]]:
        found: list[dict[str, Any]] = []
        page = 1
        while True:
            values = self.request(
                f"repos/{self.repo}/pulls?state=all&head={self.repo.split('/')[0]}:{branch}&per_page=100&page={page}"
            )
            require(
                isinstance(values, list) and len(values) <= 100, "github_pagination"
            )
            for pr in values:
                require(
                    isinstance(pr, dict)
                    and schema.integer(pr.get("number"))
                    and pr["number"] not in {p["number"] for p in found},
                    "github_pagination",
                )
                found.append(pr)
            if len(values) < 100:
                return found
            page += 1

    def get(self, number: int) -> dict[str, Any]:
        result = self.request(f"repos/{self.repo}/pulls/{number}")
        require(isinstance(result, dict), "github_invalid_response")
        return dict(result)

    def files(self, number: int) -> list[Any]:
        # We accept exactly three files. A full page (or any extra file) fails
        # closed, so pagination cannot hide an acceptable larger diff.
        result = self.request(f"repos/{self.repo}/pulls/{number}/files?per_page=100")
        require(isinstance(result, list), "github_invalid_response")
        return list(result)

    def create(self, branch: str, title: str, body: str) -> dict[str, Any]:
        result = self.request(
            f"repos/{self.repo}/pulls",
            "POST",
            {
                "base": "main",
                "head": branch,
                "title": title,
                "body": body,
                "maintainer_can_modify": False,
            },
        )
        require(isinstance(result, dict), "github_invalid_response")
        return dict(result)

    def merge(self, number: int, head: str) -> None:
        try:
            result = self.request(
                f"repos/{self.repo}/pulls/{number}/merge",
                "PUT",
                {"merge_method": "merge", "sha": head},
            )
        except GitHubHTTPError as exc:
            if not 400 <= exc.status < 500:
                # A server/proxy failure may follow a committed merge. Only
                # definite client refusals authorize clearing merge_attempted.
                raise
            code = "merge_conflict" if exc.status in {405, 409} else "merge_rejected"
            if exc.status == 409:
                try:
                    observed = self.get(number)
                except (ValueError, OSError, subprocess.SubprocessError):
                    # The mutation was definitely refused even when the
                    # follow-up read is unavailable. Never mark it ambiguous.
                    pass
                else:
                    if (observed.get("head") or {}).get("sha") != head:
                        code = "pr_identity_changed"
            raise MergeRejected(exc.status, code, exc.detail) from None
        if not isinstance(result, dict) or result.get("merged") is not True:
            raise GitHubHTTPError(200, "merge response did not confirm merged=true")


class Publisher:
    def __init__(self, root: Path, path: Path, github: GitHub | None = None) -> None:
        self.root = root.resolve()
        self.path = path
        self.github = github

    def git(self, *args: str, cwd: Path | None = None) -> bytes:
        result = subprocess.run(
            ["git", *args],
            cwd=cwd or self.root,
            capture_output=True,
            check=False,
            timeout=60,
        )
        if result.returncode:
            raise PublicationError(
                "git_failed", result.stderr.decode("utf-8", errors="replace")[:65536]
            )
        return result.stdout

    def main(self) -> str:
        self.git(
            "fetch",
            "--no-tags",
            "--no-write-fetch-head",
            "origin",
            "refs/heads/main:refs/remotes/origin/main",
        )
        return self.git("rev-parse", "origin/main").decode().strip()

    def worktree(self, row: dict[str, Any]) -> Path:
        common = Path(
            self.git("rev-parse", "--path-format=absolute", "--git-common-dir")
            .decode()
            .strip()
        ).resolve()
        require(common.name == ".git", "invalid_repository_layout")
        parent = common.parent / ".claude/worktrees"
        require(parent.resolve() == parent, "worktree_symlink")
        return (
            parent
            / f"worker-log-{row['source']['public_key']}-{row['source']['sequence']:04d}"
        )

    def verify(
        self, row: dict[str, Any], files: dict[str, bytes], head: str | None = None
    ) -> None:
        head = head or row["head"]
        require(
            self.git("rev-list", "--parents", "-n", "1", head).decode().split()
            == [head, row["base"]],
            "unexpected_commit_parent",
        )
        expected = {row["directory"] + "/" + name for name in files}
        schema.additions(
            self.git(
                "diff", "--raw", "--abbrev=40", "--no-renames", "-z", row["base"], head
            ),
            expected,
        )
        for name, raw in files.items():
            target = row["directory"] + "/" + name
            tree = self.git("ls-tree", head, "--", target).decode().split()
            require(tree[:2] == ["100644", "blob"], "unsafe_file_type")
            require(
                self.git("show", f"{head}:{target}") == raw, "committed_content_changed"
            )

    def commit(self, row: dict[str, Any], files: dict[str, bytes]) -> None:
        tree = self.worktree(row)
        if not tree.exists():
            tree.parent.mkdir(parents=True, exist_ok=True)
            branch_exists = (
                subprocess.run(
                    [
                        "git",
                        "show-ref",
                        "--verify",
                        "--quiet",
                        f"refs/heads/{row['branch']}",
                    ],
                    cwd=self.root,
                    capture_output=True,
                    check=False,
                ).returncode
                == 0
            )
            if branch_exists:
                head = self.git("rev-parse", row["branch"]).decode().strip()
                self.verify(row, files, head)
                self.git("worktree", "add", str(tree), row["branch"])
            else:
                self.git("worktree", "add", "-b", row["branch"], str(tree), row["base"])
        require(
            not tree.is_symlink()
            and self.git("branch", "--show-current", cwd=tree).decode().strip()
            == row["branch"],
            "worktree_changed",
        )
        require(
            self.git(
                "rev-parse", "--path-format=absolute", "--git-common-dir", cwd=tree
            )
            == self.git("rev-parse", "--path-format=absolute", "--git-common-dir"),
            "worktree_changed",
        )
        require(
            not self.git(
                "status", "--porcelain", "--untracked-files=all", cwd=tree
            ).strip(),
            "worktree_dirty",
        )
        current = self.git("rev-parse", "HEAD", cwd=tree).decode().strip()
        if current != row["base"]:
            self.verify(row, files, current)
        else:
            target = tree / row["directory"]
            for ancestor in (target, *target.parents):
                if ancestor == tree:
                    break
                require(not ancestor.is_symlink(), "worktree_symlink")
            target.mkdir(parents=True, exist_ok=False)
            for name, raw in files.items():
                with (target / name).open("xb") as stream:
                    stream.write(raw)
            self.git(
                "add",
                "--",
                *[row["directory"] + "/" + n for n in sorted(files)],
                cwd=tree,
            )
            self.git(
                "commit",
                "-m",
                f"Publish worker log for issue {row['source']['issue']} turn {row['source']['sequence']}",
                cwd=tree,
            )
            current = self.git("rev-parse", "HEAD", cwd=tree).decode().strip()
            self.verify(row, files, current)
        row.update(head=current, stage="committed")
        journal.atomic(self.path, row)

    def pr(self, row: dict[str, Any]) -> dict[str, Any]:
        assert self.github is not None
        value = self.github.get(row["pr_number"])
        schema.identity(value, row, self.github.repo)
        require(
            value["number"] == row["pr_number"],
            "pr_identity_changed",
            "PR identity mismatch: number",
        )
        require(
            type(value.get("changed_files")) is int and value["changed_files"] == 3,
            "pr_identity_changed",
            "PR identity mismatch: changed_files",
        )
        return value

    def verify_main(self, row: dict[str, Any], main: str) -> None:
        result = subprocess.run(
            ["git", "merge-base", "--is-ancestor", row["base"], main],
            cwd=self.root,
            capture_output=True,
            check=False,
            timeout=60,
        )
        require(
            result.returncode in {0, 1},
            "git_failed",
            result.stderr.decode("utf-8", errors="replace")[:65536],
        )
        require(
            result.returncode == 0,
            "base_not_ancestor",
            f"Pinned base {row['base']} is not an ancestor of main {main}",
        )
        require(
            not self.git("ls-tree", main, "--", row["directory"]).strip(),
            "record_already_exists",
        )

    def confirm_main(
        self, row: dict[str, Any], files: dict[str, bytes], pr: dict[str, Any]
    ) -> None:
        require(
            pr.get("merged") is True
            and pr.get("state") == "closed"
            and schema.digest(pr.get("merge_commit_sha"), 40),
            "merge_unconfirmed",
        )
        main = self.main()
        self.git("merge-base", "--is-ancestor", row["head"], main)
        self.git("merge-base", "--is-ancestor", pr["merge_commit_sha"], main)
        for name, raw in files.items():
            require(
                self.git("show", f"{main}:{row['directory']}/{name}") == raw,
                "main_content_changed",
            )
        row.update(
            state="merged",
            stage="merged",
            error=None,
            merge_commit=pr["merge_commit_sha"],
            main_observed=main,
        )
        journal.atomic(self.path, row)
        tree = self.worktree(row)
        if (
            tree.exists()
            and not tree.is_symlink()
            and self.git("rev-parse", "HEAD", cwd=tree).decode().strip() == row["head"]
            and not self.git(
                "status", "--porcelain", "--untracked-files=all", cwd=tree
            ).strip()
        ):
            self.git("worktree", "remove", str(tree))

    def publish(self, *, dry_run: bool = False, retry: bool = False) -> dict[str, Any]:
        with nullcontext() if dry_run else journal.locked(self.path):
            row = journal.read(self.path)
            try:
                if (
                    retry
                    and not dry_run
                    and row["state"] == "blocked"
                    and "receipt" in row
                    and "files" not in row
                ):
                    recovered = render(row["source"], row["receipt"])
                    row.update(
                        files={k: v.decode("utf-8") for k, v in recovered.items()},
                        state="ready",
                        error=None,
                    )
                    journal.atomic(self.path, row)
                require(
                    row["state"] in {"ready", "pending", "merged"}
                    or retry
                    and row["state"] == "blocked"
                    and "files" in row,
                    row.get("error") or "final_not_ready",
                )
                files = {k: v.encode("utf-8") for k, v in row["files"].items()}
                require(
                    files == render(row["source"], row["receipt"]),
                    "prepared_content_changed",
                )
                guard = guard_for(row["source"], row["receipt"]["record"])
                meta = schema.validate(files, guard)
                if dry_run:
                    return {
                        **journal.view(row),
                        "dry_run": True,
                        "directory": schema.path(meta),
                    }
                if self.github is None:
                    self.github = GitHub(meta["repo"])
                require(
                    self.github.repo.casefold() == meta["repo"].casefold(),
                    "repository_changed",
                )
                repository = self.github.repository()
                origin = self.git("remote", "get-url", "origin").decode().strip()
                require(
                    canonical_remote(origin)
                    in {
                        canonical_remote(u)
                        for u in (
                            repository.get("clone_url"),
                            repository.get("ssh_url"),
                        )
                        if isinstance(u, str)
                    },
                    "repository_changed",
                )
                main = self.main()
                if "base" not in row:
                    row.update(
                        base=main,
                        directory=schema.path(meta),
                        branch=f"worker-log/{meta['delegation_key']}/{meta['turn']['sequence']:04d}",
                        state="pending",
                        error=None,
                    )
                    journal.atomic(self.path, row)
                require(
                    row["directory"] == schema.path(meta)
                    and row["branch"]
                    == f"worker-log/{meta['delegation_key']}/{meta['turn']['sequence']:04d}",
                    "publication_identity_changed",
                )
                if "head" not in row:
                    self.commit(row, files)
                self.verify(row, files)
                expected_body = f"<!-- worker-log:v1 key={meta['delegation_key']} turn={meta['turn']['sequence']} head={row['head']} -->\n\nRefs #{meta['parent']}, #{meta['issue']}\n\n検査済みworkerログ3ファイルの追加。専用入口が固定HEADとmainへの保存を照合する。\n"
                require(
                    row.get("pr_body", expected_body) == expected_body,
                    "publication_identity_changed",
                )
                row["pr_body"] = expected_body
                guard.check([row["branch"], row["directory"], row["pr_body"]])
                journal.atomic(self.path, row)
                if "pr_number" not in row:
                    remote = (
                        self.git(
                            "ls-remote",
                            "--heads",
                            "origin",
                            f"refs/heads/{row['branch']}",
                        )
                        .decode()
                        .split()
                    )
                    require(
                        not remote
                        or remote == [row["head"], f"refs/heads/{row['branch']}"],
                        "remote_head_changed",
                    )
                    self.git(
                        "push", "origin", f"{row['head']}:refs/heads/{row['branch']}"
                    )
                    row["stage"] = "pushed"
                    journal.atomic(self.path, row)
                    existing = self.github.find(row["branch"])
                    require(len(existing) <= 1, "multiple_log_prs")
                    if existing:
                        created = existing[0]
                    else:
                        require(
                            not row.get("pr_create_attempted"), "pr_creation_uncertain"
                        )
                        row.update(pr_create_attempted=True, stage="pr_create")
                        journal.atomic(self.path, row)
                        created = self.github.create(
                            row["branch"],
                            f"workerログ: #{meta['issue']} turn {meta['turn']['sequence']}",
                            row["pr_body"],
                        )
                    schema.identity(created, row, self.github.repo)
                    row.update(pr_number=created["number"], stage="pr_open")
                    journal.atomic(self.path, row)
                pr = self.pr(row)
                if pr.get("merged") is True:
                    self.confirm_main(row, files, pr)
                    return journal.view(row)
                require(pr.get("state") == "open", "closed_unmerged")
                require(not row.get("merge_attempted"), "merge_response_uncertain")
                # Another parent's independent additions may have advanced main.
                main = self.main()
                self.verify_main(row, main)
                require(pr.get("mergeable") is not False, "merge_conflict")
                self.verify(row, files)
                schema.pr_additions(
                    self.github.files(row["pr_number"]),
                    {row["directory"] + "/" + name for name in files},
                )
                fresh = self.pr(row)
                require(
                    fresh.get("state") == "open",
                    "pr_identity_changed",
                    "PR identity mismatch: state",
                )
                require(fresh.get("mergeable") is not False, "merge_conflict")
                # base.sha is GitHub's PR snapshot, not the current main ref.
                # Revalidate old blocked records through every guard above;
                # neither --retry nor a stale base SHA bypasses identity checks.
                if fresh["base"].get("sha") != main:
                    row["diagnostic"] = json.dumps(
                        {
                            "reason": "stale_pr_base_sha",
                            "pr_base": fresh["base"].get("sha"),
                            "base": row["base"],
                            "main": main,
                        },
                        sort_keys=True,
                    )
                row.update(
                    state="pending",
                    error=None,
                    merge_attempted=True,
                    stage="merge_attempted",
                )
                row.pop("merge_http_status", None)
                journal.atomic(self.path, row)
                self.github.merge(row["pr_number"], row["head"])
                row["merge_http_status"] = 200
                self.confirm_main(row, files, self.pr(row))
                return journal.view(row)
            except (
                ValueError,
                OSError,
                KeyError,
                TypeError,
                AttributeError,
                UnicodeError,
                subprocess.SubprocessError,
            ) as exc:
                code = (
                    str(exc)
                    if isinstance(exc, PublicationError)
                    else "merge_response_uncertain"
                    if row.get("merge_attempted")
                    and isinstance(exc, subprocess.SubprocessError)
                    else "git_failed"
                    if isinstance(exc, subprocess.SubprocessError)
                    else "publication_failed"
                )
                if dry_run:
                    return {**journal.view(row), "dry_run": True, "error": code}
                if isinstance(exc, PublicationError) and exc.detail:
                    row["diagnostic"] = exc.detail
                if isinstance(exc, GitHubHTTPError):
                    row["merge_http_status"] = exc.status
                if isinstance(exc, MergeRejected):
                    row.update(
                        merge_attempted=False,
                        stage="merge_rejected",
                        merge_http_status=exc.status,
                    )
                elif row.get("merge_attempted") and code in {
                    "github_failed",
                    "github_invalid_response",
                }:
                    code = "merge_response_uncertain"
                # Transient transport errors remain retryable, but an ambiguous
                # POST/merge is reconciled before ever issuing another mutation.
                pending = not isinstance(exc, MergeRejected) and code in {
                    "github_failed",
                    "git_failed",
                    "github_invalid_response",
                    "pr_creation_uncertain",
                    "merge_response_uncertain",
                    "merge_unconfirmed",
                }
                journal.fail(
                    self.path,
                    row,
                    code,
                    state="pending" if pending and "files" in row else "blocked",
                )
                return journal.view(row)
