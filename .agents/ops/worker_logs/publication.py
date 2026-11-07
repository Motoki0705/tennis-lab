"""A dedicated addition-only PR entry, independent of the normal CI/review gate."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

from . import journal, schema
from .journal import PublicationError, require
from .sources import guard_for, render


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
        if payload is not None:
            command += ["--input", "-"]
        result = subprocess.run(
            command,
            input=json.dumps(payload).encode() if payload is not None else None,
            capture_output=True,
            check=False,
            timeout=60,
        )
        if result.returncode:
            raise PublicationError(
                "github_failed", result.stderr.decode("utf-8", errors="replace")[:65536]
            )
        try:
            return json.loads(result.stdout, object_pairs_hook=journal.unique)
        except (ValueError, UnicodeError):
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
        result = self.request(
            f"repos/{self.repo}/pulls/{number}/merge",
            "PUT",
            {"merge_method": "merge", "sha": head},
        )
        require(
            isinstance(result, dict) and result.get("merged") is True,
            "merge_unconfirmed",
        )


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
            value["number"] == row["pr_number"] and value.get("changed_files") == 3,
            "pr_identity_changed",
        )
        return value

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
        with journal.locked(self.path):
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
                self.git("merge-base", "--is-ancestor", row["base"], main)
                require(
                    not self.git("ls-tree", main, "--", row["directory"]).strip(),
                    "record_already_exists",
                )
                if pr.get("mergeable") is None:
                    return journal.view(row)
                require(pr.get("mergeable") is True, "merge_conflict")
                fresh = self.pr(row)
                require(
                    fresh.get("state") == "open"
                    and fresh.get("mergeable") is True
                    and fresh["base"]["sha"] == main,
                    "pr_identity_changed",
                )
                self.verify(row, files)
                row.update(merge_attempted=True, stage="merge_attempted")
                journal.atomic(self.path, row)
                self.github.merge(row["pr_number"], row["head"])
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
                    else "publication_failed"
                )
                if dry_run:
                    return {**journal.view(row), "dry_run": True, "error": code}
                if isinstance(exc, PublicationError) and exc.detail:
                    row["diagnostic"] = exc.detail
                # Transient transport errors remain retryable, but an ambiguous
                # POST/merge is reconciled before ever issuing another mutation.
                pending = code in {
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
