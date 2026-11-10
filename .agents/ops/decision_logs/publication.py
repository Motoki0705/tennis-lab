"""Explicit, batched publication. Never invokes a runner or capture adapter."""

from __future__ import annotations

import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

from worker_logs.journal import PublicationError, atomic, locked, require
from worker_logs.publication import GitHub as WorkerGitHub
from worker_logs.publication import Publisher as GitTransport
from worker_logs.publication import canonical_remote
from worker_logs.safety import Guard
from worker_logs.schema import digest, integer, json_bytes, sha

from . import publication_policy as policy
from . import publication_state as storage
from .journal import Journal
from .navigation import ROOT, event_path, generate
from .schema import new_id


def now() -> str:
    return datetime.now(UTC).isoformat()


class GitHub(WorkerGitHub):
    def files(self, number: int) -> list[Any]:
        """Batches can exceed 100 files; no page or duplicate may be hidden."""
        found: list[Any] = []
        names: set[str] = set()
        page = 1
        while True:
            values = self.request(
                f"repos/{self.repo}/pulls/{number}/files?per_page=100&page={page}"
            )
            require(
                isinstance(values, list) and len(values) <= 100, "github_pagination"
            )
            for value in values:
                require(
                    isinstance(value, dict)
                    and isinstance(value.get("filename"), str)
                    and value["filename"] not in names,
                    "github_pagination",
                )
                names.add(value["filename"])
                found.append(value)
            if len(values) < 100:
                return found
            page += 1


class Publisher:
    def __init__(
        self, root: Path, journal: Journal, github: GitHub | None = None
    ) -> None:
        self.root, self.journal = root.resolve(), journal
        self.path = journal.path.with_name("publication.json")
        self.github = github or GitHub(journal.repo)
        self.transport = GitTransport(self.root, self.path)
        self.root = Path(self.git("rev-parse", "--show-toplevel").decode().strip())
        self.transport = GitTransport(self.root, self.path)
        self.document: dict[str, Any] = {}
        self._snapshots: dict[str, dict[str, bytes]] = {}

    def git(self, *args: str, cwd: Path | None = None) -> bytes:
        return cast(bytes, self.transport.git(*args, cwd=cwd))

    def main(self) -> str:
        return cast(str, self.transport.main())

    def snapshot(self, ref: str) -> dict[str, bytes]:
        require(digest(ref, 40), "unsafe_reference")
        if ref in self._snapshots:
            return dict(self._snapshots[ref])
        files = {}
        raw = self.git("ls-tree", "-r", "-z", ref, "--", ROOT)
        for entry in raw.rstrip(b"\0").split(b"\0") if raw else []:
            metadata, name = entry.split(b"\t", 1)
            mode, kind, oid = metadata.split()
            require(mode == b"100644" and kind == b"blob", "unsafe_file_type")
            files[name.decode("utf-8")] = self.git(
                "cat-file", "blob", oid.decode("ascii")
            )
        self._snapshots[ref] = files
        return dict(files)

    def save(self) -> None:
        atomic(self.path, self.document)

    def expected(self, batch: dict[str, Any]) -> dict[str, bytes]:
        return {event_path(p["record"]): json_bytes(p) for p in batch["payloads"]}

    def plan(self, batch: dict[str, Any], base: str) -> dict[str, bytes]:
        before = self.snapshot(base)
        expected = self.expected(batch)
        for name, raw in expected.items():
            require(name not in before or before[name] == raw, "record_already_exists")
        union = {**before, **expected}
        union.update(generate(union))
        return {n: raw for n, raw in union.items() if before.get(n) != raw}

    def verify(
        self, batch: dict[str, Any], revision: dict[str, Any], head: str | None = None
    ) -> dict[str, str]:
        head = head or revision["head"]
        require(digest(head, 40), "journal_invalid")
        require(
            self.git("rev-list", "--parents", "-n", "1", head).decode().split()
            == [head, revision["base"]],
            "unexpected_commit_parent",
        )
        diff = policy.changes(
            self.git(
                "diff",
                "--raw",
                "--abbrev=40",
                "--no-renames",
                "-z",
                revision["base"],
                head,
            )
        )
        policy.verify(
            self.snapshot(revision["base"]),
            self.snapshot(head),
            diff,
            parent=batch["parent"],
            expected=self.expected(batch),
        )
        return diff

    def worktree(self, batch: dict[str, Any], revision: int | None = None) -> Path:
        common = Path(
            self.git("rev-parse", "--path-format=absolute", "--git-common-dir")
            .decode()
            .strip()
        ).resolve()
        require(common.name == ".git", "invalid_repository_layout")
        directory = common.parent / ".claude/worktrees"
        require(directory.resolve() == directory, "worktree_symlink")
        index = revision if revision is not None else len(batch["revisions"])
        return directory / f"decision-log-{batch['key']}-{index:04d}"

    def cleanup(self, batch: dict[str, Any]) -> int:
        retained = 0
        for index, revision in enumerate(batch["revisions"], 1):
            tree = self.worktree(batch, index)
            if not tree.exists() and not tree.is_symlink():
                continue
            if (
                tree.is_symlink()
                or revision["head"] is None
                or self.git(
                    "rev-parse", "--path-format=absolute", "--git-common-dir", cwd=tree
                )
                != self.git("rev-parse", "--path-format=absolute", "--git-common-dir")
                or self.git("rev-parse", "HEAD", cwd=tree).decode().strip()
                != revision["head"]
                or self.git(
                    "status", "--porcelain", "--untracked-files=all", cwd=tree
                ).strip()
            ):
                retained += 1
                continue
            self.git("worktree", "remove", str(tree))
        return retained

    def commit(self, batch: dict[str, Any], revision: dict[str, Any]) -> None:
        files = self.plan(batch, revision["base"])
        require(bool(files), "unsafe_diff")
        tree = self.worktree(batch)
        require(not tree.is_symlink(), "worktree_symlink")
        if not tree.exists():
            tree.parent.mkdir(parents=True, exist_ok=True)
            self.git("worktree", "add", "--detach", str(tree), revision["base"])
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
        head = self.git("rev-parse", "HEAD", cwd=tree).decode().strip()
        if head == revision["base"]:
            for name, raw in files.items():
                target = tree / name
                require(
                    not any(
                        p.is_symlink()
                        for p in (target, *target.parents)
                        if p != tree.parent
                    ),
                    "worktree_symlink",
                )
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(raw)
            # Only the fully validated public file set is force-added; old main's
            # ignore rules may predate this new publication format.
            self.git("add", "--force", "--", *sorted(files), cwd=tree)
            self.git(
                "commit",
                "-m",
                f"Publish parent {batch['parent']} decision records",
                cwd=tree,
            )
            head = self.git("rev-parse", "HEAD", cwd=tree).decode().strip()
        self.verify(batch, revision, head)
        revision["head"] = head
        batch.update(stage="committed", state="pending", error=None)
        self.save()

    def remote(self, batch: dict[str, Any]) -> str | None:
        raw = (
            self.git("ls-remote", "--heads", "origin", f"refs/heads/{batch['branch']}")
            .decode()
            .split()
        )
        require(
            not raw or len(raw) == 2 and raw[1] == f"refs/heads/{batch['branch']}",
            "remote_head_changed",
        )
        return raw[0] if raw else None

    def push(self, batch: dict[str, Any]) -> None:
        head = batch["revisions"][-1]["head"]
        remote = self.remote(batch)
        known = {r["head"] for r in batch["revisions"] if r["head"]}
        require(remote is None or remote in known, "remote_head_changed")
        if remote != head:
            # A clean verified replacement is generated from new main; the lease
            # prevents overwriting any unrelated remote update.
            self.git(
                "push",
                f"--force-with-lease=refs/heads/{batch['branch']}:{remote or ''}",
                "origin",
                f"{head}:refs/heads/{batch['branch']}",
            )
        batch.update(stage="pushed", state="pending", error=None)
        self.save()

    def pull_request(
        self, batch: dict[str, Any], diff: dict[str, str]
    ) -> dict[str, Any]:
        if batch["pr_number"] is None:
            found = self.github.find(batch["branch"])
            require(len(found) <= 1, "multiple_log_prs")
            if found:
                pr = found[0]
            else:
                require(not batch["pr_create_attempted"], "pr_creation_uncertain")
                batch.update(pr_create_attempted=True, stage="pr_create")
                self.save()
                pr = self.github.create(
                    batch["branch"],
                    f"親判断記録: #{batch['parent']}",
                    policy.body(batch),
                )
            require(integer(pr.get("number")), "github_invalid_response")
            batch["pr_number"] = pr["number"]
            self.save()
        pr = self.github.get(batch["pr_number"])
        policy.identity(pr, batch, batch["revisions"][-1]["head"], len(diff))
        return dict(pr)

    def note(
        self,
        batch: dict[str, Any],
        status: str,
        *,
        error: str | None = None,
        commit: str | None = None,
    ) -> None:
        current = {r["id"]: r for r in self.journal.status()}
        for payload in batch["payloads"]:
            record = payload["record"]
            state = current[record["id"]]
            if state["publication_status"] == "published" or (
                state["publication_status"] == status and state["error"] == error
            ):
                continue
            evidence = (
                []
                if commit is None
                else [
                    {
                        "kind": "public",
                        "url": f"https://github.com/{batch['repo']}/blob/{commit}/{event_path(record)}",
                        "revision": commit,
                        "content_sha256": sha(json_bytes(payload)),
                    }
                ]
            )
            self.journal.publication(
                record["id"],
                key="publication:" + new_id(),
                status=status,
                at=now(),
                evidence=evidence,
                error=error,
            )

    def confirm(self, batch: dict[str, Any], pr: dict[str, Any]) -> dict[str, Any]:
        revision = batch["revisions"][-1]
        diff = self.verify(batch, revision)
        policy.identity(pr, batch, revision["head"], len(diff))
        require(
            pr.get("merged") is True
            and pr.get("state") == "closed"
            and digest(pr.get("merge_commit_sha"), 40),
            "merge_unconfirmed",
        )
        main = self.main()
        commit = pr["merge_commit_sha"]
        self.git("merge-base", "--is-ancestor", revision["head"], main)
        self.git("merge-base", "--is-ancestor", commit, main)
        saved = self.snapshot(main)
        require(
            all(saved.get(n) == raw for n, raw in self.expected(batch).items()),
            "main_content_changed",
        )
        # Check the actual merge tree too, including concurrent additions. Later
        # legitimate main updates may of course regenerate these READMEs again.
        merged = self.snapshot(commit)
        canonical = all(merged.get(n) == raw for n, raw in generate(merged).items())
        if not canonical:
            canonical = all(saved.get(n) == raw for n, raw in generate(saved).items())
        if not canonical:
            batch.update(
                state="blocked",
                stage="readme_refresh_required",
                error="prepared_content_changed",
                merge_commit=commit,
                merge_attempted=True,
            )
            self.save()
            self.note(batch, "blocked", error="prepared_content_changed")
            return self.view(batch)
        batch.update(
            state="merged",
            stage="merged",
            error=None,
            merge_commit=commit,
            merge_attempted=True,
        )
        self.save()
        self.note(batch, "published", commit=commit)
        return self.view(batch, retained_worktrees=self.cleanup(batch))

    def view(self, batch: dict[str, Any] | None = None, **extra: Any) -> dict[str, Any]:
        status = self.journal.status()
        result = {
            "state": batch["state"] if batch else "idle",
            "stage": batch["stage"] if batch else "idle",
            "saved_to_main": bool(status)
            and (batch is None or batch["state"] == "merged")
            and all(r["publication_status"] == "published" for r in status),
            "unpublished": sum(r["publication_status"] != "published" for r in status),
            "blocked": sum(r["publication_status"] == "blocked" for r in status),
            "error": batch["error"] if batch else None,
            "pr": f"https://github.com/{batch['repo']}/pull/{batch['pr_number']}"
            if batch and batch["pr_number"]
            else None,
            "head": batch["revisions"][-1]["head"]
            if batch and batch["revisions"]
            else None,
            **extra,
        }
        Guard().check(result)
        return result

    def _publish(self, *, retry: bool) -> dict[str, Any]:
        # Preparation reuses the shared transformation; capture insertion order
        # is dependency order. No ready projection is ever re-rendered.
        for state in self.journal.status():
            if (
                state["publication_status"] == "unprepared"
                or retry
                and state["publication_status"] == "blocked"
            ):
                self.journal.prepare(
                    state["id"],
                    key="publication-prepare:" + new_id(),
                    at=now(),
                    retry=retry,
                )
        payloads = self.journal.public_payloads()
        self.document = storage.read(self.path, self.journal.repo, self.journal.parent)
        active = next(
            (b for b in self.document["batches"] if b["state"] != "merged"), None
        )
        if active and active["state"] == "blocked" and not retry:
            return self.view(active)
        repository = self.github.repository()
        require(
            self.github.repo.casefold() == self.journal.repo.casefold(),
            "repository_changed",
        )
        origin = canonical_remote(
            self.git("remote", "get-url", "origin").decode().strip()
        )
        require(
            origin
            in {
                canonical_remote(u)
                for u in (repository.get("clone_url"), repository.get("ssh_url"))
                if isinstance(u, str)
            },
            "repository_changed",
        )
        main = self.main()
        status = {r["id"]: r for r in self.journal.status()}
        # Process death can fall between recording the observed merge and
        # appending all per-record publication facts. Reconcile that batch first.
        for batch in self.document["batches"]:
            if batch["state"] == "merged" and any(
                status[p["record"]["id"]]["publication_status"] != "published"
                for p in batch["payloads"]
            ):
                reconciled = self.confirm(batch, self.github.get(batch["pr_number"]))
                if reconciled["state"] != "merged":
                    return reconciled
        status = {r["id"]: r for r in self.journal.status()}
        if active is None:
            pending = [
                p
                for p in payloads
                if status[p["record"]["id"]]["publication_status"] != "published"
            ]
            if not pending:
                last = (
                    self.document["batches"][-1] if self.document["batches"] else None
                )
                if last:
                    return self.confirm(last, self.github.get(last["pr_number"]))
                return self.view()
            key = new_id()
            active = {
                "repo": self.journal.repo,
                "parent": self.journal.parent,
                "key": key,
                "branch": storage.branch(self.journal.parent, key),
                "payloads": pending,
                "revisions": [],
                "pr_number": None,
                "pr_create_attempted": False,
                "merge_attempted": False,
                "state": "pending",
                "stage": "prepared",
                "error": None,
                "diagnostic": None,
                "merge_commit": None,
                "superseded_prs": [],
            }
            self.document["batches"].append(active)
            self.save()
        current = {p["record"]["id"]: p for p in payloads}
        require(
            all(current.get(p["record"]["id"]) == p for p in active["payloads"]),
            "prepared_content_changed",
        )
        if active["stage"] == "readme_refresh_required":
            # The merge succeeded but a last-boundary main update changed the
            # deterministic view. Preserve that PR and fixed envelopes; only an
            # explicit retry can create a README repair under a distinct branch.
            pr = self.github.get(active["pr_number"])
            observed = self.confirm(active, pr)
            if observed["state"] == "merged":
                return observed
            active["superseded_prs"].append(
                {
                    "number": active["pr_number"],
                    "branch": active["branch"],
                    "head": active["revisions"][-1]["head"],
                    "merge_commit": active["merge_commit"],
                }
            )
            active.update(
                branch=storage.branch(
                    active["parent"], active["key"], len(active["superseded_prs"])
                ),
                pr_number=None,
                pr_create_attempted=False,
                merge_attempted=False,
                state="pending",
                stage="refreshing_readmes",
                error=None,
                merge_commit=None,
            )
            self.save()
        self.note(active, "pending")
        for _ in range(3):
            if active["pr_number"] is not None:
                pr = self.github.get(active["pr_number"])
                policy.identity(pr, active, active["revisions"][-1]["head"])
                if pr.get("merged") is True:
                    return self.confirm(active, pr)
                require(pr.get("state") == "open", "closed_unmerged")
            require(not active["merge_attempted"], "merge_response_uncertain")
            revisions = active["revisions"]
            if revisions:
                require(
                    self.git("merge-base", revisions[-1]["base"], main).decode().strip()
                    == revisions[-1]["base"],
                    "base_not_ancestor",
                )
            if not revisions or revisions[-1]["base"] != main:
                revisions.append({"base": main, "head": None})
                active.update(state="pending", error=None, stage="regenerating")
                self.save()
            revision = revisions[-1]
            if revision["head"] is None:
                self.commit(active, revision)
            diff = self.verify(active, revision)
            self.push(active)
            pr = self.pull_request(active, diff)
            if pr.get("merged") is True:
                return self.confirm(active, pr)
            require(pr.get("state") == "open", "closed_unmerged")
            latest = self.main()
            if latest != main:
                main = latest
                continue
            if pr.get("mergeable") is None:
                active.update(stage="mergeability_pending", state="pending", error=None)
                self.save()
                return self.view(active)
            require(pr.get("mergeable") is True, "merge_conflict")
            self.verify(active, revision)
            policy.pr_changes(self.github.files(active["pr_number"]), diff)
            fresh = self.github.get(active["pr_number"])
            policy.identity(fresh, active, revision["head"], len(diff))
            if fresh.get("merged") is True:
                return self.confirm(active, fresh)
            require(fresh.get("state") == "open", "closed_unmerged")
            latest = self.main()
            if latest != main:
                main = latest
                continue
            if fresh.get("mergeable") is None:
                active.update(stage="mergeability_pending", state="pending", error=None)
                self.save()
                return self.view(active)
            require(fresh.get("mergeable") is True, "merge_conflict")
            active.update(merge_attempted=True, stage="merge_attempted", error=None)
            self.save()
            self.github.merge(active["pr_number"], revision["head"])
            return self.confirm(active, self.github.get(active["pr_number"]))
        active.update(stage="main_advanced", state="pending", error=None)
        self.save()
        return self.view(active)

    def publish(self, *, dry_run: bool = False, retry: bool = False) -> dict[str, Any]:
        # The read precedes lock creation to reject a symlinked private hierarchy.
        document = storage.read(self.path, self.journal.repo, self.journal.parent)
        if dry_run:
            payloads = self.journal.public_payloads()
            base = self.git("rev-parse", "origin/main").decode().strip()
            files = self.snapshot(base)
            expected = {event_path(p["record"]): json_bytes(p) for p in payloads}
            require(
                all(n not in files or files[n] == raw for n, raw in expected.items()),
                "record_already_exists",
            )
            files.update(expected)
            batch = next(
                (b for b in document["batches"] if b["state"] != "merged"), None
            )
            return self.view(
                batch,
                dry_run=True,
                generated_readmes=len(generate(files)),
                prepared_records=len(payloads),
            )
        with locked(self.path):
            try:
                return self._publish(retry=retry)
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
                active = next(
                    (
                        b
                        for b in self.document.get("batches", [])
                        if b["state"] != "merged"
                    ),
                    None,
                )
                if active:
                    transient = code in {
                        "github_failed",
                        "git_failed",
                        "github_invalid_response",
                        "pr_creation_uncertain",
                        "merge_response_uncertain",
                        "merge_unconfirmed",
                    }
                    active.update(
                        state="pending" if transient else "blocked",
                        error=code,
                        diagnostic=exc.detail
                        if isinstance(exc, PublicationError)
                        else str(exc),
                    )
                    self.save()
                    self.note(
                        active,
                        "pending" if transient else "blocked",
                        error=None if transient else code,
                    )
                return self.view(active, error=code)
