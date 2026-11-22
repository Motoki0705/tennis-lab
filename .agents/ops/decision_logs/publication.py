"""Explicit, batched publication. Never invokes a runner or capture adapter."""

from __future__ import annotations

import subprocess
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

from worker_logs.journal import PublicationError, atomic, require
from worker_logs.ownership import require_owner
from worker_logs.publication import GitHub as WorkerGitHub
from worker_logs.publication import GitHubHTTPError, MergeRejected, canonical_remote
from worker_logs.publication import Publisher as GitTransport
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
        self,
        root: Path,
        journal: Journal,
        github: GitHub | None = None,
        *,
        record_ids: set[str] | None = None,
    ) -> None:
        self.root, self.journal = root.resolve(), journal
        self.path = journal.path.with_name("publication.json")
        self.github = github or GitHub(journal.repo)
        self.transport = GitTransport(self.root, self.path)
        self.root = Path(self.git("rev-parse", "--show-toplevel").decode().strip())
        self.transport = GitTransport(self.root, self.path)
        self.document: dict[str, Any] = {}
        self._snapshots: dict[str, dict[str, bytes]] = {}
        self.record_ids = record_ids
        self.owner_check: Callable[[], bool] | None = None
        self.actor: dict[str, Any] | None = None
        self.publication_actor: dict[str, Any] | None = None

    def git(self, *args: str, cwd: Path | None = None) -> bytes:
        return cast(bytes, self.transport.git(*args, cwd=cwd))

    def main(self) -> str:
        require_owner(self.owner_check)
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
        require_owner(self.owner_check)
        atomic(self.path, self.document)

    def batch_actor(self, batch: dict[str, Any]) -> dict[str, Any] | None:
        """Use the frozen batch actor, including legacy partial confirmations."""
        ids = {p["record"]["id"] for p in batch["payloads"]}
        entries = self.journal.read()["entries"]
        actors = {
            json_bytes(e["publication_actor"]): e["publication_actor"]
            for e in entries
            if e["event"] == "publication"
            and e["record_id"] in ids
            and "publication_actor" in e
            and ("publication_actor" not in batch or e["status"] == "published")
        }
        if "publication_actor" in batch:
            actor = batch["publication_actor"]
            require(
                all(a == actor for a in actors.values()), "publication_identity_changed"
            )
        else:
            # Earlier batches did not persist their actor separately. Existing
            # publication facts are authoritative; never allocate a new request
            # from the remaining subset of that batch's records.
            if not actors:
                actors = {
                    json_bytes(
                        {
                            "parent": e["private"]["publication_actor"]["parent"],
                            "request_id": e["record"]["id"],
                        }
                    ): {
                        "parent": e["private"]["publication_actor"]["parent"],
                        "request_id": e["record"]["id"],
                    }
                    for e in entries
                    if e["event"] == "capture"
                    and e["record"]["id"] in ids
                    and "publication_actor" in e["private"]
                }
            require(len(actors) <= 1, "publication_identity_changed")
            actor = next(iter(actors.values()), None)
        if actor is not None:
            request = next(
                (
                    e
                    for e in entries
                    if e["event"] == "capture"
                    and e["record"]["id"] == actor["request_id"]
                ),
                None,
            )
            require(
                request is not None
                and request["record"]["kind"] == "action"
                and request["record"]["data"]["operation"] == "publish"
                and request["private"].get("publication_actor", {}).get("parent")
                == actor["parent"],
                "publication_identity_changed",
            )
        return cast(dict[str, Any] | None, actor)

    def request_actor(self, attribution: dict[str, Any]) -> dict[str, Any]:
        request = next(
            e
            for e in self.journal.read()["entries"]
            if e["event"] == "capture"
            and e["record"]["id"] == attribution["request_id"]
        )
        return cast(dict[str, Any], request["private"]["publication_actor"])

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
        diff = self.commit_diff(revision, head)
        policy.verify(
            self.snapshot(revision["base"]),
            self.snapshot(head),
            diff,
            parent=batch["parent"],
            expected=self.expected(batch),
        )
        return diff

    def commit_diff(
        self, revision: dict[str, Any], head: str | None = None
    ) -> dict[str, str]:
        head = head or revision["head"]
        require(digest(head, 40), "journal_invalid")
        require(
            self.git("rev-list", "--parents", "-n", "1", head).decode().split()
            == [head, revision["base"]],
            "unexpected_commit_parent",
        )
        return dict(
            policy.changes(
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
        )

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
                    "status", "--porcelain", "--untracked-files=all", "--ignored", cwd=tree
                ).strip()
            ):
                retained += 1
                continue
            require_owner(self.owner_check)
            self.git("worktree", "remove", str(tree))
        return retained

    def prepared_worktree(self, tree: Path, files: dict[str, bytes], base: str) -> None:
        """Resume only exact plan bytes or untouched base files; preserve tampering."""
        try:
            staged = policy.changes(
                self.git(
                    "diff",
                    "--cached",
                    "--raw",
                    "--abbrev=40",
                    "--no-renames",
                    "-z",
                    base,
                    cwd=tree,
                )
            )
            working = policy.changes(
                self.git(
                    "diff", "--raw", "--abbrev=40", "--no-renames", "-z", base, cwd=tree
                )
            )
        except PublicationError:
            raise PublicationError("worktree_dirty") from None
        # Ignore rules control publication paths too. They must not hide a
        # recovery file outside the frozen plan from validation or cleanup.
        others = self.git("ls-files", "--others", "-z", cwd=tree).rstrip(b"\0")
        untracked = {n.decode() for n in others.split(b"\0")} if others else set()
        require(
            set(staged) | set(working) | untracked <= files.keys(), "worktree_dirty"
        )
        before = self.snapshot(base)
        for name, raw in files.items():
            target = tree / name
            require(
                not any(p.is_symlink() for p in (target, *target.parents)),
                "worktree_dirty",
            )
            actual = target.read_bytes() if target.is_file() else None
            require(
                actual == raw or name not in staged and actual == before.get(name),
                "worktree_dirty",
            )
            if target.exists():
                require(
                    target.is_file() and not target.stat().st_mode & 0o111,
                    "worktree_dirty",
                )
            if name in staged:
                require(self.git("show", ":" + name, cwd=tree) == raw, "worktree_dirty")

    def commit(self, batch: dict[str, Any], revision: dict[str, Any]) -> None:
        require_owner(self.owner_check)
        files = self.plan(batch, revision["base"])
        require(bool(files), "unsafe_diff")
        require_owner(self.owner_check)
        tree = self.worktree(batch)
        require(not tree.is_symlink(), "worktree_symlink")
        if not tree.exists():
            require_owner(self.owner_check)
            tree.parent.mkdir(parents=True, exist_ok=True)
            require_owner(self.owner_check)
            self.git("worktree", "add", "--detach", str(tree), revision["base"])
        require(
            self.git(
                "rev-parse", "--path-format=absolute", "--git-common-dir", cwd=tree
            )
            == self.git("rev-parse", "--path-format=absolute", "--git-common-dir"),
            "worktree_changed",
        )
        head = self.git("rev-parse", "HEAD", cwd=tree).decode().strip()
        dirty = self.git(
            "status", "--porcelain", "--untracked-files=all", "--ignored", cwd=tree
        ).strip()
        if head == revision["base"]:
            self.prepared_worktree(tree, files, revision["base"])
        else:
            require(not dirty, "worktree_dirty")
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
                require_owner(self.owner_check)
                target.parent.mkdir(parents=True, exist_ok=True)
                require_owner(self.owner_check)
                target.write_bytes(raw)
            # Only the fully validated public file set is force-added; old main's
            # ignore rules may predate this new publication format.
            require_owner(self.owner_check)
            self.git("add", "--force", "--", *sorted(files), cwd=tree)
            require_owner(self.owner_check)
            self.git(
                "commit",
                "-m",
                f"Publish parent {batch['parent']} decision records",
                cwd=tree,
            )
            head = self.git("rev-parse", "HEAD", cwd=tree).decode().strip()
        self.verify(batch, revision, head)
        require_owner(self.owner_check)
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
            require_owner(self.owner_check)
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
                require_owner(self.owner_check)
                batch.update(pr_create_attempted=True, stage="pr_create")
                self.save()
                require_owner(self.owner_check)
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

    def reconcile_pr(self, batch: dict[str, Any]) -> dict[str, Any]:
        pr = self.github.get(batch["pr_number"])
        observed = (pr.get("head") or {}).get("sha")
        known = next(
            (r for r in batch["revisions"] if r["head"] and r["head"] == observed),
            None,
        )
        if known is None:
            raise PublicationError("pr_identity_changed")
        policy.identity(pr, batch, observed)
        if pr.get("merged") is True:
            # Confirmation still requires the latest revision. Never rewrite a
            # closed PR or forget a revision that was not applied to it.
            return dict(pr)
        require(pr.get("state") == "open", "closed_unmerged")
        latest = batch["revisions"][-1]
        if known is not latest:
            require(not batch["merge_attempted"], "merge_response_uncertain")
            self.verify(batch, known)
            # Recover before planning another main revision: a commit may exist
            # locally even though its journal HEAD and remote push were lost.
            if latest["head"] is None:
                self.commit(batch, latest)
            diff = self.verify(batch, latest)
            self.push(batch)
            return self.pull_request(batch, diff)
        policy.identity(pr, batch, latest["head"])
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
                state["publication_status"] == status
                and state["error"] == error
                and state.get("publication_actor") == self.publication_actor
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
                publication_actor=self.publication_actor,
                owner_check=self.owner_check,
            )

    def confirm(self, batch: dict[str, Any], pr: dict[str, Any]) -> dict[str, Any]:
        self.publication_actor = self.batch_actor(batch)
        revision = batch["revisions"][-1]
        status = {r["id"]: r for r in self.journal.status()}
        # A completed publication already passed the full policy before its
        # merge fact and every per-record publication fact were persisted.
        # Its old README bytes may belong to an earlier renderer. Reconcile
        # that historical fact without pretending it is a new exception: the
        # exact PR/HEAD/merge, Git structure, ancestry and current envelopes are
        # still checked below. Every new or unconfirmed revision uses verify().
        # A repair resets merge_commit to None and gets the full current check.
        confirmed = (
            batch["stage"] in {"merged", "readme_refresh_required"}
            and batch["merge_commit"] is not None
            and batch["merge_commit"] == pr.get("merge_commit_sha")
            and all(
                status[p["record"]["id"]]["publication_status"] == "published"
                for p in batch["payloads"]
            )
        )
        diff = self.commit_diff(revision) if confirmed else self.verify(batch, revision)
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
        # The current main may include worker files that arrived after the
        # decision batch merged. Their derived navigation must be current too;
        # the old merge tree being canonical is not proof about today's view.
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
            and not extra.get("error")
            and all(r["publication_status"] == "published" for r in status),
            "unpublished": sum(r["publication_status"] != "published" for r in status),
            "blocked": sum(r["publication_status"] == "blocked" for r in status),
            "error": batch["error"] if batch else None,
            **(
                {"merge_http_status": batch["merge_http_status"]}
                if batch and "merge_http_status" in batch
                else {}
            ),
            "pr": f"https://github.com/{batch['repo']}/pull/{batch['pr_number']}"
            if batch and batch["pr_number"]
            else None,
            "head": batch["revisions"][-1]["head"]
            if batch and batch["revisions"]
            else None,
            **extra,
            **(
                {"publication_actor": self.publication_actor}
                if self.publication_actor
                else {}
            ),
        }
        Guard().check(result)
        return result

    def owner_mismatch(self) -> dict[str, Any]:
        """Report durable progress, excluding any mutation refused by save()."""
        self.document = storage.read(self.path, self.journal.repo, self.journal.parent)
        active = next(
            (b for b in self.document["batches"] if b["state"] != "merged"), None
        )
        return self.view(active, error="log_owner_mismatch", saved_to_main=False)

    def _publish(self, *, retry: bool) -> dict[str, Any]:
        active = next(
            (b for b in self.document["batches"] if b["state"] != "merged"), None
        )
        retry_merge = bool(
            active
            and active.get("stage") == "merge_rejected"
            and active.get("merge_http_status") == 405
            and active["error"] == "merge_conflict"
            and active["merge_attempted"] is False
        )
        if active and active["state"] == "blocked" and not retry and not retry_merge:
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
        require_owner(self.owner_check)
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
        self.publication_actor = self.batch_actor(active) if active else None
        if active and active["pr_number"] is not None:
            # A lost response can leave a merged PR in a pending batch. Check
            # that fixed publication before recording any fresh request too.
            observed = self.github.get(active["pr_number"])
            if observed.get("merged") is True:
                reconciled = self.confirm(active, self.reconcile_pr(active))
                if (
                    reconciled["state"] == "merged"
                    or active["stage"] != "readme_refresh_required"
                ):
                    return reconciled
        # Reconcile completed work before deriving a new request from pending
        # records. Its actor may already be published while dependents are not.
        for state in self.journal.status():
            if self.record_ids is not None and state["id"] not in self.record_ids:
                continue
            if (
                state["publication_status"] == "unprepared"
                or retry
                and state["publication_status"] == "blocked"
            ):
                require_owner(self.owner_check)
                self.journal.prepare(
                    state["id"],
                    key="publication-prepare:" + new_id(),
                    at=now(),
                    retry=retry,
                    owner_check=self.owner_check,
                )
        if self.actor is not None and (
            self.publication_actor is None
            or self.request_actor(self.publication_actor) != self.actor
        ):
            self.publication_actor = self.journal.publication_request(
                self.actor,
                at=now(),
                boundary=active["key"] if active else None,
                owner_check=self.owner_check,
            )
        if self.publication_actor is not None:
            request_id = self.publication_actor["request_id"]
            if self.record_ids is not None:
                self.record_ids.add(request_id)
            if (
                next(s for s in self.journal.status() if s["id"] == request_id)[
                    "publication_status"
                ]
                == "unprepared"
            ):
                self.journal.prepare(
                    request_id,
                    key="publication-prepare:" + new_id(),
                    at=now(),
                    owner_check=self.owner_check,
                )
        payloads = self.journal.public_payloads()
        if active is not None and self.publication_actor is not None:
            published = {
                r["id"]
                for r in self.journal.status()
                if r["publication_status"] == "published"
            }
            if any(p["record"]["id"] in published for p in active["payloads"]):
                # A README repair preserves the original record publication.
                # A successor's own request waits for the following batch.
                self.publication_actor = self.batch_actor(active)
            elif active.get("publication_actor") != self.publication_actor:
                active["publication_actor"] = dict(self.publication_actor)
                self.save()
        status = {r["id"]: r for r in self.journal.status()}
        if active is None:
            pending = [
                p
                for p in payloads
                if status[p["record"]["id"]]["publication_status"] != "published"
                and (self.record_ids is None or p["record"]["id"] in self.record_ids)
            ]
            if not pending:
                last = (
                    self.document["batches"][-1] if self.document["batches"] else None
                )
                if last:
                    return self.confirm(last, self.github.get(last["pr_number"]))
                return self.view()
            key = new_id()
            require_owner(self.owner_check)
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
                "publication_actor": self.publication_actor,
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
        require_owner(self.owner_check)
        self.note(active, "pending")
        for _ in range(3):
            if active["pr_number"] is not None:
                pr = self.reconcile_pr(active)
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
            require(pr.get("mergeable") is not False, "merge_conflict")
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
            require(fresh.get("mergeable") is not False, "merge_conflict")
            require_owner(self.owner_check)
            active.update(merge_attempted=True, stage="merge_attempted", error=None)
            active.pop("merge_http_status", None)
            self.save()
            require_owner(self.owner_check)
            self.github.merge(active["pr_number"], revision["head"])
            active["merge_http_status"] = 200
            return self.confirm(active, self.github.get(active["pr_number"]))
        active.update(stage="main_advanced", state="pending", error=None)
        self.save()
        return self.view(active)

    def publish(
        self,
        *,
        dry_run: bool = False,
        retry: bool = False,
        owner_check: Callable[[], bool] | None = None,
        actor: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        self.owner_check, self.actor = owner_check, actor
        self.publication_actor = None
        require_owner(owner_check)
        # The read precedes lock creation to reject a symlinked private hierarchy.
        document = storage.read(self.path, self.journal.repo, self.journal.parent)
        if dry_run:
            preview = self.journal.preview_publication(retry=retry)
            payloads = preview["payloads"]
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
            generated = generate(files)
            # generate() checks escaped Markdown after removing its fixed HTML
            # scaffolding. Do not feed those closing tags to the path guard.
            return {
                **self.view(
                    batch,
                    dry_run=True,
                    generated_readmes=len(generated),
                    prepared_records=preview["prepared_records"],
                    preview_records=len(payloads),
                    preview_blocked=preview["blocked"],
                    blocked=len(preview["blocked"]),
                ),
                "preview_readmes": {
                    name: raw.decode("utf-8") for name, raw in generated.items()
                },
            }
        with storage.publication_lock(self.path):
            # Reload under the lock before any fallible preparation. An old
            # Publisher instance must never persist its cached batch history
            # over another publisher's completed work in an error handler.
            self.document = storage.read(
                self.path, self.journal.repo, self.journal.parent
            )
            try:
                require_owner(owner_check)
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
                    else "git_failed"
                    if isinstance(exc, subprocess.SubprocessError)
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
                if code == "log_owner_mismatch":
                    # Ownership loss is not corrupt input. Preserve the last
                    # durable batch/journal for the next authorized publisher.
                    return self.owner_mismatch()
                try:
                    require_owner(owner_check)
                except PublicationError:
                    return self.owner_mismatch()
                if active:
                    if active.get("merge_attempted") and isinstance(
                        exc, subprocess.SubprocessError
                    ):
                        code = "merge_response_uncertain"
                    if isinstance(exc, GitHubHTTPError):
                        active["merge_http_status"] = exc.status
                    if isinstance(exc, MergeRejected):
                        active.update(
                            merge_attempted=False,
                            stage="merge_rejected",
                            merge_http_status=exc.status,
                        )
                    elif active.get("merge_attempted") and code in {
                        "github_failed",
                        "github_invalid_response",
                    }:
                        code = "merge_response_uncertain"
                    transient = not isinstance(exc, MergeRejected) and code in {
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
                    try:
                        self.save()
                        self.note(
                            active,
                            "pending" if transient else "blocked",
                            error=None if transient else code,
                        )
                    except PublicationError as bookkeeping_error:
                        if str(bookkeeping_error) == "log_owner_mismatch":
                            return self.owner_mismatch()
                        raise
                return self.view(active, error=code)
