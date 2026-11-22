"""Freeze a delegation's ready turns before publishing their exact union."""

from __future__ import annotations

from collections.abc import Callable
from contextlib import nullcontext
from pathlib import Path
from typing import Any

from . import journal, schema
from .journal import require
from .ownership import require_owner
from .publication import Publisher
from .reconciliation import worker_row
from .sources import guard_for, render

MAX_TURNS = 32  # Three additions per turn fit in one GitHub files page.
PUBLICATION_FIELDS = (
    "state",
    "stage",
    "error",
    "pr_number",
    "branch",
    "base",
    "head",
    "merge_commit",
    "main_observed",
    "merge_http_status",
    "merge_attempted",
    "pr_create_attempted",
)


def payload(manifest: Path, batch: dict[str, Any]) -> tuple[dict[str, bytes], set[str]]:
    source = batch["source"]
    members = batch["members"]
    require(
        isinstance(members, list)
        and 0 < len(members) <= MAX_TURNS
        and all(schema.integer(n) for n in members)
        and members == sorted(set(members))
        and members[0] == source["sequence"]
        and manifest.name == f"{members[0]}.json",
        "publication_identity_changed",
    )
    files: dict[str, bytes] = {}
    directories: set[str] = set()
    for n in members:
        path = manifest.parent.parent / str(n) / "journal.json"
        turn = worker_row(path, source["repo"])
        src = turn["source"]
        require(
            all(
                src[k] == source[k]
                for k in ("repo", "parent", "issue", "public_key", "repository_root")
            )
            and turn.get("publication_mode") == "delegation_batch",
            "journal_identity_changed",
        )
        original = render(src, turn["receipt"])
        require(
            original == {k: v.encode("utf-8") for k, v in turn["files"].items()},
            "prepared_content_changed",
        )
        meta = schema.validate(original, guard_for(src, turn["receipt"]["record"]))
        directory = schema.path(meta)
        require(
            meta["turn"]["sequence"] == n
            and meta["delegation_key"] == "d-" + src["public_key"]
            and meta["issue"] == src["issue"]
            and meta["parent"] == src["parent"]
            and meta["repo"].casefold() == src["repo"].casefold(),
            "journal_identity_changed",
        )
        directories.add(directory)
        files.update(
            {Path(directory).name + "/" + name: raw for name, raw in original.items()}
        )
    return files, directories


def publish(
    path: Path,
    *,
    dry_run: bool,
    retry: bool,
    owner_check: Callable[[], bool] | None,
    publisher: Callable[[Path, Path], Publisher],
    only_turn: bool = False,
) -> dict[str, Any]:
    parent = path.parent.parent
    # A manifest binds every member before the first Git/GitHub mutation. A
    # crash before copying the PR receipt to the last turn is therefore replayable.
    with nullcontext() if dry_run else journal.locked(parent / "publication.json"):
        row = journal.read(path)
        src = row["source"]
        row = worker_row(path, src["repo"])
        seq = src["sequence"]
        manifests = parent / "batches"
        require(not manifests.is_symlink(), "private_storage_symlink")
        if row.get("individual_publication") or (
            "base" in row and "publication_batch" not in row
        ):
            return publisher(Path(src["repository_root"]), path).publish(
                dry_run=dry_run, retry=retry
            )
        if row["state"] == "merged":
            return journal.view(row)
        manifest = None
        claimed: set[int] = set()
        for entry in manifests.glob("*.json"):
            frozen = journal.read(entry)
            members = frozen["members"]
            require(not claimed.intersection(members), "publication_identity_changed")
            claimed.update(members)
            if seq in members:
                manifest = entry
        if manifest is None:
            require("publication_batch" not in row, "journal_unavailable")
            if (
                retry
                and row["state"] == "blocked"
                and "receipt" in row
                and "files" not in row
            ):
                recovered = render(src, row["receipt"])
                if not dry_run:
                    with journal.locked(path):
                        row.update(
                            files={k: v.decode("utf-8") for k, v in recovered.items()},
                            state="ready",
                            stage="receipt_verified",
                            error=None,
                        )
                        journal.atomic(path, row)
            require(
                row["state"] == "ready" or retry and "files" in row,
                row.get("error") or "final_not_ready",
            )
            members = [seq]
            if not only_turn:
                for candidate in sorted(parent.glob("*/journal.json")):
                    if len(members) == MAX_TURNS:
                        break
                    if candidate == path:
                        continue
                    item = worker_row(candidate, src["repo"])
                    n = item["source"]["sequence"]
                    if (
                        item.get("publication_mode") == "delegation_batch"
                        and item["state"] == "ready"
                        and "base" not in item
                        and not item.get("individual_publication")
                        and n not in claimed
                    ):
                        members.append(n)
            if len(members) == 1:
                # Preserve single-turn PR identity, including orphan recovery.
                # Pin that choice before any mutation, even if more turns arrive later.
                if not dry_run:
                    require_owner(owner_check)
                    with journal.locked(path):
                        row = worker_row(path, src["repo"])
                        row["individual_publication"] = True
                        journal.atomic(path, row)
                return publisher(Path(src["repository_root"]), path).publish(
                    dry_run=dry_run, retry=retry
                )
            members.sort()
            first = worker_row(parent / str(members[0]) / "journal.json", src["repo"])
            manifest = manifests / f"{members[0]}.json"
            frozen = {
                **first,
                "state": "ready",
                "stage": "receipt_verified",
                "members": members,
            }
            payload(manifest, frozen)
            if dry_run:
                return {
                    **journal.view(row),
                    "dry_run": True,
                    "turns": members,
                    "directory": schema.path(journal.decode(row["files"]["meta.json"])),
                }
            require_owner(owner_check)
            journal.atomic(manifest, frozen)
        pub = publisher(Path(src["repository_root"]), manifest)
        if not dry_run:
            require_owner(owner_check)
        result = pub.publish(dry_run=dry_run, retry=retry)
        frozen = journal.read(manifest)
        if dry_run:
            return {
                **journal.view(row),
                "dry_run": True,
                "turns": frozen["members"],
                "error": result.get("error"),
            }
        for n in frozen["members"]:
            target = parent / str(n) / "journal.json"
            with journal.locked(target):
                member = worker_row(target, src["repo"])
                was_merged = member["state"] == "merged"
                member.update({k: frozen[k] for k in PUBLICATION_FIELDS if k in frozen})
                member["publication_batch"] = frozen["members"][0]
                journal.atomic(target, member)
            if n != seq and member["state"] == "merged" and not was_merged:
                from decision_logs.observations import worker_log

                worker_log(target, journal.view(member), retry=retry)
        return journal.view(journal.read(path))
