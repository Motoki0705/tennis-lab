"""Private publication batches. Capture journal facts remain separate records."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from worker_logs.journal import PUBLIC_ERRORS, PublicationError, decode, locked, require
from worker_logs.schema import digest, fields, integer

from .projection import validate_public
from .schema import public_id


def branch(parent: int, key: str, repairs: int = 0) -> str:
    suffix = f"-repair-{repairs}" if repairs else ""
    return f"decision-log/parent-{parent}/{key}{suffix}"


@contextmanager
def publication_lock(path: Path) -> Iterator[None]:
    """Do not expose a private storage path when lock or persistence fails."""
    try:
        with locked(path):
            yield
    except OSError:
        raise PublicationError("journal_unavailable") from None


def read(path: Path, repo: str, parent: int) -> dict[str, Any]:
    require(
        not any(p.is_symlink() for p in (path, *path.parents)),
        "private_storage_symlink",
    )
    try:
        raw = path.read_bytes()
    except FileNotFoundError:
        return {"version": 1, "repo": repo, "parent": parent, "batches": []}
    except OSError:
        raise PublicationError("journal_unavailable") from None
    row = decode(raw)
    fields(row, {"version", "repo", "parent", "batches"})
    require(
        type(row["version"]) is int
        and row["version"] == 1
        and row["repo"] == repo
        and type(row["parent"]) is int
        and row["parent"] == parent,
        "journal_identity_changed",
    )
    require(isinstance(row["batches"], list), "journal_invalid")
    keys: set[str] = set()
    active = 0
    for batch in row["batches"]:
        fields(
            batch,
            {
                "repo",
                "parent",
                "key",
                "branch",
                "payloads",
                "revisions",
                "pr_number",
                "pr_create_attempted",
                "merge_attempted",
                "state",
                "stage",
                "error",
                "diagnostic",
                "merge_commit",
                "superseded_prs",
            }
            | ({"merge_http_status"} if "merge_http_status" in batch else set()),
        )
        if "merge_http_status" in batch:
            require(
                type(batch["merge_http_status"]) is int
                and 100 <= batch["merge_http_status"] <= 599,
                "journal_invalid",
            )
        public_id(batch["key"])
        require(
            batch["key"] not in keys
            and isinstance(batch["superseded_prs"], list)
            and batch["branch"]
            == branch(parent, batch["key"], len(batch["superseded_prs"]))
            and batch["repo"] == repo
            and type(batch["parent"]) is int
            and batch["parent"] == parent,
            "publication_identity_changed",
        )
        keys.add(batch["key"])
        for index, previous in enumerate(batch["superseded_prs"]):
            fields(previous, {"number", "branch", "head", "merge_commit"})
            require(
                integer(previous["number"])
                and previous["branch"] == branch(parent, batch["key"], index)
                and digest(previous["head"], 40)
                and digest(previous["merge_commit"], 40),
                "journal_pr",
            )
        require(
            isinstance(batch["payloads"], list) and bool(batch["payloads"]),
            "journal_invalid",
        )
        payloads = [validate_public(p) for p in batch["payloads"]]
        # Dependencies may already have been published by an earlier batch.
        require(
            len({p["record"]["id"] for p in payloads}) == len(payloads),
            "journal_invalid",
        )
        require(
            all(
                p["record"]["scope"]["repo"] == repo
                and p["record"]["scope"]["parent"] == parent
                for p in payloads
            ),
            "journal_identity_changed",
        )
        require(batch["state"] in {"pending", "blocked", "merged"}, "journal_state")
        active += batch["state"] != "merged"
        require(
            batch["error"] is None or batch["error"] in PUBLIC_ERRORS, "journal_invalid"
        )
        require(
            isinstance(batch["stage"], str)
            and (batch["diagnostic"] is None or isinstance(batch["diagnostic"], str)),
            "journal_invalid",
        )
        require(
            all(
                type(batch[k]) is bool
                for k in ("pr_create_attempted", "merge_attempted")
            ),
            "journal_invalid",
        )
        require(batch["pr_number"] is None or integer(batch["pr_number"]), "journal_pr")
        require(
            batch["merge_commit"] is None or digest(batch["merge_commit"], 40),
            "journal_invalid",
        )
        require(isinstance(batch["revisions"], list), "journal_invalid")
        for revision in batch["revisions"]:
            fields(revision, {"base", "head"})
            require(
                digest(revision["base"], 40)
                and (revision["head"] is None or digest(revision["head"], 40)),
                "journal_invalid",
            )
        if batch["state"] == "merged":
            require(
                batch["pr_number"] is not None
                and batch["merge_attempted"]
                and batch["merge_commit"] is not None
                and bool(batch["revisions"])
                and batch["revisions"][-1]["head"] is not None,
                "journal_state",
            )
    require(active <= 1, "journal_state")
    return dict(row)
