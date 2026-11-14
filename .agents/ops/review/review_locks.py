"""Cross-run PR leases. A dead owner never makes an unfinished execution safe."""

from __future__ import annotations

import fcntl
from collections.abc import Iterator, Mapping, Sequence
from contextlib import ExitStack, contextmanager
from pathlib import Path
from typing import Any

from parent.policy import digest, object_value
from parent.session import ParentSession
from parent_review_policy import execution_stages, load_json, request_manifest
from review_inputs import fixed_files
from review_interruption import CLOSED, confirmed_exit


@contextmanager
def target_locks(directory: Path, request: Mapping[str, Any]) -> Iterator[None]:
    """Acquire PRs in numeric order before run locks and the registry transaction.

    Hold through cleanup/postcheck/publication. Never unlink flock files: a new
    inode would allow a second owner. Unrelated PRs have independent leases.
    """
    locks = directory / "review-targets"
    locks.mkdir(mode=0o700, parents=True, exist_ok=True)
    repo = digest(request["repo"].casefold())
    with ExitStack() as stack:
        for n in sorted(p["number"] for p in request["prs"]):
            lock = stack.enter_context((locks / f"{repo}-pr-{n}.lock").open("a"))
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise ValueError(
                    f"PR #{n} review is already active in another run"
                ) from exc
        yield


def require_finished_overlaps(
    directory: Path,
    reviews: Sequence[Mapping[str, Any]],
    request: Mapping[str, Any],
) -> None:
    """Read frozen registrations outside registry, holding all target leases.

    SIGKILL releases flock, but may leave a reviewer alive. A durable running
    execution remains a blocker until recovery or explicit closure confirms exit.
    Do not infer liveness from wall time or a reusable PID.
    """
    numbers = {p["number"] for p in request["prs"]}
    for row in reviews:
        other = request_manifest(row["request"], instructed=False)
        if other["review_run_id"] == request["review_run_id"]:
            continue
        if other["repo"].casefold() != request[
            "repo"
        ].casefold() or not numbers.intersection(p["number"] for p in other["prs"]):
            continue
        path = directory / "reviews" / other["review_run_id"] / "execution.json"
        with fixed_files([path]):
            if not path.exists():
                continue
            execution = object_value(load_json(path.read_text(encoding="utf-8")))
            execution_stages(execution)
            if (
                execution.get("status")
                not in {
                    "running",
                    "published",
                    "publication_failed",
                    "preparation_failed",
                    CLOSED,
                }
                or type(execution.get("agent_invocations")) is not int
                or execution["agent_invocations"] not in {0, 1}
            ):
                raise ValueError("unknown overlapping review execution state")
            if execution.get("request_sha256") != digest(other):
                raise ValueError("overlapping review execution/request mismatch")
            if not confirmed_exit(execution, other):
                raise ValueError(
                    "PR review is active, interrupted or cleanup is unconfirmed in another run; "
                    "confirm its exit and resume-publication or close-interrupted first"
                )


def finished_overlaps(session: ParentSession, request: Mapping[str, Any]) -> None:
    """Fence registrations from every parent that can overlap these target PRs."""

    def registrations() -> list[Mapping[str, Any]]:
        numbers = {p["number"] for p in request["prs"]}
        return [
            row
            for parent in session.store.parents.values()
            for row in parent["reviews"]
            if row["request"]["repo"].casefold() == request["repo"].casefold()
            and numbers.intersection(p["number"] for p in row["request"]["prs"])
        ]

    before = digest(registrations())
    with session.gh.io_scope():
        require_finished_overlaps(session.store.directory, registrations(), request)
    if digest(registrations()) != before:
        session.store.conflict(
            "overlapping review registrations changed during verification"
        )


@contextmanager
def run_lock(directory: Path) -> Iterator[None]:
    with (directory / "run.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError(
                "review run is active; retry after its completion"
            ) from exc
        yield
