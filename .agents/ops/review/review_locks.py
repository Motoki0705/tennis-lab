"""PR leases for automatic progressions; registry locks are never held here."""

from __future__ import annotations

import fcntl
import os
from collections.abc import Iterator
from contextlib import ExitStack, contextmanager
from pathlib import Path
from typing import Any

from lib.ops_review.progression import read, save
from parent.policy import digest


@contextmanager
def pr_lease(directory: Path, repo: str, number: int) -> Iterator[Path]:
    leases = directory / "progression-targets"
    leases.mkdir(mode=0o700, parents=True, exist_ok=True)
    name = digest([repo.casefold(), number])
    with (leases / (name + ".lock")).open("a") as lock:
        os.chmod(lock.name, 0o600)
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError("PR already has an active progression") from exc
        yield leases / (name + ".json")


@contextmanager
def targets(directory: Path, state: dict[str, Any]) -> Iterator[None]:
    receipts = []
    with ExitStack() as stack:
        for target in sorted(state["targets"].values(), key=lambda t: t["number"]):
            receipt = stack.enter_context(
                pr_lease(directory, state["request"]["repo"], target["number"])
            )
            if receipt.exists():
                previous = read(receipt)["progression"]
                if previous != state["id"]:
                    prior = read(directory / "progressions" / previous / "state.json")
                    if (
                        state["request"].get("kind") != "post_merge_confirmation"
                        or prior["targets"][str(target["number"])]["stage"]
                        != "completed"
                    ):
                        raise ValueError(
                            "PR has an unfinished progression; continue that request"
                        )
            receipts.append(receipt)
        for receipt in receipts:
            save(receipt, {"progression": state["id"]})
        yield
