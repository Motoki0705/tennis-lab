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
def targets(directory: Path, state: dict[str, Any]) -> Iterator[None]:
    leases = directory / "progression-targets"
    leases.mkdir(mode=0o700, parents=True, exist_ok=True)
    with ExitStack() as stack:
        for target in sorted(state["targets"].values(), key=lambda t: t["number"]):
            name = digest([state["request"]["repo"].casefold(), target["number"]])
            lock = stack.enter_context((leases / (name + ".lock")).open("a"))
            os.chmod(lock.name, 0o600)
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise ValueError("PR already has an active progression") from exc
            receipt = leases / (name + ".json")
            if receipt.exists():
                previous = read(receipt)["progression"]
                if previous != state["id"]:
                    prior = read(directory / "progressions" / previous / "state.json")
                    if prior["targets"][str(target["number"])]["stage"] != "completed":
                        raise ValueError(
                            "PR has an unfinished progression; continue that request"
                        )
            save(receipt, {"progression": state["id"]})
        yield
