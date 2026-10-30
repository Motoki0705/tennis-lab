"""Compose runner observations with explicit, process-local GitHub refreshes."""

from __future__ import annotations

from copy import deepcopy
from threading import Lock
from typing import Any

from worker_status import collect_status

from .web_data import BoardReader


class ExecutionReader:
    def __init__(self, board: BoardReader) -> None:
        self.board = board
        self.remote: dict[str, Any] | None = None
        self.lock = Lock()

    def _cached_github(self, observation: dict[str, Any]) -> None:
        if self.remote is None:
            return
        observation["github"] = deepcopy(self.remote["github"])
        previous = {
            (row["repo"], row["parent"], row["generation"]): row
            for row in self.remote["parents"]
        }
        for parent in observation["parents"]:
            cached = previous.get(
                (parent["repo"], parent["parent"], parent["generation"])
            )
            if cached is None:
                continue
            children = {row["child"]: row for row in cached["children"]}
            for child in parent["children"]:
                old = children.pop(child["child"], None)
                # Do not attach an old stop/launch observation to a redelegation.
                if old is not None and old["delegation"] == child["delegation"]:
                    child["github"] = deepcopy(old["github"])
            # Formal GitHub children may have no local task yet. Preserve their
            # last explicit observation, including its original local timestamp.
            for old in children.values():
                if old["run"] is not None or old["delegation"] is not None:
                    continue
                child = deepcopy(old)
                child["cached_local_observation"] = True
                parent["children"].append(child)
            parent["children"].sort(key=lambda row: row["child"])

    def read(self, *, refresh: bool = False) -> dict[str, Any]:
        with self.lock:
            observation = collect_status(
                state_dir=self.board.store.directory.parent / "parent-sessions",
                github=refresh,
            )
            if refresh:
                for parent in observation["parents"]:
                    for child in parent["children"]:
                        child["github"]["fetched_at"] = observation["github"][
                            "fetched_at"
                        ]
            else:
                self._cached_github(observation)
            # This also validates the complete response before any cache is
            # replaced. Failure must retain the last successful GitHub snapshot.
            observation = self.board.execution_roster(observation, refresh=refresh)
            if refresh:
                self.remote = deepcopy(observation)
            return observation
