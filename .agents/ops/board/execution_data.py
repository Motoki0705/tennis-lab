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
        self.refreshing = False
        self.refresh_revision = 0
        self.refresh_result: str | None = None
        self.lock = Lock()

    def _cached_github(
        self, observation: dict[str, Any], remote: dict[str, Any] | None
    ) -> None:
        if remote is None:
            return
        observation["github"] = deepcopy(remote["github"])
        previous = {
            (row["repo"], row["parent"], row["generation"]): row
            for row in remote["parents"]
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

    def _refresh_state(self) -> dict[str, Any]:
        """Caller holds the cache lock; never perform source reads here."""
        return {
            "epoch": self.board.epoch,
            "status": "running" if self.refreshing else "idle",
            "revision": self.refresh_revision,
            "result": self.refresh_result,
        }

    def _local(self) -> dict[str, Any]:
        observation = collect_status(
            state_dir=self.board.store.directory.parent / "parent-sessions",
            github=False,
        )
        with self.lock:
            # Published snapshots are immutable. Copy/merge outside the lock,
            # alongside the fresh local observation, without blocking refresh.
            remote = self.remote
            state = self._refresh_state()
        self._cached_github(observation, remote)
        observation["github_refresh"] = state
        return self.board.execution_roster(observation)

    def read(self, *, refresh: bool = False) -> dict[str, Any]:
        claimed = False
        with self.lock:
            if refresh and not self.refreshing:
                self.refreshing = claimed = True
                self.refresh_revision += 1
        if not claimed:
            # A duplicate explicit refresh receives a local observation with
            # status=running; the HTTP route returns 202 instead of queuing it.
            return self._local()
        try:
            observation = collect_status(
                state_dir=self.board.store.directory.parent / "parent-sessions",
                github=True,
            )
            for parent in observation["parents"]:
                for child in parent["children"]:
                    child["github"]["fetched_at"] = observation["github"]["fetched_at"]
            observation = self.board.execution_roster(observation, refresh=True)
            published = deepcopy(observation)
        except Exception:
            with self.lock:
                self.refreshing = False
                self.refresh_result = "failed"
                self.refresh_revision += 1
            raise
        with self.lock:
            self.remote = published
            self.refreshing = False
            self.refresh_result = "succeeded"
            self.refresh_revision += 1
            observation["github_refresh"] = self._refresh_state()
        return observation
