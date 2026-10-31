"""Compose runner observations with explicit, process-local GitHub/Herdr reads."""

from __future__ import annotations

from copy import deepcopy
from threading import Lock
from typing import Any

from worker_status import collect_status
from worker_status.task_view import prefer_local_title

from . import runtime
from .web_data import BoardReader


class ExecutionReader:
    """Local runner records on every read; GitHub and Herdr only when requested.

    GitHub is read by the explicit refresh only. Herdr is read by that refresh
    and by `herdr=True` (the page's first display of the execution tab). Both
    keep their last successful observation in this process; failures never
    replace it with an empty result.
    """

    def __init__(self, board: BoardReader) -> None:
        self.board = board
        self.remote: dict[str, Any] | None = None
        self.herdr: dict[str, Any] | None = None
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
                    prefer_local_title(child)
            # Formal GitHub children may have no local task yet. Preserve their
            # last explicit observation, including its original local timestamp.
            for old in children.values():
                if old["run"] is not None or old["delegation"] is not None:
                    continue
                child = deepcopy(old)
                child["cached_local_observation"] = True
                parent["children"].append(child)
            parent["children"].sort(key=lambda row: row["child"])

    @staticmethod
    def _attach_herdr(
        observation: dict[str, Any], herdr: dict[str, Any] | None
    ) -> None:
        """Join cached Herdr agents to parents of the same assignment generation."""
        observation["herdr"] = {
            "status": "observed" if herdr is not None else "not_requested",
            "observed_at": herdr["observed_at"] if herdr is not None else None,
        }
        used: set[int] = set()
        for parent in observation["parents"]:
            joined: dict[str, Any] = {
                "status": "not_requested",
                "method": None,
                "session": None,
            }
            if herdr is not None:
                match = herdr["parents"].get(f"{parent['repo']}#{parent['parent']}")
                if (
                    match is None
                    or match["generation"] != parent["generation"]
                    or match["assignment"] != parent["assignment"]
                ):
                    # Registered or handed off after the observation: never reuse
                    # another generation's terminal. The next Herdr read decides.
                    joined["status"] = "not_observed"
                elif match["status"] == "matched":
                    used.add(match["index"])
                    joined.update(
                        status="matched",
                        method=match["method"],
                        session=deepcopy(herdr["sessions"][match["index"]]),
                    )
                else:
                    joined["status"] = match["status"]
            parent["herdr"] = joined
        observation["sessions"] = (
            [
                deepcopy(row)
                for index, row in enumerate(herdr["sessions"])
                if index not in used
            ]
            if herdr is not None
            else []
        )

    def _refresh_state(self) -> dict[str, Any]:
        """Caller holds the cache lock; never perform source reads here."""
        return {
            "epoch": self.board.epoch,
            "status": "running" if self.refreshing else "idle",
            "revision": self.refresh_revision,
            "result": self.refresh_result,
        }

    def _publish_herdr(self, herdr: dict[str, Any]) -> None:
        """Caller holds the cache lock. A slower, older read never wins."""
        if self.herdr is None or herdr["observed_at"] >= self.herdr["observed_at"]:
            self.herdr = herdr

    def _local(self) -> dict[str, Any]:
        observation = collect_status(
            state_dir=self.board.store.directory.parent / "parent-sessions",
            github=False,
        )
        with self.lock:
            # Published snapshots are immutable. Copy/merge outside the lock,
            # alongside the fresh local observation, without blocking refresh.
            remote, herdr = self.remote, self.herdr
            state = self._refresh_state()
        self._cached_github(observation, remote)
        self._attach_herdr(observation, herdr)
        observation["github_refresh"] = state
        return self.board.execution_names(observation)

    def read(self, *, refresh: bool = False, herdr: bool = False) -> dict[str, Any]:
        if herdr and not refresh:
            # Herdr reads are short and idempotent: no refresh claim is needed.
            # A failure is an HTTP error; the cached observation is retained.
            observed = runtime.sessions(self.board.env)
            with self.lock:
                self._publish_herdr(observed)
            return self._local()
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
            observed = runtime.sessions(self.board.env)
            self._attach_herdr(observation, observed)
            observation = self.board.execution_names(observation)
            published = deepcopy(observation)
        except Exception:
            with self.lock:
                self.refreshing = False
                self.refresh_result = "failed"
                self.refresh_revision += 1
            raise
        with self.lock:
            self.remote = published
            self._publish_herdr(observed)
            self.refreshing = False
            self.refresh_result = "succeeded"
            self.refresh_revision += 1
            observation["github_refresh"] = self._refresh_state()
        return observation
