"""Decision publication fixtures sharing the existing real-git worker-log kit."""

from __future__ import annotations

import copy
import importlib
from pathlib import Path
from typing import Any, cast

from tests.support.agents_ops.decision_logs import AT, history, journal
from tests.support.agents_ops.worker_logs import FakeGitHub, LogEnv, git

publication: Any = importlib.import_module("decision_logs.publication")
navigation: Any = importlib.import_module("decision_logs.navigation")
policy: Any = importlib.import_module("decision_logs.publication_policy")


class DecisionGitHub(FakeGitHub):
    """Reflect pushed branch updates and the complete batch's changed-file count."""

    def refresh(self, number: int) -> None:
        pr = self.rows[number]
        if pr["state"] == "open":
            pr["head"]["sha"] = git(self.upstream, "rev-parse", pr["head"]["ref"])
            pr["changed_files"] = len(self.files(number))

    def get(self, number: int) -> dict[str, Any]:
        self.refresh(number)
        return cast(dict[str, Any], super().get(number))

    def get_without_count(self, number: int) -> dict[str, Any]:
        self.refresh(number)
        return cast(dict[str, Any], super().get_without_count(number))


class DecisionEnv(LogEnv):
    def __init__(self, root: Path, monkeypatch: Any) -> None:
        super().__init__(root, monkeypatch)
        self.github = DecisionGitHub(self.root, self.upstream)

    def capture(
        self,
        rows: list[dict[str, Any]] | None = None,
        *,
        parent: int = 1,
        prepare: bool = False,
    ) -> Any:
        log = journal.Journal(self.state, "o/r", parent)
        for row in copy.deepcopy(rows if rows is not None else history()):
            row["scope"]["parent"] = parent
            log.capture(row["id"], row)
            if prepare:
                log.prepare(row["id"], key="prepare:" + row["id"], at=AT)
        return log

    def decision_publisher(self, log: Any) -> Any:
        return publication.Publisher(self.root, log, self.github)

    def upstream_files(self) -> dict[str, bytes]:
        log = journal.Journal(self.state, "o/r", 1)
        return cast(
            dict[str, bytes], self.decision_publisher(log).snapshot(self.github.main())
        )

    def advance(self, files: dict[str, bytes]) -> str:
        git(self.root, "fetch", "origin")
        git(self.root, "merge", "--ff-only", "origin/main")
        for name, raw in files.items():
            target = self.root / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(raw)
        git(self.root, "add", "--force", "--", *sorted(files))
        git(self.root, "commit", "-qm", "independent main change")
        git(self.root, "push", "origin", "main")
        return cast(str, git(self.root, "rev-parse", "HEAD"))
