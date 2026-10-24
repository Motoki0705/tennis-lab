"""Short deterministic collector/dispatcher adapters, never an alternate agent launcher."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any

from lib.opsproc import run_process_group

from .api import CycleAPI
from .store import load


class CommandHooks:
    def __init__(self, api: CycleAPI) -> None:
        self.api = api

    def __call__(
        self, role: str, operation: str, cycle: dict[str, Any] | None, timeout: float
    ) -> dict[str, Any]:
        command = self.api.snapshot()["integrations"][role]
        if not command:
            raise ValueError(f"daily_ci {role} integration is not configured")
        payload = {
            "schema_version": 1,
            "operation": operation,
            "repository": self.api.store.repository,
            "state_dir": str(self.api.store.root.resolve()),
            "cycle": cycle,
        }
        with tempfile.TemporaryDirectory(
            prefix="hook-", dir=self.api.store.directory
        ) as name:
            path = Path(name) / "input.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            response = run_process_group(
                [*command, "--input", str(path)],
                cwd=Path.cwd(),
                timeout=timeout,
                term_grace=1,
            )
        value = load(response)
        if (
            not isinstance(value, dict)
            or type(value.get("schema_version")) is not int
            or value["schema_version"] != 1
        ):
            raise ValueError(f"invalid daily_ci {role} adapter response")
        return value
