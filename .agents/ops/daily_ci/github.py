"""Bounded GitHub REST boundary. Dispatch is never retried here."""

from __future__ import annotations

import json
import subprocess
from datetime import timedelta
from typing import Any, Protocol
from urllib.parse import urlencode

from .model import instant, positive, sha
from .store import load


class Actions(Protocol):
    def main_sha(self) -> str: ...

    def dispatch(
        self, workflow: str, inputs: dict[str, str], timeout: float
    ) -> int | None: ...

    def runs(self, workflow: str, requested_at: str) -> list[dict[str, Any]]: ...

    def run(self, run_id: int) -> dict[str, Any]: ...

    def cancel(self, run_id: int) -> None: ...


class GitHub:
    def __init__(self, repository: str) -> None:
        self.prefix = f"repos/{repository}"

    def request(
        self,
        method: str,
        path: str,
        payload: dict[str, Any] | None = None,
        *,
        timeout: float = 20,
    ) -> Any:
        args = [
            "gh",
            "api",
            "--hostname",
            "github.com",
            "--method",
            method,
            "-H",
            "X-GitHub-Api-Version: 2022-11-28",
            f"{self.prefix}/{path}",
        ]
        if payload is not None:
            args.extend(["--input", "-"])
        result = subprocess.run(
            args,
            input=json.dumps(payload) if payload is not None else None,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
        if result.returncode:
            raise ValueError(f"GitHub {method} {path}: {result.stderr.strip()[:1000]}")
        return load(result.stdout) if result.stdout.strip() else None

    def main_sha(self) -> str:
        return sha(self.request("GET", "git/ref/heads/main")["object"]["sha"])

    def dispatch(
        self, workflow: str, inputs: dict[str, str], timeout: float
    ) -> int | None:
        response = self.request(
            "POST",
            f"actions/workflows/{workflow}/dispatches",
            {"ref": "main", "inputs": inputs, "return_run_details": True},
            timeout=timeout,
        )
        return positive(response["workflow_run_id"]) if response else None

    def runs(self, workflow: str, requested_at: str) -> list[dict[str, Any]]:
        after = (instant(requested_at) - timedelta(seconds=30)).isoformat()
        rows: list[dict[str, Any]] = []
        seen: set[int] = set()
        for page in range(1, 11):
            query = urlencode(
                {
                    "event": "workflow_dispatch",
                    "created": f">={after}",
                    "per_page": 100,
                    "page": page,
                }
            )
            response = self.request("GET", f"actions/workflows/{workflow}/runs?{query}")
            if (
                not isinstance(response, dict)
                or type(response.get("total_count")) is not int
                or response["total_count"] > 1000
                or not isinstance(response.get("workflow_runs"), list)
            ):
                raise ValueError("incomplete/capped dispatch reconciliation")
            for row in response["workflow_runs"]:
                identifier = positive(row["id"])
                if identifier in seen:
                    raise ValueError("run pagination changed; reconcile again")
                seen.add(identifier)
                rows.append(row)
            if len(response["workflow_runs"]) < 100:
                if len(rows) != response["total_count"]:
                    raise ValueError("incomplete dispatch reconciliation pages")
                return rows
        raise ValueError("dispatch reconciliation exceeded pagination limit")

    def run(self, run_id: int) -> dict[str, Any]:
        result = self.request("GET", f"actions/runs/{positive(run_id)}")
        if not isinstance(result, dict):
            raise ValueError("missing workflow run")
        return result

    def cancel(self, run_id: int) -> None:
        self.request("POST", f"actions/runs/{positive(run_id)}/cancel")
