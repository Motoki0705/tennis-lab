"""Read the latest audited full main ancestor's informational growth artifact."""

from __future__ import annotations

import io
import json
import subprocess
import time
import zipfile
from pathlib import Path
from typing import Any

from .growth import validate_snapshot
from .plan import _api


def previous_main(
    root: Path, repo: str, ancestor: str, current_run: str
) -> tuple[dict[str, Any] | None, str | None]:
    """Missing evidence is a visible reason, never an empty successful comparison."""
    problems = []
    deadline = time.monotonic() + 90
    try:
        runs = json.loads(
            _api(
                f"repos/{repo}/actions/workflows/ci.yml/runs?branch=main&event=push&status=success&per_page=30"
            )
        )["workflow_runs"]
        for run in runs:
            if time.monotonic() >= deadline:
                problems.append("main baseline lookup exceeded its 90-second budget")
                break
            if (
                str(run["id"]) == current_run
                or run["event"] != "push"
                or run["head_branch"] != "main"
                or run["conclusion"] != "success"
            ):
                continue
            ancestry = subprocess.run(
                ["git", "merge-base", "--is-ancestor", run["head_sha"], ancestor],
                cwd=root,
                capture_output=True,
                timeout=10,
            )
            if ancestry.returncode != 0:
                if ancestry.returncode != 1:
                    return None, f"cannot verify ancestor for main run {run['id']}"
                continue
            artifacts = json.loads(
                _api(f"repos/{repo}/actions/runs/{run['id']}/artifacts?per_page=100")
            )["artifacts"]
            name = f"cpu-test-coverage-{run['id']}-{run['run_attempt']}"
            matching = [a for a in artifacts if a["name"] == name and not a["expired"]]
            if len(matching) != 1:
                return (
                    None,
                    f"main run {run['id']}: coverage artifact missing/expired/ambiguous; latest full baseline cannot be verified",
                )
            with zipfile.ZipFile(
                io.BytesIO(
                    _api(f"repos/{repo}/actions/artifacts/{matching[0]['id']}/zip")
                )
            ) as archive:
                coverage = json.loads(archive.read("coverage.json"))
                if (
                    coverage["schema_version"] != 2
                    or coverage["source_sha"] != run["head_sha"]
                    or str(coverage["run_id"]) != str(run["id"])
                    or str(coverage["attempt"]) != str(run["run_attempt"])
                    or coverage["event"] != "push"
                    or coverage["ref"] != "refs/heads/main"
                ):
                    return (
                        None,
                        f"main run {run['id']}: incompatible coverage schema or source/run/attempt",
                    )
                if (
                    coverage["success"] is not True
                    or coverage["full_cpu_success"] is not True
                ):
                    problems.append(
                        f"main run {run['id']}: no verified full CPU success"
                    )
                    continue
                # Once the latest full success is known, never silently replace
                # its missing/new-schema data with an older compatible run.
                try:
                    snapshot = validate_snapshot(
                        json.loads(archive.read("growth.json"))
                    )
                    if any(
                        str(snapshot[key]) != str(coverage[key])
                        for key in (
                            "source_sha",
                            "run_id",
                            "attempt",
                            "universe_count",
                            "requested_count",
                            "deferred_count",
                        )
                    ):
                        raise ValueError(
                            "growth evidence differs from coverage identity/counts"
                        )
                    if (
                        snapshot["deferred_count"]
                        or len(snapshot["nodeids"]) != snapshot["requested_count"]
                    ):
                        raise ValueError(
                            "growth evidence is not the full CPU inventory"
                        )
                except (ValueError, KeyError, TypeError, AttributeError) as error:
                    return (
                        None,
                        f"main run {run['id']}: missing/invalid growth evidence: {type(error).__name__}: {error}",
                    )
            return snapshot, None
    except (
        OSError,
        subprocess.SubprocessError,
        ValueError,
        KeyError,
        TypeError,
        AttributeError,
        zipfile.BadZipFile,
    ) as error:
        problems.append(f"main baseline unavailable: {type(error).__name__}: {error}")
    return None, "; ".join(
        problems
    ) or "no verified full main ancestor run in the latest 30 successful main push runs"
