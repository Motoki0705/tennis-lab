"""Select unambiguous lane/shard artifacts through the run-wide REST inventory."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any


def inspect_artifacts(
    artifacts: list[dict[str, Any]], run_id: str, count: int
) -> dict[str, Any]:
    """Require one live artifact per name; never merge or pick among duplicates."""
    if not 1 <= count <= 32:
        raise ValueError("shard count must be between 1 and 32")
    plan_name = f"cpu-lane-plan-{run_id}"
    shard_prefix = f"cpu-test-measurements-{run_id}-shard-"
    expected = [plan_name, *(f"{shard_prefix}{i}" for i in range(count))]
    selected: list[dict[str, Any]] = []
    issues: list[str] = []
    for artifact in artifacts:
        name = artifact["name"]
        if name != plan_name and not name.startswith(shard_prefix):
            continue
        if type(artifact.get("expired")) is not bool:
            raise ValueError(f"artifact expiration is missing/invalid: {name}")
        if artifact["expired"]:
            continue
        if name not in expected:
            issues.append(f"unexpected shard artifact: {name} (id {artifact['id']})")
        row = {
            key: artifact[key]
            for key in ("id", "name", "created_at", "size_in_bytes", "expired")
        }
        if (
            type(row["id"]) is not int
            or row["id"] <= 0
            or type(row["size_in_bytes"]) is not int
            or row["size_in_bytes"] < 0
            or not isinstance(row["created_at"], str)
            or not row["created_at"]
        ):
            raise ValueError(f"invalid artifact metadata: {name}")
        selected.append(row)
    names = Counter(row["name"] for row in selected)
    for name in expected:
        if names[name] != 1:
            issues.append(
                f"expected exactly one non-expired artifact for {name}, found {names[name]}"
            )
    if len({row["id"] for row in selected}) != len(selected):
        issues.append("duplicate artifact IDs in run inventory")
    order = {name: index for index, name in enumerate(expected)}
    selected.sort(key=lambda row: (order.get(row["name"], count + 1), row["id"]))
    return {
        "schema_version": 1,
        "run_id": run_id,
        "shard_count": count,
        "success": not issues,
        "issues": issues,
        "artifacts": selected,
    }


def fetch_artifacts(repository: str, run_id: str) -> list[dict[str, Any]]:
    """List every page of the run's artifacts, including earlier attempts."""
    pages = json.loads(
        subprocess.check_output(
            [
                "gh",
                "api",
                f"repos/{repository}/actions/runs/{run_id}/artifacts?per_page=100",
                "--paginate",
                "--slurp",
            ],
            stderr=subprocess.PIPE,
            timeout=60,
        )
    )
    if not isinstance(pages, list) or not pages:
        raise ValueError("invalid paginated run artifact response")
    artifacts = []
    for page in pages:
        if not isinstance(page, dict) or not isinstance(page.get("artifacts"), list):
            raise ValueError("invalid run artifact page")
        artifacts.extend(page["artifacts"])
    return artifacts


def read_manifest(path: Path, run_id: str, count: int) -> dict[str, Any]:
    manifest: dict[str, Any] = json.loads(path.read_text())
    if (
        manifest["schema_version"] != 1
        or manifest["run_id"] != run_id
        or manifest["shard_count"] != count
    ):
        raise ValueError("artifact manifest schema/run/shard count mismatch")
    if manifest["success"] is not True:
        raise ValueError("artifact selection failed: " + "; ".join(manifest["issues"]))
    checked = inspect_artifacts(manifest["artifacts"], run_id, count)
    if checked != manifest:
        raise ValueError("artifact manifest differs from the checked download set")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--count", required=True, type=int)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    try:
        manifest = inspect_artifacts(
            fetch_artifacts(args.repo, args.run_id), args.run_id, args.count
        )
    except (
        OSError,
        ValueError,
        KeyError,
        TypeError,
        AttributeError,
        subprocess.SubprocessError,
    ) as error:
        detail = (
            error.stderr.decode(errors="replace").strip()
            if isinstance(error, subprocess.CalledProcessError) and error.stderr
            else str(error)
        )
        manifest = {
            "schema_version": 1,
            "run_id": args.run_id,
            "shard_count": args.count,
            "success": False,
            "issues": [f"run artifact inventory unavailable/invalid: {detail}"],
            "artifacts": [],
        }
    document = json.dumps(manifest, indent=2) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(document)
    print(document, end="")
    if manifest["success"] and (output := os.environ.get("GITHUB_OUTPUT")):
        with Path(output).open("a") as stream:
            print(f"plan_artifact_id={manifest['artifacts'][0]['id']}", file=stream)
            print(
                "shard_artifact_ids="
                + ",".join(str(row["id"]) for row in manifest["artifacts"][1:]),
                file=stream,
            )
    return 0 if manifest["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
