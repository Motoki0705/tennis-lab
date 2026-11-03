"""Plan CPU lanes using only Python's standard library, git and read-only gh API."""

from __future__ import annotations

import argparse
import io
import json
import os
import subprocess
import time
import zipfile
from pathlib import Path
from typing import Any

from .lanes import digest, load_registry, make_plan, validate_plan


def parse_diff(raw: bytes) -> list[dict[str, Any]]:
    """Parse NUL-delimited name-status, retaining both sides of a rename."""
    if raw and not raw.endswith(b"\0"):
        raise ValueError("unterminated git diff")
    fields = iter(raw.decode("utf-8").split("\0")[:-1])
    changes = []
    for status in fields:
        try:
            paths = [next(fields)]
            if status.startswith(("R", "C")):
                paths.append(next(fields))
        except StopIteration as error:
            raise ValueError("truncated git name-status record") from error
        changes.append({"status": status, "paths": paths})
    return changes


def git_changes(root: Path, base: str, head: str) -> list[dict[str, Any]]:
    return parse_diff(
        subprocess.check_output(
            ["git", "diff", "--name-status", "-z", "--find-renames", base, head, "--"],
            cwd=root,
            stderr=subprocess.PIPE,
            timeout=30,
        )
    )


def _api(endpoint: str) -> bytes:
    return subprocess.check_output(
        ["gh", "api", endpoint],
        stderr=subprocess.PIPE,
        timeout=30,
    )


def main_successes(
    root: Path, repo: str, ancestor: str, registry: dict[str, Any]
) -> dict[str, Any]:
    """A green job alone is insufficient: require the full, audited main artifact.

    Search is bounded. Expired/missing/incompatible evidence remains unknown;
    neither a skipped suite nor another attempt's artifact establishes success.
    """
    missing = {suite["id"] for suite in registry["suites"]}
    found: dict[str, Any] = {}
    problems: list[str] = []
    if not missing:
        return found
    deadline = time.monotonic() + 90
    try:
        runs = json.loads(
            _api(
                f"repos/{repo}/actions/workflows/ci.yml/runs?branch=main&event=push&status=success&per_page=30"
            )
        )["workflow_runs"]
        for run in runs:
            if not missing:
                break
            if time.monotonic() >= deadline:
                problems.append("main success lookup exceeded its 90-second budget")
                break
            if (
                run["event"] != "push"
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
                    problems.append(f"cannot verify main ancestor {run['head_sha']}")
                continue
            artifacts = json.loads(
                _api(f"repos/{repo}/actions/runs/{run['id']}/artifacts?per_page=100")
            )["artifacts"]
            name = f"cpu-test-coverage-{run['id']}-{run['run_attempt']}"
            matching = [a for a in artifacts if a["name"] == name and not a["expired"]]
            if len(matching) != 1:
                continue
            with zipfile.ZipFile(
                io.BytesIO(
                    _api(f"repos/{repo}/actions/artifacts/{matching[0]['id']}/zip")
                )
            ) as archive:
                coverage = json.loads(archive.read("coverage.json"))
            if (
                coverage.get("success") is not True
                or coverage.get("full_cpu_success") is not True
                or coverage.get("source_sha") != run["head_sha"]
                or str(coverage.get("run_id")) != str(run["id"])
                or str(coverage.get("attempt")) != str(run["run_attempt"])
                or coverage.get("event") != "push"
                or coverage.get("ref") != "refs/heads/main"
                or coverage.get("policy_sha256") != digest(registry)
            ):
                continue
            for suite in sorted(missing):
                result = coverage.get("suite_results", {}).get(suite, {})
                if (
                    result.get("verified_success") is True
                    and result.get("counts", {}).get("passed", 0) > 0
                ):
                    found[suite] = {
                        "commit": run["head_sha"],
                        "run_id": str(run["id"]),
                        "attempt": str(run["run_attempt"]),
                        "url": run["html_url"],
                        "counts": result["counts"],
                        "policy_sha256": digest(registry),
                    }
            missing.difference_update(found)
    except (
        OSError,
        subprocess.SubprocessError,
        ValueError,
        KeyError,
        TypeError,
        zipfile.BadZipFile,
    ) as error:
        problems.append(
            f"main success evidence unavailable: {type(error).__name__}: {error}"
        )
    reason = (
        "; ".join(problems)
        if problems
        else "no compatible full success artifact in the latest 30 successful main runs (missing/expired/unverified)"
    )
    return {
        **found,
        **{suite: {"commit": None, "reason": reason} for suite in sorted(missing)},
    }


def build_plan(
    root: Path, environment: dict[str, str], payload: dict[str, Any]
) -> dict[str, Any]:
    registry, files = load_registry(root)
    event = environment["GITHUB_EVENT_NAME"]
    sha = environment["GITHUB_SHA"]
    if (
        subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True
        ).strip()
        != sha
    ):
        raise ValueError("plan checkout differs from GITHUB_SHA")
    base = head = None
    changes: list[dict[str, Any]] = []
    diff_error = None
    if event == "pull_request":
        try:
            base = payload["pull_request"]["base"]["sha"]
            head = payload["pull_request"]["head"]["sha"]
            if (
                not isinstance(base, str)
                or not isinstance(head, str)
                or not base
                or not head
            ):
                raise ValueError("missing PR base/head")
            changes = git_changes(root, base, head)
            # Reject malformed git evidence before deciding to defer anything.
            validate_plan(
                make_plan(
                    registry,
                    files,
                    source_sha=sha,
                    event=event,
                    base_sha=base,
                    head_sha=head,
                    changes=changes,
                )
            )
        except (
            OSError,
            subprocess.SubprocessError,
            UnicodeError,
            ValueError,
            KeyError,
            TypeError,
        ) as error:
            changes = []
            diff_error = f"{type(error).__name__}: {error}"
    history = main_successes(
        root, environment["GITHUB_REPOSITORY"], base or sha, registry
    )
    return validate_plan(
        make_plan(
            registry,
            files,
            source_sha=sha,
            event=event,
            ref=environment["GITHUB_REF"],
            run_id=environment["GITHUB_RUN_ID"],
            attempt=environment["GITHUB_RUN_ATTEMPT"],
            base_sha=base,
            head_sha=head,
            changes=changes,
            diff_error=diff_error,
            main_successes=history,
        ),
        root=root,
    )


def markdown(plan: dict[str, Any]) -> str:
    lines = [
        "## CPU lane plan",
        "",
        f"Scope: **{plan['scope']}**.",
        "Reasons: " + "; ".join(plan["reasons"]),
        f"Requested path suites: {', '.join(plan['requested_suites']) or '(none)'}.",
        "Unregistered tests always run.",
        "",
        "| Deferred suite | Owner | Reason | Last verified full main success |",
        "|---|---|---|---|",
    ]
    for suite in plan["registry"]["suites"]:
        if suite["id"] not in plan["deferred_suites"]:
            continue
        previous = plan["main_successes"].get(suite["id"], {})
        evidence = (
            f"[{previous['commit']}]({previous['url']})"
            if previous.get("commit")
            else "unknown: " + previous.get("reason", "no verified main evidence")
        )
        lines.append(
            f"| {suite['id']} | {suite['owner']} | {suite['reason']} | {evidence} |"
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = json.loads(Path(os.environ["GITHUB_EVENT_PATH"]).read_text())
    plan = build_plan(Path.cwd(), dict(os.environ), payload)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n")
    document = markdown(plan)
    args.output.with_suffix(".md").write_text(document)
    print(document)
    if destination := os.environ.get("GITHUB_STEP_SUMMARY"):
        with Path(destination).open("a") as stream:
            stream.write(document)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
