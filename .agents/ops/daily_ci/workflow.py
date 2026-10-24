"""Small stdlib-only Actions runner. Inputs are data, commands are argv arrays."""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path
from typing import Any

from .evidence import PLANS, digest_file, read_json, validate_identity
from .model import instant, utcnow, window


def save(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        json.dump(value, fh, ensure_ascii=False, indent=2)
        fh.flush()
        os.fsync(fh.fileno())
    tmp.replace(path)


def git(root: Path, *args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=root, text=True).strip()


def check_window(identity: dict[str, Any], now: datetime) -> None:
    cycle, start, deadline = window(now)
    if cycle != identity["cycle_id"] or not start <= now < deadline:
        raise ValueError("outside daily CI maintenance window")


def guard(
    identity: dict[str, Any], root: Path, env: dict[str, str], now: datetime
) -> None:
    validate_identity(identity)
    check_window(identity, now)
    if (
        env.get("GITHUB_EVENT_NAME") != "workflow_dispatch"
        or env.get("GITHUB_REF") != "refs/heads/main"
        or env.get("GITHUB_SHA") != identity["target_sha"]
        or env.get("GITHUB_WORKFLOW_SHA") != identity["target_sha"]
        or env.get("GITHUB_WORKFLOW_REF")
        != (
            f"{identity['repository']}/.github/workflows/{identity['workflow']}@refs/heads/main"
        )
        or git(root, "rev-parse", "HEAD") != identity["target_sha"]
    ):
        raise ValueError("event/workflow/main/checkout SHA mismatch")
    # Full hashes were validated before reaching argv. Preserve the whole interval,
    # including changes from failed previous days; never substitute event.before.
    for before, after in (
        (identity["target_sha"], "refs/remotes/origin/main"),
        (identity["baseline_sha"], identity["target_sha"]),
    ):
        if subprocess.run(
            ["git", "merge-base", "--is-ancestor", before, after], cwd=root, check=False
        ).returncode:
            raise ValueError("target is not on main or baseline is not its ancestor")


def initialize(path: Path, root: Path, env: dict[str, str]) -> None:
    if path.exists():
        raise ValueError("evidence already exists; never overwrite an attempt")
    identity: dict[str, Any] = {
        key: env.get("DAILY_CI_" + key.upper(), "")
        for key in (
            "workflow",
            "shard",
            "cycle_id",
            "generation",
            "request_id",
            "target_sha",
            "baseline_sha",
            "deadline_at",
        )
    }
    identity.update(
        repository=env.get("GITHUB_REPOSITORY", ""),
        job=env.get("GITHUB_JOB", ""),
        run_id=env.get("GITHUB_RUN_ID", ""),
        attempt=env.get("GITHUB_RUN_ATTEMPT", ""),
        checkout_sha=git(root, "rev-parse", "HEAD"),
    )
    plan = PLANS.get((identity["workflow"], identity["job"], identity["shard"]), ())
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "identity": identity,
        "started_at": utcnow().isoformat(),
        "finished_at": None,
        "validation_error": None,
        "files": {},
        "checks": {
            name: {
                "outcome": "not_run",
                "started_at": None,
                "finished_at": None,
                "exit_code": None,
                "argv": [],
            }
            for name in plan
        },
    }
    try:
        for key in ("generation", "run_id", "attempt"):
            value = identity[key]
            if not value.isascii() or not value.isdecimal() or str(int(value)) != value:
                raise ValueError(f"invalid {key}")
            identity[key] = int(value)
        guard(identity, root, env, utcnow())
        check_window(identity, utcnow())
    except (ValueError, subprocess.CalledProcessError) as exc:
        manifest["validation_error"] = str(exc)
        raise
    finally:
        save(path, manifest)
        write_junit(path, manifest)


def write_junit(path: Path, manifest: dict[str, Any]) -> None:
    suite = ET.Element(
        "testsuite", name="daily-ci-checks", tests=str(len(manifest["checks"]))
    )
    for name, check in manifest["checks"].items():
        case = ET.SubElement(
            suite, "testcase", name=name, classname=str(manifest["identity"]["job"])
        )
        if check["outcome"] != "passed":
            ET.SubElement(
                case,
                "failure" if check["outcome"] == "failed" else "skipped",
                message=check["outcome"],
            )
    ET.ElementTree(suite).write(
        path.parent / "checks.junit.xml", encoding="utf-8", xml_declaration=True
    )


def run_check(path: Path, root: Path, name: str, argv: list[str], cwd: Path) -> int:
    manifest = read_json(path)
    if manifest["validation_error"] or manifest["finished_at"]:
        raise ValueError("job was rejected or already finalized")
    guard(manifest["identity"], root, dict(os.environ), utcnow())
    check = manifest["checks"][name]
    if check["outcome"] != "not_run" or not argv:
        raise ValueError("check already started or command missing")
    check.update(outcome="incomplete", started_at=utcnow().isoformat(), argv=argv)
    save(path, manifest)
    env = {**os.environ, "DAILY_CI_MANIFEST": str(path)}
    # Git ancestry checks and durable bookkeeping may themselves cross 09:00.
    check_window(manifest["identity"], utcnow())
    process = subprocess.Popen(argv, cwd=cwd, env=env, start_new_session=True)
    try:
        seconds = (
            instant(manifest["identity"]["deadline_at"]) - utcnow()
        ).total_seconds()
        code = process.wait(timeout=max(0, seconds))
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
        code = 124
    check.update(
        outcome="passed" if code == 0 else "failed",
        exit_code=code,
        finished_at=utcnow().isoformat(),
    )
    save(path, manifest)
    write_junit(path, manifest)
    return code


def finalize(path: Path, root: Path) -> None:
    manifest = read_json(path)
    try:
        guard(manifest["identity"], root, dict(os.environ), utcnow())
        check_window(manifest["identity"], utcnow())
    except (ValueError, subprocess.CalledProcessError) as exc:
        manifest["validation_error"] = str(exc)
    manifest["finished_at"] = utcnow().isoformat()
    write_junit(path, manifest)
    manifest["files"] = {
        p.name: digest_file(p)
        for p in sorted(path.parent.iterdir())
        if p.is_file() and p.name != "manifest.json" and not p.name.endswith(".tmp")
    }
    save(path, manifest)
    if manifest["validation_error"]:
        raise ValueError(manifest["validation_error"])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=Path(".daily-ci-evidence"))
    sub = parser.add_subparsers(dest="operation", required=True)
    sub.add_parser("init")
    sub.add_parser("finalize")
    run = sub.add_parser("run")
    run.add_argument("--cwd", type=Path, default=Path("."))
    run.add_argument("check")
    run.add_argument("argv", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    root = Path.cwd()
    path = args.directory.resolve() / "manifest.json"
    try:
        if args.operation == "init":
            initialize(path, root, dict(os.environ))
        elif args.operation == "finalize":
            finalize(path, root)
        else:
            argv = args.argv[1:] if args.argv[:1] == ["--"] else args.argv
            return run_check(path, root, args.check, argv, args.cwd)
        return 0
    except (
        KeyError,
        TypeError,
        ValueError,
        OSError,
        subprocess.CalledProcessError,
    ) as exc:
        print(f"daily CI evidence: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
