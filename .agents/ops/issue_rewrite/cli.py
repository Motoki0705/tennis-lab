"""Workflow commands; no raw CLI errors, model output, or private paths on stdout."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, cast

# The dependency's public module lives in lib, independently of agent_exec.sh.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))

from agent_output import OutputSchema
from agy_headless import DEFAULT_AGY
from parent.judgement_comments import read_json

from .github import GitHub
from .pipeline import (
    PACKAGE,
    Handoff,
    apply,
    bundle_for,
    fail,
    generate,
    outcome,
    prepare,
)
from .publication import publication_errors, summary, write_json

PUBLIC_FAILURES = {
    "body_conflict_retry_exhausted",
    "interrupted_during_write",
    "patch_response_mismatch",
    "missing_validated_rewrite",
    "bundle_repository_mismatch",
    "verifier_did_not_pass",
    "artifact_invariants_failed",
    "failed_bundle_cannot_be_applied",
    "github_request_failed",
    "rewrite_label_conflict",
}


def read_bundle(path: Path, repo: str, number: int, attempt: int) -> Handoff:
    text = path.read_text(encoding="utf-8")
    if publication_errors(text):
        raise ValueError("unsafe_artifact")
    value: Any = read_json(text)
    OutputSchema(PACKAGE / "schemas/handoff.schema.json", agent="agy").validate(value)
    if (value["repo"].casefold(), value["issue_number"], value["attempt"]) != (
        repo.casefold(),
        number,
        attempt,
    ):
        raise ValueError("artifact_identity_mismatch")
    return cast(Handoff, value)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "rewrite", "apply", "fail"))
    parser.add_argument("--repo", required=True)
    parser.add_argument(
        "--enabled-since",
        required=True,
        help="explicit workflow vars.ISSUE_REWRITE_ENABLED_SINCE (empty means unset)",
    )
    parser.add_argument("--issue-number", required=True, type=int)
    parser.add_argument("--attempt", type=int, choices=(1, 2), default=1)
    parser.add_argument("--manual", action="store_true")
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--bundle", type=Path)
    parser.add_argument("--agy", type=Path, default=DEFAULT_AGY)
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path(os.environ["GITHUB_STEP_SUMMARY"])
        if os.environ.get("GITHUB_STEP_SUMMARY")
        else None,
    )
    args = parser.parse_args(argv)
    if args.issue_number < 1 or (args.command == "rewrite" and args.run_dir is None):
        parser.error(
            "positive issue number and a private rewrite run directory are required"
        )
    if args.command != "prepare" and args.bundle is None:
        parser.error("--bundle is required for rewrite, apply and fail")
    if args.manual and args.command != "prepare":
        parser.error("--manual is supported only by prepare")
    bundle = bundle_for(args.repo, args.issue_number, args.attempt)
    code = 0
    try:
        gh = GitHub(args.repo, enabled_since=args.enabled_since or None)
        if args.command == "prepare":
            report = prepare(gh, args.issue_number, manual=args.manual)
        elif args.command == "rewrite":
            bundle = generate(
                gh,
                args.issue_number,
                args.attempt,
                run_dir=args.run_dir,
                agy=args.agy,
            )
            if publication_errors(json.dumps(bundle, ensure_ascii=False)):
                # Do not upload a rejected body, input, or sensitive diagnostic.
                bundle = outcome(
                    bundle_for(gh.repo, args.issue_number, args.attempt),
                    "failed",
                    "unsafe_artifact",
                )
            OutputSchema(PACKAGE / "schemas/handoff.schema.json", agent="agy").validate(
                bundle
            )
            write_json(args.bundle, bundle)
            report = bundle["report"]
            code = int(bundle["status"] == "failed")
        elif args.command == "apply":
            bundle = read_bundle(args.bundle, gh.repo, args.issue_number, args.attempt)
            report = apply(gh, bundle)
        else:
            # A checkout/upload/runtime failure can leave no artifact at all.
            if args.bundle.exists():
                try:
                    bundle = read_bundle(
                        args.bundle, gh.repo, args.issue_number, args.attempt
                    )
                except (ValueError, OSError):
                    # A rejected artifact is never used, including its report.
                    # Still finalize the freshly fetched issue and expose why.
                    outcome(bundle, "failed", "invalid_failure_artifact")
            status, reason = fail(gh, args.issue_number)
            if reason:
                outcome(bundle, status, reason)
            report = outcome(bundle, status, "workflow_job_failed")["report"]
    except (ValueError, OSError, KeyError, TypeError, subprocess.TimeoutExpired) as exc:
        reason = (
            str(exc) if str(exc) in PUBLIC_FAILURES else "workflow_processing_error"
        )
        report = outcome(
            bundle_for(args.repo, args.issue_number, args.attempt), "failed", reason
        )["report"]
        code = 1
    summary(args.summary, report)
    if os.environ.get("GITHUB_OUTPUT"):
        with Path(os.environ["GITHUB_OUTPUT"]).open("a", encoding="utf-8") as handle:
            handle.write(
                f"status={report['status']}\nneeds_retry={'true' if report['status'] == 'retry_required' else 'false'}\n"
            )
    print(json.dumps({"issue_number": args.issue_number, "status": report["status"]}))
    return code
