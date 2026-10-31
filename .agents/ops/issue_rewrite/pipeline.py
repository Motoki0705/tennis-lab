"""Read-only generation and separately authorized, bounded GitHub application."""

from __future__ import annotations

import json
import subprocess
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, TypedDict, cast

from agy_headless import DEFAULT_AGY, AgyError, load_config, run_headless
from issues.format import CHILD_LABEL, validate_body
from issues.rewrite_state import (
    REWRITE_DONE_LABEL,
    REWRITE_FAILED_LABEL,
    REWRITE_FINAL_LABELS,
    REWRITE_SKIPPED_LABEL,
    rewrite_state,
)
from parent.execution import PROGRESS_SECTIONS

from .github import GitHub, has_execution_record, uninterrupted
from .invariants import check_invariants
from .publication import (
    diagnostic_differences,
    public_text,
    publication_errors,
    write_json,
)

PACKAGE = Path(__file__).resolve().parent
USAGE_KEYS = {
    "input_tokens",
    "output_tokens",
    "thinking_tokens",
    "cache_read_tokens",
    "cache_write_tokens",
    "total_tokens",
}


class Handoff(TypedDict):
    schema_version: int
    repo: str
    issue_number: int
    attempt: int
    status: str
    source: dict[str, Any] | None
    body: str | None
    verification: dict[str, Any] | None
    report: dict[str, Any]


def issue_state(gh: GitHub, issue: dict[str, Any]) -> str:
    return cast(
        str,
        rewrite_state(
            [item["name"] for item in issue["labels"]],
            issue["created_at"],
            gh.enabled_since,
        ),
    )


def eligible(state: str, manual: bool) -> bool:
    return state == "pending" or (manual and state == "failed")


def stable_issue(
    gh: GitHub,
    number: int,
    *,
    clock: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> dict[str, Any]:
    """Observe 60 seconds without a body change, with a 10 minute hard limit."""
    issue: dict[str, Any] = gh.issue(number)
    started = unchanged_since = clock()
    while True:
        now = clock()
        if now - started > 600:
            raise ValueError("body_stability_timeout")
        if now - unchanged_since >= 60:
            return issue
        if now - started >= 600:
            raise ValueError("body_stability_timeout")
        sleep(min(10, 60 - (now - unchanged_since), 600 - (now - started)))
        current = gh.issue(number)
        if current["body"] != issue["body"]:
            unchanged_since = clock()
        issue = current


def bundle_for(repo: str, number: int, attempt: int) -> Handoff:
    return {
        "schema_version": 1,
        "repo": repo,
        "issue_number": number,
        "attempt": attempt,
        "status": "failed",
        "source": None,
        "body": None,
        "verification": None,
        "report": {
            "issue_number": number,
            "status": "failed",
            "reasons": [],
            "turns": [],
        },
    }


def outcome(bundle: Handoff, status: str, reason: str | None = None) -> Handoff:
    bundle["status"] = bundle["report"]["status"] = status
    if reason:
        bundle["report"]["reasons"].append({"rule": reason, "detail": reason})
    return bundle


def generate(
    gh: GitHub,
    number: int,
    attempt: int,
    *,
    run_dir: Path,
    manual: bool,
    agy: Path = DEFAULT_AGY,
    timeout: float = 300,
    wait: Callable[[GitHub, int], dict[str, Any]] | None = None,
) -> Handoff:
    run_dir.mkdir(parents=True, mode=0o700)  # existing runs must never be reused
    bundle = bundle_for(gh.repo, number, attempt)
    report = bundle["report"]
    try:
        issue = gh.issue(number)
        state = issue_state(gh, issue)
        comments = gh.comments(number)
        write_json(
            run_dir / "initial-input.json", {"issue": issue, "comments": comments}
        )
        if not eligible(state, manual):
            return outcome(bundle, "not_target", f"state_{state}")
        publisher = gh.repo.split("/")[0]
        if has_execution_record(comments, publisher, gh.repo, number):
            return outcome(bundle, "skipped", "execution_record_exists")
        issue = (stable_issue if wait is None else wait)(gh, number)
        comments = gh.comments(number)
        history = gh.history(number)
        write_json(
            run_dir / "input.json",
            {"issue": issue, "comments": comments, "history": history},
        )
        if not eligible(issue_state(gh, issue), manual):
            return outcome(bundle, "not_target", "state_changed_during_wait")
        if has_execution_record(comments, publisher, gh.repo, number):
            return outcome(bundle, "skipped", "execution_record_exists")
        original = issue["body"]
        if not isinstance(original, str) or history["body"] != original:
            return outcome(bundle, "failed", "snapshot_changed")
        labels = [item["name"] for item in issue["labels"]]
        is_child = CHILD_LABEL in labels
        errors = validate_body(original, is_parent=True, is_child=is_child)
        if errors:
            report["reasons"].extend(
                {"rule": "invalid_source", "detail": public_text(error)}
                for error in errors
            )
            return outcome(bundle, "failed")
        if publication_errors(original):
            return outcome(bundle, "failed", "unsafe_source_for_artifact")
        source = {
            "title": issue["title"],
            "body": original,
            "labels": labels,
            "created_at": issue["created_at"],
            "author": issue["user"]["login"],
            "history": history,
        }
        config = load_config()
        report["version"], report["model"] = config.version, config.model
        context = {
            "repo": gh.repo,
            "issue_number": number,
            "url": f"https://github.com/{gh.repo}/issues/{number}",
            "issue": {
                key: source[key]
                for key in ("title", "body", "labels", "created_at", "author")
            },
            "comments": comments,
            "managed_sections": sorted(PROGRESS_SECTIONS),
        }

        def turn(role: str, revision: int, inputs: dict[str, Any]) -> Any:
            prompt = (PACKAGE / "prompts" / f"{role}.md").read_text(encoding="utf-8")
            prompt += "\n\n入力（JSON）:\n" + json.dumps(inputs, ensure_ascii=False)
            prompt_file = run_dir / f"{role}-{revision}.md"
            prompt_file.write_text(prompt, encoding="utf-8")
            prompt_file.chmod(0o600)
            result = run_headless(
                agy=agy,
                cwd=Path.cwd(),
                prompt_file=prompt_file,
                json_schema=PACKAGE / "schemas" / f"{role}.schema.json",
                log_file=run_dir / f"{role}-{revision}.log",
                timeout=timeout,
            )
            write_json(run_dir / f"{role}-{revision}.json", result.structured_output)
            report["turns"].append(
                {
                    "role": role,
                    "revision": revision,
                    "usage": None
                    if result.usage is None
                    else {k: v for k, v in result.usage.items() if k in USAGE_KEYS},
                    "duration_seconds": result.duration_seconds,
                    "elapsed_seconds": result.elapsed_seconds,
                    "denied_action_count": len(result.denied_actions),
                }
            )
            return result.structured_output

        feedback: dict[str, Any] = {}
        for revision in (1, 2):
            writer = turn("writer", revision, {**context, **feedback})
            candidate = writer["body"]
            unsafe = publication_errors(candidate)
            if unsafe:
                return outcome(bundle, "failed", unsafe[0])
            problems = check_invariants(original, candidate, is_child=is_child)
            verification = None
            if not problems:
                verification = turn(
                    "verifier", revision, {**context, "rewritten": candidate}
                )
                if verification["verdict"] != "pass" or verification["differences"]:
                    problems = [
                        {
                            "rule": "verifier_rejected",
                            "detail": "semantic verification failed",
                        }
                    ]
                    report["reasons"].append(
                        {
                            "rule": "semantic_differences",
                            "revision": revision,
                            "differences": diagnostic_differences(
                                verification, original, candidate
                            ),
                        }
                    )
            if not problems:
                bundle["source"], bundle["body"], bundle["verification"] = (
                    source,
                    candidate,
                    verification,
                )
                return outcome(bundle, "ready")
            report["reasons"].extend(
                {**item, "detail": public_text(item["detail"]), "revision": revision}
                for item in problems
            )
            feedback = {
                "previous_body": candidate,
                "violations": problems,
                "verification": verification,
            }
        return outcome(bundle, "failed", "repair_exhausted")
    except AgyError as exc:
        write_json(run_dir / "failure.json", {"reason": exc.reason, "detail": str(exc)})
        return outcome(bundle, "failed", f"agy_{exc.reason}")
    except (ValueError, OSError, KeyError, TypeError, subprocess.TimeoutExpired) as exc:
        write_json(run_dir / "failure.json", {"detail": str(exc)})
        reason = (
            "body_stability_timeout"
            if str(exc) == "body_stability_timeout"
            else "input_or_github_error"
        )
        return outcome(bundle, "failed", reason)


def final_label(gh: GitHub, number: int, label: str) -> str:
    issue = gh.issue(number)
    labels = [item["name"] for item in issue["labels"]]
    state = issue_state(gh, issue)
    # Preserve a human's skip or a completed run, including races after checks.
    if state in {"not_target", "done", "skipped"}:
        return state
    if set(labels) & REWRITE_FINAL_LABELS == {label}:
        return label.removeprefix("rewrite:")
    gh.request(
        "PATCH",
        f"repos/{gh.repo}/issues/{number}",
        {
            "labels": [name for name in labels if name not in REWRITE_FINAL_LABELS]
            + [label],
        },
    )
    return label.removeprefix("rewrite:")


def apply(gh: GitHub, bundle: Handoff, *, ref: str, manual: bool) -> dict[str, Any]:
    from agent_output import OutputSchema

    number, attempt = bundle["issue_number"], bundle["attempt"]
    if bundle["repo"].casefold() != gh.repo.casefold():
        raise ValueError("bundle_repository_mismatch")
    if bundle["status"] == "failed":
        raise ValueError("failed_bundle_cannot_be_applied")
    if bundle["status"] == "not_target":
        return bundle["report"]
    current = gh.issue(number)
    if not eligible(issue_state(gh, current), manual):
        return outcome(bundle, "not_target", "state_changed_before_apply")["report"]
    if has_execution_record(
        gh.comments(number), gh.repo.split("/")[0], gh.repo, number
    ):
        state = final_label(gh, number, REWRITE_SKIPPED_LABEL)
        return outcome(bundle, state, "execution_record_exists")["report"]
    if bundle["status"] == "skipped":
        # A record removed since generation is not permission to apply stale input.
        return outcome(bundle, final_label(gh, number, REWRITE_SKIPPED_LABEL))["report"]
    source, candidate, verification = (
        bundle["source"],
        bundle["body"],
        bundle["verification"],
    )
    if (
        source is None
        or not isinstance(candidate, str)
        or not isinstance(verification, dict)
    ):
        raise ValueError("missing_validated_rewrite")
    OutputSchema(PACKAGE / "schemas/verifier.schema.json", agent="agy").validate(
        verification
    )
    if verification["verdict"] != "pass" or verification["differences"]:
        raise ValueError("verifier_did_not_pass")
    if check_invariants(
        source["body"], candidate, is_child=CHILD_LABEL in source["labels"]
    ):
        raise ValueError("artifact_invariants_failed")
    before = gh.history(number)
    current = gh.issue(
        number
    )  # final uncached body comparison immediately before PATCH
    if not eligible(issue_state(gh, current), manual):
        return outcome(bundle, "not_target", "state_changed_before_patch")["report"]
    if has_execution_record(
        gh.comments(number), gh.repo.split("/")[0], gh.repo, number
    ):
        state = final_label(gh, number, REWRITE_SKIPPED_LABEL)
        return outcome(bundle, state, "execution_record_exists")["report"]
    if (
        current["body"] != source["body"]
        or before["body"] != source["body"]
        or before["edits"] != source["history"]["edits"]
    ):
        if attempt == 1:
            gh.request(
                "POST",
                f"repos/{gh.repo}/actions/workflows/issue-rewrite.yml/dispatches",
                {
                    "ref": ref,
                    "inputs": {"issue_number": str(number), "attempt": "2"},
                },
            )
            return outcome(bundle, "retry_dispatched", "body_changed_before_patch")[
                "report"
            ]
        raise ValueError("body_conflict_retry_exhausted")
    if candidate != source["body"]:
        patched = gh.request(
            "PATCH", f"repos/{gh.repo}/issues/{number}", {"body": candidate}
        )
        if not isinstance(patched, dict) or patched.get("body") != candidate:
            raise ValueError("patch_response_mismatch")
        if not uninterrupted(before, gh.history(number), candidate):
            # Never roll back over another edit. GitHub history retains all versions.
            raise ValueError("interrupted_during_write")
    elif gh.history(number) != before:
        raise ValueError("interrupted_during_write")
    return outcome(bundle, final_label(gh, number, REWRITE_DONE_LABEL))["report"]


def fail(gh: GitHub, number: int) -> tuple[str, str | None]:
    current = gh.issue(number)
    state = issue_state(gh, current)
    if state in {"not_target", "done", "skipped"}:
        return state, None
    comments = gh.comments(number)
    try:
        recorded = has_execution_record(
            comments, gh.repo.split("/")[0], gh.repo, number
        )
    except ValueError:
        # Parsing failure is itself a failure, never a successful skip/rewrite.
        return final_label(gh, number, REWRITE_FAILED_LABEL), "invalid_execution_record"
    label = REWRITE_SKIPPED_LABEL if recorded else REWRITE_FAILED_LABEL
    return final_label(
        gh, number, label
    ), "execution_record_exists" if recorded else None
