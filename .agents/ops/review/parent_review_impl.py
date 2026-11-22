"""One read-only broad review, invoked only by the automatic progression."""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path
from typing import Any, Literal

from dispatcher.runtime import Runtime, capacity_refusal
from lib.agent_events import runtime_errors_failed
from lib.d_capacity import CapacityError
from lib.ops_review.parent_review_policy import marker, parse_output
from lib.ops_review.progression import (
    read,
    review_attempt,
    review_launch_directory,
    save,
)
from lib.ops_review.review_policy import neutralize_markers
from lib.ops_review.review_workspace import build_prompt, workspace
from lib.review_settings import runtime_view
from parent.github import GitHub
from parent.policy import object_value
from parent.session import ParentSession
from parent.snapshot import git, snapshot
from parent.state import Store

HERE = Path(__file__).resolve().parent
OPS = HERE.parent


def save_execution(
    path: Path,
    status: Literal["running", "completed", "capacity_deferred"],
    attempt: int,
) -> None:
    save(
        path,
        {
            "status": status,
            "agent_invocations": int(status != "capacity_deferred"),
            "attempt": attempt,
        },
    )


def check_runtime_request(
    launch: Path, materials: dict[str, Any], record: dict[str, Any]
) -> None:
    runtime = launch / "agent-runtime"
    contract = record.get("contract", {})
    settings = contract.get("settings", {})
    if (
        record.get("final_path") != str(runtime / "final.md")
        or record.get("run_dir") != str(runtime)
        or record.get("prompt_sha256")
        != hashlib.sha256((launch / "prompt.md").read_bytes()).hexdigest()
        or contract.get("review_selection") != materials["review_selection"]
        or contract.get("mode") != "read-only"
        or settings.get("agent") != "codex"
        or settings.get("profile") != "review"
    ):
        raise ValueError("saved review runtime does not match its request")


def completed_result(
    review_dir: Path, launch: Path, materials: dict[str, Any], record: dict[str, Any]
) -> dict[str, Any]:
    if (
        record.get("result") != "completed"
        or record.get("exit_code") != 0
        or record.get("cleanup_complete") is not True
        or runtime_errors_failed(record)
    ):
        raise ValueError("review runtime did not complete and clean up")
    check_runtime_request(launch, materials, record)
    final = launch / "agent-runtime/final.md"
    result = parse_output(
        final.read_text(), [entry["number"] for entry in materials["prs"]]
    )
    result.update(
        reviewed_heads={
            str(entry["number"]): entry["head_sha"] for entry in materials["prs"]
        },
        runtime=runtime_view(record),
    )
    save(review_dir / "result.json", result)
    return result


def run_review(state: dict[str, Any], directory: Path, gh: GitHub) -> dict[str, Any]:
    from lib.opsproc import CommandError, run_process_group

    request = state["request"]
    review_dir = directory / f"review-{state['review']['round']}"
    review_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    result_path = review_dir / "result.json"
    if result_path.exists():
        return read(result_path)
    execution_path = review_dir / "execution.json"
    runner = Runtime(
        Path(state.get("options", {}).get("agent_exec", OPS / "lib/agent_exec.sh"))
    )
    attempt = 0
    if execution_path.exists():
        attempt = review_attempt(execution_path)
        launch = review_launch_directory(review_dir, attempt)
        runtime = launch / "agent-runtime"
        if (
            not (launch / "request.json").is_file()
            or not (runtime / "result.json").is_file()
        ):
            raise ValueError(
                "recovery_required: missing review launch or runtime evidence"
            )
        saved = read(launch / "request.json")
        if (
            saved.get("review_run_id") != request["review_run_id"]
            or saved.get("repo") != request["repo"]
            or [p["number"] for p in saved["prs"]]
            != [p["number"] for p in request["prs"]]
        ):
            raise ValueError("saved review intent does not match progression")
        proof = runner.inspect(runtime)
        if not runner.ended(proof):
            raise ValueError("recovery_required: reviewer termination is unconfirmed")
        record = proof["record"]
        if capacity_refusal(record):
            check_runtime_request(launch, saved, record)
            save_execution(execution_path, "capacity_deferred", attempt)
            attempt += 1
        else:
            result = completed_result(review_dir, launch, saved, record)
            save_execution(execution_path, "completed", attempt)
            return result
    from lib.review_settings import choose, load_policy

    policy = load_policy(None)
    if any(
        (previous / "execution.json").exists()
        for previous in directory.glob("review-*")
        if previous != review_dir
    ):
        raise ValueError("a previous review round already ran; recover its result")
    launch = review_launch_directory(review_dir, attempt)
    launch.mkdir(parents=True, exist_ok=True, mode=0o700)
    materials = dict(request)
    materials["review_selection"] = state["review"].get(
        "selection", request["review_selection"]
    )
    upgrading = materials["review_selection"]["policy"] != policy
    if upgrading:
        # Old requests awaiting their first launch adopt the current Sol pin.
        materials["review_selection"] = choose(policy)
    materials["prs"] = []
    diffs = {}
    prs: list[dict[str, Any]] = []
    issues: set[int] = {request["parent_issue"]}
    dependency_prs: set[int] = set()
    for target in request["prs"]:
        pr = object_value(gh.get(f"repos/{gh.repo}/pulls/{target['number']}"))
        entry, diff = snapshot(pr)
        materials["prs"].append(entry)
        diffs[pr["number"]] = diff
        prs.append(
            {
                "pr": pr,
                "comments": gh.rows(f"repos/{gh.repo}/issues/{pr['number']}/comments"),
            }
        )
        issues.update(target["issues"])
        issues.update(d["issue"] for d in target["dependencies"])
        dependency_prs.update(n for d in target["dependencies"] for n in d["prs"])
    for n in sorted(dependency_prs - {p["number"] for p in materials["prs"]}):
        prs.append(
            {
                "pr": object_value(gh.get(f"repos/{gh.repo}/pulls/{n}")),
                "comments": gh.comments(n),
            }
        )
    # Follow numbered ancestry and saved evidence bindings, never a repository index.
    pending = list(issues)
    while pending:
        parent = gh.get(
            f"repos/{gh.repo}/issues/{pending.pop()}/parent", missing_parent=True
        )
        if parent is not None:
            n = gh.identity(parent)
            if n not in issues:
                issues.add(n)
                pending.append(n)
    login = gh.authenticated_login()
    bindings: dict[str, int | None] = {}
    evidence_numbers: set[int] = set()
    for n in sorted(issues):
        binding = gh.evidence_issues().find_bound(
            n, login, required=n == request["parent_issue"]
        )
        bindings[str(n)] = binding.number if binding else None
        if binding:
            evidence_numbers.add(binding.number)
    issues.update(evidence_numbers)
    context = {
        "instruction": request["instruction"],
        "issues": [
            {
                "issue": object_value(gh.get(f"repos/{gh.repo}/issues/{n}")),
                "comments": gh.rows(f"repos/{gh.repo}/issues/{n}/comments"),
            }
            for n in sorted(issues)
        ],
        "prs": prs,
        "evidence_bindings": bindings,
    }
    root = Path(state["checkout"])
    template = (HERE / "prompt_template.md").read_text(encoding="utf-8")
    prompt = launch / "prompt.md"
    prompt.write_text(
        build_prompt(materials, context, diffs, template), encoding="utf-8"
    )
    save(launch / "request.json", materials)
    save(launch / "context.json", context)
    save(launch / "review-selection.json", materials["review_selection"])
    config = dict(state.get("options", {}))
    if state["review"].get("config_file"):
        config["review_config_file"] = state["review"]["config_file"]
    if upgrading:
        config.pop("review_config_file", None)
    retrieval = {
        "repo": str(root),
        "repository": request["repo"],
        "run_id": request["review_run_id"],
        "commits": sorted(
            {
                p[k]
                for p in materials["prs"]
                for k in ("head_sha", "base_sha", "merge_base_sha")
            }
        ),
        "github_scope": {
            "issues": sorted(issues),
            "prs": sorted(p["pr"]["number"] for p in prs),
            "ci_heads": sorted({p["pr"]["head"]["sha"] for p in prs}),
        },
        "journal_dir": str(launch / "retrieval"),
        "cpu_tests": config.get("cpu_tests", {}),
        "isolation_proof": config.get("isolation_proof"),
    }
    save(launch / "retrieval-config.json", retrieval)
    from decision_logs.review_capture import reviewed, start_review

    runtime = launch / "agent-runtime"
    operation, runtime_env = start_review(directory.parents[1], state, runtime)
    mcp = launch / "mcp-config.json"
    save(
        mcp,
        {
            "review": {
                "command": sys.executable,
                "args": [
                    "-m",
                    "lib.review_tools.review_server",
                    "--config",
                    str(launch / "retrieval-config.json"),
                ],
                "env": {
                    "PYTHONPATH": str(OPS),
                    **(
                        {"DECISION_LOG_RETRIEVAL": str(operation.path)}
                        if operation
                        else {}
                    ),
                },
                "tool_timeout_sec": 960,
            }
        },
    )
    main_root = Path(
        git("rev-parse", "--path-format=absolute", "--git-common-dir", cwd=root)
        .decode()
        .strip()
    ).parent
    with workspace(main_root, materials, context, diffs) as cwd:
        # No HTTP occurs between the saved intent and the common runner.
        save_execution(execution_path, "running", attempt)
        command = [
            str(runner.runner),
            "--agent",
            "codex",
            "--profile",
            "review",
            "--review-selection-file",
            str(launch / "review-selection.json"),
            *(
                ["--config-file", config["review_config_file"]]
                if config.get("review_config_file")
                else []
            ),
            "--cwd",
            str(cwd),
            "--prompt-file",
            str(prompt),
            "--out",
            str(launch / "agent_output.md"),
            "--mcp-config",
            str(mcp),
            "--json-schema",
            str(HERE / "result_schema.json"),
            "--mode",
            "read-only",
            "--log-dir",
            str(launch),
            "--run-dir",
            str(runtime),
        ]
        try:
            run_process_group(
                command,
                cwd=root,
                timeout=config.get("agent_timeout", 1800),
                env=runtime_env,
            )
        except CommandError:
            if (runtime / "result.json").is_file():
                record = read(runtime / "result.json")
                if reason := capacity_refusal(record):
                    proof = runner.inspect(runtime)
                    if runner.ended(proof) and proof["record"] == record:
                        check_runtime_request(launch, materials, record)
                        save_execution(execution_path, "capacity_deferred", attempt)
                        raise CapacityError(reason) from None
            raise
    result = completed_result(
        review_dir, launch, materials, read(runtime / "result.json")
    )
    reviewed(operation, result)
    save_execution(execution_path, "completed", attempt)
    # Display copies are not merge evidence. Save the result first; a restarted
    # controller returns it without replaying publication or launching an agent.
    public = {
        "schema_version": 2,
        "kind": "broad_review",
        "parent_issue": request["parent_issue"],
        "review_run_id": request["review_run_id"],
        **result,
    }
    body = marker(public) + "\n\n" + neutralize_markers(result["summary"])
    for finding in result["major"]:
        body += "\n\n- " + neutralize_markers(
            finding["title"] + ": " + finding["detail"]
        )
    errors = []
    try:
        if len(body) > 60000 or str(root) in body or str(directory) in body:
            raise ValueError(
                "review display exceeds its limit or contains private paths"
            )
        registry = Store(directory.parents[1])
        with registry.consistent_snapshot():
            ParentSession(gh, registry, request["parent_issue"], state["owner"]).public(
                body
            )
    except (ValueError, OSError) as exc:
        result["publication_errors"] = [{"error": str(exc)}]
        save(result_path, result)
        return result
    for p in materials["prs"]:
        try:
            gh.write(
                "POST", f"repos/{gh.repo}/issues/{p['number']}/comments", {"body": body}
            )
        except (ValueError, OSError) as exc:
            errors.append({"pr": p["number"], "error": str(exc)})
    if errors:
        result["publication_errors"] = errors
        save(result_path, result)
    return result
