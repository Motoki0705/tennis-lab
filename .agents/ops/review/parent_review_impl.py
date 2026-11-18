"""One read-only broad review, invoked only by the automatic progression."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

from lib.ops_review.parent_review_policy import marker, parse_output
from lib.ops_review.progression import read, save
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


def completed_result(review_dir: Path, materials: dict[str, Any]) -> dict[str, Any]:
    runtime = review_dir / "agent-runtime"
    record = read(runtime / "result.json")
    final = runtime / "final.md"
    # A transient reconnect may be reported as runtime_error even after a
    # clean CLI exit. Require the terminal event and exact final in that case.
    completed = record.get("result") == "completed" and record.get("exit_code") == 0
    runtime_errors = (record.get("runtime") or {}).get("runtime_errors", [])
    if (
        not completed
        and runtime_errors
        and all(
            e.get("type") == "error"
            and e.get("message", "").startswith("Reconnecting...")
            for e in runtime_errors
        )
        and record.get("turn_end_reason") == "runtime_error"
        and record.get("cli_exit_code") == 0
    ):
        events = [
            json.loads(line)
            for line in (runtime / "events.jsonl").read_text().splitlines()
            if line.strip()
        ]
        messages = [
            e["item"]["text"]
            for e in events
            if e.get("type") == "item.completed"
            and e.get("item", {}).get("type") == "agent_message"
        ]
        completed = bool(
            events
            and events[-1].get("type") == "turn.completed"
            and messages
            and messages[-1].strip() == final.read_text().strip()
        )
    if not completed or record.get("cleanup_complete") is not True:
        raise ValueError("review runtime did not complete and clean up")
    if (
        record.get("final_path") != str(final)
        or record.get("run_dir") != str(runtime)
        or record.get("prompt_sha256")
        != hashlib.sha256((review_dir / "prompt.md").read_bytes()).hexdigest()
        or record.get("contract", {}).get("review_selection")
        != materials["review_selection"]
    ):
        raise ValueError("saved review runtime does not match its request")
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
    from lib.opsproc import run_process_group

    request = state["request"]
    review_dir = directory / f"review-{state['review']['round']}"
    review_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    result_path = review_dir / "result.json"
    if result_path.exists():
        return read(result_path)
    execution_path = review_dir / "execution.json"
    if execution_path.exists():
        if (
            not (review_dir / "request.json").is_file()
            or not (review_dir / "agent-runtime/result.json").is_file()
        ):
            raise ValueError(
                "recovery_required: missing review launch or runtime evidence"
            )
        saved = read(review_dir / "request.json")
        if (
            saved.get("review_run_id") != request["review_run_id"]
            or saved.get("repo") != request["repo"]
            or [p["number"] for p in saved["prs"]]
            != [p["number"] for p in request["prs"]]
        ):
            raise ValueError("saved review intent does not match progression")
        from dispatcher.runtime import Runtime

        runner = Runtime(
            Path(state.get("options", {}).get("agent_exec", OPS / "lib/agent_exec.sh"))
        )
        if not runner.ended(runner.inspect(review_dir / "agent-runtime")):
            raise ValueError("recovery_required: reviewer termination is unconfirmed")
        result = completed_result(review_dir, saved)
        save(execution_path, {"status": "completed", "agent_invocations": 1})
        return result
    from lib.review_settings import choose, load_policy

    policy = load_policy(None)
    if any(
        (previous / "execution.json").exists()
        for previous in directory.glob("review-*")
        if previous != review_dir
    ):
        raise ValueError("a previous review round already ran; recover its result")
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
    prompt = review_dir / "prompt.md"
    prompt.write_text(
        build_prompt(materials, context, diffs, template), encoding="utf-8"
    )
    save(review_dir / "request.json", materials)
    save(review_dir / "context.json", context)
    save(review_dir / "review-selection.json", materials["review_selection"])
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
        "journal_dir": str(review_dir / "retrieval"),
        "cpu_tests": config.get("cpu_tests", {}),
        "isolation_proof": config.get("isolation_proof"),
    }
    save(review_dir / "retrieval-config.json", retrieval)
    from decision_logs.review_capture import reviewed, start_review

    operation, runtime_env = start_review(
        directory.parents[1], state, review_dir / "agent-runtime"
    )
    mcp = review_dir / "mcp-config.json"
    save(
        mcp,
        {
            "review": {
                "command": sys.executable,
                "args": [
                    "-m",
                    "lib.review_tools.review_server",
                    "--config",
                    str(review_dir / "retrieval-config.json"),
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
    runtime = review_dir / "agent-runtime"
    main_root = Path(
        git("rev-parse", "--path-format=absolute", "--git-common-dir", cwd=root)
        .decode()
        .strip()
    ).parent
    with workspace(main_root, materials, context, diffs) as cwd:
        # No HTTP occurs between the saved intent and the common runner.
        save(execution_path, {"status": "running", "agent_invocations": 1})
        run_process_group(
            [
                config.get("agent_exec", str(OPS / "lib/agent_exec.sh")),
                "--agent",
                "codex",
                "--profile",
                "review",
                "--review-selection-file",
                str(review_dir / "review-selection.json"),
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
                str(review_dir / "agent_output.md"),
                "--mcp-config",
                str(mcp),
                "--json-schema",
                str(HERE / "result_schema.json"),
                "--mode",
                "read-only",
                "--log-dir",
                str(review_dir),
                "--run-dir",
                str(runtime),
            ],
            cwd=root,
            timeout=config.get("agent_timeout", 1800),
            env=runtime_env,
        )
    result = completed_result(review_dir, materials)
    reviewed(operation, result)
    save(execution_path, {"status": "completed", "agent_invocations": 1})
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
