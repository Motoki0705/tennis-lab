"""One read-only broad review, invoked only by the automatic progression."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from lib.review_settings import runtime_view
from parent.github import GitHub
from parent.policy import object_value
from parent.session import ParentSession
from parent.snapshot import git, snapshot
from parent.state import Store

from review.parent_review_policy import marker, parse_output
from review.progression import read, save
from review.review_policy import neutralize_markers
from review.review_workspace import build_prompt, workspace

HERE = Path(__file__).resolve().parent
OPS = HERE.parent


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
        raise ValueError(
            "recovery_required: reviewer launch has no saved completed result"
        )
    for previous in directory.glob("review-*"):
        if previous == review_dir or not (previous / "execution.json").exists():
            continue
        from dispatcher.runtime import Runtime

        runner = Path(
            state.get("options", {}).get("agent_exec", str(OPS / "lib/agent_exec.sh"))
        )
        evidence = Runtime(runner).inspect(previous / "agent-runtime")
        if not Runtime(runner).ended(evidence):
            raise ValueError(
                "recovery_required: previous reviewer termination is unconfirmed"
            )
    materials = dict(request)
    materials["review_selection"] = state["review"].get(
        "selection", request["review_selection"]
    )
    materials["prs"] = []
    diffs = {}
    prs = []
    issues = {request["parent_issue"]}
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
    context = {
        "instruction": request["instruction"],
        "issues": [
            {
                "issue": gh.issue(n),
                "comments": gh.rows(f"repos/{gh.repo}/issues/{n}/comments"),
            }
            for n in sorted(issues)
        ],
        "prs": prs,
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
            "prs": [p["number"] for p in materials["prs"]],
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
    record = read(runtime / "result.json")
    if (
        record.get("result") != "completed"
        or record.get("exit_code") != 0
        or record.get("cleanup_complete") is not True
    ):
        raise ValueError("review runtime did not complete and clean up")
    result = parse_output(
        (runtime / "final.md").read_text(encoding="utf-8"),
        [p["number"] for p in materials["prs"]],
    )
    result.update(
        reviewed_heads={str(p["number"]): p["head_sha"] for p in materials["prs"]},
        runtime=runtime_view(record),
    )
    save(result_path, result)
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
