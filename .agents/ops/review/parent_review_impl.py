"""Run and recover the progression's read-only broad review."""

from __future__ import annotations

import fcntl
import hashlib
import sys
from collections.abc import Callable, Iterator
from contextlib import AbstractContextManager, ExitStack, contextmanager, nullcontext
from pathlib import Path
from typing import Any

from lib.ops_review.parent_review_policy import marker, parse_output
from lib.ops_review.progression import read, save
from lib.ops_review.review_policy import neutralize_markers
from lib.ops_review.review_workspace import (
    build_prompt,
    require_clean_workspace,
    workspace,
    workspace_path,
)
from lib.review_settings import runtime_view
from parent.github import GitHub
from parent.policy import closing_issues, object_value
from parent.session import ParentSession
from parent.snapshot import snapshot
from parent.state import Store

HERE = Path(__file__).resolve().parent
OPS = HERE.parent
MAX_REVIEW_ATTEMPTS = 3
RETRYABLE_END_REASONS = {"runtime_error", "startup_failed", "usage_limit"}


def saved_request(review_dir: Path, request: dict[str, Any]) -> dict[str, Any]:
    saved = read(review_dir / "request.json")
    if (
        any(
            saved.get(key) != request[key]
            for key in ("review_run_id", "repo", "parent_issue", "instruction")
        )
        or [p["number"] for p in saved["prs"]] != [p["number"] for p in request["prs"]]
        or {
            **saved["authorization"],
            "generation": request["authorization"]["generation"],
        }
        != request["authorization"]
    ):
        raise ValueError("saved review intent does not match progression")
    return saved


def require_matching_runtime(
    review_dir: Path, materials: dict[str, Any], record: dict[str, Any]
) -> None:
    runtime = review_dir / "agent-runtime"
    contract = object_value(record.get("contract"))
    if (
        record.get("final_path") != str(runtime / "final.md")
        or record.get("run_dir") != str(runtime)
        or record.get("prompt_sha256")
        != hashlib.sha256((review_dir / "prompt.md").read_bytes()).hexdigest()
        or contract.get("mode") != "read-only"
        or contract.get("review_selection") != materials["review_selection"]
    ):
        raise ValueError("saved review runtime does not match its request")


def ended_runtime(state: dict[str, Any], runtime: Path) -> dict[str, Any]:
    from dispatcher.runtime import Runtime

    runner = Runtime(
        Path(state.get("options", {}).get("agent_exec", OPS / "lib/agent_exec.sh"))
    )
    evidence = runner.inspect(runtime)
    if not runner.ended(evidence):
        raise ValueError(
            "recovery_required: reviewer termination or cleanup is unconfirmed"
        )
    return object_value(evidence["record"])


def require_prior_failures(
    state: dict[str, Any], directory: Path, review_dir: Path
) -> None:
    previous = {
        p.name
        for p in directory.glob("review-*")
        if p != review_dir
        and ((p / "execution.json").exists() or (p / "result.json").exists())
    }
    failed = {
        f"review-{entry['round']}"
        for entry in state.get("review_history", [])
        if "failure" in entry
    }
    expected = {f"review-{n}" for n in range(1, state["review"]["round"])}
    if (
        previous != failed
        or failed != expected
        or any("result" in entry for entry in state.get("review_history", []))
        or any((directory / name / "result.json").exists() for name in previous)
    ):
        raise ValueError("a previous review round already ran; recover its result")


def target_pr(
    gh: GitHub, target: dict[str, Any], retry_heads: dict[str, str] | None
) -> dict[str, Any]:
    number = target["number"]
    pr = object_value(gh.get(f"repos/{gh.repo}/pulls/{number}"))
    if retry_heads is not None and (
        pr.get("number") != number
        or pr.get("state") != "open"
        or pr.get("merged")
        or pr["head"]["sha"] != retry_heads[str(number)]
        or pr["head"]["ref"] != target["branch"]
        or pr["base"]["ref"] != target["base_ref"]
        or any(
            pr[side]["repo"]["full_name"].casefold() != gh.repo.casefold()
            for side in ("head", "base")
        )
        or closing_issues(pr.get("body") or "", gh.repo) != set(target["issues"])
    ):
        raise ValueError(
            f"PR #{number}: retry requires the previous review target and HEAD"
        )
    return pr


def retry_failure(
    state: dict[str, Any], review_dir: Path, materials: dict[str, Any]
) -> dict[str, Any]:
    execution = read(review_dir / "execution.json")
    guarded = execution.get("guarded_launch") is True
    runtime = review_dir / "agent-runtime"
    if guarded:
        # The controller and its Popen child share this flock across fork/exec.
        # Even a launcher that has yet to create its runtime directory holds it.
        guard = review_dir / "launch.lock"
        if not guard.is_file() or guard.is_symlink():
            raise ValueError("recovery_required: reviewer launch lock is missing")
        with guard.open("rb") as stream:
            try:
                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise ValueError(
                    "recovery_required: reviewer launcher is still active"
                ) from exc
    if (runtime / "result.json").is_file():
        record = ended_runtime(state, runtime)
        require_matching_runtime(review_dir, materials, record)
        if record["result"] == "completed":
            raise ValueError("review runtime completed; recover it with merge-continue")
        if record.get("turn_end_reason") not in RETRYABLE_END_REASONS:
            raise ValueError(
                "previous review did not end with a retryable runtime failure"
            )
        return {
            key: record.get(key)
            for key in ("turn_end_reason", "error", "agent_spawned")
        }
    if not guarded or runtime.exists() or runtime.is_symlink():
        raise ValueError(
            "recovery_required: reviewer termination or runtime state is unconfirmed"
        )
    return {
        "turn_end_reason": "interrupted_before_runtime",
        "error": "reviewer launcher ended before creating its runtime",
        "agent_spawned": False,
    }


def reserve_retry(
    state: dict[str, Any], directory: Path, gh: GitHub, failed_round: int
) -> None:
    review = state["review"]
    if failed_round != review["round"]:
        raise ValueError("--retry-review must identify the current failed round")
    if (
        review["stage"] not in {"running", "parent_wait"}
        or "result" in review
        or any((p / "result.json").exists() for p in directory.glob("review-*"))
    ):
        raise ValueError(
            "review retry requires an attempted round without a saved result"
        )
    if failed_round >= MAX_REVIEW_ATTEMPTS:
        raise ValueError(f"review attempt limit reached ({MAX_REVIEW_ATTEMPTS})")
    review_dir = directory / f"review-{failed_round}"
    if not (review_dir / "execution.json").is_file():
        raise ValueError("review retry requires a saved launch intent for this round")
    require_prior_failures(state, directory, review_dir)
    request = state["request"]
    materials = saved_request(review_dir, request)
    failure = retry_failure(state, review_dir, materials)
    require_clean_workspace(Path(state["checkout"]), request["review_run_id"])
    heads = {str(p["number"]): p["head_sha"] for p in materials["prs"]}
    for target in request["prs"]:
        target_pr(gh, target, heads)
    state.setdefault("review_history", []).append({**review, "failure": failure})
    state["review"] = {
        "stage": "requested",
        "round": failed_round + 1,
        "retry_heads": heads,
        "selection": materials["review_selection"],
        **({"config_file": review["config_file"]} if "config_file" in review else {}),
    }


def completed_result(
    review_dir: Path, materials: dict[str, Any], record: dict[str, Any]
) -> dict[str, Any]:
    runtime = review_dir / "agent-runtime"
    final = runtime / "final.md"
    from lib.agent_events import runtime_errors_failed

    if (
        record.get("result") != "completed"
        or record.get("exit_code") != 0
        or record.get("cleanup_complete") is not True
        or runtime_errors_failed(record)
    ):
        raise ValueError("review runtime did not complete and clean up")
    require_matching_runtime(review_dir, materials, record)
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


def run_review(
    state: dict[str, Any],
    directory: Path,
    gh: GitHub,
    *,
    retry_launch: Callable[[Callable[[], None]], AbstractContextManager[None]],
) -> dict[str, Any]:
    from dispatcher.runtime import gate
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
        saved = saved_request(review_dir, request)
        record = ended_runtime(state, review_dir / "agent-runtime")
        result = completed_result(review_dir, saved, record)
        save(execution_path, {**read(execution_path), "status": "completed"})
        return publish_result(state, directory, gh, saved, result_path, result)
    from lib.review_settings import choose, load_policy

    policy = load_policy(None)
    require_prior_failures(state, directory, review_dir)
    retry_heads = state["review"].get("retry_heads")
    materials = dict(request)
    materials["review_selection"] = state["review"].get(
        "selection", request["review_selection"]
    )
    upgrading = materials["review_selection"]["policy"] != policy
    if upgrading:
        if retry_heads is not None:
            raise ValueError(
                "review selection policy changed since the previous attempt"
            )
        # Old requests awaiting their first launch adopt the current Sol pin.
        materials["review_selection"] = choose(policy)
    materials["prs"] = []
    diffs = {}
    prs: list[dict[str, Any]] = []
    issues: set[int] = {request["parent_issue"]}
    dependency_prs: set[int] = set()
    for target in request["prs"]:
        pr = target_pr(gh, target, retry_heads)
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
    if (
        upgrading
        or materials["review_selection"]["policy"]
        != request["review_selection"]["policy"]
    ):
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
    scratch = workspace_path(root, request["review_run_id"])
    execution = {
        "status": "running",
        "agent_invocations": 1,
        "guarded_launch": True,
    }

    def verify_heads() -> None:
        for target in request["prs"]:
            target_pr(gh, target, retry_heads)

    @contextmanager
    def launch_retry() -> Iterator[None]:
        with retry_launch(verify_heads):
            save(execution_path, execution)
            yield

    with ExitStack() as launch:
        lock_fd = launch.enter_context(gate(review_dir / "launch.lock"))
        if lock_fd is None:
            raise ValueError("reviewer launcher is already active")
        # The runner inherits the lock even if the controller dies. A retry
        # commits its intent at launch, after the worktrees and final checks.
        if retry_heads is None:
            save(execution_path, execution)
        cwd = launch.enter_context(workspace(scratch, materials, context, diffs))
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
            pass_fds=(lock_fd,),
            launch_context=launch_retry if retry_heads is not None else nullcontext,
        )
    result = completed_result(review_dir, materials, read(runtime / "result.json"))
    reviewed(operation, result)
    save(execution_path, {**execution, "status": "completed"})
    return publish_result(state, directory, gh, materials, result_path, result)


def publish_result(
    state: dict[str, Any],
    directory: Path,
    gh: GitHub,
    materials: dict[str, Any],
    result_path: Path,
    result: dict[str, Any],
) -> dict[str, Any]:
    # Display copies are not merge evidence. Save the result first; a restarted
    # controller returns it without replaying publication or launching an agent.
    request = state["request"]
    root = Path(state["checkout"])
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
        if related := finding.get("related_prs"):
            targets = ", ".join(f"#{n}" for n in finding["prs"])
            references = ", ".join(f"#{n}" for n in related)
            body += f"\n  対象 PR {targets}。参照 PR {references}。"
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
