"""One explicitly requested judgment. Scheduling, reservation and launch live elsewhere."""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
from pathlib import Path
from typing import Any

from parent.execution import POLICY_VERSION, execution_input, record_autonomous, sha256
from parent.github import GitHub
from parent.policy import digest, nonempty, number, object_value
from parent.session import ParentSession
from parent.state import Store, state_directory

from .agent import RUNNER, judge
from .agent_input import write_agent_input
from .collect import SnapshotGitHub, assert_fresh, collect, is_container, material
from .records import DETAIL_KEYS, comment_body, ignored_comments, owned_comment
from .schema import REQUIREMENTS, execution_context, source_url, strings, validate

CANDIDATE_KEYS = {
    "schema_version",
    "repo",
    "parent",
    "route",
    "priority",
    "basis_sha256",
    "generation",
    "input_sha256",
    "target_sha256",
    "code_commit",
    "requirements",
    "dependencies",
    "rationale",
    "sources",
}


def inputs_for(snapshot: dict[str, Any], store: Store, n: int) -> dict[str, Any]:
    session = ParentSession(SnapshotGitHub(snapshot), store, n, None)
    ignored = ignored_comments(
        snapshot["issues"][str(n)]["comments"], session.login, session.gh.repo, n
    )
    inputs: dict[str, Any] = execution_input(
        session, ignored, execution_context(snapshot, n)
    )
    return inputs


def publish_decision(
    session: ParentSession, decision: dict[str, Any], snapshot: dict[str, Any]
) -> str:
    detail = {k: decision[k] for k in DETAIL_KEYS}
    detail["sources"] = [source_url(snapshot, s) for s in decision["sources"]]
    body = comment_body(session.gh.repo, session.parent, detail)
    session.public(body)
    existing = [
        c
        for c in session.gh.comments(session.parent)
        if owned_comment(c, session.login, session.gh.repo, session.parent)
    ]
    if len(existing) > 1:
        raise ValueError("multiple owned judgement comments; reconcile them explicitly")
    if existing and existing[0]["body"] == body:
        return "unchanged"
    if existing:
        session.gh.write(
            "PATCH",
            f"repos/{session.gh.repo}/issues/comments/{number(existing[0]['id'])}",
            {"body": body},
        )
        return "updated"
    session.gh.write("POST", session.endpoint + "/comments", {"body": body})
    return "created"


def assert_candidate_fresh(
    gh: GitHub,
    store: Store,
    root: Path,
    snapshot: dict[str, Any],
    candidate: dict[str, Any],
) -> None:
    """Assignment controller calls under the SAME lock as validate_candidate/reservation."""
    if set(candidate) != CANDIDATE_KEYS:
        raise ValueError("unknown or malformed candidate schema")
    n = number(candidate["parent"])
    number(candidate["priority"])
    number(candidate["generation"])
    sha256(candidate["basis_sha256"])
    if (
        type(candidate.get("schema_version")) is not int
        or candidate["schema_version"] != 1
        or type(snapshot.get("schema_version")) is not int
        or snapshot["schema_version"] != 1
        or n not in snapshot["subjects"]
        or candidate.get("repo") != snapshot["repo"]
        or candidate.get("input_sha256") != snapshot["input_sha256"]
        or snapshot["input_sha256"] != digest(material(snapshot))
        or candidate.get("target_sha256") != digest(material(snapshot, n))
        or candidate.get("code_commit") != snapshot["code"]["commit"]
    ):
        raise ValueError("candidate does not match the fixed judgement input")
    raw = snapshot["issues"][str(n)]
    route_kind = (
        "parent"
        if raw["children"] or "issue:parent" in {x["name"] for x in raw["labels"]}
        else "standalone"
    )
    if (
        raw["parent"] is not None
        or is_container(raw)
        or candidate["route"] != route_kind
        or not isinstance(candidate["dependencies"], list)
        or any(type(d) is not int for d in candidate["dependencies"])
        or candidate["dependencies"] != raw["dependencies"]
    ):
        raise ValueError("candidate route/dependencies differ from the collected issue")
    requirements = object_value(candidate["requirements"])
    if set(requirements) != REQUIREMENTS:
        raise ValueError("malformed candidate requirements")
    for key in REQUIREMENTS:
        nonempty(requirements[key], key)
    inputs = inputs_for(snapshot, store, n)
    if (
        inputs != snapshot["execution_inputs"][str(n)]
        or digest(
            autonomous_basis(
                requirements,
                nonempty(candidate["rationale"], "rationale"),
                strings(candidate["sources"], "sources"),
                inputs,
            )
        )
        != candidate["basis_sha256"]
    ):
        raise ValueError(
            "candidate requirements/sources do not match its execution basis"
        )
    assert_fresh(gh, store, root, snapshot, n)


def autonomous_basis(
    requirements: dict[str, Any],
    rationale: str,
    sources: list[str],
    inputs: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "kind": "autonomous_decision",
        **requirements,
        "rationale": rationale,
        "sources": sources,
        "input_sha256": digest(inputs),
        "policy_version": POLICY_VERSION,
    }


def run(
    gh: GitHub,
    store: Store,
    root: Path,
    run_dir: Path,
    *,
    apply: bool = False,
    timeout: float = 1800,
    runner: Path = RUNNER,
) -> dict[str, Any]:
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("timeout must be a finite positive number")
    root, run_dir = root.resolve(), run_dir.resolve()
    run_dir.mkdir(parents=True, mode=0o700, exist_ok=False)
    result: dict[str, Any] = {
        "schema_version": 1,
        "result": "failed",
        "input_sha256": None,
        "decisions": [],
        "candidates": [],
        "publications": [],
        "diagnostics": [],
    }
    try:
        with store.locked():
            snapshot = collect(gh, store, root)
            snapshot["input_sha256"] = digest(material(snapshot))
            result["input_sha256"] = snapshot["input_sha256"]
            snapshot["execution_inputs"] = {
                str(n): inputs_for(snapshot, store, n)
                for n in snapshot["subjects"]
                if not is_container(snapshot["issues"][str(n)])
            }
        Store.write_json(run_dir / "input.json", snapshot)
        write_agent_input(snapshot, run_dir / "agent-input")
        # Always one fresh turn, even if material equals a previous scheduled run.
        raw = judge(root, run_dir, timeout, runner)
        Store.write_json(run_dir / "agent-output.json", raw)
        decisions, diagnostics = validate(raw, snapshot)
        result["decisions"] = decisions
        result["diagnostics"].extend(diagnostics)
        with store.locked():
            assert_fresh(gh, store, root, snapshot)
            # Check public payloads before the first irreversible publication.
            for decision in decisions:
                ParentSession(gh, store, decision["issue"], None).public(decision)
        if not apply:
            result["result"] = "preview"
        else:
            candidates = []
            for decision in decisions:
                n = decision["issue"]
                if is_container(snapshot["issues"][str(n)]):
                    continue
                with store.locked():
                    assert_fresh(gh, store, root, snapshot, n)
                    session = ParentSession(gh, store, n, None)
                    action = publish_decision(session, decision, snapshot)
                    result["publications"].append({"issue": n, "action": action})
                    Store.write_json(run_dir / "result.json", result)
                    current = assert_fresh(gh, store, root, snapshot, n)
                    if decision["disposition"] != "startable":
                        continue
                    ignored = ignored_comments(
                        current["issues"][str(n)]["comments"], session.login, gh.repo, n
                    )
                    # Display-only comments may have changed since the agent read them.
                    inputs = execution_input(
                        ParentSession(SnapshotGitHub(current), store, n, None),
                        ignored,
                        execution_context(snapshot, n),
                    )
                    if inputs != snapshot["execution_inputs"][str(n)]:
                        raise ValueError(
                            "execution input differs from the agent's fixed input"
                        )
                    basis = autonomous_basis(
                        decision["requirements"],
                        decision["rationale"],
                        [source_url(snapshot, s) for s in decision["sources"]],
                        inputs,
                    )
                    recorded = record_autonomous(
                        session,
                        {"basis": basis, "inputs": inputs, "ignored_comments": ignored},
                    )
                    candidates.append(
                        {
                            "schema_version": 1,
                            "repo": snapshot["repo"],
                            **recorded,
                            "route": "parent"
                            if snapshot["issues"][str(n)]["children"]
                            or "issue:parent"
                            in {x["name"] for x in snapshot["issues"][str(n)]["labels"]}
                            else "standalone",
                            "priority": decision["priority"],
                            "input_sha256": snapshot["input_sha256"],
                            "target_sha256": digest(material(snapshot, n)),
                            "code_commit": snapshot["code"]["commit"],
                            "requirements": decision["requirements"],
                            "dependencies": decision["dependencies"],
                            "rationale": basis["rationale"],
                            "sources": basis["sources"],
                        }
                    )
            with store.locked():
                assert_fresh(gh, store, root, snapshot)
            result["result"] = "recorded"
            result["candidates"] = sorted(candidates, key=lambda c: c["priority"])
    except (
        ValueError,
        OSError,
        subprocess.SubprocessError,
        KeyError,
        TypeError,
    ) as exc:
        result["result"] = "failed"
        result["candidates"] = []
        result["diagnostics"].append(f"{type(exc).__name__}: {exc}")
    Store.write_json(run_dir / "result.json", result)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", help="owner/repo; default: gh repo view")
    parser.add_argument("--cwd", type=Path, default=Path.cwd())
    parser.add_argument(
        "--run-dir", type=Path, required=True, help="new private artifact directory"
    )
    parser.add_argument(
        "--parent-state-dir", type=Path, help="existing shared registry directory"
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="publish judgments/bases; never assign or launch parents",
    )
    parser.add_argument("--timeout", type=float, default=1800)
    parser.add_argument("--agent-exec", type=Path, default=RUNNER)
    args = parser.parse_args(argv)
    os.umask(0o077)
    try:
        result = run(
            GitHub(args.repo),
            Store(args.parent_state_dir or state_directory()),
            args.cwd,
            args.run_dir,
            apply=args.apply,
            timeout=args.timeout,
            runner=args.agent_exec.resolve(),
        )
    except (ValueError, OSError, subprocess.SubprocessError) as exc:
        result = {
            "schema_version": 1,
            "result": "failed",
            "candidates": [],
            "diagnostics": [f"{type(exc).__name__}: {exc}"],
        }
    # Detailed prose, CLI/session identifiers and absolute paths stay in private artifacts.
    print(
        json.dumps(
            {
                "schema_version": 1,
                "result": result["result"],
                "candidate_count": len(result["candidates"]),
                "diagnostic_count": len(result["diagnostics"]),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 2 if result["result"] == "failed" else 0


if __name__ == "__main__":
    raise SystemExit(main())
