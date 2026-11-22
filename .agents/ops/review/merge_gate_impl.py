"""Automatic review-to-merge lifecycle and its detached controller entry point."""

from __future__ import annotations

import argparse
import fcntl
import json
import math
import os
import subprocess
import sys
import time
import uuid
from collections.abc import Iterator
from contextlib import ExitStack, contextmanager
from pathlib import Path
from typing import Any

from lib.ops_review.progression import (
    KNOWLEDGE_MERGE,
    Engine,
    new_state,
    read,
    review_attempt,
    review_launch_directory,
    save,
)
from lib.review_settings import run_id
from parent.execution import authorization
from parent.policy import closing_issues, dependencies, nonempty, object_value
from parent.session import ParentSession
from parent.snapshot import snapshot
from parent.state import Store


def state_path(store: Store, identifier: str) -> Path:
    return store.directory / "progressions" / run_id(identifier) / "state.json"


@contextmanager
def controller_lock(path: Path) -> Iterator[None]:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    with path.with_suffix(".lock").open("a") as lock:
        os.chmod(lock.name, 0o600)
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError(
                "automatic progression controller is already running"
            ) from exc
        yield


def knowledge_states(store: Store, row: dict[str, Any]) -> Iterator[dict[str, Any]]:
    for identifier in row.get("knowledge_merges", []):
        yield read(state_path(store, identifier))


def registered(
    session: ParentSession, prs: list[int], *, kind: str = "review_and_merge"
) -> dict[str, Any] | None:
    if not prs or len(set(prs)) != len(prs):
        raise ValueError("review needs a nonempty set of unique PRs")
    # Reusing the request cannot restart a paused controller. That always needs
    # merge-continue, even after the original process or WSL has disappeared.
    wanted = set(prs)
    for key, owner in session.store.parents.items():
        if key.rpartition("#")[0] != session.gh.repo.casefold():
            continue
        requests = [
            prior["request"]
            for prior in owner.get("reviews", [])
            if prior["request"].get("schema_version") == 4
        ]
        requests.extend(
            state["request"] for state in knowledge_states(session.store, owner)
        )
        for request in reversed(requests):
            numbers = {p["number"] for p in request["prs"]}
            if not wanted.intersection(numbers):
                continue
            if (
                key != session.key
                or wanted != numbers
                or request.get("kind", "review_and_merge") != kind
            ):
                raise ValueError(
                    f"PR is already managed by another progression: {request['review_run_id']}"
                )
            state = read(state_path(session.store, request["review_run_id"]))
            return {
                "review_run_id": state["id"],
                "reused": True,
                "status": state["status"],
            }
    return None


def register(
    session: ParentSession,
    prs: list[int],
    instruction: str,
    reason: str | None,
    selection: dict[str, Any],
    options: dict[str, Any],
) -> dict[str, Any]:
    from lib.ops_review import knowledge_records

    timeout = options.get("agent_timeout", 1800)
    if type(timeout) not in {int, float} or not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("agent timeout must be positive and finite")
    session.gh.targeted_history = True
    row = session.agreed()
    existing = registered(session, prs)
    if existing:
        return existing
    auth = authorization(session)
    parent, children = session.work_items()
    entries = []
    for n in sorted(prs):
        pr = object_value(session.gh.get(f"repos/{session.gh.repo}/pulls/{n}"))
        issues = sorted(
            closing_issues(pr.get("body") or "", session.gh.repo) & set(children)
        )
        if pr.get("number") != n or pr.get("state") != "open" or not issues:
            raise ValueError("review requires related open PRs")
        if any(
            pr[side]["repo"]["full_name"].casefold() != session.gh.repo.casefold()
            for side in ("head", "base")
        ):
            raise ValueError("review requires same-repository PRs")
        with session.gh.io_scope():
            scope, _ = knowledge_records.assess(Path.cwd(), session.gh.repo, pr)
            if scope.records_only:
                raise ValueError(
                    f"PR #{n}: knowledge records only; use merge-knowledge --prs {n}"
                )
            entry, _ = snapshot(pr)
        entry["issues"] = issues
        deps = sorted(
            {
                d
                for child in issues
                for d in row["tasks"]
                .get(str(child), {})
                .get("dependencies", dependencies(children[child].body))
            }
            - set(issues)
        )
        entry["dependencies"] = [
            {"issue": d, "prs": [p["number"] for p in session.gh.pulls({d})]}
            for d in deps
        ]
        workers = [
            (child, row["tasks"].get(str(child), {}))
            for child in issues
            if row["tasks"].get(str(child), {}).get("delegated")
        ]
        if len(workers) > 1:
            raise ValueError(
                "PR has multiple creating workers; identify one before requesting review"
            )
        entry["worker"] = None
        if workers:
            child, task = workers[0]
            if task.get("agent") != "codex" or task["branch"] != entry["branch"]:
                raise ValueError(
                    "PR does not belong to the registered creating Codex worker"
                )
            from lib.ops_review.progression_workers import source as worker_source

            entry["worker"] = {
                "kind": "dispatcher" if row.get("dispatch") else "manual",
                "child": child,
                "delegation_id": task["delegation_id"],
                "progression_id": uuid.uuid4().hex,
                "source": worker_source(session, child),
            }
        entries.append(entry)
    identifier = f"parent-{session.parent}-{uuid.uuid4().hex}"
    request = {
        "schema_version": 4,
        "review_run_id": identifier,
        "repo": session.gh.repo,
        "parent_issue": session.parent,
        "authorization": auth,
        "review_selection": selection,
        "prs": entries,
        "instruction": nonempty(instruction, "review instruction"),
        "reason": reason or "親が対象PR全体のレビューと自動進行を依頼",
    }
    state = new_state(
        request,
        owner=nonempty(session.session, "parent identity"),
        generation=row["generation"],
    )
    state.update(checkout=str(Path.cwd()), options=options)
    # Parent registration serializes new requests. Existing legacy rows remain
    # untouched and can be inspected, but cannot authorize this progression.
    for entry in entries:
        worker = entry["worker"]
        if worker and worker["kind"] == "dispatcher":
            from dispatcher.repairs import register_progression

            if not options.get("dispatcher_state_dir"):
                raise ValueError(
                    "dispatcher progression requires --dispatcher-state-dir"
                )
            register_progression(
                session,
                worker["progression_id"],
                child=worker["child"],
                pr=entry["number"],
            )
    # Network reads released the registry lock. Recheck all owners immediately
    # before publishing the reservation, under the same lock as the registry save.
    existing = registered(session, prs)
    if existing:
        return existing
    path = state_path(session.store, identifier)
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=False)
    save(path, state)
    row["reviews"].append({"request": request, "state": "requested"})
    row["latest_progression_id"] = identifier
    session.store.save()
    from decision_logs.review_capture import bind_request

    bind_request(identifier)
    return {"review_run_id": identifier, "reused": False, "status": "requested"}


def latest(store: Store, repo: str, parent: int, identifier: str | None = None) -> Path:
    with store.consistent_snapshot():
        row = store.parents.get(f"{repo.casefold()}#{parent}", {})
        found = [
            r["request"]["review_run_id"]
            for r in row.get("reviews", [])
            if r["request"].get("schema_version") == 4
        ]
        found.extend(row.get("post_merge_confirmations", []))
        found.extend(row.get("knowledge_merges", []))
    if identifier is None:
        if not found:
            raise ValueError("no automatic progression is registered")
        identifier = row.get("latest_progression_id", found[-1])
    if identifier not in found:
        raise ValueError("progression is not registered with this parent")
    return state_path(store, identifier)


def review_usage(path: Path, state: dict[str, Any]) -> dict[str, Any]:
    """One row per launch, including legacy rounds. Missing measurements stay null."""
    rounds = []
    for directory in sorted(path.parent.glob("review-*"), key=lambda d: d.name):
        execution_path = directory / "execution.json"
        if not execution_path.exists():
            continue
        review_round = int(directory.name.removeprefix("review-"))
        try:
            latest_attempt = review_attempt(execution_path)
        except (OSError, ValueError):
            rounds.append(
                {
                    "round": review_round,
                    "started_at": None,
                    "error": "execution_record_unavailable",
                }
            )
            continue
        for attempt in range(latest_attempt + 1):
            runtime = (
                review_launch_directory(directory, attempt)
                / "agent-runtime/result.json"
            )
            row: dict[str, Any] = {
                "round": review_round,
                "attempt": attempt,
                "started_at": None,
            }
            try:
                record = read(runtime) if runtime.is_file() else {}
            except (OSError, ValueError):
                row["error"] = "runtime_record_unavailable"
            else:
                usage = (record.get("runtime") or {}).get("usage", {}).get(
                    "value"
                ) or {}
                start, finish = record.get("started_at"), record.get("finished_at")
                row.update(
                    started_at=start,
                    agent_spawned=record.get("agent_spawned"),
                    duration_seconds=finish - start
                    if finish is not None and start is not None
                    else None,
                    **{
                        key: usage.get(key)
                        for key in (
                            "input_tokens",
                            "cached_input_tokens",
                            "output_tokens",
                        )
                    },
                )
            rounds.append(row)
    starts = [r["started_at"] for r in rounds if r["started_at"] is not None]
    return {
        "launch_intents": len(rounds),
        "confirmed_launches": sum(r.get("agent_spawned") is True for r in rounds),
        "rounds": rounds,
        "seconds_to_observed_merge": {
            n: t["merged_at"] - min(starts)
            if starts and t.get("merged_at") is not None
            else None
            for n, t in state["targets"].items()
        },
    }


def view(path: Path) -> dict[str, Any]:
    state = read(path)
    return {
        "review_run_id": state["id"],
        "status": state["status"],
        "kind": state["request"].get("kind", "review_and_merge"),
        **(
            {"evidence": state["request"]["evidence"]}
            if state["request"].get("kind") == KNOWLEDGE_MERGE
            else {}
        ),
        "review": state["review"],
        "prs": [
            {k: v for k, v in t.items() if k != "worker"}
            for t in state["targets"].values()
        ],
        "events": state["events"],
        "review_usage": review_usage(path, state),
        "decisions": state["decisions"],
    }


def start(path: Path) -> None:
    # The child takes a nonblocking lifetime lock before doing any work. Two
    # competing launchers can never run two reviewers/workers or merge twice.
    state = read(path)
    with (path.parent / "controller.log").open("ab") as log:
        subprocess.Popen(
            [
                sys.executable,
                str(Path(__file__).with_name("merge_gate.py")),
                "--state",
                str(path),
            ],
            cwd=state["checkout"],
            env={
                **os.environ,
                "PYTHONPATH": os.pathsep.join(
                    filter(
                        None,
                        [
                            str(Path(__file__).resolve().parents[1]),
                            os.environ.get("PYTHONPATH"),
                        ],
                    )
                ),
            },
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=log,
            start_new_session=True,
        )


def continue_progression(
    path: Path,
    store: Store,
    identity: str,
    instruction: str,
    *,
    prs: list[int] | None = None,
) -> None:
    from lib.ops_review.knowledge_progression import refresh_evidence
    from lib.ops_review.progression_backend import LiveBackend
    from lib.ops_review.progression_workers import refresh
    from lib.ops_review.review_locks import lease_targets
    from parent.review_handoff import require_followup_ended

    with controller_lock(path):
        state = read(path)
        backend = LiveBackend(state, store)
        with store.consistent_snapshot():
            session = ParentSession(backend.gh, store, backend.parent, identity)
            current = authorization(session)
            # A partial continuation still starts the controller for the
            # entire progression, whose leases cover every registered PR.
            for target in lease_targets(state):
                for child in target["issues"]:
                    require_followup_ended(session.row, child)
            saved = state["request"]["authorization"]
            if current != {**saved, "generation": current["generation"]}:
                raise ValueError("progression execution authority changed")
            if (
                state["owner"] != identity
                or state["generation"] != current["generation"]
            ):
                if current["generation"] <= state["generation"] or not any(
                    event.get("event") == "handoff"
                    and event.get("generation") == current["generation"]
                    for event in session.history()
                ):
                    raise ValueError(
                        "review progression requires a recorded owner handoff"
                    )
                state.setdefault("owner_history", []).append(
                    {"owner": state["owner"], "generation": state["generation"]}
                )
                state.update(owner=identity, generation=current["generation"])
                state["request"]["authorization"] = current
        backend.authorize()
        engine = Engine(path, backend)
        # Carry the ownership handoff checked above into the next saved state.
        engine.state = state

        def prepare(value: dict[str, Any]) -> None:
            if engine.knowledge:
                refresh_evidence(backend, value, prs, instruction)
            else:
                refresh(backend, value, prs, instruction)

        engine.continue_run(
            instruction,
            prs=prs,
            prepare=prepare,
        )
    start(path)


def wait(path: Path, *, after: int = 0, timeout: float | None = None) -> dict[str, Any]:
    if timeout is not None and (not math.isfinite(timeout) or timeout <= 0):
        raise ValueError("timeout must be positive")
    deadline = time.monotonic() + timeout if timeout is not None else None
    while True:
        state = read(path)
        if (
            any(e["id"] > after for e in state["events"])
            or state["status"] == "completed"
        ):
            return view(path)
        if deadline is not None and time.monotonic() >= deadline:
            return {**view(path), "wait_timed_out": True}
        time.sleep(0.25)


def completion(store: Store, repo: str, parent: int, pr: int) -> dict[str, Any]:
    row = store.read_data().get("parents", {}).get(f"{repo.casefold()}#{parent}", {})
    identifiers = [
        r["request"]["review_run_id"]
        for r in row.get("reviews", [])
        if r["request"].get("schema_version") == 4
    ]
    identifiers.extend(row.get("post_merge_confirmations", []))
    identifiers.extend(row.get("knowledge_merges", []))
    for identifier in reversed(identifiers):
        state = read(state_path(store, identifier))
        target = state["targets"].get(str(pr))
        if (
            target
            and target["stage"] == "completed"
            and (
                target.get("issues_confirmed")
                or (
                    state["request"].get("kind") == KNOWLEDGE_MERGE
                    and target.get("implementation_pr")
                    and not target["issues"]
                )
            )
            and target.get("main_ci", {}).get("status") == "success"
        ):
            return object_value(target)
    raise ValueError(f"PR #{pr}: automatic post-merge confirmation is missing")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state", type=Path, required=True)
    args = parser.parse_args()
    from lib.ops_review.progression_backend import LiveBackend
    from lib.ops_review.review_locks import PRLeaseConflict, targets

    path = args.state.resolve()
    store = Store(path.parents[2])
    try:
        with controller_lock(path), ExitStack() as leases:
            state = read(path)
            engine = Engine(path, LiveBackend(state, store))
            try:
                leases.enter_context(targets(store.directory, state))
            except PRLeaseConflict as exc:
                engine.block_start(exc.number, str(exc))
                result = engine.state
            else:
                result = engine.run()
            print(json.dumps(view(path), ensure_ascii=False))
            return 0 if result["status"] == "completed" else 1
    except (ValueError, OSError) as exc:
        print(str(exc), file=sys.stderr)
        return 2
