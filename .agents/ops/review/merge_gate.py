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
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from lib.review_settings import run_id
from parent.execution import authorization
from parent.policy import closing_issues, dependencies, nonempty, object_value
from parent.session import ParentSession
from parent.snapshot import snapshot
from parent.state import Store

from review.progression import Engine, new_state, read, save


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


def register(
    session: ParentSession,
    prs: list[int],
    instruction: str,
    reason: str | None,
    selection: dict[str, Any],
    options: dict[str, Any],
) -> dict[str, Any]:
    timeout = options.get("agent_timeout", 1800)
    if type(timeout) not in {int, float} or not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("agent timeout must be positive and finite")
    session.gh.targeted_history = True
    row = session.agreed()
    if not prs or len(set(prs)) != len(prs):
        raise ValueError("review needs a nonempty set of unique PRs")
    # Reusing the request cannot restart a paused controller. That always needs
    # merge-continue, even after the original process or WSL has disappeared.
    for prior in row["reviews"]:
        request = prior["request"]
        if request.get("schema_version") == 4 and set(prs) & {
            p["number"] for p in request["prs"]
        }:
            state = read(state_path(session.store, request["review_run_id"]))
            if state["status"] != "completed":
                if set(prs) != {p["number"] for p in request["prs"]}:
                    raise ValueError("PR is already managed by another progression")
                return {
                    "review_run_id": state["id"],
                    "reused": True,
                    "status": state["status"],
                }
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
            entry["worker"] = {
                "kind": "dispatcher" if row.get("dispatch") else "manual",
                "child": child,
                "delegation_id": task["delegation_id"],
                "progression_id": uuid.uuid4().hex,
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
    path = state_path(session.store, identifier)
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=False)
    save(path, state)
    row["reviews"].append({"request": request, "state": "requested"})
    session.store.save()
    from decision_logs.capture import current

    capture = current()
    if capture is not None:
        try:
            with capture.transaction():
                capture.row["review_run_id"] = identifier
                capture._save()
        except Exception:
            capture.mark_failed(lost_source="review.binding")
    return {"review_run_id": identifier, "reused": False, "status": "requested"}


def latest(store: Store, repo: str, parent: int, identifier: str | None = None) -> Path:
    with store.consistent_snapshot():
        row = store.parents.get(f"{repo.casefold()}#{parent}", {})
        found = [
            r["request"]["review_run_id"]
            for r in row.get("reviews", [])
            if r["request"].get("schema_version") == 4
        ]
    if identifier is None:
        if not found:
            raise ValueError("no automatic progression is registered")
        identifier = found[-1]
    if identifier not in found:
        raise ValueError("progression is not registered with this parent")
    return state_path(store, identifier)


def view(path: Path) -> dict[str, Any]:
    state = read(path)
    return {
        "review_run_id": state["id"],
        "status": state["status"],
        "review": state["review"],
        "prs": [
            {k: v for k, v in t.items() if k != "worker"}
            for t in state["targets"].values()
        ],
        "events": state["events"],
        "decisions": state["decisions"],
    }


def start(path: Path) -> None:
    # The child takes a nonblocking lifetime lock before doing any work. Two
    # competing launchers can never run two reviewers/workers or merge twice.
    state = read(path)
    with (path.parent / "controller.log").open("ab") as log:
        subprocess.Popen(
            [sys.executable, "-m", "review.merge_gate", "--state", str(path)],
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
    rereview: bool = False,
    prs: list[int] | None = None,
    review_selection: dict[str, Any] | None = None,
    review_config_file: str | None = None,
) -> None:
    from review.progression_backend import LiveBackend

    with controller_lock(path):
        state = read(path)
        if state["owner"] != identity:
            raise ValueError("only the registered parent may continue progression")
        backend = LiveBackend(state, store)
        backend.authorize()
        Engine(path, backend).continue_run(
            instruction,
            rereview=rereview,
            prs=prs,
            review_selection=review_selection,
            review_config_file=review_config_file,
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
    requests = [
        r["request"]
        for r in row.get("reviews", [])
        if r["request"].get("schema_version") == 4
    ]
    for request in reversed(requests):
        state = read(state_path(store, request["review_run_id"]))
        target = state["targets"].get(str(pr))
        if (
            target
            and target["stage"] == "completed"
            and target.get("issues_confirmed")
            and target.get("main_ci", {}).get("status") == "success"
        ):
            return object_value(target)
    raise ValueError(f"PR #{pr}: automatic post-merge confirmation is missing")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state", type=Path, required=True)
    args = parser.parse_args()
    from review.progression_backend import LiveBackend

    path = args.state.resolve()
    store = Store(path.parents[2])
    try:
        with controller_lock(path):
            state = read(path)
            backend = LiveBackend(state, store)
            from review.review_locks import targets

            with targets(store.directory, state):
                result = Engine(path, backend).run()
            print(json.dumps(view(path), ensure_ascii=False))
            return 0 if result["status"] == "completed" else 1
    except (ValueError, OSError) as exc:
        print(str(exc), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
