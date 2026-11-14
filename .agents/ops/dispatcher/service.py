"""Scheduled judgments: persist intent, observe outside registry, then reconcile."""

from __future__ import annotations

import copy
import os
import signal
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from parent.github import GitHub
from parent.policy import nonempty
from parent.state import Store

from . import judgement
from .outcomes import read_result, validate_result
from .runtime import Runtime, alive, gate, load, process_identity
from .schedule import Schedule, latest_slot
from .state import DispatchStore, require_client


def require_store(store: Store, schedule: Schedule) -> None:
    if not isinstance(store, DispatchStore) or store.schedule.path != schedule.path:
        raise ValueError(
            "dispatcher requires the supplied DispatchStore and its schedule"
        )


def active_run(state: dict[str, Any]) -> dict[str, Any]:
    return next(r for r in state["runs"] if r["id"] == state["active"])


def same_run(
    store: Store, state: dict[str, Any], expected: dict[str, Any]
) -> dict[str, Any]:
    if state["active"] != expected["id"] or active_run(state) != expected:
        store.conflict("judgment run changed during runner observation")
    return active_run(state)


def runtime_identity(run: dict[str, Any]) -> str | None:
    path = Path(run["directory"]) / "agent-runtime/result.json"
    return (
        nonempty(load(path).get("run_id"), "judgment runtime run ID")
        if path.exists()
        else None
    )


def verify_runtime(store: Store, run: dict[str, Any], expected: str | None) -> None:
    try:
        current = runtime_identity(run)
    except (ValueError, OSError, KeyError, TypeError):
        store.conflict("judgment runtime could not be revalidated")
    if current != expected:
        store.conflict("judgment runtime run changed during observation")


def reconcile_judgement(state: dict[str, Any]) -> bool:
    """Observe a frozen schedule outside the registry. No durable writes here."""
    if state["active"] is None:
        return True
    run = active_run(state)
    if alive(run["controller"]):
        return False
    directory = Path(run["directory"])
    runtime_dir = directory / "agent-runtime"
    if (runtime_dir / "result.json").exists():
        runtime = Runtime(Path(state["config"]["runner"]))
        evidence = runtime.inspect(runtime_dir)
        if not runtime.ended(evidence):
            run.update(
                phase="recovery_required",
                diagnostic="前回判定runtimeの終了・回収を確認する",
            )
            return False
    if (directory / "result.json").exists():
        result = read_result(directory)
        if result["result"] != "failed" and not (runtime_dir / "result.json").exists():
            run.update(
                phase="recovery_required", diagnostic="保存済み判定のruntime証拠が欠落"
            )
            return False
        run.update(
            phase=result["result"],
            diagnostic=" / ".join(
                ["controller中断後に保存済み結果を照合", *result["diagnostics"]]
            ),
        )
        state["active"] = None
        return True
    run.update(
        phase="recovery_required",
        diagnostic="前回runの結果欠落。時刻で終了扱いにせず明示復旧する",
    )
    return False


def run_due(
    gh: GitHub,
    store: Store,
    schedule: Schedule,
    now: datetime,
    *,
    explicit: bool = False,
) -> dict[str, Any]:
    require_store(store, schedule)
    require_client(gh, store)
    with gate(schedule.directory / "judgement.lock") as fd:
        if fd is None:
            return {"result": "busy"}
        with store.locked():
            state = schedule.read()
            if not state["enabled"]:
                return {"result": "disabled"}
            if state["active"] is not None:
                active_run(state)["diagnostic"] = "controller/runtime照合要求を保存済み"
                schedule.write(state)
            expected = copy.deepcopy(state)
            previous_runtime = (
                runtime_identity(active_run(state)) if state["active"] else None
            )
            checkpoint = store.checkpoint()
        ready = reconcile_judgement(state)
        with store.locked():
            store.revalidate(checkpoint)
            if schedule.read() != expected:
                store.conflict(
                    "dispatcher schedule changed during judgment reconciliation"
                )
            if expected["active"] is not None:
                verify_runtime(store, active_run(expected), previous_runtime)
            if not ready:
                schedule.write(state)
                return {"result": "recovery_required"}
            slot = latest_slot(now).isoformat()
            if not explicit and slot <= state["cursor"]:
                schedule.write(state)
                return {"result": "not_due"}
            identifier = uuid.uuid4().hex
            directory = schedule.directory / "judgements" / identifier
            directory.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            run = {
                "id": identifier,
                "slot": slot,
                "explicit": explicit,
                "phase": "running",
                "controller": process_identity(os.getpid()),
                "directory": str(directory),
                "diagnostic": None,
                "candidates": {},
            }
            state["runs"].append(run)
            state["cursor"] = max(state["cursor"], slot)
            state["active"] = identifier
            schedule.write(state)
            expected_run = copy.deepcopy(run)
            config = copy.deepcopy(state["config"])

        was_interrupted = False

        def interrupted(signum: int, frame: object) -> None:
            # The signal handler must never call a waiting runner while a
            # transaction happens to be active. Reconcile via an explicit stop.
            nonlocal was_interrupted
            was_interrupted = True
            raise InterruptedError("scheduled judgment was explicitly stopped")

        old = {
            sig: signal.signal(sig, interrupted)
            for sig in (signal.SIGTERM, signal.SIGINT)
        }
        try:
            result = validate_result(
                judgement.run(
                    gh,
                    store,
                    Path(config["cwd"]),
                    directory,
                    apply=True,
                    timeout=config["judgement_timeout"],
                    runner=Path(config["runner"]),
                )
            )
        finally:
            for sig, handler in old.items():
                signal.signal(sig, handler)
        with store.locked():
            state = schedule.read()
            run = same_run(store, state, expected_run)
            run["diagnostic"] = "判定runtimeの終了照合要求を保存済み"
            schedule.write(state)
            expected_run = copy.deepcopy(run)
            expected_runtime = runtime_identity(run)
        diagnostic = None
        runtime_dir = directory / "agent-runtime"
        try:
            if (runtime_dir / "result.json").exists():
                runtime = Runtime(Path(config["runner"]))
                if not runtime.ended(
                    runtime.inspect(runtime_dir, stop=was_interrupted)
                ):
                    diagnostic = "judgment runtime cleanup remains unconfirmed"
            elif was_interrupted:
                diagnostic = "interrupted startup has no runtime evidence"
        except (ValueError, OSError) as exc:
            diagnostic = str(exc)
        with store.locked():
            state = schedule.read()
            run = same_run(store, state, expected_run)
            verify_runtime(store, run, expected_runtime)
            run.update(
                phase=result["result"],
                diagnostic=" / ".join(result["diagnostics"]) or None,
            )
            state["active"] = None
            if diagnostic is not None:
                run.update(phase="recovery_required", diagnostic=diagnostic)
                state["active"] = identifier
            schedule.write(state)
        return {
            "result": run["phase"],
            "run_id": identifier,
            "deferred": result["deferred"],
        }


def stop_judgement(store: Store, schedule: Schedule) -> None:
    require_store(store, schedule)
    with store.locked():
        state = schedule.read()
        state["enabled"] = False
        if state["active"] is None:
            schedule.write(state)
            return
        run = active_run(state)
        run.update(
            phase="recovery_required",
            diagnostic="停止要求済み。controllerとruntimeを照合して明示復旧する",
        )
        schedule.write(state)
        expected = copy.deepcopy(run)
        expected_runtime = runtime_identity(run)
        runner = Path(state["config"]["runner"])
    runtime_dir = Path(expected["directory"]) / "agent-runtime"
    if (runtime_dir / "result.json").exists():
        Runtime(runner).inspect(runtime_dir, stop=True)
    if alive(expected["controller"]):
        os.kill(expected["controller"]["pid"], signal.SIGTERM)
    with store.locked():
        current = same_run(store, schedule.read(), expected)
        verify_runtime(store, current, expected_runtime)


def recover_judgement(store: Store, schedule: Schedule, evidence: str) -> None:
    require_store(store, schedule)
    evidence = nonempty(evidence, "recovery evidence")
    with gate(schedule.directory / "judgement.lock") as fd:
        if fd is None:
            raise ValueError(
                "judgment controller is still running; stop it before recovery"
            )
        with store.locked():
            state = schedule.read()
            if state["active"] is None:
                raise ValueError("no interrupted judgment to recover")
            run = active_run(state)
            if alive(run["controller"]):
                raise ValueError("judgment controller is still alive")
            run.update(
                phase="recovery_required", diagnostic="明示復旧の終了照合を要求済み"
            )
            schedule.write(state)
            expected = copy.deepcopy(run)
            expected_runtime = runtime_identity(run)
            runner = Path(state["config"]["runner"])
        runtime_dir = Path(expected["directory"]) / "agent-runtime"
        if (runtime_dir / "result.json").exists():
            runtime = Runtime(runner)
            if not runtime.ended(runtime.inspect(runtime_dir, stop=True)):
                raise ValueError("judgment runtime cleanup remains unconfirmed")
        with store.locked():
            state = schedule.read()
            run = same_run(store, state, expected)
            verify_runtime(store, run, expected_runtime)
            if alive(run["controller"]):
                raise ValueError("judgment controller is still alive")
            run.update(phase="failed", diagnostic=evidence)
            state["active"] = None
            schedule.write(state)
