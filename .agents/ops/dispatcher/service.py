"""Scheduled judgment entry and recovery; no issue-change notification path."""

from __future__ import annotations

import os
import signal
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from parent.github import GitHub
from parent.state import Store

from . import judgement
from .runtime import Runtime, alive, gate, load, process_identity
from .schedule import Schedule, latest_slot


def reconcile_judgement(state: dict[str, Any]) -> bool:
    """False means the previous judgment cannot yet be superseded."""
    if state["active"] is None:
        return True
    run = next(r for r in state["runs"] if r["id"] == state["active"])
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
        result = load(directory / "result.json")
        if result.get("schema_version") == 1 and result.get("result") in {
            "recorded",
            "failed",
        }:
            if (
                result["result"] == "recorded"
                and not (runtime_dir / "result.json").exists()
            ):
                run.update(
                    phase="recovery_required",
                    diagnostic="記録済み判定のruntime証拠が欠落",
                )
                return False
            run.update(
                phase=result["result"],
                diagnostic="controller中断後に保存済み結果を照合",
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
    with gate(schedule.directory / "judgement.lock") as fd:
        if fd is None:
            return {"result": "busy"}
        with store.locked():
            state = schedule.read()
            if not state["enabled"]:
                return {"result": "disabled"}
            if not reconcile_judgement(state):
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
            # Consume only the latest missing slot, never replay a backlog.
            state["cursor"] = max(state["cursor"], slot)
            state["active"] = identifier
            schedule.write(state)
            config = state["config"]

        was_interrupted = False

        def interrupted(signum: int, frame: object) -> None:
            nonlocal was_interrupted
            was_interrupted = True
            runtime_dir = directory / "agent-runtime"
            if (runtime_dir / "result.json").exists():
                Runtime(Path(config["runner"])).inspect(runtime_dir, stop=True)
            raise InterruptedError("scheduled judgment was explicitly stopped")

        old = {
            sig: signal.signal(sig, interrupted)
            for sig in (signal.SIGTERM, signal.SIGINT)
        }
        try:
            result = judgement.run(
                gh,
                store,
                Path(config["cwd"]),
                directory,
                apply=True,
                timeout=config["judgement_timeout"],
                runner=Path(config["runner"]),
            )
        finally:
            for sig, handler in old.items():
                signal.signal(sig, handler)
        with store.locked():
            state = schedule.read()
            run = next(r for r in state["runs"] if r["id"] == identifier)
            run.update(
                phase=result["result"],
                diagnostic=" / ".join(result["diagnostics"]) or None,
            )
            state["active"] = None
            runtime_dir = directory / "agent-runtime"
            try:
                if (runtime_dir / "result.json").exists():
                    runtime = Runtime(Path(config["runner"]))
                    if not runtime.ended(runtime.inspect(runtime_dir)):
                        raise ValueError("judgment runtime cleanup remains unconfirmed")
                elif was_interrupted:
                    raise ValueError("interrupted startup has no runtime evidence")
            except (ValueError, OSError) as exc:
                run.update(phase="recovery_required", diagnostic=str(exc))
                state["active"] = identifier
            schedule.write(state)
        return {"result": run["phase"], "run_id": identifier}


def stop_judgement(store: Store, schedule: Schedule) -> None:
    with store.locked():
        state = schedule.read()
        state["enabled"] = False
        schedule.write(state)
        if state["active"] is None:
            return
        run = next(r for r in state["runs"] if r["id"] == state["active"])
        runtime_dir = Path(run["directory"]) / "agent-runtime"
        if (runtime_dir / "result.json").exists():
            Runtime(Path(state["config"]["runner"])).inspect(runtime_dir, stop=True)
        if alive(run["controller"]):
            os.kill(run["controller"]["pid"], signal.SIGTERM)
        run.update(
            phase="recovery_required",
            diagnostic="停止要求済み。controllerとruntimeを照合して明示復旧する",
        )
        schedule.write(state)


def recover_judgement(store: Store, schedule: Schedule, evidence: str) -> None:
    from parent.policy import nonempty

    with gate(schedule.directory / "judgement.lock") as fd:
        if fd is None:
            raise ValueError(
                "judgment controller is still running; stop it before recovery"
            )
        with store.locked():
            state = schedule.read()
            if state["active"] is None:
                raise ValueError("no interrupted judgment to recover")
            run = next(r for r in state["runs"] if r["id"] == state["active"])
            if alive(run["controller"]):
                raise ValueError("judgment controller is still alive")
            runtime_dir = Path(run["directory"]) / "agent-runtime"
            if (runtime_dir / "result.json").exists():
                runtime = Runtime(Path(state["config"]["runner"]))
                if not runtime.ended(runtime.inspect(runtime_dir, stop=True)):
                    raise ValueError("judgment runtime cleanup remains unconfirmed")
            run.update(
                phase="failed", diagnostic=nonempty(evidence, "recovery evidence")
            )
            state["active"] = None
            schedule.write(state)
