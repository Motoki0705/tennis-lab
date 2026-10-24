"""Purpose-driven fake parent and native transcripts; no model or GitHub writes."""

from __future__ import annotations

import hashlib
import importlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any


def stamp() -> str:
    from datetime import UTC, datetime

    return datetime.now(UTC).isoformat()


class FakeNative:
    """Emit the same record shape as the anonymized real native probe."""

    def __init__(self, runtime: Path, cwd: Path) -> None:
        self.runtime = runtime
        self.runtime.mkdir(parents=True, exist_ok=True)
        self.home = Path(os.environ["CODEX_HOME"])
        self.directory = self.home / "sessions"
        self.directory.mkdir(parents=True, exist_ok=True)
        suffix = hashlib.sha256(str(runtime).encode()).hexdigest()[:16]
        self.parent = f"parent-{suffix}"
        self.turn = f"turn-{suffix}"
        self.cwd = str(cwd)
        self.children: dict[str, list[dict[str, Any]]] = {}
        self.rows = [
            self.row(
                "session_meta",
                {
                    "id": self.parent,
                    "session_id": self.parent,
                    "source": "exec",
                    "cwd": self.cwd,
                },
            ),
            self.row(
                "event_msg",
                {
                    "type": "task_started",
                    "turn_id": self.turn,
                    "root_turn_id": self.turn,
                },
            ),
            self.row("turn_context", self.context(self.turn)),
        ]
        self.result = {
            "schema_version": 1,
            "result": "running",
            "runner": None,
            "process": None,
            "cleanup_complete": True,
            "contract": {
                "settings": {
                    "agent": "codex",
                    "selected": {
                        "model": "gpt-6-astra",
                        "effort": "max",
                        "service_tier": "default",
                    },
                },
                "mode": "read-only",
                "cwd": self.cwd,
                "runtime_home": str(self.home),
            },
            "runtime": {"session_id": self.parent, "turn_id": self.turn},
        }
        self.save()

    def row(self, kind: str, payload: dict[str, Any]) -> dict[str, Any]:
        return {"timestamp": stamp(), "type": kind, "payload": payload}

    def context(self, turn: str) -> dict[str, Any]:
        return {
            "turn_id": turn,
            "cwd": self.cwd,
            "model": "gpt-6-astra",
            "effort": "max",
            "sandbox_policy": {"type": "read-only"},
            "approval_policy": "never",
        }

    def child_id(self, name: str) -> str:
        return f"{self.parent}-{name}"

    def save(self) -> None:
        (self.runtime / "result.json").write_text(json.dumps(self.result))
        self.write(self.directory / f"rollout-{self.parent}.jsonl", self.rows)
        for name, rows in self.children.items():
            self.write(self.directory / f"rollout-{self.child_id(name)}.jsonl", rows)

    @staticmethod
    def write(path: Path, rows: list[dict[str, Any]]) -> None:
        path.write_text("".join(json.dumps(r) + "\n" for r in rows))

    def spawn(self, name: str, *, rejected: bool = False) -> None:
        call_id = f"spawn-{name}-{len(self.rows)}"
        self.rows.append(
            self.row(
                "response_item",
                {
                    "type": "function_call",
                    "namespace": "collaboration",
                    "name": "spawn_agent",
                    "call_id": call_id,
                    "arguments": json.dumps(
                        {
                            "task_name": name,
                            "message": f"Investigate {name}",
                            "fork_turns": "none",
                        }
                    ),
                },
            )
        )
        output = (
            "collab spawn failed: agent thread limit reached"
            if rejected
            else json.dumps({"task_name": f"/root/{name}"})
        )
        self.rows.append(
            self.row(
                "response_item",
                {"type": "function_call_output", "call_id": call_id, "output": output},
            )
        )
        if not rejected:
            agent_path = f"/root/{name}"
            turn = f"turn-{name}"
            self.children[name] = [
                self.row(
                    "session_meta",
                    {
                        "id": self.child_id(name),
                        "session_id": self.parent,
                        "cwd": self.cwd,
                        "source": {
                            "subagent": {
                                "thread_spawn": {
                                    "parent_thread_id": self.parent,
                                    "depth": 1,
                                    "agent_path": agent_path,
                                }
                            }
                        },
                        "parent_thread_id": self.parent,
                        "agent_path": agent_path,
                    },
                ),
                self.row(
                    "event_msg",
                    {
                        "type": "task_started",
                        "turn_id": turn,
                        "root_turn_id": self.turn,
                        "turn_attribution": {
                            "parent_turn_id": self.turn,
                            "root_turn_id": self.turn,
                            "turn_id": turn,
                        },
                    },
                ),
                self.row("turn_context", self.context(turn)),
            ]
        self.save()

    def finish_child(self, name: str, result: str) -> None:
        self.children[name] += [
            self.row(
                "response_item",
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": result}],
                },
            ),
            self.row(
                "event_msg",
                {
                    "type": "task_complete",
                    "turn_id": f"turn-{name}",
                    "root_turn_id": self.turn,
                    "last_agent_message": result,
                },
            ),
        ]
        self.save()

    def finish(self) -> None:
        self.rows.append(
            self.row(
                "event_msg",
                {
                    "type": "task_complete",
                    "turn_id": self.turn,
                    "root_turn_id": self.turn,
                    "last_agent_message": "parent done",
                },
            )
        )
        self.result["result"] = "completed"
        self.save()


def main() -> None:
    prompt, source, output = (Path(p) for p in sys.argv[1:])
    connection = json.loads(Path(os.environ["FAKE_AUDIT_MCP_CONFIG"]).read_text())[
        "ops_report"
    ]
    sys.path.insert(0, connection["env"]["PYTHONPATH"])
    AuditTools = importlib.import_module("opsreport.audit_tools").AuditTools
    Retrieval = importlib.import_module("opsreport.retrieval").Retrieval

    args = connection["args"]

    def option(name: str) -> str:
        return str(args[args.index(name) + 1])

    repo = Path(option("--repo"))
    assert Path.cwd() == repo
    runtime = FakeNative(Path(option("--runtime-dir")), repo)
    server = AuditTools(
        Retrieval(
            repo,
            option("--base-sha"),
            option("--repository"),
            option("--round-id"),
            Path(option("--journal-dir")),
            ops_memory_sha=option("--ops-memory-sha"),
        ),
        ledger_path=Path(option("--ledger-path")),
        runtime_dir=runtime.runtime,
        input_path=Path(option("--input-path")),
        ending_path=Path(option("--ending-path")),
        max_threads=int(option("--max-threads")),
    )

    def call(name: str, **kw: Any) -> Any:
        result = server.call(name, kw)
        if not result["ok"]:
            raise ValueError(f"{name}: {result['error']}")
        return result["data"]

    def plan(name: str, question: str, origin: dict[str, str]) -> None:
        if name not in server.book().snapshot["tasks"]:
            call(
                "audit_plan_task",
                id=name,
                question=question,
                scope=["big.py"],
                origin=origin,
            )

    def spawn(name: str) -> None:
        runtime.spawn(name)
        call("audit_start_task", id=name)

    def done(name: str, conclusion: str, *, failed: bool = False) -> None:
        time.sleep(0.002)
        runtime.finish_child(name, conclusion)
        evidence = []
        if not failed:
            response = server.call(
                "code_file", {"path": "big.py", "start_line": 2, "line_count": 3}
            )
            assert response["ok"], response
            key = f"e_{name}_{runtime.parent}"
            call("audit_evidence", id=key, receipt=response["receipt"])
            evidence = [key]
        call(
            "audit_finish_task",
            id=name,
            status="failed" if failed else "completed",
            conclusion=conclusion,
            evidence=evidence,
            next_questions=[],
            unconfirmed=[
                {
                    "scope": "big.py",
                    "reason": "ツール失敗で未確認",
                    "resume": "同じIDで再開",
                }
            ]
            if failed
            else [],
        )

    raw = source.read_text()
    try:
        data = json.loads(raw)
    except ValueError:
        runtime.finish()
        output.write_text(raw)
        return
    origin = {"kind": "goal", "id": option("--round-id")}
    plan("initial", "全体構造から次の検証対象を選ぶ", origin)
    plan("independent", "独立した影響範囲を確認する", origin)
    if os.environ.get("FAKE_AUDIT_SPAWN_UNAVAILABLE"):
        for name in ("initial", "independent"):
            call(
                "audit_finish_task",
                id=name,
                status="failed",
                conclusion="native spawn unavailable",
                evidence=[],
                next_questions=[],
                unconfirmed=[
                    {
                        "scope": "big.py",
                        "reason": "native spawn unavailable",
                        "resume": "起動機能を修復して同じIDで再開",
                    }
                ],
            )
        call(
            "audit_end",
            status="incomplete",
            reason="起動機能が利用不能",
            checked_scope=[],
            excluded_scope=[],
            unverified_risks=[
                {
                    "scope": "big.py",
                    "reason": "起動不能",
                    "resume": "native機能を修復する",
                }
            ],
            further_investigation=["起動不能の解消"],
        )
        runtime.finish()
        output.write_text(json.dumps({"memory_update": None}))
        return
    pending = server.book().snapshot["tasks"]
    if pending["initial"]["status"] != "completed":
        spawn("initial")
    if pending["independent"]["status"] != "completed":
        spawn("independent")
    if os.environ.get("FAKE_AUDIT_REJECT_ONCE"):
        plan("busy", "別の独立した調査を継続する", origin)
        spawn("busy")
        plan("capacity", "枠が空いた後に独立調査を行う", origin)
        runtime.spawn("capacity", rejected=True)
        call(
            "audit_finish_task",
            id="capacity",
            status="failed",
            conclusion="agent thread limit reached",
            evidence=[],
            next_questions=[],
            unconfirmed=[
                {
                    "scope": "big.py",
                    "reason": "枠不足",
                    "resume": "先行子の完了を受領してから再試行",
                }
            ],
        )
    if os.environ.get("FAKE_AGENT_HANG"):
        time.sleep(300)
    if os.environ.get("FAKE_AGENT_FAIL"):
        raise RuntimeError("fake agent: quota exhausted")
    discovery = os.environ.get("FAKE_AUDIT_DISCOVERY", "dynamic")
    if pending["initial"]["status"] != "completed":
        done("initial", discovery)
    if os.environ.get("FAKE_AUDIT_REJECT_ONCE"):
        spawn("capacity")
        done("capacity", "空き枠で確認できた")
    # The first returned result, including a prior verified result on resume,
    # decides the next question. There is no fixed area/round count in production.
    first = server.book().snapshot["tasks"]["initial"]["conclusion"]
    initial = server.book().snapshot["tasks"]["initial"]
    initial_finding = {
        "id": "initial_finding",
        "task_id": "initial",
        "kind": "design",
        "status": "suspected",
        "conclusion": f"初期結果 {first} の利用経路に疑いがある",
        "evidence": initial["evidence"],
        "counterevidence": [],
        "details": {},
    }
    if "initial_finding" not in server.book().snapshot["findings"]:
        call("audit_finding", record=initial_finding)
    previous = "initial"
    for question in (
        f"refs_{first}",
        f"config_{first}",
        f"impact_{first}",
        f"verify_{first}",
    ):
        plan(
            question,
            f"先行結果 {previous} に基づき {question} を確認する",
            {"kind": "finding", "id": "initial_finding"}
            if previous == "initial"
            else {"kind": "task", "id": previous},
        )
        if server.book().snapshot["tasks"][question]["status"] != "completed":
            spawn(question)
            done(question, f"{question} を確認済み")
        previous = question
    if pending["independent"]["status"] != "completed":
        done(
            "independent",
            "独立範囲の確認",
            failed=bool(os.environ.get("FAKE_AUDIT_FAIL")),
        )
    if os.environ.get("FAKE_AUDIT_REJECT_ONCE"):
        done("busy", "独立調査の確認完了")
    # Findings are snapshots in the ledger, never an aggregate final JSON.
    task = server.book().snapshot["tasks"][previous]
    if server.book().snapshot["findings"]["initial_finding"]["status"] != "refuted":
        call(
            "audit_finding",
            record={
                **initial_finding,
                "status": "refuted",
                "counterevidence": task["evidence"],
            },
        )
    props = [
        {
            "title": "放置PRの対応を判断する",
            "category": "code-debt",
            "related_parent": None,
        },
        *data.get("proposals", []),
    ]
    for i, prop in enumerate(props):
        key = f"f{i}"
        row = {
            "id": key,
            "task_id": previous,
            "kind": "ops" if prop["category"] == "control-rules" else "design",
            "status": "suspected",
            "conclusion": prop["title"],
            "evidence": task["evidence"],
            "counterevidence": [],
            "details": {},
        }
        current = server.book().snapshot["findings"].get(key)
        if current is None:
            call("audit_finding", record=row)
        if current is None or current["status"] != "confirmed":
            call("audit_finding", record={**row, "status": "confirmed"})
        if prop.get("related_parent") and not any(
            h["finding_id"] == key for h in server.book().snapshot["handoffs"].values()
        ):
            call(
                "audit_handoff",
                record={
                    "id": f"route_{key}",
                    "finding_id": key,
                    "kind": "proposal",
                    "reference": "範囲確認待ち",
                    "related_parents": [prop["related_parent"]],
                    "reason": "既存親の範囲確認へ戻す",
                },
            )
    gaps = [
        g for t in server.book().snapshot["tasks"].values() for g in t["unconfirmed"]
    ]
    call(
        "audit_end",
        status="incomplete" if gaps else "completed",
        reason="発見から利用経路・影響確認まで調べた",
        checked_scope=["big.py"],
        excluded_scope=["外部submodule内部"],
        unverified_risks=gaps,
        further_investigation=["失敗タスクを再開"] if gaps else [],
    )
    runtime.finish()
    output.write_text(
        json.dumps(
            {"memory_update": data["memory_update"]} if "memory_update" in data else {}
        )
    )


if __name__ == "__main__":
    main()
