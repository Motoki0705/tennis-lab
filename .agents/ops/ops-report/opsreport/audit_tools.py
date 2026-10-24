"""Bounded MCP ledger operations; all paths and permissions are host-owned."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, cast

from . import audit, handoff, memory_pr, native, shell
from .ledger import Ledger, LedgerError, _end
from .retrieval import (
    MAX_RESPONSE_BYTES,
    PAGE,
    STRING,
    TEXT,
    Retrieval,
    RetrievalError,
    Retrieved,
    Tool,
    encode,
    strict_json,
    text_page,
)

OBJECT = {"type": "object"}
STRINGS = {"type": "array", "items": {"type": "string", "minLength": 1}}
GAPS = {
    "type": "array",
    "items": {
        "type": "object",
        "properties": {k: STRING for k in ("scope", "reason", "resume")},
        "required": ["scope", "reason", "resume"],
        "additionalProperties": False,
    },
}
ENDING = {
    "status": {"enum": ["completed", "incomplete"]},
    "reason": STRING,
    "checked_scope": STRINGS,
    "excluded_scope": STRINGS,
    "unverified_risks": GAPS,
    "further_investigation": STRINGS,
}


class AuditTool(Tool):
    def public(self) -> dict[str, Any]:
        result = cast(dict[str, Any], super().public())
        result["annotations"]["readOnlyHint"] = self.name in {
            "audit_status",
            "audit_original",
            "audit_context",
            "audit_previous_handoffs",
        }
        return result


class AuditTools:
    """Wrap retrieval without changing its GET-only API or journal protocol."""

    def __init__(
        self,
        retrieval: Retrieval,
        *,
        ledger_path: Path,
        runtime_dir: Path,
        input_path: Path,
        ending_path: Path,
        max_threads: int,
    ) -> None:
        self.retrieval = retrieval
        self.ledger_path = ledger_path
        self.runtime_dir = runtime_dir
        self.input_path = input_path
        self.ending_path = ending_path
        self.max_threads = max_threads
        for path in (ledger_path, runtime_dir, input_path, ending_path):
            if not path.is_absolute() or path.resolve().is_relative_to(retrieval.repo):
                raise ValueError(
                    "audit state must be absolute and outside the agent checkout"
                )
        self.book()
        if type(max_threads) is not int or max_threads < 2:
            raise ValueError("audit tools require OPS_REPORT_MAX_THREADS >= 2")
        definitions = [
            AuditTool(
                "audit_status",
                "Bounded ledger summary, unfinished tasks and follow-up candidates. Start here on resume.",
                self.status,
                {**PAGE},
            ),
            AuditTool(
                "audit_original",
                "Read a versioned ledger ref from audit_status. Large originals require smaller character ranges.",
                self.original,
                {"ref": OBJECT, **TEXT},
                ("ref",),
            ),
            AuditTool(
                "audit_context",
                "Read pinned memory decisions/persistence or previous round references in bounded ranges. Dedicated memory content uses record_file(scope=ops_memory).",
                self.context,
                {"section": {"enum": ["ops_memory", "previous_reports"]}, **TEXT},
                ("section",),
            ),
            AuditTool(
                "audit_previous_handoffs",
                "Read a prior completed round's handoffs and current report adoption/triage annotations. Does not start work or grant approval.",
                self.previous_handoffs,
                {"slot": STRING, **PAGE},
                ("slot",),
            ),
            AuditTool(
                "audit_plan_task",
                "Register a distinct question before native spawn. id must also be the native task_name; origin identifies the goal, task or finding.",
                self.plan,
                {"id": STRING, "question": STRING, "scope": STRINGS, "origin": OBJECT},
                ("id", "question", "scope", "origin"),
            ),
            AuditTool(
                "audit_start_task",
                "After spawn_agent returns, resolve /root/<id> to its actual native session ID from runtime metadata. Enforces the existing concurrency pin. Read the returned task record.",
                self.start,
                {"id": STRING},
                ("id",),
            ),
            AuditTool(
                "audit_finish_task",
                "Record one child's result immediately. Completed requires evidence receipts and no gaps; spawn failure uses failed with reason/resume. A failed/interrupted task retries with the same id.",
                self.finish,
                {
                    "id": STRING,
                    "status": {
                        "enum": ["completed", "failed", "interrupted", "cancelled"]
                    },
                    "conclusion": STRING,
                    "evidence": STRINGS,
                    "next_questions": {"type": "array", "items": OBJECT},
                    "unconfirmed": GAPS,
                },
                (
                    "id",
                    "status",
                    "conclusion",
                    "evidence",
                    "next_questions",
                    "unconfirmed",
                ),
            ),
            AuditTool(
                "audit_evidence",
                "Register a successful retrieval receipt as immutable evidence. Source, acquired time and target/PR head are derived from the durable receipt, never from a claim.",
                self.evidence,
                {"id": STRING, "receipt": STRING},
                ("id", "receipt"),
            ),
            AuditTool(
                "audit_finding",
                "Record a finding snapshot. Fields: id, task_id, kind, status, conclusion, evidence, counterevidence, details. Start suspected/unverified; confirmation follows ledger validation (see ops-report README).",
                self.finding,
                {"record": OBJECT},
                ("record",),
            ),
            AuditTool(
                "audit_handoff",
                "Record an existing issue/PR or proposed owner, without GitHub writes or agreement. Fields: id, finding_id, kind, reference, related_parents, reason.",
                self.handoff,
                {"record": OBJECT},
                ("record",),
            ),
            AuditTool(
                "audit_end",
                "Request an end with checked/excluded scopes and unresolved risks. The host verifies native transcripts/receipts and prepares proposal IDs before actually ending/publishing. No fixed task count is a completion criterion.",
                self.end,
                ENDING,
                tuple(ENDING),
            ),
        ]
        if "cpu_test_status" not in retrieval.tools:
            retrieval.tools["cpu_test_status"] = Tool(
                "cpu_test_status",
                "Inspect whether the host supplied an approved CPU-test allowlist and isolation proof.",
                lambda: self.response(
                    {
                        "enabled": False,
                        "unverified": {
                            "cpu_allowlist": "No host-approved CPU-test allowlist was supplied"
                        },
                    }
                ),
                {},
            )
        if "cpu_tests" in retrieval.tools:
            tool = retrieval.tools["cpu_tests"]
            retrieval.tools[tool.name] = AuditTool(
                tool.name,
                tool.description,
                tool.handler,
                tool.properties,
                tool.required,
            )
        self.tools = {**retrieval.tools, **{t.name: t for t in definitions}}
        for tool in definitions:
            retrieval.tools[tool.name] = tool

    def book(self) -> Ledger:
        return Ledger.load(
            self.ledger_path,
            slot=self.retrieval.round_id,
            base_sha=self.retrieval.base_sha,
        )

    def call(self, name: str, arguments: Any) -> dict[str, Any]:
        return cast(dict[str, Any], self.retrieval.call(name, arguments))

    def response(self, data: Any) -> Retrieved:
        if len(encode(data)) > MAX_RESPONSE_BYTES - 2000:
            raise RetrievalError("RESPONSE_LIMIT", "request a smaller ledger range")
        return Retrieved(
            data, [{"kind": "audit-ledger", "slot": self.retrieval.round_id}]
        )

    def status(self, page: int = 1, per_page: int = 20) -> Retrieved:
        return self.response(
            self.book().summary(limit=per_page, offset=(page - 1) * per_page)
        )

    def original(self, ref: dict[str, Any], **ranges: Any) -> Retrieved:
        data, pagination = text_page(
            json.dumps(self.book().original(ref), ensure_ascii=False), **ranges
        )
        result = self.response(data)
        result.pagination = pagination
        return result

    def context(self, section: str, **ranges: Any) -> Retrieved:
        pinned = strict_json(self.input_path.read_bytes())
        value = pinned[section]
        if section == "ops_memory":
            value = {k: v for k, v in value.items() if k != "files"}
        data, pagination = text_page(json.dumps(value, ensure_ascii=False), **ranges)
        result = self.response(data)
        result.pagination = pagination
        return result

    def previous_handoffs(
        self, slot: str, page: int = 1, per_page: int = 20
    ) -> Retrieved:
        pinned = strict_json(self.input_path.read_bytes())
        prior = next(
            (
                r
                for r in pinned["previous_reports"]
                if r["slot"] == slot and r["status"] == "complete"
            ),
            None,
        )
        if prior is None:
            raise LedgerError("not a prior completed round")
        path = self.ledger_path.with_name(f"{slot}.ledger.json")
        old = Ledger.load(path, slot=slot, base_sha=prior["base_sha"])
        body = shell.gh_json(
            ["issue", "view", str(prior["number"]), "--json", "body"],
            cwd=self.retrieval.repo,
        )["body"]
        return self.response(
            handoff.reobservation_input(
                old,
                report_issue=prior["number"],
                report_body=body,
                limit=per_page,
                offset=(page - 1) * per_page,
            )
        )

    def writable(self) -> Ledger:
        if self.ending_path.exists():
            raise LedgerError(
                "end already requested; return the final memory_update JSON"
            )
        return self.book()

    def plan(
        self, id: str, question: str, scope: list[str], origin: dict[str, Any]
    ) -> Retrieved:
        if not re.fullmatch(r"[a-z][a-z0-9_]{0,79}", id):
            raise LedgerError(
                "task id must be a native task_name: lowercase letters, digits, underscores"
            )
        book = self.writable()
        book.record_task(
            {
                "id": id,
                "question": question,
                "scope": scope,
                "origin": origin,
                "agent_id": None,
                "status": "planned",
                "started_at": None,
                "finished_at": None,
                "conclusion": "",
                "evidence": [],
                "next_questions": [],
                "unconfirmed": [],
            },
            at=audit.timestamp(),
        )
        return self.response(book.snapshot["tasks"][id])

    def start(self, id: str) -> Retrieved:
        book = self.writable()
        task = book.snapshot["tasks"][id]
        if task["status"] not in {"planned", "failed", "interrupted"}:
            raise LedgerError("task is already running/finished; do not spawn it again")
        if (
            sum(t["status"] == "running" for t in book.snapshot["tasks"].values())
            >= self.max_threads
        ):
            raise LedgerError(
                "OPS_REPORT_MAX_THREADS exhausted; receive a finished child before another launch"
            )
        result = strict_json((self.runtime_dir / "result.json").read_bytes())
        if result["contract"]["mode"] != "read-only" or result["contract"][
            "cwd"
        ] != str(self.retrieval.repo):
            raise LedgerError("native runtime sandbox/worktree mismatch")
        parent = result["runtime"]["session_id"]
        if not isinstance(parent, str) or not parent:
            raise LedgerError("native parent session not yet recorded")
        matches = [
            meta
            for _, meta in native._child_paths(
                Path(result["contract"]["runtime_home"]), parent
            )
            if meta.get("agent_path") == f"/root/{id}"
        ]
        if len(matches) != 1:
            raise LedgerError(
                "native child metadata missing/ambiguous; do not substitute task_name for session ID"
            )
        meta = matches[0]
        if (
            meta.get("cwd") != str(self.retrieval.repo)
            or meta.get("session_id") != parent
        ):
            raise LedgerError("native child metadata/worktree mismatch")
        book.record_task(
            {
                **task,
                "agent_id": meta["id"],
                "status": "running",
                "started_at": audit.timestamp(),
                "finished_at": None,
                "conclusion": "",
                "evidence": [],
                "next_questions": [],
                "unconfirmed": [],
            },
            at=audit.timestamp(),
        )
        return self.response(book.snapshot["tasks"][id])

    def finish(
        self,
        id: str,
        status: str,
        conclusion: str,
        evidence: list[str],
        next_questions: list[dict[str, Any]],
        unconfirmed: list[dict[str, Any]],
    ) -> Retrieved:
        book = self.writable()
        task = book.snapshot["tasks"][id]
        book.record_task(
            {
                **task,
                "status": status,
                "finished_at": audit.timestamp(),
                "conclusion": conclusion,
                "evidence": evidence,
                "next_questions": next_questions,
                "unconfirmed": unconfirmed,
            },
            at=audit.timestamp(),
        )
        return self.response(book.snapshot["tasks"][id])

    def evidence(self, id: str, receipt: str) -> Retrieved:
        acquired = audit.receipts(
            self.retrieval.journal.path,
            slot=self.retrieval.round_id,
            base_sha=self.retrieval.base_sha,
        )
        found = acquired.get(receipt)
        if (
            found is None
            or not found["response"]["ok"]
            or found["request"]["tool"].startswith("audit_")
            or found["request"]["tool"].startswith("cpu_")
        ):
            raise LedgerError(
                "evidence requires a successful source retrieval receipt; CPU-test evidence is recorded automatically"
            )
        row = audit.evidence_record(id, found)
        book = self.writable()
        book.record_evidence(row, at=audit.timestamp())
        return self.response(row)

    def finding(self, record: dict[str, Any]) -> Retrieved:
        book = self.writable()
        book.record_finding(record, at=audit.timestamp())
        return self.response({"ref": book.reference("finding", record["id"])})

    def handoff(self, record: dict[str, Any]) -> Retrieved:
        book = self.writable()
        book.record_handoff(record, at=audit.timestamp())
        return self.response({"ref": book.reference("handoff", record["id"])})

    def end(self, **record: Any) -> Retrieved:
        book = self.writable()
        _end(record, book.snapshot)
        memory_pr._atomic_json(self.ending_path, record)
        return self.response(
            {
                "requested": record["status"],
                "verification": "pending host native/receipt reconciliation",
            }
        )
