"""Parent operations. No agent execution: delegation is a claim for the caller."""

from __future__ import annotations

import copy
import json
import re
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from issues import (
    CHILD_LABEL,
    PARENT_LABEL,
    STANDALONE_LABEL,
    TASKS_HEADING,
    ChildTask,
    Issue,
    ParentRef,
    child_title,
    index_proposals,
    parse_proposal_source,
    parse_sections,
    parse_title,
    render_child_tasks,
    task_status,
    validate_body,
)
from issues.cli import render_spec

from . import execution, proposals, work_targets, worker_preflight
from .github import GitHub
from .launch import DEFAULT_LAUNCH_POLICY, launch_policy, plan_launches, reason
from .policy import (
    PHASES,
    child_stopped,
    closing_issues,
    dependencies,
    digest,
    marker,
    next_agreement_revision,
    nonempty,
    note_body,
    number,
    object_value,
    phase,
    raw_section,
    registration,
    replace_section,
    review_state,
)
from .state import Store

TASK_MARKER = re.compile(
    r"(?m)^<!-- parent-task:v1 parent=([1-9][0-9]*) key=([a-z0-9][a-z0-9-]{0,63}) spec=([0-9a-f]{64}) -->$"
)


def task_title(issue: Issue) -> str:
    """Public task name shown by worker status; recorded with the local task.

    Parent/child issues use the summary of their validated title. A standalone
    issue has no title convention, so its whole single-line title is the name.
    """
    if {PARENT_LABEL, CHILD_LABEL} & set(issue.labels):
        return parse_title(issue.title).summary
    return nonempty(issue.title, "issue title").strip()


def private_run_paths(task: dict[str, Any], directory: Path) -> set[str]:
    return {
        path
        for run in task.get("worker_runs", [])
        for path in (run["run_dir"], str((directory / run["run_dir"]).resolve()))
    } | {entry["stderr_file"] for entry in task.get("launch_failures", [])}


def visible_task(task: dict[str, Any], private_paths: set[str]) -> dict[str, Any]:
    """Runner locations are private registry data, including external paths."""
    paths = sorted(private_paths, key=len, reverse=True)

    def redact(value: Any) -> Any:
        if isinstance(value, str):
            for path in paths:
                value = value.replace(path, "[private run]")
        elif isinstance(value, dict):
            return {key: redact(item) for key, item in value.items()}
        elif isinstance(value, list):
            return [redact(item) for item in value]
        return value

    return {
        key: redact(value)
        for key, value in task.items()
        if key
        not in {
            "worker_runs",
            "model_reason",
            "preflight",
            "launch_failures",
            "manual_launcher",
        }
    } | (
        {"worker_phase": worker_preflight.summary(task)}
        if task.get("delegated")
        else {}
    )


class ParentSession:
    def __init__(
        self, gh: GitHub, store: Store, parent: int, session: str | None
    ) -> None:
        self.gh, self.store, self.parent, self.session = gh, store, parent, session
        self.key = f"{gh.repo.casefold()}#{parent}"
        self.endpoint = f"repos/{gh.repo}/issues/{parent}"
        self.gh.parent = parent
        self.gh.evidence_directory = store.directory / "evidence-issues"
        self.store.focus(self.key)
        self.login = nonempty(object_value(gh.get("user")).get("login"), "gh login")

    def detached(self) -> ParentSession:
        """Retain frozen local context with an explicit registry-external client."""
        session = copy.copy(self)
        session.gh = self.gh.detached()
        return session

    @contextmanager
    def unlocked(self) -> Iterator[ParentSession]:
        with self.gh.io_scope():
            yield self.detached()

    @property
    def row(self) -> dict[str, Any]:
        return object_value(self.store.parents.get(self.key, {}))

    def owner(self) -> dict[str, Any]:
        row = self.row
        if not row or row["session"] != self.session or row["status"] != "active":
            raise ValueError(
                f"{self.key}: 担当未指定・別セッション・終了済み・引継ぎ待ち ({row.get('status', 'unassigned')})"
            )
        return row

    def history(self) -> list[dict[str, Any]]:
        return self.gh.record_history(self.parent, self.login).session_events()

    def agreed(self) -> dict[str, Any]:
        row = self.owner()
        self.gh.issue(self.parent)
        execution.current_basis(self)
        return row

    def run_paths(self) -> set[str]:
        return {
            path
            for row in self.store.parents.values()
            for task in row["tasks"].values()
            for path in private_run_paths(task, self.store.directory)
        }

    def public(self, value: Any) -> None:
        serialized = json.dumps(value, ensure_ascii=False)
        secrets = {self.session.partition(":")[2]} if self.session else set()
        secrets.add(str(self.store.directory))
        for row in self.store.parents.values():
            secrets.add(row["session"].partition(":")[2])
            secrets.update(task.get("worktree", "") for task in row["tasks"].values())
        secrets.update(self.run_paths())
        if any(secret and secret in serialized for secret in secrets):
            raise ValueError(
                "public payload contains private session/worktree information"
            )

    def post(self, event: dict[str, Any], text: str) -> dict[str, Any]:
        from .record_writer import write_record

        data = {"parent": self.parent, **event}
        body = marker(data) + "\n\n" + text
        self.public(body)
        response = write_record(self, body)
        from decision_logs.parent_sources import comment

        comment(self, data, body, response)
        return response

    def note(self, body: str) -> dict[str, Any]:
        row = self.owner()
        text = note_body(body)
        sha = digest(text)
        response = self.post(
            {"event": "note", "generation": row["generation"], "body_sha256": sha},
            text,
        )
        from .record_writer import url

        return {"parent": self.parent, "body_sha256": sha, "url": url(self, response)}

    def _assignment_summary(self, summary: str | None) -> str:
        """Reject missing/invalid display input before any assignment is saved."""
        from herdr_labels.service import validate_summary

        if summary is None:
            summary = self.row.get("display_summary")
        if summary is None:
            raise ValueError(
                f"parent #{self.parent} has no sidebar summary; "
                "specify --summary <summary> when registering or accepting a handoff"
            )
        return validate_summary(summary)

    def register(self, summary: str | None = None) -> dict[str, Any]:
        if self.session is None:
            raise ValueError("session identity missing")
        summary = self._assignment_summary(summary)
        registration(self.store.parents, self.key, self.session)
        if self.gh.issue(self.parent)["state"] != "open":
            raise ValueError("cannot assign a closed parent issue")
        if not self.row or self.row["status"] != "active":
            previous = self.row
            self.store.parents[self.key] = {
                **previous,
                "session": self.session,
                "status": "active",
                "generation": previous.get("generation", 0) + 1,
                "tasks": previous.get("tasks", {}),
                "reviews": previous.get("reviews", []),
                "display_summary": summary,
            }
            self.row.pop("owner_runtime", None)
            self.store.save()
        if self.capture_owner_runtime():
            self.store.save()
        if self.row.get("display_summary") != summary:
            self.row["display_summary"] = summary
            self.store.save()
        self.post(
            {"event": "assigned", "generation": self.row["generation"]},
            "担当開始。担当指定と要件合意は別段階です。要件を確認してから分割・委任します。",
        )
        return self.status()

    def capture_owner_runtime(self) -> bool:
        from lib.identity import native_runtime

        runtime = native_runtime()
        bound = (
            None
            if runtime is None
            else {
                **runtime,
                "session": self.row["session"],
                "generation": self.row["generation"],
            }
        )
        previous = self.row.pop("owner_runtime", None)
        if bound is not None:
            self.row["owner_runtime"] = bound
        return previous != bound

    def set_summary(self, text: str) -> dict[str, Any]:
        """Save the parent's own sidebar summary; display sync projects it."""
        from herdr_labels.service import validate_summary

        row = self.owner()
        row["display_summary"] = validate_summary(text)
        self.store.save()
        return {"parent": self.parent, "display_summary": row["display_summary"]}

    def status(self) -> dict[str, Any]:
        from .notifications import publication_status

        row = self.row
        _, items = self.work_items()
        history = self.history()
        run_paths = self.run_paths()
        return {
            "repo": self.gh.repo,
            "parent": self.parent,
            "assignment": row.get("status", "unassigned"),
            "display_summary": row.get("display_summary"),
            "current_session_is_owner": bool(row and row["session"] == self.session),
            "phase": PHASES[phase(history)],
            "notifications": publication_status(
                self.gh.record_history(self.parent, self.login)
            ),
            "tasks": {
                n: {
                    **visible_task(task, run_paths),
                    "worker_phase": worker_preflight.observe(self, int(n))
                    if int(n) in items
                    else worker_preflight.summary(task),
                    "current_work_target": int(n) in items,
                }
                for n, task in row.get("tasks", {}).items()
            },
            "launch_plan": self.launch_plan(),
            "review": self.review_status(history),
            "proposal_receipts": proposals.receipts(history),
            "proposal_agreements": proposals.agreed_receipts(history),
            "input_changes": execution.input_changes(self),
            "improvements": self.improvement_status(),
            "completion": self.completion_status(),
            "execution_basis": next(
                (
                    e["basis"]
                    if e["event"] == "execution_basis"
                    else execution.human_basis(e)
                    for e in reversed(history)
                    if e["event"] in {"execution_basis", "agreed"}
                ),
                None,
            ),
        }

    def review_status(self, history: list[dict[str, Any]]) -> Any:
        if not self.row.get("reviews"):
            return "未指示"
        run = self.row["reviews"][-1]
        public_state = review_state(history, run["request"])
        return [
            {
                **run,
                "state": public_state or run["state"],
                "public_recorded": public_state is not None,
            }
        ]

    def improvement_status(self) -> list[dict[str, Any]]:
        from .improvements import status

        return status(self)

    def completion_status(self) -> dict[str, Any]:
        from .completion import status

        return status(self)

    def lifecycle(self, action: str, note: str) -> dict[str, Any]:
        row = self.row
        nonempty(note, "handoff/release note")
        self.public(note)
        if not (row.get("status") == action and row.get("session") == self.session):
            row = self.owner()
        event = {"event": action, "generation": row["generation"]}
        if action == "ended":
            from .notifications import completion

            record = completion(self)
            if record:
                event["completion_ref"] = record.url
                note = f"担当終了。全体の成果・未達・残件: {record.url}"
        if row.get("status") == action and row.get("session") == self.session:
            if not any(
                e["body"] == marker({"parent": self.parent, **event}) + "\n\n" + note
                for e in self.history()
            ):
                raise ValueError("lifecycle retry differs from its recorded evidence")
            self.sync_progress()
            return self.status()
        self.owner()
        self.post(event, note)
        row["status"] = action
        self.store.save()
        self.sync_progress()
        return self.status()

    def accept(
        self, note: str, recover: bool = False, *, summary: str | None = None
    ) -> dict[str, Any]:
        if self.session is None:
            raise ValueError("session identity missing")
        summary = self._assignment_summary(summary)
        row = self.row
        same_retry = (
            row
            and row["status"] == "active"
            and row["session"] == self.session
            and row.get("handoff_note_sha256") == digest(note)
        )
        if same_retry:
            if row.get("display_summary") != summary:
                row["display_summary"] = summary
                self.store.save()
            self.post(
                {
                    "event": "handoff",
                    "generation": row["generation"],
                    "recovery": row.get("handoff_recovery", False),
                },
                note,
            )
            return self.reconcile()
        recoverable = bool(
            row
            and row["status"] == "active"
            and row["session"] != self.session
            and recover
        )
        if not row or (
            row["status"] not in {"handoff_pending", "ended", "released"}
            and not recoverable
        ):
            raise ValueError(
                "handoff --accept requires a released/ended/pending assignment; unexpected interruption requires explicit --recover and inspected handoff evidence"
            )
        for key, other in self.store.parents.items():
            if (
                key != self.key
                and other["session"] == self.session
                and other["status"] in {"active", "handoff_pending"}
            ):
                raise ValueError(f"既に担当中: {key}")
        nonempty(note, "handoff note")
        self.public(note)
        inherited_authority = None
        if row.get("dispatch") is not None:
            try:
                inherited_authority = execution.observed_authorization(self)
            except execution.ExecutionBasisStale:
                inherited_authority = (
                    None  # Ownership transfers; execution stays suspended.
                )
        # A retry after a failed comment must keep the claim and reconcile; it must not switch owners.
        row.update(
            session=self.session,
            status="active",
            generation=row["generation"] + 1,
            handoff_note_sha256=digest(note),
            handoff_recovery=recover,
            display_summary=summary,
        )
        self.capture_owner_runtime()
        if row.get("dispatch") is not None:
            from .development import transfer_controller

            if inherited_authority is not None:
                inherited_authority = {
                    **inherited_authority,
                    "generation": row["generation"],
                }
            transfer_controller(row, inherited_authority, note, takeover=True)
        self.store.save()
        self.post(
            {"event": "handoff", "generation": row["generation"], "recovery": recover},
            note,
        )
        return self.reconcile()

    def checking(self) -> dict[str, Any]:
        self.owner()
        from .supplement import require_final

        require_final(self.gh, self.parent)
        history = self.history()
        latest = next(
            (
                e
                for e in reversed(history)
                if e["event"]
                in {"checking", "agreed", "execution_basis", "basis_revoked"}
            ),
            {},
        )
        if latest.get("event") == "checking":
            return self.status()
        self.post(
            {
                "event": "checking",
                "generation": self.row["generation"],
                "revision": next_agreement_revision(history),
            },
            "要件確認中。目的・範囲・期待する動作・受入条件と不明点を人間と認識合わせします。",
        )
        return self.status()

    def revoke_basis(self, reason: str) -> dict[str, Any]:
        self.owner()
        reason = nonempty(reason, "revocation reason")
        history = self.history()
        latest = next(
            (
                e
                for e in reversed(history)
                if e["event"]
                in {"checking", "agreed", "execution_basis", "basis_revoked"}
            ),
            {},
        )
        revision = (
            latest["revision"]
            if latest.get("event") == "basis_revoked"
            and latest.get("reason") == reason
            and latest.get("generation") == self.row["generation"]
            else max(
                (e["revision"] for e in history if e["event"] == "basis_revoked"),
                default=0,
            )
            + 1
        )
        self.post(
            {
                "event": "basis_revoked",
                "generation": self.row["generation"],
                "reason": reason,
                "revision": revision,
            },
            "実行根拠を取り消しました。人間との要件確認・合意まで自律着手しません。\n\n"
            + reason,
        )
        return self.status()

    def agree(
        self,
        agreement: dict[str, Any] | None,
        report: int | None = None,
        pid: str | None = None,
        improvement_approval: str | None = None,
    ) -> dict[str, Any]:
        self.owner()
        from .supplement import require_final

        require_final(self.gh, self.parent)
        from . import improvements

        binding = improvements.require(self)
        if improvement_approval is not None:
            if agreement is not None or report is not None or pid is not None:
                raise ValueError(
                    "use an improvement approval reference without a second agreement spec"
                )
            agreement, binding = improvements.agreement(self, improvement_approval)
        elif binding is not None:
            raise ValueError(
                "agree requires --improvement-approval; a generic agreement cannot bypass the candidate decision"
            )
        if agreement is None:
            raise ValueError(
                "agreement spec or improvement approval reference is required"
            )
        if (report is None) != (pid is None):
            raise ValueError("proposal agreement requires both --report and --proposal")
        fields = {"purpose", "scope", "behavior", "acceptance", "human_confirmation"}
        if set(agreement) != fields:
            raise ValueError(f"agreement requires exactly {sorted(fields)}")
        agreement = {k: nonempty(v, k) for k, v in agreement.items()}
        history = self.history()
        sha = digest(agreement)
        last_phase = next(
            (
                e
                for e in reversed(history)
                if e.get("event") in {"agreed", "checking", "basis_revoked"}
            ),
            {},
        )
        if report is not None and pid is not None:
            self.agreed()
            source = proposals.selected_source(self, report, pid)
            receipt = proposals.require_receipt(self, source)
            if receipt in proposals.agreed_receipts(history):
                previous = next(
                    e
                    for e in reversed(history)
                    if e.get("event") == "proposal_agreed"
                    and e.get("source") == receipt["source"]
                )
                if previous["agreement_sha256"] != sha:
                    raise ValueError(
                        "proposal agreement changed; run basis-revoke before revising requirements"
                    )
            event = {
                "event": "proposal_agreed",
                **receipt,
                "revision": last_phase.get("revision"),
            }
        else:
            if phase(history) == "agreed" and last_phase.get("agreement_sha256") == sha:
                self.agreed()  # Retrying agree must not silently accept new comments.
                if binding is not None:
                    from decision_logs.parent_sources import (
                        improvement_approval as capture_approval,
                    )

                    capture_approval(self, binding)
                return self.status()
            from .agreement_comments import snapshot as comment_inputs

            event = {
                "event": "agreed",
                "revision": next_agreement_revision(history),
                "comment_inputs": comment_inputs(self),
                "requirements_sha256": digest(
                    execution.requirement_body(
                        self.gh.issue(self.parent).get("body") or ""
                    )
                ),
                "acceptance_snapshot": {
                    heading: raw_section(
                        self.gh.issue(self.parent).get("body") or "", heading
                    )
                    for heading in ("全体の完了条件", "Acceptance checklist")
                    if raw_section(
                        self.gh.issue(self.parent).get("body") or "", heading
                    )
                },
            }
            if binding is not None:
                event["improvement_approval"] = binding
        titles = {
            "purpose": "目的",
            "scope": "範囲",
            "behavior": "期待する動作",
            "acceptance": "受入条件",
        }
        text = "## 要件合意\n\n" + "\n\n".join(
            f"{title}: {agreement[k]}" for k, title in titles.items()
        )
        text += "\n\n人間の確認（会話からの引用）:\n" + "\n".join(
            "> " + line for line in agreement["human_confirmation"].splitlines()
        )
        if report is not None:
            text = f"対象提案: #{report} / `{pid}`\n\n" + text
        require_final(self.gh, self.parent)
        self.post(
            {
                **event,
                "agreement_sha256": sha,
                "agreement": agreement,
            },
            text,
        )
        self.agreed()
        if binding is not None:
            from decision_logs.parent_sources import (
                improvement_approval as capture_approval,
            )

            capture_approval(self, binding)
        return self.status()

    def family(self) -> tuple[Issue, dict[int, Issue]]:
        family = self.gh.family(self.parent)
        from issues.evidence import require_work_issue

        for item in family.values():
            require_work_issue({"labels": item.labels, "body": item.body})
        parent = family[self.parent]
        return parent, {n: family[n] for n in parent.children}

    def reviewed_targets(self) -> set[int]:
        reviewed_targets: set[int] = set()
        for review in self.row.get("reviews", []):
            requirements = {int(n) for n in review["request"].get("requirements", {})}
            if requirements:
                reviewed_targets.update(requirements - {self.parent} or {self.parent})
        return reviewed_targets

    def work_items(self) -> tuple[Issue, dict[int, Issue]]:
        parent, children = self.family()
        items = work_targets.resolve(
            parent,
            children,
            self.row.get("tasks", {}),
            managed="dispatch" in self.row,
            reviewed_targets=self.reviewed_targets(),
        )
        return parent, items

    def update_body(self, n: int, body: str, expected_body: str) -> None:
        from .supplement import require_final

        self.public(body)
        raw = require_final(self.gh, n)
        if raw["body"] != expected_body and raw["body"] != body:
            raise ValueError(
                f"#{n}: body changed during the operation; reconcile before retry"
            )
        labels = {x["name"] for x in raw["labels"]}
        errors = validate_body(
            body,
            is_parent=PARENT_LABEL in labels,
            is_child=CHILD_LABEL in labels,
        )
        if errors:
            raise ValueError(f"#{n}: invalid issue body: {errors}")
        if raw["body"] != body:
            self.gh.write("PATCH", f"repos/{self.gh.repo}/issues/{n}", {"body": body})

    def sync_progress(self) -> dict[int, str]:
        parent, children = self.work_items()
        if PARENT_LABEL not in parent.labels:
            if parent.children:
                raise ValueError(
                    "parent role label missing; migrate explicitly using issues"
                )
            if parent.parent is not None:
                return {t["child"]: t["state"] for t in self.launch_plan()["tasks"]}
        plan = self.launch_plan()
        states = {t["child"]: t["state"] for t in plan["tasks"]}
        tasks = [
            ChildTask(
                n,
                parse_title(child.title).summary
                if PARENT_LABEL in parent.labels
                else child.title,
                task_status(child),
            )
            for n, child in children.items()
        ]
        # Keep the observed names of existing local tasks for read-only status
        # views. Formal children without a local task are not added here.
        renamed = False
        for task in tasks:
            local = self.row["tasks"].get(str(task.number))
            if local is not None and local.get("title") != task.title:
                local["title"] = task.title
                renamed = True
        if renamed:
            self.store.save()
        # Self-work is not a GitHub sub-issue, nor a row in its own checklist.
        body = (
            replace_section(parent.body, TASKS_HEADING, render_child_tasks(tasks))
            if parent.children
            else parent.body
        )
        labels = {
            "continue": "継続",
            "stopped": "停止",
            "held": "依存保留",
            "completed": "完了",
            "cancelled": "取消",
        }
        reservations = ", ".join(f"#{n}" for n in plan["active_reservations"]) or "なし"
        limit = plan["policy"]["max_parallel_children"]
        subject = "担当Issue" if self.parent in children else "子"
        lines = [
            f"{subject}の起動予約: {reservations}。上限: {limit if limit is not None else '全対象'}。"
        ]
        for task in plan["tasks"]:
            detail = " / ".join(
                r["message"].replace("\n", " ") for r in task["reasons"]
            )
            if not detail:
                detail = "起動可能"
                if task["stacked_on"]:
                    detail += (
                        "（stacked branch: "
                        + ", ".join(f"PR #{p['pr']}" for p in task["stacked_on"])
                        + "）"
                    )
            lines.append(f"- #{task['child']}: {labels[task['state']]} — {detail}")
        if self.parent in children:
            lines.insert(
                0,
                f"未分割。自身 #{self.parent} を作業対象として進める。全体受入は別途照合する。",
            )
        summary = "\n".join(lines) if states else "未分割（子issue未作成）"
        from .notifications import progress_lines

        history = self.gh.record_history(self.parent, self.login)
        summary = "\n".join(progress_lines(self, history)) + "\n\n" + summary
        start, end = "<!-- parent-progress:v1 -->", "<!-- /parent-progress -->"
        current = raw_section(body, "進捗概要")
        block = start + "\n" + summary + "\n" + end
        if start in current or end in current:
            if (
                current.count(start) != 1
                or current.count(end) != 1
                or current.index(start) > current.index(end)
            ):
                raise ValueError("invalid managed progress block; reconcile explicitly")
            current = (
                current[: current.index(start)]
                + block
                + current[current.index(end) + len(end) :]
            )
        else:
            current = block + (
                "\n\n既存の記述（自動更新の対象外）:\n" + current.strip()
                if current.strip()
                else ""
            )
        self.update_body(
            self.parent, replace_section(body, "進捗概要", current), parent.body
        )
        return states

    def create_child(self, value: dict[str, Any]) -> dict[str, Any]:
        self.agreed()  # Must precede render, search and every possible write.
        if self.gh.issue(self.parent)["state"] != "open":
            raise ValueError("parent issue is not open for creating new children")
        auth = execution.authorization(self)
        if (
            set(value) - {"key", "spec", "dependencies"}
            or not {"key", "spec"} <= value.keys()
        ):
            raise ValueError(
                "child manifest requires key/spec, optional dependencies only"
            )
        key = nonempty(value["key"], "task key")
        if not re.fullmatch(r"[a-z0-9][a-z0-9-]{0,63}", key):
            raise ValueError("invalid task key")
        spec = object_value(value["spec"])
        parent, children = self.family()
        previous_self = self.row["tasks"].get(str(self.parent))
        if self.parent in self.reviewed_targets() or (
            previous_self is not None
            and not work_targets.stopped_self_work(previous_self)
        ):
            raise work_targets.WorkTargetsChanged(
                "cannot split an issue with existing self-work; stop the worker and "
                "explicitly reconcile its execution/results and PR ownership first"
            )
        if (
            parent.parent is None
            and not children
            and any(
                self.parent in closing_issues(p.get("body") or "", self.gh.repo)
                for p in self.gh.pulls({self.parent})
            )
        ):
            raise work_targets.WorkTargetsChanged(
                "cannot split an issue with an existing self-work PR; reconcile it first"
            )
        if STANDALONE_LABEL in parent.labels:
            raise ValueError(
                f"migrate the parent using issues and remove {STANDALONE_LABEL} before splitting"
            )
        own = parse_title(parent.title)
        if spec.get("role") != "child" or spec.get("parent") != {
            "identifier": own.identifier,
            "number": self.parent,
        }:
            raise ValueError("child spec must name this parent and its identifier")
        if PARENT_LABEL not in parent.labels or validate_body(
            parent.body, is_parent=True, is_child=parent.parent is not None
        ):
            raise ValueError("migrate the parent using issues before splitting")
        draft = render_spec(spec)
        deps = value.get("dependencies", dependencies(draft["body"]))
        if not isinstance(deps, list) or len(deps) != len(set(deps)):
            raise ValueError("dependencies must be unique issue numbers")
        for dep in deps:
            if number(dep) not in children:
                raise ValueError("new child dependency is not an existing child")
        if "dependencies" in value:
            draft["body"] = replace_section(
                draft["body"],
                "依存関係",
                "\n".join(f"- #{n}" for n in deps) or "なし。",
            )
        sha = digest(value)
        task_marker = (
            f"<!-- parent-task:v1 parent={self.parent} key={key} spec={sha} -->"
        )
        body = draft["body"].rstrip() + "\n\n" + task_marker + "\n"
        self.public({**draft, "body": body})
        found = []
        existing = self.gh.issue_rows()
        source = parse_proposal_source(body)
        if source is not None:
            proposals.require_source(self, source)
            proposal_index = index_proposals(
                [(number(raw["number"]), raw.get("body") or "") for raw in existing]
            )
            previous_number = proposal_index.get(
                (source.report_number, source.proposal_id)
            )
            if previous_number is not None:
                previous_body = next(
                    raw["body"] for raw in existing if raw["number"] == previous_number
                )
                if parse_proposal_source(previous_body) != source:
                    raise ValueError(
                        f"proposal already exists as #{previous_number} with a different option"
                    )
                markers = list(TASK_MARKER.finditer(previous_body))
                if len(markers) > 1 or any(
                    int(m[1]) == self.parent and m[2] == key and m[3] != sha
                    for m in markers
                ):
                    raise ValueError(
                        "existing task key has a different spec or duplicate metadata"
                    )
                previous_parent = self.gh.get(
                    f"repos/{self.gh.repo}/issues/{previous_number}/parent",
                    missing_parent=True,
                )
                # A renamed/reparented existing proposal remains the same work.
                # Do not silently move it or recreate it under another task key.
                if previous_parent is not None:
                    actual_parent = self.gh.identity(previous_parent)
                    reused: dict[str, Any] = {
                        "child": previous_number,
                        "reused": True,
                        "actual_parent": actual_parent,
                    }
                    try:
                        reused["progress"] = self.sync_progress()
                    except work_targets.WorkTargetsChanged as exc:
                        if actual_parent == self.parent:
                            raise
                        # This is an identity lookup for already reparented
                        # work, not admission to execute in the old parent.
                        reused.update(progress={}, target_conflict=str(exc))
                    return reused
                if not any(
                    int(m[1]) == self.parent and m[2] == key
                    for m in TASK_MARKER.finditer(previous_body)
                ):
                    raise ValueError(
                        f"proposal already exists as #{previous_number}; reconcile/migrate the orphan before retry"
                    )
        for raw in existing:
            matches = list(TASK_MARKER.finditer(raw.get("body") or ""))
            for match in matches:
                if int(match[1]) == self.parent and match[2] == key:
                    if match[3] != sha or len(matches) != 1:
                        raise ValueError(
                            "existing task key has a different spec or duplicate metadata"
                        )
                    found.append(raw)
        if len(found) > 1:
            raise ValueError("duplicate existing task key")
        if found:
            child = self.gh.issue(number(found[0]["number"]))
            if child["title"] != child_title(
                ParentRef(str(own.identifier), self.parent), str(spec["summary"])
            ):
                raise ValueError("existing child title no longer matches its spec")
        else:
            if execution.authorization(self) != auth:
                raise ValueError("execution basis changed before child creation")
            if source is not None:
                proposals.require_source(self, source)
            child = self.gh.write(
                "POST",
                f"repos/{self.gh.repo}/issues",
                {"title": draft["title"], "body": body, "labels": draft["labels"]},
            )
        n = number(child["number"])
        base = f"repos/{self.gh.repo}/issues/{n}"
        current_parent = self.gh.get(base + "/parent", missing_parent=True)
        if (
            current_parent is not None
            and self.gh.identity(current_parent) != self.parent
        ):
            raise ValueError("existing child belongs to another parent")
        current_labels = {label["name"] for label in child["labels"]}
        labels = (current_labels | set(draft["labels"])) - {STANDALONE_LABEL}
        if labels != current_labels:
            self.gh.write("PATCH", base, {"labels": sorted(labels)})
        if current_parent is None:
            self.gh.write(
                "POST",
                self.endpoint + "/sub_issues",
                {"sub_issue_id": number(child["id"])},
            )
            # The relation write invalidates the parent. Also discard the
            # child's cached "no parent" when no label PATCH was needed.
            self.gh.invalidate(base)
        task = self.row["tasks"].setdefault(str(n), {})
        task["dependencies"] = deps
        task["title"] = parse_title(child["title"]).summary
        task.setdefault("creation_authorization", auth)
        self.store.save()
        states = self.sync_progress()
        if source is not None:
            proposals.require_source(self, source)
        return {"child": n, "reused": bool(found), "progress": states}

    def launch_plan(
        self,
        overrides: dict[int, list[int]] | None = None,
        restart_child: int | None = None,
    ) -> dict[str, Any]:
        parent, items = self.work_items()
        children, external_deps = work_targets.dependency_graph(parent, items)
        standalone_deps: list[int] | None = None
        blockers = []
        if self.parent in items:
            standalone_deps = external_deps
            if (
                overrides
                and list(overrides.get(self.parent, standalone_deps)) != standalone_deps
            ):
                raise ValueError(
                    "revise standalone dependencies and its execution basis before delegating"
                )
            for dep in standalone_deps:
                raw = self.gh.issue(dep)
                if raw["state"] != "closed" or raw.get("state_reason") != "completed":
                    blockers.append(
                        reason(
                            "dependency_not_completed",
                            f"先行依存 #{dep} のclosed/completedを確認する。",
                        )
                    )
            overrides = None
        if self.row.get("status") != "active":
            blockers.append(reason("parent_assignment", "親の担当がactiveでない。"))
        if parent.state != "open" and restart_child is None:
            blockers.append(reason("parent_closed", "親issueがopenでない。"))
        if phase(self.history()) not in {"agreed", "autonomous"}:
            blockers.append(
                reason("execution_basis_required", "親の有効な実行根拠を記録する。")
            )
        pulls = self.gh.pulls()
        work_targets.require_separate_self_prs(parent, pulls, self.gh.repo)
        plan = plan_launches(
            children,
            self.row.get("tasks", {}),
            pulls,
            self.gh.repo,
            launch_policy(self.row.get("launch_policy", DEFAULT_LAUNCH_POLICY)),
            blockers,
            overrides,
            restart_child,
        )
        if standalone_deps is not None:
            plan["tasks"][0]["dependencies"] = standalone_deps
        return plan

    def configure_delegation(self, limit: int | None, evidence: str) -> dict[str, Any]:
        row = self.owner()
        policy = launch_policy(
            {
                "schema_version": 1,
                "max_parallel_children": limit,
                "reason": nonempty(evidence, "child launch limit reason"),
            }
        )
        self.public(policy)
        row["launch_policy"] = policy
        self.store.save()
        self.sync_progress()
        return {"launch_plan": self.launch_plan()}

    def record_worker_ended(
        self, child: int, delegation_id: str, evidence: str
    ) -> dict[str, Any]:
        row = self.owner()
        task = row["tasks"].get(str(child), {})
        if (
            not task.get("delegated")
            or task.get("delegation_id", "legacy") != delegation_id
        ):
            raise ValueError("worker-end evidence must match the current delegation_id")
        task["worker_ended"] = {
            "delegation_id": delegation_id,
            "evidence": nonempty(evidence, "worker-end evidence"),
        }
        self.store.save()
        self.sync_progress()
        return {"child": child, "launch_plan": self.launch_plan()}

    def reconcile(self) -> dict[str, Any]:
        self.owner()
        _, children = self.work_items()
        initial_plan = self.launch_plan()
        states = {t["child"]: t["state"] for t in initial_plan["tasks"]}
        prs = self.gh.pulls()
        run_paths = self.run_paths()
        from .policy import closing_issues

        result = []
        for n, child in sorted(children.items()):
            task = {
                **self.row["tasks"].get(str(n), {}),
                "dependencies": next(
                    t["dependencies"] for t in initial_plan["tasks"] if t["child"] == n
                ),
            }
            matching = [
                p for p in prs if n in closing_issues(p.get("body") or "", self.gh.repo)
            ]
            if task.get("branch") and any(
                p["head"]["ref"] != task["branch"]
                for p in matching
                if p["state"] == "open"
            ):
                raise ValueError(
                    f"#{n}: PR branch differs from recorded worktree branch"
                )
            result.append(
                {
                    "child": n,
                    "state": states[n],
                    "assignment": visible_task(task, run_paths),
                    "prs": [
                        {
                            "number": p["number"],
                            "state": p["state"],
                            "merged": p.get("merged_at") is not None,
                            "branch": p["head"]["ref"],
                        }
                        for p in matching
                    ],
                    "verification": parse_sections(child.body).get(
                        "成果物・関連PR", "未記録"
                    ),
                    "worker_phase": worker_preflight.observe(self, n),
                    "artifacts_accepted": bool(
                        task.get("verification", {}).get("evidence")
                    ),
                    "artifact_status": "成果の検証記録を照合する",
                    "commits_since_planning": self._planning_commits(task),
                }
            )
        plan = self.launch_plan()
        for task, planned in zip(result, plan["tasks"], strict=True):
            if task["child"] != planned["child"]:
                raise ValueError("child set changed during reconciliation")
            task.update({k: planned[k] for k in ("action", "reasons", "stacked_on")})
        return {
            "parent": self.parent,
            "phase": PHASES[phase(self.history())],
            "tasks": result,
            "launch_plan": plan,
            "review": self.review_status(self.history()),
        }

    def _planning_commits(self, task: dict[str, Any]) -> bool | None:
        value = task.get("preflight", {})
        turns = value.get("turns", [])
        if not turns or not task.get("worktree"):
            return None
        from .snapshot import git

        start = turns[-1].get("head_at_start")
        with self.gh.io_scope():
            return (
                git("rev-parse", "HEAD", cwd=Path(task["worktree"])).decode().strip()
                != start
                if start
                else None
            )

    def delegate(
        self,
        child: int,
        agent: str,
        branch: str,
        worktree: Path,
        deps: list[int] | None,
        restart: str | None,
        *,
        continue_existing: bool = False,
    ) -> dict[str, Any]:
        self.agreed()
        auth = execution.authorization(self)
        parent, children = self.work_items()
        if child not in children:
            raise ValueError("delegate requires a direct child or the standalone issue")
        from .improvements import require as require_improvement

        require_improvement(self, child)
        from .adopted_tasks import check_task_adoptions

        check_task_adoptions(self.gh, parent.body)
        check_task_adoptions(self.gh, children[child].body)
        source = parse_proposal_source(children[child].body)
        if source is not None:
            proposals.require_source(self, source)
        if deps is not None:
            if len(deps) != len(set(deps)):
                raise ValueError("dependencies must be unique")
            for dep in deps:
                number(dep)
        override = {child: deps} if deps is not None else None
        initial_plan = self.launch_plan(override)
        states = {t["child"]: t["state"] for t in initial_plan["tasks"]}
        if states[child] != "continue" and not (
            continue_existing and states[child] == "completed"
        ):
            raise ValueError(f"#{child} cannot be delegated: {states[child]}")
        reconciled = next(t for t in self.reconcile()["tasks"] if t["child"] == child)
        previous = self.row["tasks"].get(str(child), {})
        if reconciled["prs"] and not continue_existing:
            raise ValueError(
                "existing PR: reconcile instead of starting another worker"
            )
        if continue_existing and (
            not restart
            or not previous.get("delegated")
            or previous.get("worker_ended", {}).get("delegation_id")
            != previous.get("delegation_id")
            or previous.get("branch") != branch
            or previous.get("worktree") != str(worktree.resolve())
            or previous.get("agent") != agent
            or len(reconciled["prs"]) > 1
            or any(p["state"] != "open" and not p["merged"] for p in reconciled["prs"])
        ):
            raise ValueError(
                "continuation requires the ended worker, same worktree and restart evidence"
            )
        if previous.get("delegated") and not restart:
            raise ValueError(
                "already delegated; reconcile the existing worker/worktree before an explicit restart"
            )
        if restart:
            nonempty(restart, "restart evidence")
        planned = next(
            t
            for t in self.launch_plan(override, child if restart else None)["tasks"]
            if t["child"] == child
        )
        if continue_existing:
            plan = self.launch_plan(override, child)
            limit = plan["policy"]["max_parallel_children"]
            if limit is not None and len(plan["active_reservations"]) >= limit:
                raise ValueError("implementation reservation limit reached")
        reasons = [
            r
            for r in planned["reasons"]
            if not (
                continue_existing
                and r["code"] in {"existing_pr", "completed", "parent_closed"}
            )
        ]
        if reasons:
            raise ValueError(
                f"#{child} cannot be delegated: "
                + "; ".join(r["message"] for r in reasons)
            )
        if agent not in {"claude", "codex"} or not branch.startswith(agent + "/"):
            raise ValueError("agent/branch prefix mismatch")
        from .snapshot import git, validate_worktree

        with self.gh.io_scope():
            validate_worktree(worktree, branch)
            for base in planned["stacked_on"]:
                git(
                    "merge-base",
                    "--is-ancestor",
                    base["head_sha"],
                    "HEAD",
                    cwd=worktree,
                )
        for n, task in self.row["tasks"].items():
            if n != str(child) and (
                task.get("branch") == branch
                or task.get("worktree") == str(worktree.resolve())
            ):
                raise ValueError("each child requires a dedicated branch/worktree")
        task = {
            **previous,
            "title": task_title(children[child]),
            "agent": agent,
            "branch": branch,
            "worktree": str(worktree.resolve()),
            "delegated": True,
            "delegation_id": uuid.uuid4().hex,
            "worker_runs": previous.get("worker_runs", []),
            "stacked_on": planned["stacked_on"],
            "dependencies": deps if deps is not None else planned["dependencies"],
            "stopped": False,
            "authorization": auth,
            "requirements_sha256": digest(
                execution.requirement_body(
                    replace_section(
                        children[child].body,
                        "依存関係",
                        "\n".join(f"- #{n}" for n in deps) or "なし。",
                    )
                    if deps is not None
                    else children[child].body
                )
            ),
        }
        if continue_existing:
            previous_phase = copy.deepcopy(worker_preflight.state(previous))
            previous_phase["continuations"].append(
                {"from": previous_phase["delegation_id"], "to": task["delegation_id"]}
            )
            previous_phase["delegation_id"] = task["delegation_id"]
            task["preflight"] = previous_phase
        else:
            task["preflight"] = worker_preflight.new(task["delegation_id"])
        if restart:
            task["restart_evidence"] = restart
        task.pop("worker_ended", None)
        # A new delegation makes its own selection. Historical runs retain theirs.
        task.pop("model", None)
        task.pop("model_reason", None)
        # The final admission check must observe edits made during local validation.
        with self.gh.fresh_reads():
            if source is not None:
                proposals.require_source(self, source)
            check_task_adoptions(self.gh, self.gh.issue(self.parent)["body"])
            check_task_adoptions(self.gh, self.gh.issue(child)["body"])
            if execution.authorization(self) != auth:
                raise ValueError("execution basis changed before delegation")
            fresh = next(
                (
                    t
                    for t in self.launch_plan(override, child if restart else None)[
                        "tasks"
                    ]
                    if t["child"] == child
                ),
                None,
            )
        if fresh != planned:
            raise ValueError(
                "child launch inputs changed before delegation; reconcile before retry"
            )
        # Claim durably before returning permission to launch; crash never causes automatic relaunch.
        self.row["tasks"][str(child)] = task
        self.store.save()
        content = raw_section(children[child].body, "成果物・関連PR")
        assignment = f"担当: {agent}\n\nbranch: `{branch}`\n\nPR・検証結果: 未作成。既存の委任記録を照合して再開する。"
        body = children[child].body
        if deps is not None:
            body = replace_section(
                body, "依存関係", "\n".join(f"- #{n}" for n in deps) or "なし。"
            )
        body = replace_section(
            body,
            "成果物・関連PR",
            (content + "\n\n" + assignment).strip(),
        )
        self.update_body(child, body, children[child].body)
        self.sync_progress()
        return {
            "action": "launch",
            "child": child,
            "assignment": visible_task(task, self.run_paths()),
            "parent": parent.number,
            "launch_plan": self.launch_plan(),
        }

    def stop_or_resume(
        self, child: int, detail: dict[str, Any], resume: bool
    ) -> dict[str, Any]:
        self.owner()
        conflict = None
        try:
            _, children = self.work_items()
        except work_targets.WorkTargetsChanged as exc:
            # A concurrent split must not prevent stopping the already saved
            # self-worker. This grants no permission to resume or launch work.
            if resume or child != self.parent or str(child) not in self.row["tasks"]:
                raise
            parent, _ = self.family()
            children = {self.parent: parent}
            conflict = str(exc)
        existing = self.row["tasks"].get(str(child), {})
        if child not in children or not (
            task_status(children[child]) == "open"
            or (
                task_status(children[child]) == "completed"
                and existing.get("delegated")
            )
        ):
            raise ValueError(
                "stop/resume requires an open target or a completed target with an existing worker"
            )
        required = {"reason", "facts", "impact", "resume_condition", "next_action"}
        if set(detail) != required:
            raise ValueError(f"stop/resume requires exactly {sorted(required)}")
        detail = {key: nonempty(value, key) for key, value in detail.items()}
        task = self.row["tasks"].setdefault(str(child), {})
        operation = {"resume": resume, "detail_sha256": digest(detail)}
        if (
            resume
            and not child_stopped(children[child].body, task.get("stopped", False))
            and task.get("last_operation") != operation
        ):
            raise ValueError("child is not stopped")
        action = "再開" if resume else "停止"
        text = (
            action
            + "\n\n"
            + "\n\n".join(f"{key}: {value}" for key, value in detail.items())
        )
        text += f"\n\n<!-- parent-task-status:v1 state={'resumed' if resume else 'stopped'} detail={digest(detail)} -->"
        current = raw_section(children[child].body, "作業状況・停止事項")
        already_recorded = current.rstrip().endswith(text)
        operation_number = len(
            re.findall(r"(?m)^<!-- parent-task-status:v1 ", current)
        ) + (0 if already_recorded else 1)
        self.update_body(
            child,
            replace_section(
                children[child].body,
                "作業状況・停止事項",
                current if already_recorded else current + "\n\n" + text,
            ),
            children[child].body,
        )
        # Save after the detailed record, before parent notification: retries can repair progress.
        task["stopped"] = not resume
        task["last_status"] = detail
        task["last_operation"] = operation
        self.store.save()
        states = self.sync_progress() if conflict is None else {}
        held = (
            ", ".join(f"#{n}" for n, state in states.items() if state == "held")
            or "なし"
        )
        continuing = (
            ", ".join(f"#{n}" for n, state in states.items() if state == "continue")
            or "なし"
        )
        self.post(
            {
                "event": "child_resumed" if resume else "child_stopped",
                "child": child,
                "detail_sha256": digest(detail),
                "operation": operation_number,
            },
            f"{action}: #{child} — {detail['reason']}\n\n全体への影響: 依存保留 {held}。継続 {continuing}。\n\n次の対応: {detail['next_action']}。詳細・再開条件は #{child}。",
        )
        return {
            "child": child,
            "progress": states,
            **({"target_conflict": conflict} if conflict else {}),
        }

    def improvement(
        self, child: int | None, value: dict[str, Any], key: str | None = None
    ) -> dict[str, Any]:
        from .improvements import record

        return record(self, child, value, key)

    def record_result(self, child: int, evidence: str) -> dict[str, Any]:
        self.owner()
        nonempty(evidence, "verification evidence")
        accepted_turn_id = worker_preflight.require_implementation(self, child)
        # During checking this remains a factual record, without implementation
        # authorization. In an execution phase, stale inputs are an explicit error.
        auth = (
            execution.authorization(self)
            if phase(self.history()) in {"agreed", "autonomous"}
            else None
        )
        parent, children = self.work_items()
        if child not in children:
            raise ValueError("result requires a direct child or the standalone issue")
        from .policy import closing_issues

        pulls = self.gh.pulls()
        work_targets.require_separate_self_prs(parent, pulls, self.gh.repo)
        prs = [
            p
            for p in pulls
            if child in closing_issues(p.get("body") or "", self.gh.repo)
        ]
        if not prs:
            raise ValueError("no related PR; record a PR closing this child first")
        task = self.row["tasks"].get(str(child), {})
        if task.get("branch") and any(
            p["head"]["ref"] != task["branch"] for p in prs if p["state"] == "open"
        ):
            raise ValueError("result PR branch differs from delegated branch")
        refs = ", ".join(
            f"#{p['number']} ({p['state']})"
            for p in sorted(prs, key=lambda p: p["number"])
        )
        quoted = "\n".join("> " + line for line in evidence.splitlines())
        text = f"成果PR: {refs}\n\n検証結果:\n{quoted}"
        body = children[child].body
        current = raw_section(body, "成果物・関連PR")
        with self.gh.fresh_reads():
            if set(self.work_items()[1]) != set(children):
                raise work_targets.WorkTargetsChanged(
                    "work targets changed before result registration"
                )
        if not current.endswith(text):
            self.update_body(
                child,
                replace_section(
                    body, "成果物・関連PR", (current + "\n\n" + text).strip()
                ),
                body,
            )
        self.row["tasks"].setdefault(str(child), {})["verification"] = {
            "prs": [p["number"] for p in prs],
            "evidence": evidence,
            "authorization": auth,
            "requirements_sha256": digest(execution.requirement_body(body)),
        }
        recorded = self.row["tasks"][str(child)]
        if recorded.get("delegated") and "preflight" in recorded:
            recorded["preflight"]["accepted_turn_id"] = accepted_turn_id
        self.store.save()
        from decision_logs.parent_sources import verification

        verification(self, child)
        return {
            "child": child,
            "prs": [p["number"] for p in prs],
            "progress": self.sync_progress(),
        }
