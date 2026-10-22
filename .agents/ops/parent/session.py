"""Parent operations. No agent execution: delegation is a claim for the caller."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from issues import (
    CHILD_LABEL,
    PARENT_LABEL,
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

from . import execution, proposals
from .github import GitHub
from .policy import (
    PHASES,
    child_stopped,
    dependencies,
    digest,
    events,
    marker,
    nonempty,
    number,
    object_value,
    phase,
    progress,
    raw_section,
    registration,
    replace_section,
    review_state,
)
from .state import Store

TASK_MARKER = re.compile(
    r"(?m)^<!-- parent-task:v1 parent=([1-9][0-9]*) key=([a-z0-9][a-z0-9-]{0,63}) spec=([0-9a-f]{64}) -->$"
)


class ParentSession:
    def __init__(
        self, gh: GitHub, store: Store, parent: int, session: str | None
    ) -> None:
        self.gh, self.store, self.parent, self.session = gh, store, parent, session
        self.key = f"{gh.repo.casefold()}#{parent}"
        self.endpoint = f"repos/{gh.repo}/issues/{parent}"
        self.login = nonempty(object_value(gh.get("user")).get("login"), "gh login")

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
        return events(self.gh.comments(self.parent), self.login, self.parent)

    def agreed(self) -> dict[str, Any]:
        row = self.owner()
        raw = self.gh.issue(self.parent)
        if raw["state"] != "open":
            raise ValueError("parent issue is not open")
        execution.current_basis(self)
        return row

    def public(self, value: Any) -> None:
        serialized = json.dumps(value, ensure_ascii=False)
        secrets = {self.session.partition(":")[2]} if self.session else set()
        secrets.add(str(self.store.directory))
        for row in self.store.parents.values():
            secrets.add(row["session"].partition(":")[2])
            secrets.update(task.get("worktree", "") for task in row["tasks"].values())
        if any(secret and secret in serialized for secret in secrets):
            raise ValueError(
                "public payload contains private session/worktree information"
            )

    def post(self, event: dict[str, Any], text: str) -> None:
        data = {"parent": self.parent, **event}
        body = marker(data) + "\n\n" + text
        self.public(body)
        if any(item["body"] == body for item in self.history()):
            return
        self.gh.write("POST", self.endpoint + "/comments", {"body": body})

    def register(self) -> dict[str, Any]:
        if self.session is None:
            raise ValueError("session identity missing")
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
            }
            self.store.save()
        self.post(
            {"event": "assigned", "generation": self.row["generation"]},
            "担当開始。担当指定と要件合意は別段階です。要件を確認してから分割・委任します。",
        )
        return self.status()

    def status(self) -> dict[str, Any]:
        row = self.row
        history = self.history()
        return {
            "repo": self.gh.repo,
            "parent": self.parent,
            "assignment": row.get("status", "unassigned"),
            "current_session_is_owner": bool(row and row["session"] == self.session),
            "phase": PHASES[phase(history)],
            "tasks": row.get("tasks", {}),
            "review": self.review_status(history),
            "proposal_receipts": proposals.receipts(history),
            "proposal_agreements": proposals.agreed_receipts(history),
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

    def lifecycle(self, action: str, note: str) -> dict[str, Any]:
        row = self.owner()
        nonempty(note, "handoff/release note")
        self.post({"event": action, "generation": row["generation"]}, note)
        row["status"] = action
        self.store.save()
        return self.status()

    def accept(self, note: str, recover: bool = False) -> dict[str, Any]:
        if self.session is None:
            raise ValueError("session identity missing")
        row = self.row
        same_retry = (
            row
            and row["status"] == "active"
            and row["session"] == self.session
            and row.get("handoff_note_sha256") == digest(note)
        )
        if same_retry:
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
        # A retry after a failed comment must keep the claim and reconcile; it must not switch owners.
        row.update(
            session=self.session,
            status="active",
            generation=row["generation"] + 1,
            handoff_note_sha256=digest(note),
            handoff_recovery=recover,
        )
        self.store.save()
        self.post(
            {"event": "handoff", "generation": row["generation"], "recovery": recover},
            note,
        )
        return self.reconcile()

    def checking(self) -> dict[str, Any]:
        self.owner()
        history = self.history()
        if phase(history) == "checking":
            return self.status()
        revision = sum(e.get("event") == "checking" for e in history) + 1
        self.post(
            {
                "event": "checking",
                "generation": self.row["generation"],
                "revision": revision,
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
        agreement: dict[str, Any],
        report: int | None = None,
        pid: str | None = None,
    ) -> dict[str, Any]:
        self.owner()
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
                        "proposal agreement changed; run checking before revising requirements"
                    )
            event = {
                "event": "proposal_agreed",
                **receipt,
                "revision": last_phase.get("revision"),
            }
        else:
            if phase(history) == "agreed" and last_phase.get("agreement_sha256") == sha:
                self.agreed()  # Retrying agree must not silently accept new comments.
                return self.status()
            if phase(history) != "checking":
                raise ValueError(
                    "run checking and obtain human confirmation before recording agreement"
                )
            from .agreement_comments import snapshot as comment_inputs

            event = {
                "event": "agreed",
                "revision": last_phase.get("revision"),
                "comment_inputs": comment_inputs(self),
            }
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
        self.post(
            {
                **event,
                "agreement_sha256": sha,
                "agreement": agreement,
            },
            text,
        )
        self.agreed()
        return self.status()

    def family(self) -> tuple[Issue, dict[int, Issue]]:
        family = self.gh.family(self.parent)
        parent = family[self.parent]
        return parent, {n: family[n] for n in parent.children}

    def update_body(self, n: int, body: str, expected_body: str | None = None) -> None:
        self.public(body)
        raw = self.gh.issue(n)
        if (
            expected_body is not None
            and raw["body"] != expected_body
            and raw["body"] != body
        ):
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
        parent, children = self.family()
        if PARENT_LABEL not in parent.labels:
            if children:
                raise ValueError(
                    "parent role label missing; migrate explicitly using issues"
                )
            return {}
        states = progress(children, self.row["tasks"])
        tasks = [
            ChildTask(n, parse_title(child.title).summary, task_status(child))
            for n, child in children.items()
        ]
        body = replace_section(parent.body, TASKS_HEADING, render_child_tasks(tasks))
        labels = {
            "continue": "継続",
            "stopped": "停止",
            "held": "依存保留",
            "completed": "完了",
            "cancelled": "取消",
        }
        summary = (
            "\n".join(f"- #{n}: {labels[state]}" for n, state in sorted(states.items()))
            or "未分割（子issue未作成）"
        )
        self.update_body(
            self.parent, replace_section(body, "進捗概要", summary), parent.body
        )
        return states

    def create_child(self, value: dict[str, Any]) -> dict[str, Any]:
        self.agreed()  # Must precede render, search and every possible write.
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
        draft["labels"] = sorted(set(draft["labels"]) | {"ai-proposed"})
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
                    return {
                        "child": previous_number,
                        "reused": True,
                        "actual_parent": self.gh.identity(previous_parent),
                        "progress": self.sync_progress(),
                    }
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
        labels = {label["name"] for label in child["labels"]} | set(draft["labels"])
        self.gh.write("PATCH", base, {"labels": sorted(labels)})
        if current_parent is None:
            self.gh.write(
                "POST",
                self.endpoint + "/sub_issues",
                {"sub_issue_id": number(child["id"])},
            )
        task = self.row["tasks"].setdefault(str(n), {})
        task["dependencies"] = deps
        task.setdefault("creation_authorization", auth)
        self.store.save()
        states = self.sync_progress()
        if source is not None:
            proposals.require_source(self, source)
        return {"child": n, "reused": bool(found), "progress": states}

    def reconcile(self) -> dict[str, Any]:
        self.owner()
        _, children = self.family()
        states = progress(children, self.row["tasks"])
        prs = self.gh.pulls()
        from .policy import closing_issues

        result = []
        for n, child in sorted(children.items()):
            task = {
                **self.row["tasks"].get(str(n), {}),
                "dependencies": dependencies(child.body),
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
            action = (
                "do_not_launch"
                if matching or task.get("delegated") or states[n] != "continue"
                else "ready"
            )
            result.append(
                {
                    "child": n,
                    "state": states[n],
                    "action": action,
                    "assignment": task,
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
                }
            )
        return {
            "parent": self.parent,
            "phase": PHASES[phase(self.history())],
            "tasks": result,
            "review": self.review_status(self.history()),
        }

    def delegate(
        self,
        child: int,
        agent: str,
        branch: str,
        worktree: Path,
        deps: list[int] | None,
        restart: str | None,
    ) -> dict[str, Any]:
        self.agreed()
        auth = execution.authorization(self)
        parent, children = self.family()
        if child not in children:
            raise ValueError("delegate requires a direct child of this parent")
        source = parse_proposal_source(children[child].body)
        if source is not None:
            proposals.require_source(self, source)
        if deps is not None:
            if len(deps) != len(set(deps)):
                raise ValueError("dependencies must be unique")
            for dep in deps:
                number(dep)
        override = {child: deps} if deps is not None else None
        states = progress(children, self.row["tasks"], override)
        if states[child] != "continue":
            raise ValueError(f"#{child} cannot be delegated: {states[child]}")
        reconciled = next(t for t in self.reconcile()["tasks"] if t["child"] == child)
        previous = self.row["tasks"].get(str(child), {})
        if reconciled["prs"]:
            raise ValueError(
                "existing PR: reconcile instead of starting another worker"
            )
        if previous.get("delegated") and not restart:
            raise ValueError(
                "already delegated; reconcile the existing worker/worktree before an explicit restart"
            )
        if restart:
            nonempty(restart, "restart evidence")
        if agent not in {"claude", "codex"} or not branch.startswith(agent + "/"):
            raise ValueError("agent/branch prefix mismatch")
        from .snapshot import validate_worktree

        validate_worktree(worktree, branch)
        for n, task in self.row["tasks"].items():
            if n != str(child) and (
                task.get("branch") == branch
                or task.get("worktree") == str(worktree.resolve())
            ):
                raise ValueError("each child requires a dedicated branch/worktree")
        task = {
            **previous,
            "agent": agent,
            "branch": branch,
            "worktree": str(worktree.resolve()),
            "delegated": True,
            "dependencies": deps
            if deps is not None
            else dependencies(children[child].body),
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
        if restart:
            task["restart_evidence"] = restart
        if source is not None:
            proposals.require_source(self, source)
        if execution.authorization(self) != auth:
            raise ValueError("execution basis changed before delegation")
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
            "assignment": task,
            "parent": parent.number,
        }

    def stop_or_resume(
        self, child: int, detail: dict[str, Any], resume: bool
    ) -> dict[str, Any]:
        self.owner()
        _, children = self.family()
        if child not in children or task_status(children[child]) != "open":
            raise ValueError("stop/resume requires an open direct child")
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
        states = self.sync_progress()
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
        return {"child": child, "progress": states}

    def improvement(self, child: int, value: dict[str, Any]) -> dict[str, Any]:
        self.owner()
        _, children = self.family()
        if child not in children or set(value) != {"purpose", "evidence", "scope"}:
            raise ValueError(
                "improvement requires a direct child and purpose/evidence/scope"
            )
        text = "\n\n".join(f"{key}: {nonempty(v, key)}" for key, v in value.items())
        section = "次の改善候補"
        current = raw_section(children[child].body, section)
        if text not in current:
            self.update_body(
                child,
                replace_section(
                    children[child].body, section, (current + "\n\n" + text).strip()
                ),
                children[child].body,
            )
        return {"child": child, "recorded": True, "current_scope_changed": False}

    def record_result(self, child: int, evidence: str) -> dict[str, Any]:
        self.owner()
        nonempty(evidence, "verification evidence")
        # During checking this remains a factual record, without implementation
        # authorization. In an execution phase, stale inputs are an explicit error.
        auth = (
            execution.authorization(self)
            if phase(self.history()) in {"agreed", "autonomous"}
            else None
        )
        _, children = self.family()
        if child not in children:
            raise ValueError("result requires a direct child")
        from .policy import closing_issues

        prs = [
            p
            for p in self.gh.pulls()
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
        text = f"成果PR: {refs}\n\n検証結果:\n{evidence}"
        body = children[child].body
        current = raw_section(body, "成果物・関連PR")
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
        self.store.save()
        return {
            "child": child,
            "prs": [p["number"] for p in prs],
            "progress": self.sync_progress(),
        }
