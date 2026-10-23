"""Journaled issue publication. Never reserve an owner or launch development."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

from issues import (
    NO_CHILDREN,
    TASKS_HEADING,
    ChildTask,
    Issue,
    ProposalSource,
    bind_task,
    check_consistency,
    index_proposals,
    parse_child_tasks,
    parse_proposal_source,
    parse_sections,
    parse_task_metadata,
    parse_title,
    render_child_tasks,
    task_status,
    validate_body,
)
from parent.github import GitHub
from parent.policy import digest, replace_section
from parent.state import Store

from . import organization as org
from . import schedule
from .proposals import annotate_children, annotate_handoffs, parse_report


def _binding(row: dict[str, Any]) -> dict[str, Any]:
    return {k: row[k] for k in ("source", "proposal_sha256")}


def _source(tx: dict[str, Any], name: str) -> ProposalSource | None:
    for d in tx["plan"]["decisions"]:
        if d["action"] == "issue" and d["target"] == name:
            return ProposalSource(**tx["fixed"]["proposals"][d["proposal"]]["source"])
    return None


def _numbers(tx: dict[str, Any], snapshot: dict[str, Any]) -> dict[str, int]:
    numbers = dict(tx["numbers"])
    nodes = {node["key"]: node for node in tx["plan"]["nodes"]}
    found: dict[str, int] = {}
    for raw in snapshot["issues"].values():
        metadata = parse_task_metadata(raw["body"])
        if metadata is None or metadata["plan_sha256"] != tx["plan_sha256"]:
            continue
        name = metadata["node"]
        if name not in nodes or name in found:
            raise ValueError("unknown/duplicate task identity in GitHub")
        expected = org.bindings(tx["plan"], tx["fixed"]["proposals"], name)
        if metadata["bindings"] != expected or parse_proposal_source(
            raw["body"]
        ) != _source(tx, name):
            raise ValueError(
                "task source/digest does not match the saved organization plan"
            )
        found[name] = raw["number"]
    for name, n in numbers.items():
        if found.get(name) != n:
            raise ValueError(
                "a journaled task disappeared or lost its provenance; do not recreate it"
            )
    numbers.update(found)
    for node in nodes.values():
        if node["role"] == "reuse":
            name, n = node["key"], node["issue"]
            if name in numbers and numbers[name] != n:
                raise ValueError("reused task identity changed")
    return numbers


def _check_sources(gh: GitHub, tx: dict[str, Any]) -> None:
    reports = {}
    targets = {d["proposal"] for d in tx["plan"]["decisions"] if d["action"] == "issue"}
    for name, row in tx["fixed"]["proposals"].items():
        if tx["complete"] and name not in targets:
            continue
        source = ProposalSource(**row["source"])
        if source.report_number not in reports:
            raw = gh.issue(source.report_number)
            meta, parsed = parse_report(raw["body"])
            if raw["state"].lower() != "open" or meta.status != "complete":
                raise ValueError("source report is closed or incomplete")
            reports[source.report_number] = {p.pid: p for p in parsed}
        prop = reports[source.report_number].get(source.proposal_id)
        if prop is None or org.proposal_input(source.report_number, prop) != row:
            raise ValueError(
                f"{source.proposal_id}: adoption/option/proposal/related parent changed"
            )


def _guard(gh: GitHub, store: Store, root: Path, tx: dict[str, Any]) -> dict[str, Any]:
    """Subtract only this journal's exact writes; preserve every external input."""
    _check_sources(gh, tx)
    current = org.collect(gh, store, root)
    numbers = _numbers(tx, current)
    baseline = tx["fixed"]["snapshot"]
    normalized = deepcopy(current)
    nodes = {node["key"]: node for node in tx["plan"]["nodes"]}
    new_ids = {n for n in numbers.values() if str(n) not in baseline["issues"]}
    for name, n in numbers.items():
        node = nodes[name]
        raw = normalized["issues"][str(n)]
        draft = tx["drafts"].get(name)
        if draft is None:
            raise ValueError("task exists without a saved publication draft")
        if org.occupied(current, n, completed_reuse=node["role"] == "reuse"):
            raise ValueError(
                "task acquired an owner/assignee/PR during issue publication"
            )
        expected_body = draft["body"]
        actual_body = raw["body"]
        if node["role"] == "parent":
            # The checklist is the only part populated after creating children.
            actual_body = replace_section(actual_body, TASKS_HEADING, NO_CHILDREN)
            expected_body = replace_section(expected_body, TASKS_HEADING, NO_CHILDREN)
        if actual_body != expected_body:
            raise ValueError(
                "task content changed during publication; reconcile the existing issue"
            )
        if node["role"] == "child":
            p = node["parent"]
            desired = numbers.get(p) if isinstance(p, str) else p
            if raw["parent"] not in (None, desired):
                raise ValueError("task belongs to another parent; do not reparent it")
        elif str(n) not in baseline["issues"] and raw["parent"] is not None:
            raise ValueError("new root unexpectedly belongs to another parent")
        if str(n) in baseline["issues"]:
            raw["body"] = baseline["issues"][str(n)]["body"]
            raw["source_reports"] = baseline["issues"][str(n)]["source_reports"]
        else:
            if raw["comments"] or {v["name"] for v in raw["labels"]} != set(
                draft["labels"]
            ):
                raise ValueError(
                    "new task instructions/labels changed during publication"
                )
            if raw["state"] == "closed" and raw["state_reason"] != "completed":
                raise ValueError(
                    "new task was cancelled; reconcile instead of recreating"
                )
            normalized["issues"].pop(str(n))
    # Existing parent checklists may contain exactly the old rows plus our children.
    for n, raw in normalized["issues"].items():
        original = baseline["issues"].get(n)
        own_children = set(raw["children"]) & new_ids
        if own_children:
            if original is None:
                raise ValueError("unexpected existing parent")
            original_tasks = parse_child_tasks(
                parse_sections(original["body"])[TASKS_HEADING]
            )
            allowed = {t.number: t for t in original_tasks}
            for child in own_children:
                item = current["issues"][str(child)]
                allowed[child] = ChildTask(
                    child,
                    parse_title(item["title"]).summary,
                    task_status(
                        Issue(
                            child,
                            item["title"],
                            item["body"],
                            frozenset(),
                            state=item["state"],
                            state_reason=item["state_reason"],
                        )
                    ),
                )
            tasks = parse_child_tasks(parse_sections(raw["body"])[TASKS_HEADING])
            actual_tasks = {t.number: t for t in tasks}
            if actual_tasks not in ({t.number: t for t in original_tasks}, allowed):
                raise ValueError("existing parent checklist changed outside this plan")
            raw["body"] = original["body"]
        raw["children"] = [child for child in raw["children"] if child not in new_ids]
    normalized["subjects"] = [n for n in normalized["subjects"] if n not in new_ids]
    if org.material(normalized) != org.material(baseline):
        raise ValueError(
            "organization inputs/ownership/code changed; retry with current evidence"
        )
    return current


def _status(
    gh: GitHub,
    proposals: dict[str, Any],
    statuses: dict[str, str],
    links: dict[str, int],
) -> None:
    reports = sorted({row["source"]["report_number"] for row in proposals.values()})
    for n in reports:
        raw = gh.issue(n)
        _, parsed = parse_report(raw["body"])
        annotations, handoffs = {}, {}
        for prop in parsed:
            name = org.key(n, prop.pid)
            if name in statuses:
                handoffs[prop.pid] = " ".join(statuses[name].split())
            if name in links:
                if org.proposal_input(n, prop) != proposals[name]:
                    raise ValueError(
                        "proposal changed before linking the generated issue"
                    )
                actual = parse_proposal_source(gh.issue(links[name])["body"])
                if actual != ProposalSource(**proposals[name]["source"]):
                    raise ValueError(
                        "generated issue source changed before report annotation"
                    )
                annotations[prop.pid] = links[name]
        body = annotate_handoffs(annotate_children(raw["body"], annotations), handoffs)
        if body != raw["body"]:
            if gh.issue(n)["body"] != raw["body"]:
                raise ValueError("report changed before triage status update")
            gh.write("PATCH", f"repos/{gh.repo}/issues/{n}", {"body": body})


def _topological(plan: dict[str, Any]) -> list[dict[str, Any]]:
    pending = list(plan["nodes"])
    ordered: list[dict[str, Any]] = []
    while pending:
        known = {n["key"] for n in ordered}
        ready = next(
            n
            for n in pending
            if all(
                not isinstance(ref, str) or ref in known
                for ref in [n["parent"], *n["dependencies"]]
            )
        )
        ordered.append(ready)
        pending.remove(ready)
    return ordered


class Journal:
    def __init__(self, repo: str, path: Path | None = None) -> None:
        self.path = (
            path or schedule.state_dir() / f"triage-{digest(repo.casefold())[:16]}.json"
        )
        self.data: dict[str, Any] = (
            org.read_json(self.path.read_text())
            if self.path.exists() or self.path.is_symlink()
            else {"schema_version": 1, "repo": repo.casefold(), "transactions": []}
        )
        if (
            not isinstance(self.data, dict)
            or set(self.data) != {"schema_version", "repo", "transactions"}
            or type(self.data["schema_version"]) is not int
            or self.data["schema_version"] != 1
            or self.data["repo"] != repo.casefold()
            or not isinstance(self.data["transactions"], list)
        ):
            raise ValueError(
                "invalid triage journal; preserve and reconcile it explicitly"
            )
        issued: set[str] = set()
        for tx in self.data["transactions"]:
            if (
                set(tx)
                != {
                    "fixed",
                    "plan",
                    "plan_sha256",
                    "numbers",
                    "drafts",
                    "attempted",
                    "complete",
                }
                or type(tx["complete"]) is not bool
            ):
                raise ValueError("invalid triage transaction")
            fixed = tx["fixed"]
            if (
                not isinstance(fixed, dict)
                or set(fixed)
                != {"schema_version", "snapshot", "proposals", "input_sha256"}
                or type(fixed["schema_version"]) is not int
                or fixed["schema_version"] != 1
            ):
                raise ValueError("unknown or malformed saved organization input")
            if fixed["input_sha256"] != digest(
                {
                    "snapshot": org.material(fixed["snapshot"]),
                    "proposals": fixed["proposals"],
                }
            ) or tx["plan_sha256"] != digest(tx["plan"]):
                raise ValueError("triage journal content digest mismatch")
            org.validate(tx["plan"], fixed)
            node_keys = {node["key"] for node in tx["plan"]["nodes"]}
            if (
                not isinstance(tx["numbers"], dict)
                or not isinstance(tx["drafts"], dict)
                or not set(tx["numbers"]) <= node_keys
                or not set(tx["drafts"]) <= node_keys
            ):
                raise ValueError("invalid saved task identities/drafts")
            for name, n in tx["numbers"].items():
                if type(n) is not int or n < 1 or name not in tx["drafts"]:
                    raise ValueError("invalid saved issue number")
            for name, draft in tx["drafts"].items():
                if (
                    not isinstance(draft, dict)
                    or set(draft) != {"title", "body", "labels"}
                    or not isinstance(draft["title"], str)
                    or not isinstance(draft["body"], str)
                    or not isinstance(draft["labels"], list)
                ):
                    raise ValueError("invalid saved issue draft")
                metadata = parse_task_metadata(draft["body"])
                if (
                    metadata is None
                    or metadata
                    != {
                        "plan_sha256": tx["plan_sha256"],
                        "node": name,
                        "bindings": org.bindings(tx["plan"], fixed["proposals"], name),
                    }
                    or parse_proposal_source(draft["body"]) != _source(tx, name)
                ):
                    raise ValueError("saved issue draft provenance changed")
            if (
                not isinstance(tx["attempted"], list)
                or any(not isinstance(name, str) for name in tx["attempted"])
                or len(set(tx["attempted"])) != len(tx["attempted"])
                or not set(tx["attempted"]) <= node_keys
            ):
                raise ValueError("invalid triage creation attempts")
            for d in tx["plan"]["decisions"]:
                if d["action"] == "issue":
                    if d["proposal"] in issued:
                        raise ValueError(
                            "proposal occurs in multiple issue transactions"
                        )
                    issued.add(d["proposal"])

    def save(self) -> None:
        org.save_json(self.path, self.data)

    def add(self, fixed: dict[str, Any], plan: dict[str, Any]) -> dict[str, Any]:
        tx = {
            "fixed": fixed,
            "plan": plan,
            "plan_sha256": digest(plan),
            "numbers": {},
            "drafts": {},
            "attempted": [],
            "complete": False,
        }
        self.data["transactions"].append(tx)
        self.save()
        return tx

    def check_tasks(self, snapshot: dict[str, Any]) -> None:
        """Require saved plans for context-only tasks as well as index anchors."""
        checked: set[str] = set()
        for raw in snapshot["issues"].values():
            metadata = parse_task_metadata(raw["body"])
            if metadata is None:
                continue
            plan_sha = metadata["plan_sha256"]
            matches = [
                tx for tx in self.data["transactions"] if tx["plan_sha256"] == plan_sha
            ]
            if len(matches) != 1:
                reason = (
                    "no journal transaction"
                    if not matches
                    else "multiple journal transactions"
                )
                raise ValueError(
                    f"managed task issue #{raw['number']} has {reason} for plan {plan_sha}; "
                    "restore the triage journal from saved organization-input.json, "
                    "organization-plan.json, publication drafts and GitHub ops-report-task "
                    "markers before retrying; do not create new issues"
                )
            if plan_sha not in checked:
                _numbers(matches[0], snapshot)
                checked.add(plan_sha)


def _apply(
    gh: GitHub, store: Store, root: Path, journal: Journal, tx: dict[str, Any]
) -> dict[str, int]:
    if tx["complete"]:
        _check_sources(gh, tx)
        current = org.collect(gh, store, root)
        numbers = _numbers(tx, current)
    else:
        current = _guard(gh, store, root, tx)
        numbers = _numbers(tx, current)
        tx["numbers"] = numbers
        journal.save()
        for node in _topological(tx["plan"]):
            name = node["key"]
            if name in numbers:
                continue
            if node["role"] != "reuse" and name in tx["attempted"]:
                raise ValueError(
                    "prior issue creation outcome is unknown and no matching issue was found; reconcile GitHub before retrying"
                )
            current = _guard(gh, store, root, tx)
            bound = org.bindings(tx["plan"], tx["fixed"]["proposals"], name)
            if node["role"] == "reuse":
                n = node["issue"]
                raw = current["issues"][str(n)]
                draft = {
                    "title": raw["title"],
                    "body": raw["body"],
                    "labels": [label["name"] for label in raw["labels"]],
                }
            else:
                draft = org.render_node(node, numbers, tx["fixed"], tx["plan"])
            draft["body"] = bind_task(
                replace_section(
                    draft["body"],
                    "採用提案の出典",
                    org.source_section(tx["fixed"]["proposals"], bound, gh.repo),
                ),
                plan=tx["plan_sha256"],
                node=name,
                bindings=bound,
                source=_source(tx, name),
            )
            errors = validate_body(
                draft["body"],
                is_parent="issue:parent" in draft["labels"],
                is_child="issue:child" in draft["labels"],
            )
            if errors:
                raise ValueError("invalid publication draft: " + "; ".join(errors))
            tx["drafts"][name] = draft
            journal.save()  # An ambiguous create response can always find this exact node.
            _guard(gh, store, root, tx)
            if node["role"] == "reuse":
                created = gh.write(
                    "PATCH",
                    f"repos/{gh.repo}/issues/{node['issue']}",
                    {"body": draft["body"]},
                )
            else:
                tx["attempted"].append(name)
                journal.save()
                created = gh.write("POST", f"repos/{gh.repo}/issues", draft)
            n = gh.identity(created)
            numbers[name] = n
            tx["numbers"] = numbers
            journal.save()
            _guard(gh, store, root, tx)
        for node in tx["plan"]["nodes"]:
            if node["role"] != "child":
                continue
            current = _guard(gh, store, root, tx)
            n = numbers[node["key"]]
            p = node["parent"]
            parent = numbers[p] if isinstance(p, str) else p
            actual = current["issues"][str(n)]["parent"]
            if actual is None:
                gh.write(
                    "POST",
                    f"repos/{gh.repo}/issues/{parent}/sub_issues",
                    {"sub_issue_id": gh.issue(n)["id"]},
                )
                _guard(gh, store, root, tx)
            elif actual != parent:
                raise ValueError(
                    "existing task belongs to another parent; do not reparent it"
                )
        parents = {
            numbers[n["parent"]] if isinstance(n["parent"], str) else n["parent"]
            for n in tx["plan"]["nodes"]
            if n["role"] == "child"
        }
        for p in parents:
            _guard(gh, store, root, tx)
            family = gh.family(p)
            parent_issue = family[p]
            tasks = [
                ChildTask(
                    n, parse_title(family[n].title).summary, task_status(family[n])
                )
                for n in parent_issue.children
            ]
            body = replace_section(
                parent_issue.body, TASKS_HEADING, render_child_tasks(tasks)
            )
            if body != parent_issue.body:
                gh.write("PATCH", f"repos/{gh.repo}/issues/{p}", {"body": body})
            _guard(gh, store, root, tx)
        for n in numbers.values():
            problems = check_consistency(gh.family(n))
            if problems:
                raise ValueError(f"published task family is inconsistent: {problems}")
        tx["numbers"] = numbers
        tx["complete"] = True
        journal.save()
    links = {
        d["proposal"]: numbers[d["target"]]
        for d in tx["plan"]["decisions"]
        if d["action"] == "issue"
    }
    return links


def run(
    *,
    gh: GitHub,
    store: Store,
    root: Path,
    work_dir: Path,
    candidates: list[tuple[int, Any]],
    dry_run: bool,
    result: Any,
    journal_path: Path | None = None,
) -> None:
    """Caller holds the shared triage lock and the existing registry lock."""
    journal = Journal(gh.repo, journal_path)
    work_dir.chmod(0o700)
    try:
        snapshot = org.collect(gh, store, root)
    except ValueError as exc:
        result.errors.append(str(exc))
        return
    journal.check_tasks(snapshot)
    candidate_keys = {org.key(n, p.pid) for n, p in candidates}
    issued = {
        d["proposal"]
        for tx in journal.data["transactions"]
        for d in tx["plan"]["decisions"]
        if d["action"] == "issue"
    }
    for tx in journal.data["transactions"]:
        active = candidate_keys & set(tx["fixed"]["proposals"])
        if not active or not any(
            d["action"] == "issue" for d in tx["plan"]["decisions"]
        ):
            continue
        try:
            if dry_run:
                continue
            links = _apply(gh, store, root, journal, tx)
            statuses = {
                k: f"作業issue #{n}。着手判断・親割当は次の08:00・20:00 JST予定回。"
                for k, n in links.items()
            }
            _status(gh, tx["fixed"]["proposals"], statuses, links)
            result.linked_existing.extend(
                (k.split("/", 1)[1], n) for k, n in links.items()
            )
        except ValueError as exc:
            result.errors.append(str(exc))
            statuses = {
                k: f"保留: {exc}。根拠: 保存済み出典との再照合。再開条件: 採用・本文・担当と既存issueを照合して解消。次の対応: 元の採用に戻すか別提案として整理。既存issueは保持。"
                for k in active
            }
            if not dry_run:
                _status(gh, tx["fixed"]["proposals"], statuses, {})
    pending = {}
    for n, prop in candidates:
        name = org.key(n, prop.pid)
        if name in issued:
            continue
        selected, _ = prop.selection()
        if selected:
            pending[name] = org.proposal_input(n, prop)
    if not pending:
        return
    snapshot = org.collect(gh, store, root)
    journal.check_tasks(snapshot)
    index = index_proposals(
        (int(n), raw["body"]) for n, raw in snapshot["issues"].items()
    )
    if any(
        (r["source"]["report_number"], r["source"]["proposal_id"]) in index
        for r in pending.values()
    ):
        raise ValueError(
            "managed proposal has no journal; restore the journal rather than creating another issue"
        )
    fixed = {"schema_version": 1, "snapshot": snapshot, "proposals": pending}
    fixed["input_sha256"] = digest(
        {"snapshot": org.material(snapshot), "proposals": pending}
    )
    cached = next(
        (
            tx
            for tx in reversed(journal.data["transactions"])
            if tx["fixed"]["input_sha256"] == fixed["input_sha256"]
        ),
        None,
    )
    if cached is None:
        org.save_json(work_dir / "organization-input.json", fixed)
        plan = org.validate(org.execute(root, work_dir), fixed)
        org.save_json(work_dir / "organization-plan.json", plan)
        fresh = org.collect(gh, store, root)
        if org.material(fresh) != org.material(snapshot):
            raise ValueError("proposal organization inputs changed before publication")
        if dry_run:
            return
        tx = journal.add(fixed, plan)
    else:
        tx = cached
    if dry_run:
        return
    try:
        links = _apply(gh, store, root, journal, tx)
    except ValueError as exc:
        result.errors.append(str(exc))
        _status(
            gh,
            pending,
            {
                name: f"保留: {exc}。根拠: Issue整理入力・出典・担当の再照合。再開条件: 現在の採用と既存issueの相違を解消する。次の対応: 保存済み計画から不足分を再開する。既存issueは保持。"
                for name in pending
            },
            {},
        )
        return
    statuses = {}
    for d in tx["plan"]["decisions"]:
        name = d["proposal"]
        if d["action"] == "hold":
            statuses[name] = (
                f"保留: {d['reason']}。根拠: {' / '.join(d['evidence'])}。再開条件: {d['resume_condition']}。次の対応: {d['next_action']}"
            )
            result.unassigned.append(name.split("/", 1)[1])
        else:
            statuses[name] = (
                f"作業issue #{links[name]}。着手判断・親割当は次の08:00・20:00 JST予定回。"
            )
            result.linked_existing.append((name.split("/", 1)[1], links[name]))
    _status(gh, pending, statuses, links)
