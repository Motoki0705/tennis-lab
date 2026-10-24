"""Pinned, read-only proposal organization and strict task-plan validation."""

from __future__ import annotations

import json
import os
import re
import subprocess
import tempfile
from copy import deepcopy
from pathlib import Path
from typing import Any, cast

from dispatcher import collect as inputs
from issues import (
    CHILD_LABEL,
    PARENT_LABEL,
    Issue,
    ParentRef,
    ProposalSource,
    check_consistency,
    child_title,
    parse_proposal_source,
    parse_title,
    render_child_body,
    render_parent_body,
    render_standalone_body,
)
from parent.github import GitHub
from parent.policy import (
    child_stopped,
    digest,
    events,
    nonempty,
    number,
    object_value,
    phase,
)
from parent.state import Store, unique_object

from .proposals import ParsedProposal, parse_report

RUNNER = Path(__file__).resolve().parents[2] / "lib/agent_exec.sh"


def read_json(text: str) -> Any:
    def reject(value: str) -> None:
        raise ValueError(f"non-finite JSON: {value}")

    return json.loads(text, object_pairs_hook=unique_object, parse_constant=reject)


def save_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    fd, name = tempfile.mkstemp(dir=path.parent, prefix=path.name + ".", suffix=".tmp")
    temporary = Path(name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def key(report: int, pid: str) -> str:
    return f"{report}/{pid}"


def proposal_input(report: int, prop: ParsedProposal) -> dict[str, Any]:
    prop.requirements()
    selected, option = prop.selection()
    if not selected or prop.deferred_reason():
        raise ValueError(f"{prop.pid}: proposal must be adopted, without deferral")
    return {
        "source": {"report_number": report, "proposal_id": prop.pid, "option": option},
        "proposal_sha256": digest(prop.snapshot()),
        "proposal": prop.snapshot(),
        "related_parent": prop.parent(),
    }


def collect(gh: GitHub, store: Store, root: Path) -> dict[str, Any]:
    rows = gh.issue_rows()
    seeds = inputs.unique_numbers(rows, "all issues")
    snapshot = inputs.collect(gh, store, root, seeds=seeds)
    if set(snapshot["issues"]) != {str(n) for n in seeds}:
        raise ValueError("all-issue listing changed during proposal collection")
    return cast(dict[str, Any], snapshot)


def material(snapshot: dict[str, Any]) -> dict[str, Any]:
    """Keep instructions and adoption; ignore only triage's report display fields."""
    result = deepcopy(snapshot)
    for raw in result["issues"].values():
        if inputs.is_container(raw):
            # Parse even incomplete containers; malformed reports never become tasks.
            parse_report(raw["body"])
            raw["body"] = re.sub(r"(?m)^  - 引渡し: .*\n?", "", raw["body"])
            raw["body"] = re.sub(
                r"(?m)(^-[^\n]*\*\*R-[^\n]*) → #[1-9][0-9]*$", r"\1", raw["body"]
            )
    return result


def occupied(
    snapshot: dict[str, Any], n: int, *, completed_reuse: bool = False
) -> bool:
    raw = snapshot["issues"][str(n)]
    if (
        completed_reuse
        and raw["state"] == "closed"
        and raw["state_reason"] == "completed"
    ):
        # Provenance on completed work reserves no worker and preserves historical owners.
        return bool(
            "needs-human" in {label["name"] for label in raw["labels"]}
            or child_stopped(raw["body"])
            or any(p["state"] == "open" and n in p["issues"] for p in snapshot["pulls"])
        )
    root = inputs.route(snapshot["issues"], n)
    family: set[int] = set()
    pending = [root]
    while pending:
        current = pending.pop()
        if current not in family:
            family.add(current)
            pending.extend(snapshot["issues"][str(current)]["children"])
    return bool(
        any(snapshot["issues"][str(k)]["assignees"] for k in family)
        or any(
            "needs-human"
            in {label["name"] for label in snapshot["issues"][str(k)]["labels"]}
            or child_stopped(snapshot["issues"][str(k)]["body"])
            or phase(
                events(snapshot["issues"][str(k)]["comments"], snapshot["publisher"], k)
            )
            in {"checking", "agreed"}
            for k in family
        )
        or any(
            int(k) in family or set(row["tasks"]) & {str(n) for n in family}
            for k, row in snapshot["owners"].items()
            if row["status"] != "ended"
            or any(v["delegated"] for v in row["tasks"].values())
        )
        or any(
            p["state"] == "open" and set(p["issues"]) & family
            for p in snapshot["pulls"]
        )
    )


def validate_existing_family(snapshot: dict[str, Any], n: int) -> None:
    root = inputs.route(snapshot["issues"], n)
    family = {}
    for key, raw in snapshot["issues"].items():
        if inputs.route(snapshot["issues"], int(key)) == root:
            family[int(key)] = Issue(
                int(key),
                raw["title"],
                raw["body"],
                frozenset(label["name"] for label in raw["labels"]),
                raw["parent"],
                tuple(raw["children"]),
                raw["state"],
                raw["state_reason"],
            )
    problems = check_consistency(family)
    if problems:
        raise ValueError(f"existing issue family requires reconciliation: {problems}")


def execute(root: Path, run_dir: Path) -> Any:
    prompt = Path(__file__).parents[1] / "organize_prompt.md"
    prompt_path = run_dir / "organize-prompt.md"
    prompt_path.write_text(
        prompt.read_text(encoding="utf-8")
        + f"\n\n固定入力: {run_dir / 'organization-input.json'}\n",
        encoding="utf-8",
    )
    runtime_dir = run_dir / "organization-runtime"
    runner = Path(os.environ.get("OPS_REPORT_ORGANIZER_EXEC") or RUNNER).resolve()
    process = subprocess.run(
        [
            str(runner),
            "--role",
            "implementation",
            "--mode",
            "read-only",
            "--cwd",
            str(root),
            "--task-id",
            "adopted-proposal-organization",
            "--prompt-file",
            str(prompt_path),
            "--run-dir",
            str(runtime_dir),
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    (run_dir / "organization-runner.stdout").write_text(
        process.stdout, encoding="utf-8"
    )
    (run_dir / "organization-runner.stderr").write_text(
        process.stderr, encoding="utf-8"
    )
    if process.returncode:
        raise ValueError(f"proposal organizer runner failed: exit {process.returncode}")
    summary = object_value(read_json(process.stdout))
    evidence = object_value(read_json((runtime_dir / "result.json").read_text()))
    if (
        any(
            r.get("schema_version") != 1
            or r.get("result") != "completed"
            or r.get("turn_end_reason") != "completed"
            or r.get("exit_code") != 0
            or r.get("task_success") is not None
            for r in (summary, evidence)
        )
        or not evidence.get("run_id")
        or summary.get("run_id") != evidence["run_id"]
    ):
        raise ValueError("organizer did not provide matching completed turn evidence")
    contract = object_value(evidence.get("contract"))
    settings = object_value(contract.get("settings"))
    runtime = object_value(evidence.get("runtime"))
    effective = object_value(runtime.get("effective"))
    selected = object_value(settings.get("selected"))
    if (
        contract.get("mode") != "read-only"
        or contract.get("cwd") != str(root)
        or contract.get("task_id") != "adopted-proposal-organization"
        or settings.get("role") != "implementation"
        or settings.get("agent") != "codex"
        or settings.get("profile") is not None
        or settings.get("overrides") != {}
        or selected != settings.get("defaults")
        or runtime.get("runtime_errors")
        or evidence.get("cleanup_complete") is not True
        or any(
            object_value(effective.get(k)).get("status") != "observed"
            or effective[k].get("value") != selected.get(k)
            for k in ("model", "effort")
        )
    ):
        raise ValueError(
            "organizer runtime contract/effective settings/cleanup mismatch"
        )
    return read_json((runtime_dir / "final.md").read_text(encoding="utf-8"))


def _keys(row: dict[str, Any], fields: str) -> None:
    if set(row) != set(fields.split()):
        raise ValueError(f"organization schema keys must be: {fields}")


def validate(plan: Any, fixed: dict[str, Any]) -> dict[str, Any]:
    plan = object_value(plan)
    _keys(plan, "schema_version input_sha256 nodes decisions")
    if (
        type(plan["schema_version"]) is not int
        or plan["schema_version"] != 1
        or plan["input_sha256"] != fixed["input_sha256"]
    ):
        raise ValueError("organization version/input digest mismatch")
    if not isinstance(plan["nodes"], list) or not isinstance(plan["decisions"], list):
        raise ValueError("organization nodes/decisions must be lists")
    nodes: dict[str, Any] = {}
    existing: set[int] = set()
    snapshot = fixed["snapshot"]
    for node in plan["nodes"]:
        node = object_value(node)
        _keys(node, "key role issue parent identifier summary sections dependencies")
        name = node["key"]
        if (
            not isinstance(name, str)
            or not re.fullmatch(r"[a-z][a-z0-9-]{0,63}", name)
            or name in nodes
        ):
            raise ValueError("invalid/duplicate organization node")
        nodes[name] = node
        sections = object_value(node["sections"])
        if any(not isinstance(v, str) for v in sections.values()) or any(
            s in sections
            for s in ("採用提案の出典", "子タスクのチェックリスト", "依存関係")
        ):
            raise ValueError("sections contain reserved fields or non-text")
        role = node["role"]
        if role == "reuse":
            n = number(node["issue"])
            if n in existing or str(n) not in snapshot["issues"]:
                raise ValueError("unknown/duplicate reused issue")
            existing.add(n)
            raw = snapshot["issues"][str(n)]
            validate_existing_family(snapshot, n)
            if (
                inputs.is_container(raw)
                or parse_proposal_source(raw["body"]) is not None
            ):
                raise ValueError("reused issue is a report or already indexed proposal")
            if raw["state"] == "closed" and raw["state_reason"] != "completed":
                raise ValueError("cancelled/unknown closed issue is not completed work")
            if (
                node["parent"] is not None
                or node["identifier"] is not None
                or node["summary"]
                or sections
                or node["dependencies"]
            ):
                raise ValueError(
                    "reuse must preserve the existing issue and formal parent"
                )
        elif role in {"standalone", "parent", "child"}:
            if node["issue"] is not None:
                raise ValueError("new node must not specify an issue number")
            nonempty(node["summary"], "task summary")
            child_title(ParentRef("task", 1), node["summary"])
            if role == "parent":
                if node["parent"] is not None:
                    raise ValueError("new parents must be top-level")
                ParentRef(node["identifier"], 1)
                render_parent_body(sections)
            else:
                if node["identifier"] is not None:
                    raise ValueError("only a parent has an identifier")
                render_standalone_body({**sections, "依存関係": "なし。"})
                parent = node["parent"]
                if role == "standalone" and parent is not None:
                    raise ValueError("standalone has no parent")
                if role == "child":
                    if type(parent) is int:
                        parent = number(parent)
                        raw = snapshot["issues"].get(str(parent))
                        if (
                            not raw
                            or inputs.is_container(raw)
                            or raw["state"] != "open"
                            or PARENT_LABEL not in {x["name"] for x in raw["labels"]}
                        ):
                            raise ValueError(
                                "existing parent must be an open development parent"
                            )
                        parsed = parse_title(raw["title"])
                        if not parsed.identifier:
                            raise ValueError(
                                "existing parent needs the common issue format"
                            )
                        validate_existing_family(snapshot, parent)
                    elif not isinstance(parent, str):
                        raise ValueError("child must name a parent")
        else:
            raise ValueError("unknown organization node role")
        if not isinstance(node["dependencies"], list) or len(
            {str(d) for d in node["dependencies"]}
        ) != len(node["dependencies"]):
            raise ValueError("dependencies must be a unique list")
    for node in nodes.values():
        if isinstance(node["parent"], str) and (
            node["parent"] not in nodes or nodes[node["parent"]]["role"] != "parent"
        ):
            raise ValueError("unknown/new parent reference")
        if (
            node["role"] == "parent"
            and sum(n["parent"] == node["key"] for n in nodes.values()) < 2
        ):
            raise ValueError("a new parent requires at least two concrete children")
        for dep in node["dependencies"]:
            if type(dep) is int and str(number(dep)) in snapshot["issues"]:
                if inputs.is_container(snapshot["issues"][str(dep)]):
                    raise ValueError("reports cannot be task dependencies")
            elif not isinstance(dep, str) or dep not in nodes:
                raise ValueError("unknown dependency")
    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(name: str) -> None:
        if name in visiting:
            raise ValueError("cyclic organization dependencies")
        if name in visited:
            return
        visiting.add(name)
        node = nodes[name]
        for ref in [node["parent"], *node["dependencies"]]:
            if isinstance(ref, str):
                visit(ref)
        visiting.remove(name)
        visited.add(name)

    for name in nodes:
        visit(name)
    proposals = fixed["proposals"]
    found: set[str] = set()
    anchors: set[str] = set()
    for decision in plan["decisions"]:
        decision = object_value(decision)
        _keys(
            decision,
            "proposal action target reason evidence resume_condition next_action",
        )
        name = decision["proposal"]
        if name not in proposals or name in found:
            raise ValueError("unknown/duplicate proposal decision")
        found.add(name)
        for field in ("reason", "resume_condition", "next_action"):
            nonempty(decision[field], field)
        if not isinstance(decision["evidence"], list) or not decision["evidence"]:
            raise ValueError("organization decision requires evidence")
        for ref in decision["evidence"]:
            if ref.startswith("issue:") and ref[6:] in snapshot["issues"]:
                continue
            prefix = f"code:{snapshot['code']['commit']}:"
            if ref.startswith(prefix) and ref[len(prefix) :] in {
                f["path"] for f in snapshot["code"]["files"]
            }:
                continue
            raise ValueError("organization evidence is not in the pinned inputs")
        target = decision["target"]
        if decision["action"] == "hold":
            if target is not None:
                raise ValueError("a hold has no target")
        elif decision["action"] == "issue":
            if target not in nodes or target in anchors:
                raise ValueError("each proposal needs its own unique index anchor")
            anchors.add(target)
            node = nodes[target]
            candidate = node["issue"] if node["role"] == "reuse" else node["parent"]
            if type(candidate) is int and occupied(
                snapshot, candidate, completed_reuse=node["role"] == "reuse"
            ):
                raise ValueError(
                    "existing owner/assignee/PR conflict; return a reasoned hold"
                )
            explicit = proposals[name]["related_parent"]
            if (
                explicit is not None
                and node["role"] != "reuse"
                and node["parent"] != explicit
            ):
                raise ValueError(
                    "do not silently replace the proposal's related parent"
                )
        else:
            raise ValueError("unknown organization action")
    if found != set(proposals):
        raise ValueError("organization decisions do not cover all proposals")
    for name in nodes:
        if not bindings(plan, proposals, name):
            raise ValueError("organization contains an unrelated task")
    return cast(dict[str, Any], plan)


def bindings(
    plan: dict[str, Any], proposals: dict[str, Any], name: str
) -> list[dict[str, Any]]:
    nodes = {n["key"]: n for n in plan["nodes"]}
    result = {}
    for decision in plan["decisions"]:
        if decision["action"] != "issue":
            continue
        target = decision["target"]
        if (
            name == target
            or nodes[name]["parent"] == target
            or nodes[target]["parent"] == name
        ):
            row = proposals[decision["proposal"]]
            result[decision["proposal"]] = {
                k: row[k] for k in ("source", "proposal_sha256")
            }
    return sorted(
        result.values(),
        key=lambda b: (b["source"]["report_number"], b["source"]["proposal_id"]),
    )


def source_section(
    proposals: dict[str, Any], bound: list[dict[str, Any]], repo: str
) -> str:
    parts = [
        "方向性の採用を記録する。Issue整理は人間の要件合意・自律着手根拠を生成しない。"
    ]
    for binding in bound:
        source = ProposalSource(**binding["source"])
        row = proposals[key(source.report_number, source.proposal_id)]
        prop = row["proposal"]
        parts.append(
            f"出典: https://github.com/{repo}/issues/{source.report_number} / {source.proposal_id}\n"
            f"選択肢: {source.option or 'なし'}\n内容SHA-256: {row['proposal_sha256']}\n"
            f"提案: {prop['title']}\n" + "\n".join(prop["detail"])
        )
    return "\n\n".join(parts)


def render_node(
    node: dict[str, Any],
    numbers: dict[str, int],
    fixed: dict[str, Any],
    plan: dict[str, Any],
) -> dict[str, Any]:
    sections = dict(node["sections"])
    sections["依存関係"] = (
        "\n".join(
            f"- #{numbers[d] if isinstance(d, str) else d}"
            for d in node["dependencies"]
        )
        or "なし。"
    )
    role = node["role"]
    labels = ["ai-proposed"]
    if role == "parent":
        from issues import parent_title

        title = parent_title(node["identifier"], node["summary"])
        body = render_parent_body(sections)
        labels.append(PARENT_LABEL)
    elif role == "child":
        p = node["parent"]
        n = numbers[p] if isinstance(p, str) else p
        identifier = (
            next(v["identifier"] for v in plan["nodes"] if v["key"] == p)
            if isinstance(p, str)
            else parse_title(fixed["snapshot"]["issues"][str(p)]["title"]).identifier
        )
        if identifier is None:
            raise ValueError("parent identifier disappeared from the fixed plan")
        parent = ParentRef(identifier, n)
        title, body = (
            child_title(parent, node["summary"]),
            render_child_body(n, sections),
        )
        labels.append(CHILD_LABEL)
    else:
        title, body = node["summary"], render_standalone_body(sections)
    return {"title": title, "body": body, "labels": labels}
