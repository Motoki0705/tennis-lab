"""Pure, deterministic navigation over public envelopes and existing worker logs."""

from __future__ import annotations

import heapq
import html
import posixpath
import re
from collections import defaultdict
from pathlib import PurePosixPath
from typing import Any
from urllib.parse import quote, urlsplit

from worker_logs import schema as worker_schema
from worker_logs.journal import decode, require
from worker_logs.safety import Guard

from .aggregation import aggregate
from .graph import references, states, validate_graph
from .projection import validate_public
from .schema import timestamp

ROOT = ".agents/logs/parents"
EVENT = re.compile(
    rf"{re.escape(ROOT)}/parent-([1-9][0-9]*)/activity-([a-f0-9]{{32}})"
    r"/event-([a-f0-9]{32})\.json"
)
WORKER = re.compile(
    rf"{re.escape(ROOT)}/parent-([1-9][0-9]*)/child-([1-9][0-9]*)"
    r"/(d-[a-f0-9]{32})/([0-9]{4,}-(?:initial|resume))"
    r"/(instruction\.md|final\.md|meta\.json)"
)
KIND = {"result": "結果", "decision": "判断", "action": "行動", "correction": "訂正"}
STATE = {
    "decided_not_executed": "判断済み未実行",
    "requested": "要求済み・実行未確認",
    "rejected_before_start": "起動前拒否",
    "running": "実行中",
    "ended": "終了（成果受入とは別）",
    "unknown": "不明",
}
OPERATION = {
    "continue_ai": "同じAIの継続",
    "start_ai": "新しいAIの起動",
    "retry_fetch": "取得だけの再試行",
    "retry_publication": "公開だけの再試行",
    "review": "レビュー",
    "merge": "merge",
    "fetch": "取得",
    "verify": "検証",
    "publish": "公開",
    "stop": "停止",
    "resume": "再開",
    "handoff": "引継ぎ",
    "record": "記録",
}


def event_path(record: dict[str, Any]) -> str:
    scope = record["scope"]
    return (
        f"{ROOT}/parent-{scope['parent']}/activity-{scope['activity_key']}"
        f"/event-{record['id']}.json"
    )


def records(files: dict[str, bytes]) -> list[dict[str, Any]]:
    rows = []
    for name, raw in sorted(files.items()):
        # Malformed event paths must not disappear from the generator's input.
        if PurePosixPath(name).name.startswith("event-"):
            match = EVENT.fullmatch(name)
            require(match is not None, "unsafe_reference")
            row = validate_public(decode(raw))["record"]
            require(name == event_path(row), "record_source")
            rows.append(row)
    ordered = validate_graph(rows)
    require(len({r["scope"]["repo"] for r in rows}) <= 1, "record_repository")
    remaining = {key: set(references(r)) for key, r in ordered.items()}
    successors: dict[str, list[str]] = defaultdict(list)
    for key, refs in remaining.items():
        for ref in refs:
            successors[ref].append(key)
    ready = [
        (timestamp(r["time"]["recorded_at"]), key)
        for key, r in ordered.items()
        if not remaining[key]
    ]
    heapq.heapify(ready)
    result = []
    while ready:
        _, key = heapq.heappop(ready)
        result.append(ordered[key])
        for successor in successors[key]:
            remaining[successor].remove(key)
            if not remaining[successor]:
                heapq.heappush(
                    ready,
                    (timestamp(ordered[successor]["time"]["recorded_at"]), successor),
                )
    return result


def prose(value: Any) -> str:
    """Render captured prose as text, never as active Markdown or HTML."""
    text = "".join(
        f"&#{ord(char)};" if char in "\\`*_{}[]()!#|" else html.escape(char, quote=True)
        for char in str(value)
    )
    return text.replace("\r\n", "<br>").replace("\r", "<br>").replace("\n", "<br>")


def link(source: str, target: str, label: str, anchor: str = "") -> str:
    relative = posixpath.relpath(target, posixpath.dirname(source))
    return f"[{prose(label)}]({relative}{anchor})"


def evidence_link(item: dict[str, Any]) -> str:
    if item["kind"] == "unavailable":
        return "根拠非公開・記録不足: " + prose(item["reason"])
    # Escape URL delimiters without turning captured prose into Markdown.
    target = quote(item["url"], safe="/:#?=&%+@,;~.-_")
    suffix = "固定版" if item["revision"] else "内容識別あり・全文復元は未保証"
    return f"[根拠]({target})（{suffix}）"


def generate(files: dict[str, bytes]) -> dict[str, bytes]:
    """Return every required README. No clock, private state, Git, or LLM calls.

    Legacy navigation is generated only for parents represented by new records.
    Other worker-only parents remain linked at their existing directories.
    """
    rows = records(files)
    if not rows:
        return {}
    by_id = {r["id"]: r for r in rows}
    labels = {r["id"]: f"{KIND[r['kind']]} {i}" for i, r in enumerate(rows, 1)}
    result_states = states(rows)
    aggregated = aggregate(rows)
    incoming: dict[str, list[str]] = defaultdict(list)
    decisions: dict[str, list[dict[str, Any]]] = defaultdict(list)
    actions: dict[str, list[dict[str, Any]]] = defaultdict(list)
    action_results: dict[str, list[str]] = defaultdict(list)
    for row in rows:
        for reference_id in references(row):
            incoming[reference_id].append(row["id"])
        if row["kind"] == "decision":
            for reference_id in row["data"]["inputs"]:
                decisions[reference_id].append(row)
        elif row["kind"] == "action" and row["data"]["plan_id"]:
            actions[row["data"]["plan_id"]].append(row)
        elif row["kind"] == "result" and row["data"]["action_id"]:
            action_results[row["data"]["action_id"]].append(row["id"])

    parents = sorted({r["scope"]["parent"] for r in rows})
    views: dict[str, list[dict[str, Any]]] = {f"{ROOT}/README.md": rows}
    titles = {f"{ROOT}/README.md": "親の結果・判断記録"}
    turns: dict[tuple[int, int, str, int], str] = {}
    record_turns: dict[str, str] = {}
    legacy: dict[str, list[str]] = defaultdict(list)
    worker_parents = set()
    for name in sorted(files):
        match = WORKER.fullmatch(name)
        if match is None:
            continue
        parent, child = int(match[1]), int(match[2])
        worker_parents.add(parent)
        if parent not in parents or match[5] != "meta.json":
            continue
        directory = name.rsplit("/", 1)[0]
        meta = worker_schema.validate(
            {n: files.get(f"{directory}/{n}", b"") for n in worker_schema.FILENAMES}
        )
        require(directory == worker_schema.path(meta), "record_source")
        key = (parent, child, match[3], meta["turn"]["sequence"])
        require(key not in turns, "record_turn")
        turns[key] = directory

    def add(path: str, title: str, row: dict[str, Any] | None = None) -> None:
        views.setdefault(path + "/README.md", [])
        titles[path + "/README.md"] = title
        if row is not None:
            views[path + "/README.md"].append(row)

    def hierarchy(
        parent: int,
        child: int,
        key: str,
        directory: str,
        row: dict[str, Any] | None = None,
    ) -> None:
        base = f"{ROOT}/parent-{parent}"
        add(f"{base}/child-{child}", f"担当issue #{child}", row)
        add(f"{base}/child-{child}/{key}", f"担当issue #{child} の委任", row)
        add(
            directory,
            f"担当issue #{child} のターン {int(directory.rsplit('/', 1)[1].split('-')[0])}",
            row,
        )

    for row in rows:
        scope = row["scope"]
        parent = scope["parent"]
        base = f"{ROOT}/parent-{parent}"
        add(base, f"親issue #{parent}", row)
        add(f"{base}/activity-{scope['activity_key']}", "活動の結果と判断", row)
        if scope["issue"] is not None:
            add(f"{base}/child-{scope['issue']}", f"担当issue #{scope['issue']}", row)
        if scope["delegation"] is not None:
            d = scope["delegation"]
            key = (parent, scope["issue"], d["key"], d["turn"])
            directory = turns.get(key) or (
                f"{base}/child-{scope['issue']}/{d['key']}/{d['turn']:04d}-"
                + ("initial" if d["turn"] == 1 else "resume")
            )
            record_turns[row["id"]] = directory + "/README.md"
            # Child row was already added above.
            add(
                f"{base}/child-{scope['issue']}/{d['key']}",
                f"担当issue #{scope['issue']} の委任",
                row,
            )
            add(directory, f"担当issue #{scope['issue']} のターン {d['turn']}", row)
    for (parent, child, delegation_key, _), directory in turns.items():
        hierarchy(parent, child, delegation_key, directory)
        if not views[directory + "/README.md"]:
            legacy[directory + "/README.md"].append(directory)
    for directories in list(legacy.values()):
        for directory in directories:
            ancestor = PurePosixPath(directory).parent
            while str(ancestor) != ROOT:
                legacy[str(ancestor) + "/README.md"].append(directory)
                ancestor = ancestor.parent
            legacy[f"{ROOT}/README.md"].append(directory)

    def ref(source: str, record_id: str) -> str:
        return link(
            source,
            event_path(by_id[record_id]).rsplit("/", 1)[0] + "/README.md",
            labels[record_id],
            "#record-" + record_id,
        )

    def latest_action(plan_id: str) -> dict[str, Any] | None:
        history = actions[plan_id]
        previous = {a["data"]["previous_action_id"] for a in history}
        return next((a for a in history if a["id"] not in previous), None)

    def plans(source: str, decision: dict[str, Any]) -> str:
        output = []
        for plan in decision["data"]["next_actions"]:
            history = actions[plan["id"]]
            latest = latest_action(plan["id"])
            status = (
                STATE[latest["data"]["state"]]
                if latest
                else "判断済み未実行（行動記録なし）"
            )
            chain = " → ".join(ref(source, a["id"]) for a in history)
            output.append(
                f"{prose(plan['summary'])}（{OPERATION[plan['operation']]} / {status}）"
                + (" " + chain if chain else "")
            )
        return "<br>".join(output) or "次の行動なし（判断の記録どおり）"

    def table_row(source: str, row: dict[str, Any]) -> str:
        key, data, scope = row["id"], row["data"], row["scope"]
        target = link(
            source,
            f"{ROOT}/parent-{scope['parent']}/README.md",
            f"親 #{scope['parent']}",
        )
        target += " / " + (
            link(
                source,
                f"{ROOT}/parent-{scope['parent']}/child-{scope['issue']}/README.md",
                f"担当 #{scope['issue']}",
            )
            if scope["issue"]
            else "親の直接活動"
        )
        if key in record_turns:
            target += " / " + link(
                source, record_turns[key], f"ターン {scope['delegation']['turn']}"
            )
        when = prose(row["time"]["recorded_at"]) + "（記録時刻）<br>" + target
        result, decision, action = "—", "—", "—"
        if row["kind"] == "result":
            state = result_states[key]
            result = f"{prose(data['summary'])}<br>{prose(data['status'])} / {prose(data['observer'])}"
            if state["parent_confirmation"] == "unrecorded":
                result += "<br>親の確認未記録"
            if data["final_report"]["status"] == "unavailable":
                result += "<br>記録不足・最終報告: " + prose(
                    data["final_report"]["reason"]
                )
            ds = decisions[key]
            decision = (
                "<br>".join(
                    ref(source, d["id"]) + ": " + prose(d["data"]["reason"]) for d in ds
                )
                or "未判断"
            )
            action = "<br>".join(plans(source, d) for d in ds) or "判断待ち"
        elif row["kind"] == "decision":
            result = "入力: " + (
                "、".join(ref(source, r) for r in data["inputs"])
                or "初回要求・合意（根拠参照）"
            )
            decision = prose(data["evaluation"]) + ": " + prose(data["reason"])
            if data["unresolved"]:
                decision += "<br>未解決: " + "、".join(
                    prose(s) for s in data["unresolved"]
                )
            action = plans(source, row)
        elif row["kind"] == "action":
            results = list(dict.fromkeys([*action_results[key], *data["result_ids"]]))
            result = "、".join(ref(source, r) for r in results) or "結果記録なし"
            decision = (
                ref(source, data["decision_id"])
                if data["decision_id"]
                else "記録不足・判断: " + prose(data["decision_missing_reason"])
            )
            action = (
                prose(data["summary"])
                + "<br>"
                + OPERATION[data["operation"]]
                + " / "
                + STATE[data["state"]]
            )
        else:
            result = "訂正対象: " + ref(source, data["target_id"])
            decision = prose(data["reason"])
            action = (
                "置換: " + ref(source, data["replacement_id"])
                if data["replacement_id"]
                else "訂正追記（元記録を保持）"
            )
        proof = [link(source, event_path(row), "記録本文")]
        proof += [evidence_link(e) for e in data.get("evidence", data.get("basis", []))]
        if row["kind"] in {"decision", "correction"}:
            proof += [evidence_link(e) for e in data["source"]["evidence"]]
        if references(row):
            proof.append(
                "前の記録: " + "、".join(ref(source, r) for r in references(row))
            )
        if incoming[key]:
            proof.append(
                "次の記録: " + "、".join(ref(source, r) for r in incoming[key])
            )
        anchor = f'<a id="record-{key}"></a>'
        return (
            "| "
            + " | ".join(
                [
                    anchor + ref(source, key) + "<br>" + when,
                    result,
                    decision,
                    action,
                    "<br>".join(proof),
                ]
            )
            + " |"
        )

    generated = {}
    for path, selected in sorted(views.items()):
        times = [timestamp(r["time"]["recorded_at"]) for r in selected]
        first, last = (
            (min(times).isoformat(), max(times).isoformat())
            if times
            else ("記録なし", "記録なし")
        )
        lines = [
            f"# {titles[path]}",
            "",
            "結果 → 親の判断と理由 → 実際の行動 → 次の結果を、参照で辿る表示です。",
            "",
            f"反映範囲: このHEADの記録集合 {len(selected)} 件。記録時刻 {first} 〜 {last}。",
            "",
            "公開待ち: このHEADのmain反映と、まだGitに無い非公開記録は `decision-log-status` で照合してください。ここでは未公開件数や最新状態を推測しません。",
            "",
        ]
        if path != f"{ROOT}/README.md":
            up = str(PurePosixPath(path).parent.parent / "README.md")
            lines += [
                link(path, up, "上の階層")
                + " · "
                + link(path, f"{ROOT}/README.md", "親一覧"),
                "",
            ]
        children = [
            p
            for p in views
            if p != path
            and PurePosixPath(p).parent.parent == PurePosixPath(path).parent
        ]
        if children:
            lines += [
                "下の階層: "
                + " · ".join(link(path, p, titles[p]) for p in sorted(children)),
                "",
            ]
        # Every non-activity level has direct links to the canonical activity.
        activity_paths = sorted(
            {event_path(r).rsplit("/", 1)[0] + "/README.md" for r in selected}
        )
        if activity_paths:
            lines += [
                "活動: "
                + " · ".join(
                    link(path, p, f"活動 {i}") for i, p in enumerate(activity_paths, 1)
                ),
                "",
            ]
        undecided = sum(
            result_states[r["id"]]["decision"] == "undecided"
            for r in selected
            if r["kind"] == "result"
        )
        pending_plans = 0
        for row in selected:
            if row["kind"] == "decision":
                for plan in row["data"]["next_actions"]:
                    latest = latest_action(plan["id"])
                    pending_plans += (
                        latest is None
                        or latest["data"]["state"] == "decided_not_executed"
                    )
        missing_reports = sum(
            r["kind"] == "result"
            and r["data"]["final_report"]["status"] == "unavailable"
            for r in selected
        )
        lines += [
            f"未判断の結果: {undecided} 件。判断済み未実行の計画: {pending_plans} 件。記録不足の最終報告: {missing_reports} 件。",
            "",
        ]
        unresolved = list(
            dict.fromkeys(
                s
                for r in selected
                if r["kind"] == "decision"
                for s in r["data"]["unresolved"]
            )
        )
        if unresolved:
            lines += [
                "記録された未解決事項（記録時点）: "
                + "、".join(prose(s) for s in unresolved),
                "",
            ]
        groups: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
        for row in selected:
            groups[(row["scope"]["parent"], row["scope"]["activity_key"])].append(row)
        for index, group in enumerate(groups.values(), 1):
            lines += [
                f"## 活動 {index}",
                "",
                "| 時刻・対象 | 結果 | 親の判断と理由 | 次の行動・実行状態 | 根拠 |",
                "| --- | --- | --- | --- | --- |",
            ]
            lines += [table_row(path, r) for r in group]
            # Grouping remains lossless: every record above keeps its own anchor.
            members = {r["id"] for r in group}
            repeated = [
                g
                for g in aggregated
                if g["count"] > 1 and set(g["record_ids"]) <= members
            ]
            for item in repeated:
                lines.append(
                    f"\n同条件の取得・確認 {item['count']} 件（{prose(item['first_at'])} 〜 {prose(item['last_at'])}）。原記録は上表に保持。"
                )
            lines.append("")
        for directory in sorted(set(legacy[path])):
            lines += [
                "導入前・判断未記録: "
                + link(path, directory + "/instruction.md", "既存の指示")
                + " · "
                + link(path, directory + "/final.md", "既存の最終報告"),
                "このHEADに新形式の対応記録なし。導入時期・公開待ち・捕捉不足の区別は未確認。",
                "",
            ]
        if path == f"{ROOT}/README.md":
            for parent in sorted(worker_parents - set(parents)):
                lines += [
                    link(
                        path,
                        f"{ROOT}/parent-{parent}",
                        f"親 #{parent} の既存workerログ",
                    )
                    + "（導入前・判断未記録）",
                    "",
                ]
        directory = path.rsplit("/", 1)[0]
        for name in ("instruction.md", "final.md", "meta.json"):
            if f"{directory}/{name}" in files:
                lines += [link(path, f"{directory}/{name}", name), ""]
        lines += ["<details><summary>反映した記録と補助情報</summary>", ""]
        for row in selected:
            lines += [
                f"- {ref(path, row['id'])}: " + link(path, event_path(row), "正本")
            ]
            if row["kind"] == "result":
                data = row["data"]
                lines += [
                    "  - 前回との差: " + prose(data["changes"]),
                    "  - 時刻・補助情報（欠測と累積値を保持、加算なし）: "
                    + prose({"time": row["time"], "auxiliary": data["auxiliary"]}),
                ]
        lines += ["", "</details>", ""]
        raw = "\n".join(lines).encode("utf-8")
        # Guard treats closing HTML tags as local paths. Strip only this
        # renderer's fixed scaffolding; captured HTML has already been escaped.
        checked = re.sub(
            r'<a id="record-[a-f0-9]{32}">|</a>|<br>|</?details>|</?summary>',
            "",
            raw.decode("utf-8"),
        )
        Guard().check([path, checked])
        generated[path] = raw
    validate_links({**files, **generated}, generated)
    return generated


def validate_links(files: dict[str, bytes], readmes: dict[str, bytes]) -> None:
    """Check all generated relative targets and record anchors in the same tree."""
    for source, raw in readmes.items():
        for target in re.findall(r"\]\(([^)]+)\)", raw.decode("utf-8")):
            if urlsplit(target).scheme:
                require(target.startswith("https://"), "unsafe_url")
                continue
            path, _, anchor = target.partition("#")
            name = posixpath.normpath(posixpath.join(posixpath.dirname(source), path))
            require(
                name.startswith(ROOT + "/")
                and (name in files or any(p.startswith(name + "/") for p in files)),
                "unsafe_reference",
            )
            if anchor:
                require(
                    name in files and f'id="{anchor}"'.encode() in files[name],
                    "invalid_log_reference",
                )
