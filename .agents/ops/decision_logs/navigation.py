"""Pure, deterministic navigation over public envelopes and existing worker logs."""

from __future__ import annotations

import heapq
import posixpath
import re
from collections import defaultdict
from pathlib import PurePosixPath
from typing import Any
from urllib.parse import urlsplit

from worker_logs.journal import decode, require
from worker_logs.safety import Guard

from .aggregation import aggregate
from .graph import references, states, validate_graph
from .presentation import (
    NOT_FINAL,
    change_fields,
    checked_unescape,
    evidence_link,
    fields,
    final_report,
    prose,
    summary,
)
from .projection import validate_public
from .schema import timestamp
from .trace_presentation import (
    latest_termination,
    observation_view,
    turn_fields,
    turn_summary,
)
from .turns import ROOT, Turn, by_prefix, inventory, turn_key

EVENT = re.compile(
    rf"{re.escape(ROOT)}/parent-([1-9][0-9]*)/activity-([a-f0-9]{{32}})"
    r"/event-([a-f0-9]{32})\.json"
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

# Bound every derived page, including indexes. Record pages use fixed chunks so
# allocating anchors does not depend recursively on the size of their links.
PAGE_BYTES = 192 * 1024
DETAIL_ROWS = 12
DETAIL_BYTES = 10 * 1024
INDEX_ROWS = 40
RECENT = 6


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


def link(source: str, target: str, label: str, anchor: str = "") -> str:
    relative = posixpath.relpath(target, posixpath.dirname(source))
    return f"[{prose(label)}]({relative}{anchor})"


def generate(files: dict[str, bytes]) -> dict[str, bytes]:
    """Return every required README. No clock, private state, Git, or LLM calls.

    Worker-only parents and event-only turns share the same logical inventory.
    """
    rows = records(files)
    turn_inventory = inventory(files, rows)
    turn_groups = by_prefix(turn_inventory)
    if not rows and not turn_inventory:
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

    parents = sorted(
        {r["scope"]["parent"] for r in rows} | {key[1] for key in turn_inventory}
    )
    views: dict[str, list[dict[str, Any]]] = {f"{ROOT}/README.md": rows}
    titles = {f"{ROOT}/README.md": "親の結果・判断記録"}
    turns = {
        key: item.directory
        for key, item in turn_inventory.items()
        if item.directory is not None
    }
    scoped_turns = {key: item.records for key, item in turn_inventory.items()}
    record_turns: dict[str, str] = {}
    artifact_directories: dict[str, str] = {}
    related_turns: dict[str, tuple[str, str]] = {}
    logical_turns: set[str] = set()
    legacy: dict[str, list[str]] = defaultdict(list)
    worker_parents = {key[1] for key in turns}

    def add(path: str, title: str, row: dict[str, Any] | None = None) -> None:
        views.setdefault(path + "/README.md", [])
        titles[path + "/README.md"] = title
        if row is not None:
            views[path + "/README.md"].append(row)

    for parent in parents:
        add(f"{ROOT}/parent-{parent}", f"親issue #{parent}")
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
            key = turn_key(scope)
            # Sequence does not determine initial/resume. A pre-start refusal
            # can be followed by a new initial attempt in the same lineage.
            directory = f"{base}/child-{scope['issue']}/{d['key']}/turn-{d['turn']:04d}"
            record_turns[row["id"]] = directory + "/README.md"
            logical_turns.add(directory + "/README.md")
            # Child row was already added above.
            add(
                f"{base}/child-{scope['issue']}/{d['key']}",
                f"担当issue #{scope['issue']} の委任",
                row,
            )
            add(directory, f"担当issue #{scope['issue']} のターン {d['turn']}", row)
    for key, directory in turns.items():
        _, parent, child, delegation_key, sequence = key
        base = f"{ROOT}/parent-{parent}/child-{child}"
        add(base, f"担当issue #{child}")
        add(f"{base}/{delegation_key}", f"担当issue #{child} の委任")
        logical = f"{base}/{delegation_key}/turn-{sequence:04d}"
        add(logical, f"担当issue #{child} のターン {sequence}")
        logical_page, physical_page = logical + "/README.md", directory + "/README.md"
        logical_turns.add(logical_page)
        add(directory, f"担当issue #{child} のworkerログ（ターン {sequence}）")
        views[physical_page] = list(scoped_turns[key])
        related_turns[physical_page] = (logical_page, "ターンの判断記録")
        related_turns[logical_page] = (physical_page, "既存workerログのフォルダー")
        artifact_directories[logical_page] = directory
        artifact_directories[physical_page] = directory
        if not scoped_turns[key]:
            legacy[physical_page].append(directory)
            legacy[logical_page].append(directory)
    for directories in list(legacy.values()):
        for directory in directories:
            ancestor = PurePosixPath(directory).parent
            while str(ancestor) != ROOT:
                legacy[str(ancestor) + "/README.md"].append(directory)
                ancestor = ancestor.parent
            legacy[f"{ROOT}/README.md"].append(directory)

    activity_groups = {
        path: selected
        for path, selected in views.items()
        if re.fullmatch(
            rf"{re.escape(ROOT)}/parent-[1-9][0-9]*/activity-[a-f0-9]{{32}}/README.md",
            path,
        )
    }
    record_pages: dict[str, str] = {}
    detail_pages: dict[str, tuple[str, list[dict[str, Any]], int, int]] = {}
    for path, selected in activity_groups.items():
        scope = selected[0]["scope"]
        commands = list(
            dict.fromkeys(
                match[1]
                for r in selected
                if (
                    match := re.match(
                        r"^([a-z][a-z0-9_.-]+):", r["data"].get("summary", "")
                    )
                )
            )
        )
        operations = list(
            dict.fromkeys(
                OPERATION[r["data"]["operation"]]
                for r in selected
                if r["kind"] == "action"
            )
        )
        plans_in_activity = [
            p
            for r in selected
            if r["kind"] == "decision"
            for p in r["data"]["next_actions"]
        ]
        operation = "・".join(commands[:2] or operations[:2])
        if not operation:
            operation = (
                plans_in_activity[0]["summary"]
                if plans_in_activity
                else checked_unescape(
                    summary(
                        selected[0]["data"].get("summary", KIND[selected[0]["kind"]]),
                        compact=True,
                    )
                )
            )
        target = (
            f"担当issue #{scope['issue']}"
            if scope["issue"]
            else f"親 #{scope['parent']} の直接活動"
        )
        if plans_in_activity:
            target += " / " + plans_in_activity[0]["target"][:100]
        titles[path] = operation[:120] + " / " + target
        count = (len(selected) + DETAIL_ROWS - 1) // DETAIL_ROWS
        for index in range(count):
            page = (
                path
                if count == 1
                else path.removesuffix("/README.md")
                + f"-records-{index + 1:04d}/README.md"
            )
            chunk = selected[index * DETAIL_ROWS : (index + 1) * DETAIL_ROWS]
            detail_pages[page] = (path, chunk, index + 1, count)
            for row in chunk:
                record_pages[row["id"]] = page

    def ref(source: str, record_id: str) -> str:
        return link(
            source,
            record_pages[record_id],
            labels[record_id],
            "#record-" + record_id,
        )

    def latest_action(plan_id: str) -> dict[str, Any] | None:
        history = actions[plan_id]
        previous = {a["data"]["previous_action_id"] for a in history}
        return next((a for a in history if a["id"] not in previous), None)

    def result_brief(source: str, record_id: str) -> str:
        data = by_id[record_id]["data"]
        return (
            ref(source, record_id)
            + " "
            + prose(data["status"])
            + " / "
            + prose(data["observer"])
            + ": "
            + summary(data["summary"], compact=True)
        )

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
            results = list(
                dict.fromkeys(
                    key
                    for a in history
                    for key in [*action_results[a["id"]], *a["data"]["result_ids"]]
                )
            )
            output.append(
                f"{prose(plan['summary'])}（{OPERATION[plan['operation']]} / {status}）"
                + (" " + chain if chain else "")
                + (
                    "<br>実行の結果: "
                    + "、".join(result_brief(source, key) for key in results[:3])
                    if results
                    else ""
                )
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
            result = f"{summary(data['summary'])}<br>{prose(data['status'])} / {prose(data['observer'])}"
            if state["parent_confirmation"] == "unrecorded":
                result += "<br>親の確認未記録"
            result += "<br>" + final_report(data)
            if "observation" in data:
                result += "<br>" + observation_view(data["observation"], scope["repo"])
            result += "<br>前回との差: " + change_fields(data["changes"])
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
                "<br>".join(result_brief(source, r) for r in data["inputs"])
                or "初回要求・合意（根拠参照）"
            )
            decision = prose(data["evaluation"]) + ": " + prose(data["reason"])
            if data["unresolved"]:
                decision += "<br>未解決: " + "、".join(
                    prose(s) for s in data["unresolved"]
                )
            for field, label in (
                ("reuse_scope", "再利用"),
                ("recheck_scope", "再確認"),
                ("resume_conditions", "再開条件"),
            ):
                if data[field]:
                    decision += (
                        "<br>" + label + ": " + "、".join(prose(s) for s in data[field])
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

    generated: dict[str, bytes] = {}
    table_header = [
        "| 時刻・対象 | 結果 | 親の判断と理由 | 次の行動・実行状態 | 根拠 |",
        "| --- | --- | --- | --- | --- |",
    ]

    def counts(selected: list[dict[str, Any]]) -> str:
        reports = [r for r in selected if r["kind"] == "result"]
        pending = sum(
            (latest := latest_action(p["id"])) is None
            or latest["data"]["state"] == "decided_not_executed"
            for r in selected
            if r["kind"] == "decision"
            for p in r["data"]["next_actions"]
        )
        undecided = sum(
            result_states[r["id"]]["decision"] == "undecided" for r in reports
        )
        unverified = sum(
            result_states[r["id"]]["parent_confirmation"] == "unrecorded"
            for r in reports
        )
        missing = sum(
            r["data"]["final_report"]["status"] == "unavailable"
            and r["data"]["final_report"]["reason"] != NOT_FINAL
            for r in reports
        )
        excluded = sum(
            r["data"]["final_report"] == {"status": "unavailable", "reason": NOT_FINAL}
            for r in reports
        )
        failed = sum(r["data"]["status"] in {"failure", "rejected"} for r in reports)
        return (
            f"{len(selected)} 記録 / 判断 {sum(r['kind'] == 'decision' for r in selected)} 件。"
            f"失敗・拒否の結果: {failed} 件。"
            f"未判断の結果: {undecided} 件。親の確認未記録: {unverified} 件。"
            f"判断済み未実行の計画: {pending} 件。記録不足の最終報告: {missing} 件（結果イベント単位）。"
            f"最終報告の観測対象外: {excluded} 件（結果イベント単位）。"
        )

    def period(selected: list[dict[str, Any]]) -> str:
        times = [timestamp(r["time"]["recorded_at"]) for r in selected]
        return (
            f"{min(times).isoformat()} 〜 {max(times).isoformat()}"
            if times
            else "記録なし"
        )

    def header(
        path: str, title: str, selected: list[dict[str, Any]], up: str | None
    ) -> list[str]:
        lines = [
            f"# {prose(title, 300)}",
            "",
            "結果 → 親の判断と理由 → 実際の行動 → 次の結果を、参照で辿る表示です。",
            "",
            f"反映範囲: このHEADの記録集合 {len(selected)} 件。記録時刻 {period(selected)}。",
            "",
            "公開待ち: このHEADのmain反映と、まだGitに無い非公開記録は `decision-log-status` で照合してください。ここでは未公開件数や最新状態を推測しません。",
            "",
        ]
        if up:
            lines += [
                link(path, up, "上の階層")
                + " · "
                + link(path, f"{ROOT}/README.md", "親一覧"),
                "",
            ]
        lines += [counts(selected), ""]
        items = selected_turns(path, selected)
        if items or path == f"{ROOT}/README.md":
            lines += [
                turn_summary(items),
                "",
                "対応数は公開worker3ファイルのあるターンを分母にしたものです。イベントのみのターンを加えた和集合とは分けます。対応イベントの存在は、親判断の記録・受入・網羅性を保証しません。",
                "",
            ]
        if path == f"{ROOT}/README.md":
            lines += [
                "対応イベントなし。導入時期・未捕捉・捕捉失敗・未公開の区別は未確認。記録の不在だけで導入前と断定せず、失われた親判断の数へ読み替えません。",
                "",
            ]
        if re.fullmatch(rf"{re.escape(ROOT)}/parent-[1-9][0-9]*/README.md", path):
            closing = latest_termination(selected)
            if closing:
                lines += [
                    "最後の受入・担当終了／引継ぎ: " + ref(path, closing["id"]),
                    "",
                    observation_view(
                        closing["data"]["observation"], closing["scope"]["repo"]
                    ),
                    "",
                ]
            else:
                lines += [
                    "最後の受入・担当終了／引継ぎ: この公開snapshotでは未確認。",
                    "",
                ]
            for gap in [
                r
                for r in selected
                if r["kind"] == "result"
                and r["data"].get("observation", {}).get("kind") == "capture_gap"
            ][-RECENT:]:
                lines += [
                    observation_view(gap["data"]["observation"], gap["scope"]["repo"])
                    + " / "
                    + ref(path, gap["id"]),
                    "",
                ]
        return lines

    def selected_turns(path: str, selected: list[dict[str, Any]]) -> list[Turn]:
        if path in turn_groups:
            return turn_groups[path]
        keys = {key for r in selected if (key := turn_key(r["scope"])) is not None}
        return [turn_inventory[key] for key in sorted(keys)]

    def save(path: str, lines: list[str]) -> None:
        raw = ("\n".join(lines) + "\n").encode("utf-8")
        require(len(raw) < PAGE_BYTES, "prepared_content_changed")
        checked = re.sub(
            r'<a id="record-[a-f0-9]{32}">|</a>|<br>|</?details>|</?summary>',
            "",
            raw.decode("utf-8"),
        )
        Guard().check([path, checked])
        generated[path] = raw

    def page_path(owner: str, tag: str, index: int) -> str:
        directory = owner.removesuffix("/README.md")
        name = f"{tag}-{index:04d}"
        if directory == ROOT:
            return f"{ROOT}/parent-index-{name}/README.md"
        if re.fullmatch(rf"{re.escape(ROOT)}/parent-[1-9][0-9]*", directory):
            return f"{directory}/activity-index-{name}/README.md"
        if re.fullmatch(
            rf"{re.escape(ROOT)}/parent-[1-9][0-9]*/child-[1-9][0-9]*", directory
        ):
            return f"{directory}/d-index-{name}/README.md"
        if PurePosixPath(directory).name.startswith("d-"):
            return f"{directory}/page-{name}/README.md"
        return f"{directory}-{name}/README.md"

    def latest(source: str, selected: list[dict[str, Any]]) -> str:
        result = next((r for r in reversed(selected) if r["kind"] == "result"), None)
        action = next((r for r in reversed(selected) if r["kind"] == "action"), None)
        output = []
        closing = latest_termination(selected)
        if closing:
            output.append(
                observation_view(
                    closing["data"]["observation"], closing["scope"]["repo"], brief=True
                )
                + " "
                + ref(source, closing["id"])
            )
        if result:
            output += [result_brief(source, result["id"])]
        if action:
            output += [
                ref(source, action["id"])
                + " "
                + OPERATION[action["data"]["operation"]]
                + " / "
                + STATE[action["data"]["state"]]
            ]
        return "<br>".join(output) or "結果・行動の記録なし"

    def catalog(
        owner: str,
        tag: str,
        title: str,
        targets: list[tuple[str, str, list[dict[str, Any]]]],
    ) -> list[str]:
        targets = sorted(
            targets,
            key=lambda item: (
                max(
                    (timestamp(r["time"]["recorded_at"]) for r in item[2]),
                    default=timestamp("1970-01-01T00:00:00+00:00"),
                ),
                item[0],
            ),
        )

        def entries(
            source: str, items: list[tuple[str, str, list[dict[str, Any]]]]
        ) -> list[str]:
            lines = [
                "| 対象・期間 | 件数と未確認 | 最新の記録 |",
                "| --- | --- | --- |",
            ]
            for target, label, selected in items:
                lines += [
                    "| "
                    + link(source, target, label[:180])
                    + "<br>"
                    + period(selected)
                    + " | "
                    + counts(selected)
                    + "<br>"
                    + turn_summary(selected_turns(target, selected))
                    + " | "
                    + latest(source, selected)
                    + " |"
                ]
            return lines

        if len(targets) <= RECENT:
            return [f"## {title}", "", *entries(owner, targets), ""] if targets else []
        count = (len(targets) + INDEX_ROWS - 1) // INDEX_ROWS
        for i in range(count):
            path = page_path(owner, tag, i + 1)
            chunk = targets[i * INDEX_ROWS : (i + 1) * INDEX_ROWS]
            selected = [r for _, _, group in chunk for r in group]
            lines = header(path, title + f"（{i + 1} / {count}）", selected, owner)
            if i:
                lines += [link(path, page_path(owner, tag, i), "前のページ"), ""]
            if i + 1 < count:
                lines += [link(path, page_path(owner, tag, i + 2), "次のページ"), ""]
            lines += entries(path, chunk)
            save(path, lines)
        return [
            f"## {title}",
            "",
            link(
                owner,
                page_path(owner, tag, 1),
                f"全{len(targets)}件の一覧（{count}ページ）",
            ),
            "",
            *entries(owner, targets[-RECENT:]),
            "",
        ]

    def bounded_row(
        path: str, row: dict[str, Any], *, anchor: bool = True
    ) -> tuple[str, str]:
        rendered = table_row(path, row)
        auxiliary = ""
        if row["kind"] == "result":
            auxiliary = (
                "- "
                + ref(path, row["id"])
                + "<br>時刻: "
                + fields(row["time"])
                + "<br>補助情報（欠測と累積値を保持、加算なし）: "
                + fields(row["data"]["auxiliary"])
            )
        if len((rendered + auxiliary).encode()) > DETAIL_BYTES:
            # A huge individual record cannot defeat pagination. Its full value
            # remains in the immutable envelope; omission is visible, not silent.
            rendered = (
                f'| <a id="record-{row["id"]}"></a>{ref(path, row["id"])}'
                + " | 表示省略・記録本文を参照 | "
                + prose(row["data"].get("reason", ""), 200)
                + " | "
                + prose(row["data"].get("state", ""))
                + " | "
                + link(path, event_path(row), "完全な記録本文")
                + " |"
            )
            auxiliary = ""
        if not anchor:
            rendered = re.sub(r'<a id="record-[a-f0-9]{32}"></a>', "", rendered)
        return rendered, auxiliary

    def detail_table(path: str, selected: list[dict[str, Any]]) -> list[str]:
        rendered = [bounded_row(path, r) for r in selected]
        lines = [*table_header, *(r[0] for r in rendered), ""]
        auxiliary = [r[1] for r in rendered if r[1]]
        if auxiliary:
            lines += [
                "<details><summary>反映した記録と補助情報</summary>",
                "",
                *auxiliary,
                "",
                "</details>",
                "",
            ]
        members = {r["id"] for r in selected}
        for item in aggregated:
            if item["count"] > 1 and set(item["record_ids"]) <= members:
                lines += [
                    f"同条件の取得・確認 {item['count']} 件（{prose(item['first_at'])} 〜 {prose(item['last_at'])}）。原記録は上表に保持。",
                    "",
                ]
        return lines

    for path, selected in sorted(views.items()):
        up = (
            str(PurePosixPath(path).parent.parent / "README.md")
            if path != f"{ROOT}/README.md"
            else None
        )
        lines = header(path, titles[path], selected, up)
        important = [r for r in selected if r["kind"] == "decision"]
        if important and path != f"{ROOT}/README.md":
            lines += [
                "## 最近の親判断",
                "",
                f"判断 {len(important)} 件。直近{min(RECENT, len(important))}件を表示。過去の判断は各活動の詳細へ。",
                "",
                *table_header,
            ]
            lines += [
                bounded_row(path, r, anchor=False)[0] for r in important[-RECENT:]
            ]
            lines += [""]
        if path != f"{ROOT}/README.md":
            lines += ["直近の観測（受入の判定とは別）: " + latest(path, selected), ""]
        children = sorted(
            p
            for p in views
            if p != path
            and PurePosixPath(p).parent.parent == PurePosixPath(path).parent
            and p not in activity_groups
        )
        lines += catalog(
            path, "children", "下の階層", [(p, titles[p], views[p]) for p in children]
        )
        activities = sorted(
            {event_path(r).rsplit("/", 1)[0] + "/README.md" for r in selected}
        )
        if path not in activity_groups and path != f"{ROOT}/README.md":
            lines += catalog(
                path,
                "activities",
                "活動別の結果と判断",
                [(p, titles[p], activity_groups[p]) for p in activities],
            )
        if path in related_turns:
            other, label = related_turns[path]
            lines += [link(path, other, label), ""]
        elif path in logical_turns:
            lines += [
                "workerログ3ファイルはこのHEADに未収録。公開待ち・記録不足・ターン種別は未確認。",
                "",
            ]
        if path in logical_turns:
            item = turn_groups[path][0]
            execution, final, public, proof = turn_fields(item)
            lines += [
                "| 実行（観測時点） | 最終報告 | 公開の所在 | 根拠 |",
                "| --- | --- | --- | --- |",
                f"| {execution} | {final} | {public} | {proof or 'このHEADの公開workerファイルと記録本文を参照'} |",
                "",
            ]
        if path in detail_pages:
            lines += ["## 全記録の詳細", "", *detail_table(path, selected)]
        elif path in activity_groups:
            count = (len(selected) + DETAIL_ROWS - 1) // DETAIL_ROWS
            first_page = path.removesuffix("/README.md") + "-records-0001/README.md"
            lines += [
                "## 全記録の詳細",
                "",
                link(path, first_page, f"全{len(selected)}記録（{count}ページ）"),
                "",
            ]
        directories = sorted(set(legacy[path]))
        if directories and path in logical_turns | set(artifact_directories):
            for directory in directories:
                lines += [
                    "対応イベントなし: "
                    + link(path, directory + "/instruction.md", "既存の指示")
                    + " · "
                    + link(path, directory + "/final.md", "既存の最終報告"),
                    "このHEADに新形式の対応記録なし。導入時期・未捕捉・捕捉失敗・未公開の区別は未確認。",
                    "",
                ]
        elif directories:
            lines += [
                f"対応イベントなし: {len(directories)} ターン。下の階層で既存の指示と最終報告を参照。",
                "",
            ]
        if path == f"{ROOT}/README.md":
            worker_targets: list[tuple[str, str, list[dict[str, Any]]]] = [
                (
                    f"{ROOT}/parent-{p}/README.md"
                    if p in parents
                    else f"{ROOT}/parent-{p}",
                    f"親 #{p} の既存workerログ"
                    + (
                        "（対応イベントあり）"
                        if any(r["scope"]["parent"] == p for r in rows)
                        else "（対応イベント未確認）"
                    ),
                    views.get(f"{ROOT}/parent-{p}/README.md", []),
                )
                for p in sorted(worker_parents)
            ]
            lines += catalog(path, "legacy", "既存workerログを持つ親", worker_targets)
        directory = artifact_directories.get(path, path.rsplit("/", 1)[0])
        for name in ("instruction.md", "final.md", "meta.json"):
            if f"{directory}/{name}" in files:
                lines += [link(path, f"{directory}/{name}", name), ""]
        save(path, lines)
    for path, (owner, selected, number, count) in sorted(detail_pages.items()):
        if path == owner:
            continue
        lines = header(
            path, titles[owner] + f" / 全記録（{number} / {count}）", selected, owner
        )
        prefix = owner.removesuffix("/README.md")
        if number > 1:
            lines += [
                link(
                    path, prefix + f"-records-{number - 1:04d}/README.md", "前のページ"
                ),
                "",
            ]
        if number < count:
            lines += [
                link(
                    path, prefix + f"-records-{number + 1:04d}/README.md", "次のページ"
                ),
                "",
            ]
        lines += detail_table(path, selected)
        save(path, lines)
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
