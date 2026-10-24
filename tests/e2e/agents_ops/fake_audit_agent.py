"""Native-parent stand-in: concurrent fake scouts, no model/API execution."""

from __future__ import annotations

import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any


def payload(
    data: dict[str, Any], collected: dict[str, Any], log: Path
) -> dict[str, Any]:
    runtime = collected["audit_runtime"]
    findings: dict[str, list[dict[str, str]]] = {
        key: [] for key in ("code", "platform", "governance", "direction")
    }
    decisions = []
    for i, prop in enumerate(data["proposals"]):
        key = "code" if prop["category"] == "code-debt" else "governance"
        topic = f"proposal-{i}"
        prop["sources"] = [f"{key}:{topic}"]
        findings[key].append(
            {
                "topic": topic,
                "position": "support",
                "conclusion": prop["title"],
                "evidence": prop["evidence"],
            }
        )
        decisions.append(
            {
                "sources": prop["sources"],
                "outcome": "propose",
                "reason": "根拠確認済み",
                "evidence": prop["evidence"],
            }
        )

    # A host wall-clock correction must not turn an offline scout interval negative.
    wall_start, monotonic_start = datetime.now(UTC), time.monotonic()

    def stamp() -> str:
        return (
            wall_start + timedelta(seconds=time.monotonic() - monotonic_start)
        ).isoformat()

    def scout(key: str) -> dict[str, Any]:
        start = stamp()
        with log.open("a") as fh:
            fh.write(json.dumps({"id": key, "event": "start", "time": start}) + "\n")
        time.sleep(float(os.environ.get("FAKE_AUDIT_SCOUT_DELAY", "0.04")))
        end = stamp()
        with log.open("a") as fh:
            fh.write(json.dumps({"id": key, "event": "finish", "time": end}) + "\n")
        row = {
            "id": key,
            "agent_type": "scout",
            "agent_id": f"fake-{key}-{start}",
            "base_sha": collected["meta"]["base_sha"],
            "started_at": start,
            "finished_at": end,
            "status": "completed",
            "conclusion": f"{key}の対象確認済み",
            "evidence": [f"{key}の対象一覧と収集データ"],
            "counterevidence": [],
            "unconfirmed": [],
            "findings": findings[key],
        }
        if key == os.environ.get("FAKE_AUDIT_FAIL"):
            row.update(
                status="interrupted",
                unconfirmed=[
                    {
                        "scope": key,
                        "reason": "ツール失敗で未確認",
                        "resume": "担当を再起動",
                    }
                ],
                findings=[],
            )
        return row

    with ThreadPoolExecutor(max_workers=3) as pool:
        results = list(pool.map(scout, runtime["requested"]))
    valid = {**runtime["previous_results"], **{r["id"]: r for r in results}}
    valid_sources = {
        f"{key}:{f['topic']}"
        for key, row in valid.items()
        if row["status"] == "completed"
        for f in row["findings"]
    }
    data["proposals"] = [
        p for p in data["proposals"] if set(p["sources"]) <= valid_sources
    ]
    data["decisions"] = [d for d in decisions if set(d["sources"]) <= valid_sources]
    data["assignments"] = results
    return data


def main() -> None:
    prompt, source, output = (Path(p) for p in sys.argv[1:])
    text = source.read_text()
    try:
        data = json.loads(text)
    except ValueError:
        output.write_text(text)
        return
    if "assignments" in data or "proposals" not in data:
        output.write_text(text)
        return
    collected = json.loads(
        re.findall(r"```json\n(.*?)\n```", prompt.read_text(), re.DOTALL)[-1]
    )
    # A faulty cwd is caught by the fake before it returns successful scouts.
    if Path.cwd() != Path(collected["audit_runtime"]["checkout"]):
        raise ValueError("parent and scouts must use the pinned checkout")
    data = payload(data, collected, output.parent / "fake_scout_events.jsonl")
    output.write_text(json.dumps(data, ensure_ascii=False))


if __name__ == "__main__":
    main()
