from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
SCRIPT = (
    ROOT
    / ".agents/skills/literature-radar/scripts/radar_status.py"
)


def load_module(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


status_guard = load_module("literature_radar_status_snapshots", SCRIPT)


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def make_repo(tmp_path: Path) -> Path:
    config = json.loads(
        (ROOT / "knowledge/literature/config.json").read_text(
            encoding="utf-8"
        )
    )
    write_json(tmp_path / "knowledge/literature/config.json", config)
    (tmp_path / "knowledge/literature/candidates").mkdir(
        parents=True,
        exist_ok=True,
    )
    (tmp_path / "knowledge/literature/status").mkdir(
        parents=True,
        exist_ok=True,
    )
    return tmp_path


def record(date: str, *, state: str) -> dict[str, Any]:
    return {
        "id": f"paper-arxiv-{date.replace('-', '')}",
        "state": state,
        "first_seen": f"{date}T01:20:00+09:00",
        "discoveries": [
            {
                "collector_id": "geometry",
                "screening": {"topic": "ball_3d"},
            }
        ],
    }


def quota(accepted: int, limit: int) -> dict[str, int]:
    return {
        "accepted": accepted,
        "limit": limit,
        "remaining": max(0, limit - accepted),
    }


def status_payload(
    config: dict[str, Any],
    *,
    date: str,
    accepted: int,
    open_candidates: int,
) -> dict[str, Any]:
    collector_limit = config["ingestion"][
        "max_candidates_per_collector_per_day"
    ]
    topic_limit = config["ingestion"][
        "max_candidates_per_topic_per_day"
    ]
    collectors = {
        collector_id: quota(
            accepted if collector_id == "geometry" else 0,
            collector_limit,
        )
        for collector_id in config["collectors"]
    }
    configured_topics = {
        topic
        for collector in config["collectors"].values()
        for topic in collector["topics"]
    }
    topics = {
        topic: quota(accepted if topic == "ball_3d" else 0, topic_limit)
        for topic in configured_topics
    }
    return {
        "schema_version": 1,
        "date": date,
        "initialized_at": f"{date}T00:05:00+09:00",
        "last_curated_at": f"{date}T01:00:00+09:00",
        "generated_at": f"{date}T01:00:00+09:00",
        "quota_mode": "enforced",
        "quota_note": None,
        "ingestion": {
            "accepted_candidates": accepted,
            "daily_limit": config["ingestion"][
                "max_candidates_total_per_day"
            ],
            "open_candidates": open_candidates,
            "open_limit": config["ingestion"]["max_open_candidates"],
            "collectors": collectors,
            "topics": topics,
        },
    }


def test_historical_status_preserves_open_backlog_snapshot(
    tmp_path: Path,
) -> None:
    repo = make_repo(tmp_path)
    config = status_guard.load_config(repo)
    write_json(
        repo / "knowledge/literature/status/2026-08-05.json",
        status_payload(
            config,
            date="2026-08-05",
            accepted=0,
            open_candidates=0,
        ),
    )
    write_json(
        repo
        / "knowledge/literature/candidates"
        / "paper-arxiv-20260806.json",
        record("2026-08-06", state="inbox"),
    )
    write_json(
        repo / "knowledge/literature/status/2026-08-06.json",
        status_payload(
            config,
            date="2026-08-06",
            accepted=1,
            open_candidates=1,
        ),
    )

    assert status_guard.validate_repository(repo) == []


def test_newest_status_validates_current_open_backlog(
    tmp_path: Path,
) -> None:
    repo = make_repo(tmp_path)
    config = status_guard.load_config(repo)
    write_json(
        repo
        / "knowledge/literature/candidates"
        / "paper-arxiv-20260806.json",
        record("2026-08-06", state="inbox"),
    )
    write_json(
        repo / "knowledge/literature/status/2026-08-06.json",
        status_payload(
            config,
            date="2026-08-06",
            accepted=1,
            open_candidates=0,
        ),
    )

    errors = status_guard.validate_repository(repo)

    assert any(
        "open_candidates must be 1, got 0" in error
        for error in errors
    )
