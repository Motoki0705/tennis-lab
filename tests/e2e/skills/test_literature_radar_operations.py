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


status_guard = load_module("literature_radar_status", SCRIPT)


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


def record(index: int, *, state: str = "promoted") -> dict[str, Any]:
    return {
        "schema_version": 1,
        "id": f"paper-arxiv-2608-{index:05d}",
        "state": state,
        "first_seen": f"2026-08-05T0{index}:20:00+09:00",
        "last_seen": f"2026-08-05T0{index}:20:00+09:00",
        "paper": {
            "title": f"Paper {index}",
            "authors": ["Example Author"],
            "year": 2026,
            "venue": "arXiv",
            "identifiers": {
                "doi": None,
                "arxiv": f"2608.{index:05d}",
                "openreview": None,
            },
            "urls": {
                "primary": "https://example.test/paper",
                "paper": "https://example.test/paper.pdf",
                "code": None,
                "project": None,
                "dataset": None,
            },
        },
        "aggregate": {},
        "discoveries": [
            {
                "collector_id": "geometry",
                "schedule_run_id": f"geometry-run-{index}",
                "discovered_at": f"2026-08-05T0{index}:20:00+09:00",
                "screening": {
                    "topic": "ball_3d",
                },
                "sources": [],
            }
        ],
        "curation": {},
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
    accepted: int,
    mode: str,
    note: str | None,
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
        "date": "2026-08-05",
        "initialized_at": "2026-08-05T00:05:00+09:00",
        "last_curated_at": "2026-08-05T01:00:00+09:00",
        "generated_at": "2026-08-05T01:00:00+09:00",
        "quota_mode": mode,
        "quota_note": note,
        "ingestion": {
            "accepted_candidates": accepted,
            "daily_limit": config["ingestion"][
                "max_candidates_total_per_day"
            ],
            "open_candidates": 0,
            "open_limit": config["ingestion"]["max_open_candidates"],
            "collectors": collectors,
            "topics": topics,
        },
    }


def test_enforced_status_rejects_quota_overflow(tmp_path: Path) -> None:
    repo = make_repo(tmp_path)
    config = status_guard.load_config(repo)
    for index in range(1, 7):
        write_json(
            repo
            / "knowledge/literature/candidates"
            / f"paper-arxiv-2608-{index:05d}.json",
            record(index),
        )
    status = status_payload(
        config,
        accepted=6,
        mode="enforced",
        note=None,
    )
    path = repo / "knowledge/literature/status/2026-08-05.json"
    write_json(path, status)

    errors = status_guard.validate_repository(repo)

    assert any(
        "collectors.geometry exceeds enforced quota" in item
        for item in errors
    )
    assert any(
        "topics.ball_3d exceeds enforced quota" in item
        for item in errors
    )


def test_historical_backfill_preserves_pre_topic_snapshot(
    tmp_path: Path,
) -> None:
    repo = make_repo(tmp_path)
    config = status_guard.load_config(repo)
    for index in range(1, 7):
        candidate = record(index)
        candidate["discoveries"][0]["screening"].pop("topic")
        write_json(
            repo
            / "knowledge/literature/candidates"
            / f"paper-arxiv-2608-{index:05d}.json",
            candidate,
        )
    status = status_payload(
        config,
        accepted=6,
        mode="historical_backfill",
        note="Collected before topic quota hardening was merged.",
    )
    write_json(
        repo / "knowledge/literature/status/2026-08-05.json",
        status,
    )

    assert status_guard.validate_repository(repo) == []


def test_historical_backfill_validates_declared_topic_arithmetic(
    tmp_path: Path,
) -> None:
    repo = make_repo(tmp_path)
    config = status_guard.load_config(repo)
    for index in range(1, 7):
        candidate = record(index)
        candidate["discoveries"][0]["screening"].pop("topic")
        write_json(
            repo
            / "knowledge/literature/candidates"
            / f"paper-arxiv-2608-{index:05d}.json",
            candidate,
        )
    status = status_payload(
        config,
        accepted=6,
        mode="historical_backfill",
        note="Collected before topic quota hardening was merged.",
    )
    status["ingestion"]["topics"]["ball_3d"]["remaining"] = 1
    write_json(
        repo / "knowledge/literature/status/2026-08-05.json",
        status,
    )

    errors = status_guard.validate_repository(repo)

    assert any(
        "topics.ball_3d.remaining must be 0" in item
        for item in errors
    )


def test_normalize_adds_explicit_default_policy(tmp_path: Path) -> None:
    repo = make_repo(tmp_path)
    config = status_guard.load_config(repo)
    status = status_payload(
        config,
        accepted=0,
        mode="enforced",
        note=None,
    )
    status.pop("quota_mode")
    status.pop("quota_note")
    path = repo / "knowledge/literature/status/2026-08-05.json"
    write_json(path, status)

    status_guard.normalize_status(repo, "2026-08-05")
    actual = status_guard.read_json(path)

    assert actual["quota_mode"] == "enforced"
    assert actual["quota_note"] is None


def test_ingest_workflow_retries_from_latest_daily_head() -> None:
    workflow = (
        ROOT / ".github/workflows/literature-radar-ingest.yml"
    ).read_text(encoding="utf-8")

    assert "Load trusted scripts from main" in workflow
    assert "for attempt in 1 2 3" in workflow
    assert "rebuilding from its latest head" in workflow
    assert "Run daily curator initialization first" not in workflow


def test_curator_contract_prevents_raw_loss_and_unsafe_yaml() -> None:
    prompt = (
        ROOT / "knowledge/literature/prompts/daily-curator.md"
    ).read_text(encoding="utf-8")

    assert "queue branchは日付をまたいでappend-only" in prompt
    assert "存在するqueue branchはreset、rebase、force updateしない" in prompt
    assert "schedule_run_id" in prompt
    assert "frontmatterを手書きのplain YAMLとして生成しない" in prompt
    assert "最新mainをparentとする新しい単一snapshot commit" in prompt
    assert "Closes #<daily-issue-number>" in prompt


def test_ci_uses_hardened_candidate_and_status_validation() -> None:
    workflow = (ROOT / ".github/workflows/ci.yml").read_text(
        encoding="utf-8"
    )

    assert "radar_hardening.py" in workflow
    assert "radar_status.py" in workflow
    assert "radar_ingest.py \\\n            validate" not in workflow
