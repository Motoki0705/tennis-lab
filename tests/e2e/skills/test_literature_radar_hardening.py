from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[3]
SCRIPT_DIR = ROOT / ".agents/skills/literature-radar/scripts"


def load_module(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


legacy = load_module("radar_ingest", SCRIPT_DIR / "radar_ingest.py")
hardening = load_module("radar_hardening", SCRIPT_DIR / "radar_hardening.py")


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")


def make_repo(tmp_path: Path) -> Path:
    config = json.loads((ROOT / "knowledge/literature/config.json").read_text())
    write_json(tmp_path / "knowledge/literature/config.json", config)
    for path in (
        "src/tasks/blcs",
        "src/tasks/ball_detection",
        "src/tennis_scene",
        "knowledge/literature/candidates",
        "knowledge/literature/digests",
        "knowledge/literature/status",
    ):
        (tmp_path / path).mkdir(parents=True, exist_ok=True)
    return tmp_path


def candidate(
    *,
    run_id: str = "geometry-20260805T012000+0900-a",
    doi: str | None = None,
    arxiv: str | None = "2608.01234v2",
    relevance: int = 85,
    evidence: str = "fulltext",
    topic: str = "ball_3d",
) -> dict[str, Any]:
    evidence_score = {
        "abstract": 5,
        "fulltext": 14,
        "fulltext-code": 18,
        "fulltext-code-data": 20,
    }[evidence]
    breakdown = {
        "task_fit": 28,
        "repo_fit": 22,
        "evidence_quality": evidence_score,
        "experiment_quality": 13,
        "adoption_feasibility": relevance - 63 - evidence_score,
    }
    sources = [
        {
            "kind": "paper",
            "url": "https://arxiv.org/abs/2608.01234",
            "checked_at": "2026-08-05T01:20:00+09:00",
        }
    ]
    urls: dict[str, str | None] = {
        "primary": "https://arxiv.org/abs/2608.01234",
        "paper": "https://arxiv.org/pdf/2608.01234",
        "code": None,
        "project": None,
        "dataset": None,
    }
    if evidence in {"fulltext-code", "fulltext-code-data"}:
        urls["code"] = "https://github.com/example/paper"
        sources.append(
            {
                "kind": "code",
                "url": "https://github.com/example/paper",
                "checked_at": "2026-08-05T01:20:00+09:00",
            }
        )
    if evidence == "fulltext-code-data":
        urls["dataset"] = "https://example.test/data"
        sources.append(
            {
                "kind": "dataset",
                "url": "https://example.test/data",
                "checked_at": "2026-08-05T01:20:00+09:00",
            }
        )
    return {
        "schema_version": 1,
        "kind": "literature_candidate",
        "collector_id": "geometry",
        "schedule_run_id": run_id,
        "discovered_at": "2026-08-05T01:20:00+09:00",
        "paper": {
            "title": "A Tennis Trajectory Paper",
            "authors": ["Ada Example", "Taro Example"],
            "year": 2026,
            "venue": "arXiv",
            "identifiers": {"doi": doi, "arXiv": arxiv, "openreview": None},
            "urls": urls,
        },
        "screening": {
            "tasks": ["blcs"],
            "topic": topic,
            "repo_paths": ["src/tasks/blcs"],
            "relevance_score": relevance,
            "score_breakdown": breakdown,
            "novelty_score": 80,
            "evidence_level": evidence,
            "summary_ja": "軌道推定手法を提案する。",
            "applicability_ja": "既存BLCS modelへ適用できる。",
            "risks_ja": "domain gapがある。",
            "candidate_experiment_ja": "固定baselineと3 seedsで比較する。",
        },
        "sources": sources,
    }


def test_datacite_arxiv_doi_is_an_arxiv_alias() -> None:
    payload = candidate(doi="10.48550/arXiv.2608.01234", arxiv=None)
    assert hardening.canonical_paper_id(payload) == "paper-arxiv-2608-01234"
    clean = hardening._sanitise_payload(payload)
    assert clean["paper"]["identifiers"] == {
        "doi": None,
        "arxiv": "2608.01234",
        "openreview": None,
    }


def test_publisher_doi_still_has_precedence() -> None:
    payload = candidate(doi="https://doi.org/10.1000/Example.42")
    assert hardening.canonical_paper_id(payload) == "paper-doi-10-1000-example-42"


def test_score_must_equal_breakdown(tmp_path: Path) -> None:
    repo = make_repo(tmp_path)
    payload = candidate()
    payload["screening"]["score_breakdown"]["task_fit"] -= 1
    with pytest.raises(legacy.CandidateError, match="score_breakdown sum"):
        hardening.validate_hardened_candidate(payload, repo, legacy.load_config(repo))


def test_abstract_only_candidate_is_rejected(tmp_path: Path) -> None:
    repo = make_repo(tmp_path)
    payload = candidate(evidence="abstract", relevance=80)
    payload["screening"]["score_breakdown"].update(
        task_fit=30, adoption_feasibility=10
    )
    with pytest.raises(legacy.CandidateError, match="below minimum"):
        hardening.validate_hardened_candidate(payload, repo, legacy.load_config(repo))


def test_all_tasks_must_belong_to_collector(tmp_path: Path) -> None:
    repo = make_repo(tmp_path)
    payload = candidate()
    payload["screening"]["tasks"].append("ball_detection")
    with pytest.raises(legacy.CandidateError, match="exceed collector"):
        hardening.validate_hardened_candidate(payload, repo, legacy.load_config(repo))


def test_alias_match_merges_datacite_and_arxiv_forms(tmp_path: Path) -> None:
    repo = make_repo(tmp_path)
    config = legacy.load_config(repo)
    first_path = repo / "first.json"
    write_json(first_path, candidate())
    first = hardening.ingest_one(first_path, repo, config)
    assert first.action == "created"

    second_path = repo / "second.json"
    write_json(
        second_path,
        candidate(
            run_id="geometry-20260805T022000+0900-b",
            doi="10.48550/arXiv.2608.01234",
            arxiv=None,
        ),
    )
    second = hardening.ingest_one(second_path, repo, config)
    assert second.action == "merged"
    assert second.paper_id == "paper-arxiv-2608-01234"
    assert len(legacy.candidate_records(repo)) == 1


def test_topic_quota_prevents_same_topic_twice(tmp_path: Path) -> None:
    repo = make_repo(tmp_path)
    config = legacy.load_config(repo)
    first_path = repo / "first.json"
    write_json(first_path, candidate())
    hardening.ingest_one(first_path, repo, config)

    other = candidate(
        run_id="geometry-20260805T022000+0900-c",
        arxiv="2608.99999",
    )
    other["paper"]["title"] = "Another Ball Paper"
    other["paper"]["urls"]["primary"] = "https://arxiv.org/abs/2608.99999"
    allowed, reason = hardening.quota_allows(
        legacy.candidate_records(repo), other, config
    )
    assert not allowed
    assert "topic daily quota" in reason


def test_legacy_digest_preamble_is_repaired(tmp_path: Path) -> None:
    repo = make_repo(tmp_path)
    config = legacy.load_config(repo)
    raw = repo / "raw.json"
    write_json(raw, candidate())
    hardening.ingest_one(raw, repo, config)
    digest = repo / "knowledge/literature/digests/2026-08-05.md"
    digest.write_text(
        "# Literature Radar — 2026-08-05\n\n"
        "この日次ダイジェストの `<!-- literature-radar:auto:start -->\n"
        "broken\n<!-- literature-radar:auto:end -->\n\n"
        "## 日次レビュー\n\n手動レビュー。\n",
        encoding="utf-8",
    )
    hardening.update_daily_digest(repo, "2026-08-05", config)
    actual = digest.read_text(encoding="utf-8")
    assert "この日次ダイジェストの `" not in actual
    assert actual.count(legacy.AUTO_START) == 1
    assert "手動レビュー。" in actual


def test_unknown_digest_marker_damage_fails_closed(tmp_path: Path) -> None:
    repo = make_repo(tmp_path)
    config = legacy.load_config(repo)
    digest = repo / "knowledge/literature/digests/2026-08-05.md"
    digest.write_text(
        f"{legacy.AUTO_START}\nA\n{legacy.AUTO_START}\nB\n{legacy.AUTO_END}\n",
        encoding="utf-8",
    )
    with pytest.raises(legacy.CandidateError, match="exactly once"):
        hardening.update_daily_digest(repo, "2026-08-05", config)


def test_status_generation_is_stable_without_count_changes(tmp_path: Path) -> None:
    repo = make_repo(tmp_path)
    config = legacy.load_config(repo)
    raw = repo / "raw.json"
    write_json(raw, candidate())
    hardening.ingest_one(raw, repo, config)
    status_path = hardening.update_daily_status(repo, "2026-08-05", config)
    first = legacy.read_json(status_path)
    hardening.update_daily_status(repo, "2026-08-05", config)
    second = legacy.read_json(status_path)
    assert first == second
    assert second["ingestion"]["accepted_candidates"] == 1
    assert second["ingestion"]["collectors"]["geometry"]["remaining"] == 1
