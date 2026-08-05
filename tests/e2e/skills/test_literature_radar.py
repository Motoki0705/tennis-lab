from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[3]
RADAR_SCRIPT = ROOT / ".agents/skills/literature-radar/scripts/radar_ingest.py"
KG_SCRIPTS = ROOT / ".agents/skills/knowledge-control/scripts"


def load_module(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


radar = load_module("literature_radar_ingest", RADAR_SCRIPT)
sys.path.insert(0, str(KG_SCRIPTS))
kg_lib = load_module("kg_lib", KG_SCRIPTS / "kg_lib.py")


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
    ):
        (tmp_path / path).mkdir(parents=True, exist_ok=True)
    return tmp_path


def candidate(
    *,
    collector: str = "geometry",
    run_id: str = "geometry-20260805T012000+0900-a",
    arxiv: str | None = "2608.01234v2",
    title: str = "A Tennis Trajectory Paper",
    task: str = "blcs",
    repo_path: str = "src/tasks/blcs",
    relevance: int = 90,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "kind": "literature_candidate",
        "collector_id": collector,
        "schedule_run_id": run_id,
        "discovered_at": "2026-08-05T01:20:00+09:00",
        "paper": {
            "title": title,
            "authors": ["Ada Example", "Taro Example"],
            "year": 2026,
            "venue": "arXiv",
            "identifiers": {"doi": None, "arxiv": arxiv, "openreview": None},
            "urls": {
                "primary": "https://arxiv.org/abs/2608.01234",
                "paper": "https://arxiv.org/pdf/2608.01234",
                "code": None,
                "project": None,
                "dataset": None,
            },
        },
        "screening": {
            "tasks": [task],
            "repo_paths": [repo_path],
            "relevance_score": relevance,
            "novelty_score": 85,
            "evidence_level": "fulltext",
            "summary_ja": "軌道推定手法を提案する。",
            "applicability_ja": "既存BLCS modelへ適用できる。",
            "risks_ja": "公式codeが未公開。",
            "candidate_experiment_ja": "既存baselineと3 seedsで比較する。",
        },
        "sources": [
            {
                "kind": "paper",
                "url": "https://arxiv.org/abs/2608.01234",
                "checked_at": "2026-08-05T01:20:00+09:00",
            }
        ],
    }


def test_canonical_id_prefers_normalized_external_identifier(tmp_path: Path) -> None:
    repo = make_repo(tmp_path)
    config = radar.load_config(repo)
    payload = candidate()
    radar.validate_raw_candidate(payload, repo, config)
    assert radar.canonical_paper_id(payload) == "paper-arxiv-2608-01234"

    payload["paper"]["identifiers"]["doi"] = "https://doi.org/10.1000/Example.42"
    assert radar.canonical_paper_id(payload) == "paper-doi-10-1000-example-42"


def test_queue_branch_collector_identity_is_enforced(tmp_path: Path) -> None:
    repo = make_repo(tmp_path)
    config = radar.load_config(repo)
    payload = candidate(collector="geometry")

    with pytest.raises(radar.CandidateError, match="does not match queue branch"):
        radar.validate_raw_candidate(
            payload, repo, config, expected_collector="perception"
        )


def test_ingest_merges_independent_collectors_without_overwriting_curation(
    tmp_path: Path,
) -> None:
    repo = make_repo(tmp_path)
    config = radar.load_config(repo)
    first_path = repo / "raw-first.json"
    first = candidate()
    write_json(first_path, first)

    result = radar.ingest_one(first_path, repo, config)
    assert result.action == "created"
    assert result.path is not None

    record = radar.read_json(result.path)
    record["state"] = "reviewed"
    record["curation"]["decision"] = "reviewed"
    radar.write_json(result.path, record)

    second_path = repo / "raw-second.json"
    second = candidate(
        collector="systems",
        run_id="systems-20260805T012000+0900-b",
        task="tennis_scene",
        repo_path="src/tennis_scene",
    )
    write_json(second_path, second)
    merged = radar.ingest_one(second_path, repo, config)

    assert merged.action == "merged"
    actual = radar.read_json(result.path)
    assert actual["state"] == "reviewed"
    assert actual["curation"]["decision"] == "reviewed"
    assert actual["aggregate"]["collector_count"] == 2
    assert actual["aggregate"]["tasks"] == ["blcs", "tennis_scene"]


def test_invalid_candidate_is_rejected_before_write(tmp_path: Path) -> None:
    repo = make_repo(tmp_path)
    config = radar.load_config(repo)
    payload = candidate(relevance=79)
    path = repo / "raw.json"
    write_json(path, payload)

    with pytest.raises(radar.CandidateError, match="below configured minimum"):
        radar.ingest_one(path, repo, config)
    assert list((repo / "knowledge/literature/candidates").glob("paper-*.json")) == []


def test_daily_digest_preserves_manual_review_section(tmp_path: Path) -> None:
    repo = make_repo(tmp_path)
    config = radar.load_config(repo)
    raw = repo / "raw.json"
    write_json(raw, candidate())
    radar.ingest_one(raw, repo, config)

    digest = radar.update_daily_digest(repo, "2026-08-05", config)
    text = digest.read_text(encoding="utf-8").replace(
        "未レビュー。", "promoted: paper-arxiv-2608-01234"
    )
    digest.write_text(text, encoding="utf-8")
    radar.update_daily_digest(repo, "2026-08-05", config)

    actual = digest.read_text(encoding="utf-8")
    assert actual.count(radar.AUTO_START) == 1
    assert "promoted: paper-arxiv-2608-01234" in actual


def node_text(meta: dict[str, Any], body: str = "## 内容\n\n本文") -> str:
    return f"---\n{yaml.safe_dump(meta, allow_unicode=True, sort_keys=False)}---\n\n{body}\n"


def test_formal_graph_accepts_paper_and_proposal_with_explicit_evidence(
    tmp_path: Path,
) -> None:
    nodes = tmp_path / "nodes"
    nodes.mkdir()
    (nodes / "run-baseline.md").write_text(
        node_text(
            {
                "id": "run-baseline",
                "type": "run",
                "title": "baseline",
                "status": "done",
                "provider": "human",
                "parents": [],
                "relations": [],
            }
        ),
        encoding="utf-8",
    )
    (nodes / "paper-arxiv-2608-01234.md").write_text(
        node_text(
            {
                "id": "paper-arxiv-2608-01234",
                "type": "paper",
                "title": "paper",
                "status": "reviewed",
                "external_ids": {
                    "doi": None,
                    "arxiv": "2608.01234",
                    "openreview": None,
                },
                "evidence_level": "fulltext-code",
                "tasks": ["blcs"],
                "repo_paths": ["src/tasks/blcs"],
                "sources": [{"kind": "paper", "url": "https://example.test/paper"}],
                "relations": [],
            }
        ),
        encoding="utf-8",
    )
    (nodes / "proposal-blcs-paper.md").write_text(
        node_text(
            {
                "id": "proposal-blcs-paper",
                "type": "proposal",
                "title": "proposal",
                "status": "supported",
                "issue": 1,
                "task": "blcs",
                "repo_paths": ["src/tasks/blcs"],
                "hypothesis": {
                    "statement": "改善する",
                    "expected_effect": "error低下",
                    "failure_condition": "改善なし",
                },
                "evaluation": {
                    "metrics": ["position_error_m"],
                    "baseline_nodes": ["run-baseline"],
                    "seeds": 3,
                    "acceptance": "平均error低下",
                },
                "evidence_runs": ["run-baseline"],
                "parents": ["run-baseline"],
                "relations": [
                    {"to": "paper-arxiv-2608-01234", "rel": "derived-from"}
                ],
            }
        ),
        encoding="utf-8",
    )

    result = kg_lib.validate(kg_lib.load_nodes(nodes))
    assert result.errors == []


def test_supported_proposal_requires_run_evidence(tmp_path: Path) -> None:
    nodes = tmp_path / "nodes"
    nodes.mkdir()
    (nodes / "paper-example.md").write_text(
        node_text(
            {
                "id": "paper-example",
                "type": "paper",
                "title": "paper",
                "status": "reviewed",
                "external_ids": {"doi": "10.1/example"},
                "evidence_level": "fulltext",
                "tasks": ["blcs"],
                "repo_paths": ["src/tasks/blcs"],
                "sources": [{"url": "https://example.test"}],
            }
        ),
        encoding="utf-8",
    )
    (nodes / "proposal-example.md").write_text(
        node_text(
            {
                "id": "proposal-example",
                "type": "proposal",
                "title": "proposal",
                "status": "supported",
                "issue": 1,
                "task": "blcs",
                "repo_paths": ["src/tasks/blcs"],
                "hypothesis": {
                    "statement": "改善する",
                    "expected_effect": "error低下",
                    "failure_condition": "改善なし",
                },
                "evaluation": {
                    "metrics": ["position_error_m"],
                    "baseline_nodes": ["paper-example"],
                    "seeds": 3,
                    "acceptance": "改善",
                },
                "evidence_runs": [],
                "relations": [{"to": "paper-example", "rel": "derived-from"}],
            }
        ),
        encoding="utf-8",
    )

    result = kg_lib.validate(kg_lib.load_nodes(nodes))
    assert any("requires non-empty 'evidence_runs'" in error for error in result.errors)
