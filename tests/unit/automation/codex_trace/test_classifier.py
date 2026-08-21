from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.automation.codex_trace.classifier import SemanticClassifier


def test_classifier_uses_action_evidence_when_reasoning_is_unreadable() -> None:
    result = SemanticClassifier().classify(
        "", tool_kinds=("apply_patch",), action_text="update implementation"
    )

    assert result.evidence_mode == "response_action"
    assert result.primary_cluster == "implementation"
    assert sum(result.probabilities.values()) == pytest.approx(1.0)


def test_classifier_loads_explicit_taxonomy(tmp_path: Path) -> None:
    path = tmp_path / "clusters.json"
    path.write_text(
        json.dumps(
            {
                "clusters": {
                    "schema_research": {
                        "keywords": ["schema"],
                        "tool_kinds": ["web"],
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    result = SemanticClassifier.from_json(path).classify("inspect the schema")

    assert result.probabilities == {"schema_research": 1.0}


def test_classifier_rejects_signal_free_cluster(tmp_path: Path) -> None:
    path = tmp_path / "clusters.json"
    path.write_text(
        json.dumps({"clusters": {"empty": {"keywords": [], "tool_kinds": []}}}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="no classification signals"):
        SemanticClassifier.from_json(path)
