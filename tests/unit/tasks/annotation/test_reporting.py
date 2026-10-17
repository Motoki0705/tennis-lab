from __future__ import annotations

import pytest

from src.tasks.annotation.__main__ import compact_status


@pytest.mark.parametrize(
    "learning",
    [
        {"status": "not_configured", "reason": "recipe missing"},
        {"status": "no_additional_teachers", "reason": "no coordinate teachers"},
        {"state": "failed", "stage": "train", "reason": "training failed"},
        {"state": "rejected", "stage": "select", "reason": "validation regression"},
    ],
)
def test_compact_status_preserves_publication_and_learning_results(
    learning: dict,
) -> None:
    status = {
        "state": "blocked",
        "annotation": {
            "complete": True,
            "products": [
                {
                    "task": "ball",
                    "product": "frames",
                    "status": "published",
                    "publication": "receipt",
                }
            ],
        },
        "workflow": {"state": "blocked", "learning": learning},
        "failures": [{"state": "failed", "reason": str(i)} for i in range(12)],
    }
    result = compact_status(status)
    assert result["annotation"] == status["annotation"]
    assert result["workflow"]["learning"] == learning
    assert result["failure_count"] == 12
    assert len(result["failures"]) == 10
