"""Read the versioned operational decision used when creating campaign plans."""

from __future__ import annotations

import tomllib
from typing import Literal

from pydantic import Field

from src.utils.paths import PROJECT_ROOT

from .contracts import Document

DECISION_PATH = (
    PROJECT_ROOT / ".agents/skills/annotation-workflow/worker-decision.toml"
)


class WorkerDecision(Document):
    schema_version: Literal["annotation.worker-decision.v1"]
    agent_model: str = Field(min_length=1)
    agent_effort: Literal["low", "medium", "high", "xhigh", "max", "ultra"]
    decided_on: str = Field(pattern=r"^\d{4}-\d{2}-\d{2}$")
    source: str = Field(min_length=1)

    def require(self, *, agent_model: str, agent_effort: str) -> None:
        if (agent_model, agent_effort) != (self.agent_model, self.agent_effort):
            raise ValueError(
                "new campaign worker must match the user decision: "
                f"{self.agent_model}/{self.agent_effort} "
                f"({self.decided_on}, {self.source}); "
                f"received {agent_model}/{agent_effort}"
            )


def load_worker_decision() -> WorkerDecision:
    """Fail on missing/invalid decisions; never use a fixture or implicit default."""
    with DECISION_PATH.open("rb") as handle:
        decision: WorkerDecision = WorkerDecision.model_validate(tomllib.load(handle))
    return decision
