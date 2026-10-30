"""Ops report (#1057): deterministic collection, agent analysis, issue triage."""

# Support the documented standalone PYTHONPATH=.agents/ops/ops-report entrypoint.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
