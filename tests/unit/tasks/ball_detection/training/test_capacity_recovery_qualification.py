"""A saved recovery boundary must be explicit and belong to the frozen plan."""

import importlib.util
import sys
from pathlib import Path

import pytest

SCRIPT = Path("knowledge/runs/run-ball-capacity-scaling-20261103/qualify_json_resume.py")
spec = importlib.util.spec_from_file_location("capacity_recovery_qualification", SCRIPT)
assert spec is not None and spec.loader is not None
qualifier = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = qualifier
spec.loader.exec_module(qualifier)


@pytest.fixture
def plan():
    return dict(stages=[
        dict(name="query-d512-l4-T64", model=dict(dim=512), budget=dict(updates=6000)),
        dict(name="query-d768-l4-T64", model=dict(dim=768), budget=dict(updates=6000)),
    ])


@pytest.mark.parametrize("stage,update,dim", [
    ("query-d512-l4-T64", 5500, 512), ("query-d768-l4-T64", 500, 768),
])
def test_restored_stage_and_update_choose_the_exact_planned_model(plan, stage, update, dim):
    entry = qualifier.verified_target(plan, stage, update,
                                     dict(update=update, checkpoint=f"checkpoint-update-{update:08d}.pt"))
    assert entry["model"]["dim"] == dim


@pytest.mark.parametrize("update,pointer_update", [(0, 0), (6000, 6000), (500, 1000)])
def test_initial_complete_or_different_saved_update_is_rejected(plan, update, pointer_update):
    with pytest.raises(ValueError):
        qualifier.verified_target(plan, "query-d768-l4-T64", update,
                                  dict(update=pointer_update, checkpoint=f"checkpoint-update-{pointer_update:08d}.pt"))


def test_wrong_pointer_and_unknown_or_duplicate_stage_are_rejected(plan):
    latest = dict(update=500, checkpoint="checkpoint-update-00005500.pt")
    with pytest.raises(ValueError, match="pointer/update"):
        qualifier.verified_target(plan, "query-d768-l4-T64", 500, latest)
    latest["checkpoint"] = "checkpoint-update-00000500.pt"
    with pytest.raises(ValueError, match="one planned"):
        qualifier.verified_target(plan, "absent", 500, latest)
    plan["stages"].append(plan["stages"][-1])
    with pytest.raises(ValueError, match="one planned"):
        qualifier.verified_target(plan, "query-d768-l4-T64", 500, latest)
