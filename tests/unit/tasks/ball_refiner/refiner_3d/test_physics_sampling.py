from types import SimpleNamespace

import pytest

from src.tasks.ball_refiner.refiner_3d.synthetic import simulation
from src.tasks.blcs.generate_dataset.simulation.targeted_velocity_sampler import (
    FULL_PHYSICS_REJECTION_PREFIX,
)


def _install_simulator(monkeypatch, errors):
    calls = []
    accepted = object()
    def generate(**kwargs):
        calls.append(kwargs)
        if errors:
            raise errors.pop(0)
        return accepted
    monkeypatch.setattr(simulation, "PhysicsConfig", lambda **_: SimpleNamespace(sample=lambda: SimpleNamespace(gravity=9.81)))
    monkeypatch.setattr(simulation, "RallyConfig", lambda **_: None)
    monkeypatch.setattr(simulation, "TargetedVelocityConfig", lambda **_: None)
    monkeypatch.setattr(simulation, "RallySimulator", lambda **_: SimpleNamespace(generate_rally=generate))
    plan = SimpleNamespace(values={"simulation": {"maximum_physics_attempts_per_rally": 3}}, physics={}, rally={}, targeted={})
    return plan, calls, accepted


def test_only_declared_physics_rejections_are_resampled_with_logged_seeds(monkeypatch):
    plan, calls, accepted = _install_simulator(monkeypatch, [RuntimeError(FULL_PHYSICS_REJECTION_PREFIX + " test rejection")])
    result, _, _, proposals = simulation.accepted_rally(plan, seed=936, index=0)
    assert result is accepted and len(calls) == 2
    assert [p["status"] for p in proposals] == ["rejected", "accepted"]
    assert proposals[0]["seed"] == 936
    assert proposals[1]["seed"] != 936
    assert "test rejection" in proposals[0]["reason"]


def test_unknown_runtime_failure_is_not_resampled(monkeypatch):
    plan, calls, _ = _install_simulator(monkeypatch, [RuntimeError("unexpected implementation failure")])
    with pytest.raises(RuntimeError, match="unexpected implementation"):
        simulation.accepted_rally(plan, seed=936, index=0)
    assert len(calls) == 1


def test_physics_budget_exhaustion_is_a_failure(monkeypatch):
    plan, calls, _ = _install_simulator(monkeypatch, [RuntimeError(FULL_PHYSICS_REJECTION_PREFIX)] * 4)
    with pytest.raises(RuntimeError, match="budget exhausted"):
        simulation.accepted_rally(plan, seed=936, index=0)
    assert len(calls) == 3
