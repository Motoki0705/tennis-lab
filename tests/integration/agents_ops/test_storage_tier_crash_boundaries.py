"""Every recorded persistence boundary must recover into an operational daily cycle."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests.support.agents_ops.storage_crashes import (
    SCENARIOS,
    finish_recovery,
    payloads,
    prepare,
    run_child,
)
from tests.support.agents_ops.storage_tier import OPS

TABLE = json.loads(
    (
        Path(__file__).resolve().parents[2]
        / "support/agents_ops/fixtures/storage-crash-boundaries.json"
    ).read_text()
)

RECOVERY = json.loads(
    (
        Path(__file__).resolve().parents[2]
        / "support/agents_ops/fixtures/storage-recovery-boundaries.json"
    ).read_text()
)


def boundaries(scenario: str, phase: str = "all") -> list:
    trace = TABLE[scenario]
    split = next(
        (i for i, value in enumerate(trace) if "transactions" in value), len(trace)
    )
    if phase in {"quarantine", "deletion"}:
        split = trace.index("json:transactions:verified") + 1
    return [
        pytest.param(i, value, id=f"{i:03}-{value}")
        for i, value in enumerate(trace)
        if phase == "all"
        or (phase == "quarantine" and i < split)
        or (phase == "deletion" and i >= split)
        or (phase == "publication" and i < split)
        or (phase == "removal" and i >= split)
    ]


def exercise(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    scenario: str,
    index: int,
    boundary: str,
) -> None:
    monkeypatch.syspath_prepend(str(OPS))
    lab, unit, action = prepare(scenario, tmp_path)
    output = tmp_path / "crash.jsonl"
    code, trace = run_child(lab, action, output, stop=index)
    assert code == 91 and not output.with_suffix(".result.json").exists(), (code, trace)
    assert trace == TABLE[scenario][: index + 1], (boundary, trace)
    temporaries = {p: p.read_bytes() for p in lab.config.state.rglob("*.tmp")}
    before = payloads(lab)
    recovery_output = tmp_path / "recovery.jsonl"
    code, recovery_trace = run_child(
        lab, lambda: lab.engine.run("recover", [], dry_run=False), recovery_output
    )
    route = RECOVERY["routes"][RECOVERY["mapping"][f"{scenario}/{index}"]]
    expected_code = 1 if scenario == "orphan-recovery" else 0
    assert code == 0 and recovery_trace == route["trace"], recovery_trace
    assert (
        json.loads(recovery_output.with_suffix(".result.json").read_text())[0]
        == expected_code
    )
    assert before <= payloads(lab)
    finish_recovery(lab, unit)
    if b"unverified stage payload must survive" in before:
        assert b"unverified stage payload must survive" in payloads(lab)
    assert all(
        p.exists() and p.read_bytes() == value for p, value in temporaries.items()
    )


@pytest.mark.parametrize("scenario", SCENARIOS)
def test_persistence_trace_has_no_uncovered_boundary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, scenario: str
) -> None:
    monkeypatch.syspath_prepend(str(OPS))
    lab, _, action = prepare(scenario, tmp_path)
    output = tmp_path / "trace.jsonl"
    code, trace = run_child(lab, action, output)
    assert code == 0 and trace == TABLE[scenario], trace
    outcome = json.loads(output.with_suffix(".result.json").read_text())
    expected = (
        1
        if scenario.startswith(("discard-", "partial-"))
        or scenario == "orphan-recovery"
        else 0
    )
    assert outcome[0] == expected, outcome


@pytest.mark.parametrize("index,boundary", boundaries("mirror-new", "all"))
def test_mirror_new_persistence_boundary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, index: int, boundary: str
) -> None:
    exercise(tmp_path, monkeypatch, "mirror-new", index, boundary)


@pytest.mark.parametrize("index,boundary", boundaries("mirror-update", "publication"))
def test_mirror_update_publication_persistence_boundary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, index: int, boundary: str
) -> None:
    exercise(tmp_path, monkeypatch, "mirror-update", index, boundary)


@pytest.mark.parametrize("index,boundary", boundaries("mirror-update", "removal"))
def test_mirror_update_removal_persistence_boundary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, index: int, boundary: str
) -> None:
    exercise(tmp_path, monkeypatch, "mirror-update", index, boundary)


@pytest.mark.parametrize("index,boundary", boundaries("hydrate", "all"))
def test_hydrate_persistence_boundary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, index: int, boundary: str
) -> None:
    exercise(tmp_path, monkeypatch, "hydrate", index, boundary)


@pytest.mark.parametrize("index,boundary", boundaries("evict", "all"))
def test_evict_persistence_boundary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, index: int, boundary: str
) -> None:
    exercise(tmp_path, monkeypatch, "evict", index, boundary)


@pytest.mark.parametrize("index,boundary", boundaries("discard-mirror", "publication"))
def test_discard_mirror_publication_persistence_boundary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, index: int, boundary: str
) -> None:
    exercise(tmp_path, monkeypatch, "discard-mirror", index, boundary)


@pytest.mark.parametrize("index,boundary", boundaries("discard-mirror", "removal"))
def test_discard_mirror_removal_persistence_boundary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, index: int, boundary: str
) -> None:
    exercise(tmp_path, monkeypatch, "discard-mirror", index, boundary)


@pytest.mark.parametrize("index,boundary", boundaries("discard-hydrate", "publication"))
def test_discard_hydrate_publication_persistence_boundary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, index: int, boundary: str
) -> None:
    exercise(tmp_path, monkeypatch, "discard-hydrate", index, boundary)


@pytest.mark.parametrize("index,boundary", boundaries("discard-hydrate", "removal"))
def test_discard_hydrate_removal_persistence_boundary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, index: int, boundary: str
) -> None:
    exercise(tmp_path, monkeypatch, "discard-hydrate", index, boundary)


@pytest.mark.parametrize("index,boundary", boundaries("partial-mirror", "all"))
def test_partial_mirror_persistence_boundary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, index: int, boundary: str
) -> None:
    exercise(tmp_path, monkeypatch, "partial-mirror", index, boundary)


@pytest.mark.parametrize("index,boundary", boundaries("partial-hydrate", "all"))
def test_partial_hydrate_persistence_boundary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, index: int, boundary: str
) -> None:
    exercise(tmp_path, monkeypatch, "partial-hydrate", index, boundary)


@pytest.mark.parametrize("index,boundary", boundaries("checkpoint-new", "all"))
def test_checkpoint_new_persistence_boundary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, index: int, boundary: str
) -> None:
    exercise(tmp_path, monkeypatch, "checkpoint-new", index, boundary)


@pytest.mark.parametrize("index,boundary", boundaries("checkpoint-mirror", "all"))
def test_checkpoint_mirror_persistence_boundary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, index: int, boundary: str
) -> None:
    exercise(tmp_path, monkeypatch, "checkpoint-mirror", index, boundary)


@pytest.mark.parametrize("index,boundary", boundaries("checkpoint-hydrate", "all"))
def test_checkpoint_hydrate_persistence_boundary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, index: int, boundary: str
) -> None:
    exercise(tmp_path, monkeypatch, "checkpoint-hydrate", index, boundary)


@pytest.mark.parametrize("index,boundary", boundaries("recover-publication", "all"))
def test_recover_publication_persistence_boundary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, index: int, boundary: str
) -> None:
    exercise(tmp_path, monkeypatch, "recover-publication", index, boundary)


@pytest.mark.parametrize("index,boundary", boundaries("recover-hydrate", "all"))
def test_recover_hydrate_persistence_boundary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, index: int, boundary: str
) -> None:
    exercise(tmp_path, monkeypatch, "recover-hydrate", index, boundary)


@pytest.mark.parametrize("index,boundary", boundaries("recover-complete-stage", "all"))
def test_recover_complete_stage_persistence_boundary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, index: int, boundary: str
) -> None:
    exercise(tmp_path, monkeypatch, "recover-complete-stage", index, boundary)


@pytest.mark.parametrize("index,boundary", boundaries("recover-evict", "all"))
def test_recover_evict_persistence_boundary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, index: int, boundary: str
) -> None:
    exercise(tmp_path, monkeypatch, "recover-evict", index, boundary)


@pytest.mark.parametrize("index,boundary", boundaries("recover-retire", "all"))
def test_recover_retire_persistence_boundary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, index: int, boundary: str
) -> None:
    exercise(tmp_path, monkeypatch, "recover-retire", index, boundary)


@pytest.mark.parametrize("index,boundary", boundaries("recover-discard", "all"))
def test_recover_discard_persistence_boundary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, index: int, boundary: str
) -> None:
    exercise(tmp_path, monkeypatch, "recover-discard", index, boundary)


@pytest.mark.parametrize("index,boundary", boundaries("recover-copy", "all"))
def test_recover_copy_persistence_boundary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, index: int, boundary: str
) -> None:
    exercise(tmp_path, monkeypatch, "recover-copy", index, boundary)


@pytest.mark.parametrize("index,boundary", boundaries("recover-checkpoint", "all"))
def test_recover_checkpoint_persistence_boundary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, index: int, boundary: str
) -> None:
    exercise(tmp_path, monkeypatch, "recover-checkpoint", index, boundary)


@pytest.mark.parametrize(
    "index,boundary", boundaries("recover-checkpoint-hydrate", "all")
)
def test_recover_checkpoint_hydrate_persistence_boundary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, index: int, boundary: str
) -> None:
    exercise(tmp_path, monkeypatch, "recover-checkpoint-hydrate", index, boundary)


@pytest.mark.parametrize(
    "index,boundary", boundaries("mirror-hardlinks", "publication")
)
def test_mirror_hardlinks_publication_persistence_boundary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, index: int, boundary: str
) -> None:
    exercise(tmp_path, monkeypatch, "mirror-hardlinks", index, boundary)


@pytest.mark.parametrize("index,boundary", boundaries("mirror-hardlinks", "removal"))
def test_mirror_hardlinks_removal_persistence_boundary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, index: int, boundary: str
) -> None:
    exercise(tmp_path, monkeypatch, "mirror-hardlinks", index, boundary)


@pytest.mark.parametrize("index,boundary", boundaries("evict-hardlinks", "quarantine"))
def test_evict_hardlinks_quarantine_persistence_boundary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, index: int, boundary: str
) -> None:
    exercise(tmp_path, monkeypatch, "evict-hardlinks", index, boundary)


@pytest.mark.parametrize("index,boundary", boundaries("evict-hardlinks", "deletion"))
def test_evict_hardlinks_deletion_persistence_boundary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, index: int, boundary: str
) -> None:
    exercise(tmp_path, monkeypatch, "evict-hardlinks", index, boundary)


@pytest.mark.parametrize("index,boundary", boundaries("checkpoint-hardlinks", "all"))
def test_checkpoint_hardlinks_persistence_boundary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, index: int, boundary: str
) -> None:
    exercise(tmp_path, monkeypatch, "checkpoint-hardlinks", index, boundary)


@pytest.mark.parametrize("index,boundary", boundaries("index-legacy", "all"))
def test_index_legacy_persistence_boundary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, index: int, boundary: str
) -> None:
    exercise(tmp_path, monkeypatch, "index-legacy", index, boundary)


@pytest.mark.parametrize("index,boundary", boundaries("orphan-recovery"))
def test_orphan_recovery_persistence_boundary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, index: int, boundary: str
) -> None:
    exercise(tmp_path, monkeypatch, "orphan-recovery", index, boundary)


@pytest.mark.parametrize("index,boundary", boundaries("repair-audit"))
def test_audit_repair_persistence_boundary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, index: int, boundary: str
) -> None:
    exercise(tmp_path, monkeypatch, "repair-audit", index, boundary)


@pytest.mark.parametrize("index,boundary", boundaries("repair-removals"))
def test_removal_journal_repair_persistence_boundary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, index: int, boundary: str
) -> None:
    exercise(tmp_path, monkeypatch, "repair-removals", index, boundary)
