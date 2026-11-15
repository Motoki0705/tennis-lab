"""Every distinct recovery mutation sequence survives a further process exit."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from tests.support.agents_ops.storage_crashes import (
    finish_recovery,
    payloads,
    prepare,
    run_child,
)
from tests.support.agents_ops.storage_tier import OPS

TABLE = json.loads(
    (
        Path(__file__).resolve().parents[2]
        / "support/agents_ops/fixtures/storage-recovery-boundaries.json"
    ).read_text()
)


def boundaries(scenario: str) -> list[Any]:
    return [
        pytest.param(key, i, label, id=f"{key}-{i:03}-{label}")
        for key, route in TABLE["routes"].items()
        if route["scenario"] == scenario
        for i, label in enumerate(route["trace"])
    ]


def exercise(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    route_id: str,
    index: int,
    boundary: str,
) -> None:
    monkeypatch.syspath_prepend(str(OPS))
    route = TABLE["routes"][route_id]
    lab, unit, action = prepare(route["scenario"], tmp_path)
    code, _ = run_child(
        lab, action, tmp_path / "origin.jsonl", stop=route["origin_index"]
    )
    assert code == 91
    before = payloads(lab)
    output = tmp_path / "recovery-crash.jsonl"
    code, trace = run_child(
        lab, lambda: lab.engine.run("recover", [], dry_run=False), output, stop=index
    )
    assert code == 91 and not output.with_suffix(".result.json").exists(), (code, trace)
    assert trace == route["trace"][: index + 1], (boundary, trace)
    assert before <= payloads(lab), "interrupted recovery discarded payload"
    finish_recovery(lab, unit)
    if b"unverified stage payload must survive" in before:
        assert b"unverified stage payload must survive" in payloads(lab)


@pytest.mark.parametrize("route_id,index,boundary", boundaries("mirror-new"))
def test_mirror_new_recovery_boundary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    route_id: str,
    index: int,
    boundary: str,
) -> None:
    exercise(tmp_path, monkeypatch, route_id, index, boundary)


@pytest.mark.parametrize("route_id,index,boundary", boundaries("mirror-update"))
def test_mirror_update_recovery_boundary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    route_id: str,
    index: int,
    boundary: str,
) -> None:
    exercise(tmp_path, monkeypatch, route_id, index, boundary)


@pytest.mark.parametrize("route_id,index,boundary", boundaries("hydrate"))
def test_hydrate_recovery_boundary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    route_id: str,
    index: int,
    boundary: str,
) -> None:
    exercise(tmp_path, monkeypatch, route_id, index, boundary)


@pytest.mark.parametrize("route_id,index,boundary", boundaries("evict"))
def test_evict_recovery_boundary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    route_id: str,
    index: int,
    boundary: str,
) -> None:
    exercise(tmp_path, monkeypatch, route_id, index, boundary)


@pytest.mark.parametrize("route_id,index,boundary", boundaries("discard-mirror"))
def test_discard_mirror_recovery_boundary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    route_id: str,
    index: int,
    boundary: str,
) -> None:
    exercise(tmp_path, monkeypatch, route_id, index, boundary)


@pytest.mark.parametrize("route_id,index,boundary", boundaries("discard-hydrate"))
def test_discard_hydrate_recovery_boundary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    route_id: str,
    index: int,
    boundary: str,
) -> None:
    exercise(tmp_path, monkeypatch, route_id, index, boundary)


@pytest.mark.parametrize("route_id,index,boundary", boundaries("partial-mirror"))
def test_partial_mirror_recovery_boundary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    route_id: str,
    index: int,
    boundary: str,
) -> None:
    exercise(tmp_path, monkeypatch, route_id, index, boundary)


@pytest.mark.parametrize("route_id,index,boundary", boundaries("partial-hydrate"))
def test_partial_hydrate_recovery_boundary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    route_id: str,
    index: int,
    boundary: str,
) -> None:
    exercise(tmp_path, monkeypatch, route_id, index, boundary)


@pytest.mark.parametrize("route_id,index,boundary", boundaries("checkpoint-new"))
def test_checkpoint_new_recovery_boundary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    route_id: str,
    index: int,
    boundary: str,
) -> None:
    exercise(tmp_path, monkeypatch, route_id, index, boundary)


@pytest.mark.parametrize("route_id,index,boundary", boundaries("checkpoint-mirror"))
def test_checkpoint_mirror_recovery_boundary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    route_id: str,
    index: int,
    boundary: str,
) -> None:
    exercise(tmp_path, monkeypatch, route_id, index, boundary)


@pytest.mark.parametrize("route_id,index,boundary", boundaries("checkpoint-hydrate"))
def test_checkpoint_hydrate_recovery_boundary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    route_id: str,
    index: int,
    boundary: str,
) -> None:
    exercise(tmp_path, monkeypatch, route_id, index, boundary)


@pytest.mark.parametrize("route_id,index,boundary", boundaries("recover-publication"))
def test_recover_publication_recovery_boundary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    route_id: str,
    index: int,
    boundary: str,
) -> None:
    exercise(tmp_path, monkeypatch, route_id, index, boundary)


@pytest.mark.parametrize("route_id,index,boundary", boundaries("recover-hydrate"))
def test_recover_hydrate_recovery_boundary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    route_id: str,
    index: int,
    boundary: str,
) -> None:
    exercise(tmp_path, monkeypatch, route_id, index, boundary)


@pytest.mark.parametrize(
    "route_id,index,boundary", boundaries("recover-complete-stage")
)
def test_recover_complete_stage_recovery_boundary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    route_id: str,
    index: int,
    boundary: str,
) -> None:
    exercise(tmp_path, monkeypatch, route_id, index, boundary)


@pytest.mark.parametrize("route_id,index,boundary", boundaries("recover-evict"))
def test_recover_evict_recovery_boundary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    route_id: str,
    index: int,
    boundary: str,
) -> None:
    exercise(tmp_path, monkeypatch, route_id, index, boundary)


@pytest.mark.parametrize("route_id,index,boundary", boundaries("recover-retire"))
def test_recover_retire_recovery_boundary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    route_id: str,
    index: int,
    boundary: str,
) -> None:
    exercise(tmp_path, monkeypatch, route_id, index, boundary)


@pytest.mark.parametrize("route_id,index,boundary", boundaries("recover-discard"))
def test_recover_discard_recovery_boundary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    route_id: str,
    index: int,
    boundary: str,
) -> None:
    exercise(tmp_path, monkeypatch, route_id, index, boundary)


@pytest.mark.parametrize("route_id,index,boundary", boundaries("recover-copy"))
def test_recover_copy_recovery_boundary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    route_id: str,
    index: int,
    boundary: str,
) -> None:
    exercise(tmp_path, monkeypatch, route_id, index, boundary)


@pytest.mark.parametrize("route_id,index,boundary", boundaries("recover-checkpoint"))
def test_recover_checkpoint_recovery_boundary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    route_id: str,
    index: int,
    boundary: str,
) -> None:
    exercise(tmp_path, monkeypatch, route_id, index, boundary)


@pytest.mark.parametrize(
    "route_id,index,boundary", boundaries("recover-checkpoint-hydrate")
)
def test_recover_checkpoint_hydrate_recovery_boundary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    route_id: str,
    index: int,
    boundary: str,
) -> None:
    exercise(tmp_path, monkeypatch, route_id, index, boundary)


@pytest.mark.parametrize("route_id,index,boundary", boundaries("mirror-hardlinks"))
def test_mirror_hardlinks_recovery_boundary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    route_id: str,
    index: int,
    boundary: str,
) -> None:
    exercise(tmp_path, monkeypatch, route_id, index, boundary)


@pytest.mark.parametrize("route_id,index,boundary", boundaries("evict-hardlinks"))
def test_evict_hardlinks_recovery_boundary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    route_id: str,
    index: int,
    boundary: str,
) -> None:
    exercise(tmp_path, monkeypatch, route_id, index, boundary)


@pytest.mark.parametrize("route_id,index,boundary", boundaries("checkpoint-hardlinks"))
def test_checkpoint_hardlinks_recovery_boundary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    route_id: str,
    index: int,
    boundary: str,
) -> None:
    exercise(tmp_path, monkeypatch, route_id, index, boundary)


@pytest.mark.parametrize("route_id,index,boundary", boundaries("index-legacy"))
def test_index_legacy_recovery_boundary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    route_id: str,
    index: int,
    boundary: str,
) -> None:
    exercise(tmp_path, monkeypatch, route_id, index, boundary)


@pytest.mark.parametrize("route_id,index,boundary", boundaries("orphan-recovery"))
def test_orphan_recovery_recovery_boundary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    route_id: str,
    index: int,
    boundary: str,
) -> None:
    exercise(tmp_path, monkeypatch, route_id, index, boundary)
