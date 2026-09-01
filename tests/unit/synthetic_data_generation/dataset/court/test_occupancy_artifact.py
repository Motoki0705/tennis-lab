"""Tests for the exact Court V4 residual-occupancy artifact contract."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

import src.synthetic_data_generation.dataset.court.occupancy_artifact as artifact_module
from src.synthetic_data_generation.dataset.court.occupancy_artifact import (
    COURT_V4_SUPPORT_OCCUPANCY_CELLS_PATH,
    COURT_V4_SUPPORT_OCCUPANCY_METADATA_PATH,
    build_court_v4_support_occupancy_snapshot,
    load_court_v4_support_occupancy,
    occupancy_cells_content_digest,
    write_court_v4_support_occupancy,
)

_SUPPORT_DIGEST = "1" * 64


def _snapshot(order: tuple[int, ...] = (0, 1, 2)):
    cells = np.asarray(
        [
            (2, -1, 3),
            (-2, 4, 1),
            (2, -1, 2),
        ],
        dtype=np.int64,
    )
    return build_court_v4_support_occupancy_snapshot(
        cells[np.asarray(order)],
        voxel_size_m=0.5,
        support_input_digest=_SUPPORT_DIGEST,
        policy_decision_id="b00_court_support_pilot_v1",
    )


def test_snapshot_is_immutable_sorted_and_order_independent() -> None:
    first = _snapshot((0, 1, 2))
    second = _snapshot((2, 0, 1))

    assert first.cells.tolist() == [[-2, 4, 1], [2, -1, 2], [2, -1, 3]]
    assert first.content_digest == second.content_digest
    assert not first.cells.flags.writeable
    with pytest.raises(ValueError, match="read-only"):
        first.cells[0, 0] = 9
    with pytest.raises(ValueError, match="cannot set WRITEABLE flag"):
        first.cells.setflags(write=True)
    assert first.identity.to_dict()["content_digest"] == first.content_digest


def test_snapshot_rejects_duplicate_cells() -> None:
    with pytest.raises(ValueError, match="unique and lexicographically sorted"):
        build_court_v4_support_occupancy_snapshot(
            np.asarray([(0, 0, 0), (0, 0, 0)], dtype=np.int64),
            voxel_size_m=0.5,
            support_input_digest=_SUPPORT_DIGEST,
            policy_decision_id="policy-v1",
        )


def test_artifact_round_trip_binds_every_authority_field(tmp_path: Path) -> None:
    diagnostics = tmp_path / "diagnostics"
    diagnostics.mkdir()
    snapshot = _snapshot()

    written = write_court_v4_support_occupancy(
        diagnostics,
        snapshot=snapshot,
        scene_id="B00",
        profile="b00",
    )
    loaded = load_court_v4_support_occupancy(
        tmp_path,
        expected_scene_id="B00",
        expected_profile="b00",
        expected_policy_decision_id="b00_court_support_pilot_v1",
        expected_support_input_digest=_SUPPORT_DIGEST,
        expected_voxel_size_m=0.5,
        expected_cell_count=3,
        expected_content_digest=snapshot.content_digest,
        maximum_cells=3,
    )

    assert written.metadata() == loaded.metadata()
    assert np.array_equal(loaded.snapshot.cells, snapshot.cells)
    metadata = json.loads(
        (tmp_path / COURT_V4_SUPPORT_OCCUPANCY_METADATA_PATH).read_text(
            encoding="utf-8"
        )
    )
    assert metadata["cells_shape"] == [3, 3]
    assert metadata["cells_dtype"] == "little_endian_int64"
    assert metadata["coordinate_space"] == "metric_scene_metres"
    assert loaded.snapshot.identity == snapshot.identity

    with pytest.raises(ValueError, match="content_digest binding disagrees"):
        load_court_v4_support_occupancy(
            tmp_path,
            expected_content_digest="0" * 64,
        )


def test_artifact_fails_closed_on_cap_coordinate_and_content_tampering(
    tmp_path: Path,
) -> None:
    diagnostics = tmp_path / "diagnostics"
    diagnostics.mkdir()
    write_court_v4_support_occupancy(
        diagnostics,
        snapshot=_snapshot(),
        scene_id="B00",
        profile="b00",
    )

    with pytest.raises(ValueError, match="exceeds configured maximum_cells"):
        load_court_v4_support_occupancy(tmp_path, maximum_cells=2)

    metadata_path = tmp_path / COURT_V4_SUPPORT_OCCUPANCY_METADATA_PATH
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["coordinate_space"] = "nht_scene_units"
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(ValueError, match="Unknown.*coordinate_space"):
        load_court_v4_support_occupancy(tmp_path)

    metadata["coordinate_space"] = "metric_scene_metres"
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    cells_path = tmp_path / COURT_V4_SUPPORT_OCCUPANCY_CELLS_PATH
    cells = np.load(cells_path, allow_pickle=False)
    cells[0, 0] -= 1
    np.save(cells_path, cells, allow_pickle=False)
    with pytest.raises(ValueError, match="content_digest disagrees"):
        load_court_v4_support_occupancy(tmp_path)


def test_artifact_rejects_unsorted_public_numeric_payload(tmp_path: Path) -> None:
    diagnostics = tmp_path / "diagnostics"
    diagnostics.mkdir()
    write_court_v4_support_occupancy(
        diagnostics,
        snapshot=_snapshot(),
        scene_id="B00",
        profile="b00",
    )
    cells_path = tmp_path / COURT_V4_SUPPORT_OCCUPANCY_CELLS_PATH
    cells = np.load(cells_path, allow_pickle=False)
    np.save(cells_path, cells[::-1], allow_pickle=False)

    with pytest.raises(ValueError, match="lexicographically sorted"):
        load_court_v4_support_occupancy(tmp_path)


def test_artifact_failure_removes_only_invocation_owned_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    diagnostics = tmp_path / "diagnostics"
    diagnostics.mkdir()
    metadata_path = tmp_path / COURT_V4_SUPPORT_OCCUPANCY_METADATA_PATH

    def fail_after_late_racer(
        path: Path,
        payload: object,
    ) -> artifact_module._OwnedPublishedFile:
        del payload
        path.write_bytes(b"unrelated racer metadata")
        raise RuntimeError("injected metadata publication failure")

    monkeypatch.setattr(artifact_module, "_write_json_atomic", fail_after_late_racer)

    with pytest.raises(RuntimeError, match="injected metadata"):
        write_court_v4_support_occupancy(
            diagnostics,
            snapshot=_snapshot(),
            scene_id="B00",
            profile="b00",
        )

    assert metadata_path.read_bytes() == b"unrelated racer metadata"
    assert not (tmp_path / COURT_V4_SUPPORT_OCCUPANCY_CELLS_PATH).exists()


def test_content_digest_hashes_exact_immutable_payload_without_rewriting() -> None:
    snapshot = _snapshot()

    assert occupancy_cells_content_digest(snapshot.cells) == snapshot.content_digest
