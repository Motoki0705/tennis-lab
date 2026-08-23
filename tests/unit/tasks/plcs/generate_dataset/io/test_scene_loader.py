"""Boundary tests for required PLCS scene scalar metadata."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.tasks.base.data import (
    COURT_COORDINATE_NORMALIZATION_METADATA_KEY,
    CourtCoordinateContractMismatchError,
    CourtCoordinateNormalizationMetadata,
    MissingCourtCoordinateMetadataError,
    MixedCourtCoordinateMetadataError,
)
from src.tasks.plcs.generate_dataset.io.scene_loader import load_scene
from src.utils.schema.court_normalization import resolve_court_coordinate_normalization

V1_NORMALIZATION = resolve_court_coordinate_normalization("v1")


def _metadata_document(version: str | None) -> dict[str, object]:
    document: dict[str, object] = {"scene_id": "scene_000001", "fps": 30.0}
    if version is not None:
        document[COURT_COORDINATE_NORMALIZATION_METADATA_KEY] = (
            CourtCoordinateNormalizationMetadata.from_contract(
                resolve_court_coordinate_normalization(version)
            ).to_dict()
        )
    return document


def test_scene_loader_requires_explicit_num_persons(tmp_path: Path) -> None:
    (tmp_path / "meta.json").write_text(json.dumps({"fps": 30.0}))
    (tmp_path / "scalars.json").write_text(json.dumps({"num_cameras": 0}))
    np.save(tmp_path / "position.npy", np.zeros((1, 3), dtype=np.float32))
    np.save(tmp_path / "rotation.npy", np.zeros((1, 2), dtype=np.float32))
    np.save(
        tmp_path / "canonical_pose_3d.npy",
        np.zeros((1, 17, 3), dtype=np.float32),
    )

    with pytest.raises(KeyError, match="num_persons"):
        load_scene(tmp_path, court_coordinate_normalization=V1_NORMALIZATION)


def test_scene_loader_rejects_legacy_visible_filenames(tmp_path: Path) -> None:
    (tmp_path / "meta.json").write_text(json.dumps({"fps": 30.0}))
    (tmp_path / "scalars.json").write_text(
        json.dumps({"num_cameras": 1, "num_persons": 1, "cam_0_params": {}})
    )
    for name, shape in {
        "position": (1, 3),
        "rotation": (1, 2),
        "canonical_pose_3d": (1, 17, 3),
        "cam_0_human_kp_uv": (1, 17, 2),
        "cam_0_human_kp_visible": (1, 17),
        "cam_0_human_visibility_ratio": (),
        "cam_0_court_kp_uv": (1, 20, 2),
        "cam_0_court_kp_visible": (1, 20),
        "cam_0_court_visibility_count": (),
    }.items():
        np.save(tmp_path / f"{name}.npy", np.zeros(shape, dtype=np.float32))

    with pytest.raises(FileNotFoundError, match="human_kp_vis"):
        load_scene(tmp_path, court_coordinate_normalization=V1_NORMALIZATION)


@pytest.mark.parametrize(
    ("root_version", "scene_version", "runtime_version", "error", "message"),
    [
        (None, None, "v2", MissingCourtCoordinateMetadataError, "legacy v1 only"),
        ("v1", None, "v1", MixedCourtCoordinateMetadataError, "mixed"),
        (None, "v1", "v1", MixedCourtCoordinateMetadataError, "mixed"),
        (
            "v2",
            "v2",
            "v1",
            CourtCoordinateContractMismatchError,
            "does not match runtime",
        ),
    ],
)
def test_public_scene_loader_rejects_root_scene_contract_before_array_payload(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    root_version: str | None,
    scene_version: str | None,
    runtime_version: str,
    error: type[Exception],
    message: str,
) -> None:
    scene = tmp_path / "scenes" / "scene_000001"
    scene.mkdir(parents=True)
    if root_version is not None:
        (tmp_path / "meta.json").write_text(
            json.dumps(_metadata_document(root_version)),
            encoding="utf-8",
        )
    (scene / "meta.json").write_text(
        json.dumps(_metadata_document(scene_version)),
        encoding="utf-8",
    )

    def _forbid_payload(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise AssertionError("PLCS array payload was read before metadata validation.")

    monkeypatch.setattr(np, "load", _forbid_payload)
    with pytest.raises(error, match=message):
        load_scene(
            scene,
            court_coordinate_normalization=resolve_court_coordinate_normalization(
                runtime_version
            ),
        )
