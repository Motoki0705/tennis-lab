"""DatasetCatalog must auto-detect forms, guard paths, and hash revisions."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from src.tasks.base.generate_dataset import (
    CAMERA_VIEW_V2_SELECTOR,
    PHYSICAL_V1_SELECTOR,
)
from src.tasks.base.visualization.review.catalog import (
    DatasetCatalog,
    DatasetCatalogError,
)

DATA_ROOT = Path("/home/kamimura/projects/tennis-lab/data")
EXPECTED_FORMS = {
    "single_object",
    "multi_object",
    "single_object_broadcast",
    "multi_object_broadcast",
    "single_object_camera_view_v2",
    "multi_object_camera_view_v2",
}


def _write_form(
    root: Path,
    task: str,
    name: str,
    *,
    mode: str,
    selector: str,
    scenes: tuple[str, ...],
    samples: bool = False,
) -> Path:
    form = root / task / name
    scenes_dir = form / "scenes"
    scenes_dir.mkdir(parents=True)
    (form / "meta.json").write_text(
        json.dumps(
            {
                "config": {
                    "generation": {"mode": mode},
                    "court_keypoints": {"selector": selector},
                }
            }
        ),
        encoding="utf-8",
    )
    for scene in scenes:
        scene_dir = scenes_dir / scene
        scene_dir.mkdir()
        (scene_dir / "meta.json").write_text("{}", encoding="utf-8")
        (scene_dir / "scalars.json").write_text("{}", encoding="utf-8")
    if samples:
        (form / "samples").mkdir()
    return form


def test_discovers_forms_scenes_and_mode(tmp_path: Path) -> None:
    _write_form(tmp_path, "blcs", "single_object", mode="single_object", selector="physical_v1", scenes=("scene_000001", "scene_000000"), samples=True)
    _write_form(tmp_path, "blcs", "multi_object", mode="multi_object", selector="physical_v1", scenes=("scene_000000",))
    catalog = DatasetCatalog(tmp_path, "blcs")
    forms = {form.name: form for form in catalog.forms()}
    assert set(forms) == {"single_object", "multi_object"}
    assert forms["single_object"].mode == "single"
    assert forms["multi_object"].mode == "multi"
    assert forms["single_object"].scene_count == 2
    assert forms["single_object"].has_samples is True
    assert forms["multi_object"].has_samples is False
    # Scene names come back sorted and are directory names only.
    assert catalog.scenes("single_object") == ("scene_000000", "scene_000001")


def test_revision_is_stable_then_changes_on_touch(tmp_path: Path) -> None:
    form = _write_form(tmp_path, "blcs", "single_object", mode="single_object", selector="physical_v1", scenes=("scene_000000",))
    catalog = DatasetCatalog(tmp_path, "blcs")
    first = catalog.revision("single_object", "scene_000000")
    assert len(first) == 20
    assert catalog.revision("single_object", "scene_000000") == first
    scalars = form / "scenes" / "scene_000000" / "scalars.json"
    os.utime(scalars, None)
    stat = scalars.stat()
    os.utime(scalars, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000))
    assert catalog.revision("single_object", "scene_000000") != first


def test_revision_changes_when_array_changes(tmp_path: Path) -> None:
    form = _write_form(tmp_path, "blcs", "single_object", mode="single_object",
                       selector="physical_v1", scenes=("scene_000000",))
    array = form / "scenes" / "scene_000000" / "position.npy"
    array.write_bytes(b"array-one")
    catalog = DatasetCatalog(tmp_path, "blcs")
    first = catalog.revision("single_object", "scene_000000")
    stat = array.stat()
    os.utime(array, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000))
    assert catalog.revision("single_object", "scene_000000") != first


def test_forms_filter_and_unknown_form_are_explicit(tmp_path: Path) -> None:
    _write_form(tmp_path, "plcs", "single_object", mode="single_object", selector="physical_v1", scenes=("scene_000000",))
    _write_form(tmp_path, "plcs", "multi_object", mode="multi_object", selector="physical_v1", scenes=("scene_000000",))
    assert [form.name for form in DatasetCatalog(tmp_path, "plcs", forms=["multi_object"]).forms()] == ["multi_object"]
    with pytest.raises(DatasetCatalogError):
        DatasetCatalog(tmp_path, "plcs", forms=["missing_form"])


def test_contract_selector_is_resolved_per_form(tmp_path: Path) -> None:
    _write_form(tmp_path, "blcs", "single_object", mode="single_object", selector="physical_v1", scenes=("scene_000000",))
    _write_form(tmp_path, "blcs", "single_object_camera_view_v2", mode="single_object", selector="camera_view_v2", scenes=("scene_000000",))
    catalog = DatasetCatalog(tmp_path, "blcs")
    assert catalog.court_contract("single_object").selector == PHYSICAL_V1_SELECTOR
    assert (
        catalog.court_contract("single_object_camera_view_v2").selector
        == CAMERA_VIEW_V2_SELECTOR
    )


@pytest.mark.parametrize("scene", ["../secret", "a/b", "..", "."])
def test_scene_paths_cannot_escape(tmp_path: Path, scene: str) -> None:
    _write_form(tmp_path, "blcs", "single_object", mode="single_object", selector="physical_v1", scenes=("scene_000000",))
    catalog = DatasetCatalog(tmp_path, "blcs")
    with pytest.raises(DatasetCatalogError):
        catalog.scene_path("single_object", scene)


def test_form_paths_cannot_escape(tmp_path: Path) -> None:
    _write_form(tmp_path, "blcs", "single_object", mode="single_object", selector="physical_v1", scenes=("scene_000000",))
    catalog = DatasetCatalog(tmp_path, "blcs")
    for escape in ("../single_object", "a/b", "..", "."):
        with pytest.raises(DatasetCatalogError):
            catalog.scenes(escape)


@pytest.mark.skipif(not (DATA_ROOT / "blcs").is_dir(), reason="BLCS data missing")
def test_real_blcs_forms_are_detected() -> None:
    catalog = DatasetCatalog(DATA_ROOT, "blcs")
    forms = {form.name: form for form in catalog.forms()}
    assert set(forms) == EXPECTED_FORMS
    assert forms["single_object"].scene_count == 1000
    assert forms["single_object"].has_samples is True
    assert forms["single_object_camera_view_v2"].has_samples is False
