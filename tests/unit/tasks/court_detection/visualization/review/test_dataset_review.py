"""Canonical dataset catalog and ground-truth contract for the Court review UI."""

from __future__ import annotations

import base64
import io
import json
import os
from pathlib import Path

import pytest
from PIL import Image

from src.tasks.court_detection.visualization.inference.service import (
    DetectionService,
)
from src.tasks.court_detection.visualization.review.datasets import (
    CourtDatasetCatalog,
)

pytestmark = pytest.mark.unit

REAL_DATA_ROOT = Path("/home/kamimura/projects/tennis-lab/data")


def _build_real_source(service: DetectionService, dataset_id: str) -> int:
    """Read one real split without hiding source-integrity failures."""
    return int(service.datasets.count(dataset_id))


def _service(project: Path) -> DetectionService:
    return DetectionService(project_root=project)


def _entry(catalog: CourtDatasetCatalog, dataset_id: str):
    return catalog.entry(dataset_id)


def test_catalog_lists_tennis_and_synthetic_datasets(
    catalog: CourtDatasetCatalog,
) -> None:
    entries = {entry.id: entry for entry in catalog.entries()}

    assert entries["tennis_court_detector/train"].available is True
    assert entries["tennis_court_detector/train"].count == 1
    assert entries["tennis_court_detector/val"].count == 1
    for split in ("train", "val", "test"):
        entry = entries[f"synthetic_court/B00/{split}"]
        assert entry.available is True
        assert entry.count == 1
        assert entry.published_schema == "canonical_court_dataset_v3"


def test_catalog_count_matches_canonical_records(catalog: CourtDatasetCatalog) -> None:
    """The cheap catalog count must equal the authoritative record count."""
    assert catalog.entries()
    for entry in catalog.entries():
        assert entry.count == len(catalog.records(entry.id))


def test_all_split_file_stamps_survive_catalog_reload(
    catalog: CourtDatasetCatalog,
) -> None:
    for _ in range(2):
        entries = catalog.entries()
        for entry in entries:
            catalog.records(entry.id)
        for entry in entries:
            for record in catalog.records(entry.id):
                catalog.verify_sample_files(entry, record.sample_id)


def test_unknown_published_schema_disables_only_that_scene(
    review_project: Path,
) -> None:
    manifest = (
        review_project
        / "data"
        / "synthetic_data_generation"
        / "scenes"
        / "B00"
        / "datasets"
        / "court"
        / "dataset.json"
    )
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["schema"] = "canonical_court_dataset_v9"
    manifest.write_text(json.dumps(payload), encoding="utf-8")

    catalog = _service(review_project).datasets
    entry = _entry(catalog, "synthetic_court/B00/val")

    assert entry.available is False
    assert entry.reason is not None and "v9" in entry.reason
    # The Tennis dataset is unaffected by the synthetic schema failure.
    assert _entry(catalog, "tennis_court_detector/val").available is True


def test_missing_schema_field_disables_scene_with_reason(review_project: Path) -> None:
    manifest = (
        review_project
        / "data"
        / "synthetic_data_generation"
        / "scenes"
        / "B00"
        / "datasets"
        / "court"
        / "dataset.json"
    )
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload.pop("schema")
    manifest.write_text(json.dumps(payload), encoding="utf-8")

    entry = _entry(_service(review_project).datasets, "synthetic_court/B00/train")

    assert entry.available is False
    assert entry.reason is not None


def test_tennis_quarantine_is_applied_and_stale_exclusions_fail_loudly(
    review_project: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from src.tasks.court_detection.visualization.review import datasets

    preset = datasets.tennis_source_preset_path()
    preset.write_text(
        preset.read_text(encoding="utf-8").replace(
            "excluded_sample_ids: []", "excluded_sample_ids: [sample]"
        ),
        encoding="utf-8",
    )
    catalog = _service(review_project).datasets
    assert _entry(catalog, "tennis_court_detector/train").count == 0

    # A stale exclusion that matches no annotation must stop the source instead
    # of being silently inherited.
    preset.write_text(
        preset.read_text(encoding="utf-8").replace(
            "excluded_sample_ids: [sample]", "excluded_sample_ids: [missing-sample]"
        ),
        encoding="utf-8",
    )
    stale = _service(review_project).datasets
    with pytest.raises(ValueError, match="exactly one annotation record"):
        stale.records("tennis_court_detector/train")


def test_missing_dense_layers_warn_per_layer_and_keep_keypoints(
    catalog: CourtDatasetCatalog,
) -> None:
    service = _service(catalog.project_root)
    scene = service.scenes("tennis_court_detector/train", limit=1)["items"][0]["id"]

    preview = service.preview(scene)
    ground_truth = preview["items"][0]["gt"]

    assert [entry["label"] for entry in ground_truth["points"]][:2] == [
        "far_doubles_left",
        "far_doubles_right",
    ]
    assert ground_truth["rasters"] == []
    assert len(preview["warnings"]) == 3
    assert all(
        "ground truth" in note and "利用できません" in note
        for note in preview["warnings"]
    )


def test_stale_dense_layer_is_reported_not_substituted(
    materialized_catalog: CourtDatasetCatalog,
) -> None:
    service = _service(materialized_catalog.project_root)
    scene = service.scenes("tennis_court_detector/val", limit=1)["items"][0]["id"]
    entry = materialized_catalog.entry("tennis_court_detector/val")
    record = materialized_catalog.records(entry.id)[0]
    seg_path = record.dense_target_refs["seg"]
    Image.new("L", (64, 48), color=3).save(seg_path)

    preview = service.preview(scene)
    names = [raster["name"] for raster in preview["items"][0]["gt"]["rasters"]]

    assert names == ["line", "semantic_line"]
    assert any(
        "seg の ground truth" in note and "利用できません" in note
        for note in preview["warnings"]
    )
    assert any("stale" in note for note in preview["warnings"])


def test_preview_uses_original_pixels_and_source_sized_rasters(
    materialized_catalog: CourtDatasetCatalog,
) -> None:
    service = _service(materialized_catalog.project_root)
    scene = service.scenes("tennis_court_detector/val", limit=1)["items"][0]["id"]
    record = materialized_catalog.records("tennis_court_detector/val")[0]

    preview = service.preview(scene)

    assert (preview["width"], preview["height"]) == (32, 24)
    assert preview["frames"] == 1
    expected = json.loads((Path(record.annotation_path)).read_text(encoding="utf-8"))[
        0
    ]["kps"][0]
    point = preview["items"][0]["gt"]["points"][0]
    assert (point["x"], point["y"]) == (expected[0], expected[1])
    assert [raster["name"] for raster in preview["items"][0]["gt"]["rasters"]] == [
        "seg",
        "line",
        "semantic_line",
    ]
    for raster in preview["items"][0]["gt"]["rasters"]:
        image = Image.open(io.BytesIO(base64.b64decode(raster["data"].split(",")[1])))
        assert image.size == (preview["width"], preview["height"])
        assert image.mode == "RGBA"


def test_preview_and_scenes_reject_path_traversal(
    materialized_catalog: CourtDatasetCatalog,
) -> None:
    service = _service(materialized_catalog.project_root)

    with pytest.raises(ValueError):
        service.scenes("../../etc")
    with pytest.raises(ValueError):
        service.preview("../../etc/passwd")
    # Scene IDs are matched against catalog record IDs, so a traversal-shaped
    # sample can only ever fail to resolve; no path is ever built from it.
    with pytest.raises((ValueError, FileNotFoundError)):
        service.preview("tennis_court_detector/train::../images/sample")
    with pytest.raises(FileNotFoundError):
        service.preview("tennis_court_detector/train::does-not-exist")


def test_preview_rejects_multi_frame_requests(
    materialized_catalog: CourtDatasetCatalog,
) -> None:
    service = _service(materialized_catalog.project_root)
    scene = service.scenes("tennis_court_detector/train", limit=1)["items"][0]["id"]

    with pytest.raises(ValueError, match="count は 1"):
        service.preview(scene, count=4)
    with pytest.raises(ValueError, match="start は 0"):
        service.preview(scene, start=1)


def test_image_is_a_real_jpeg_of_the_source_frame(
    materialized_catalog: CourtDatasetCatalog,
) -> None:
    service = _service(materialized_catalog.project_root)
    scene = service.scenes("tennis_court_detector/train", limit=1)["items"][0]["id"]

    payload = service.image(scene, 0)
    image = Image.open(io.BytesIO(payload))

    assert payload[:2] == b"\xff\xd8"
    assert image.format == "JPEG"
    assert image.size == (32, 24)


def test_source_file_changing_after_catalog_is_refused(
    materialized_catalog: CourtDatasetCatalog,
) -> None:
    service = _service(materialized_catalog.project_root)
    scene = service.scenes("tennis_court_detector/train", limit=1)["items"][0]["id"]
    record = materialized_catalog.records("tennis_court_detector/train")[0]
    service.preview(scene)

    # Rewrite the frame with the same dimensions so only the stat identifies the
    # change; the review must refuse to serve a file it did not stat.
    Image.new("RGB", (32, 24), color=(7, 8, 9)).save(record.image_path)

    with pytest.raises(ValueError, match="disk 上で変化"):
        service.preview(scene)

    service.catalog()
    assert service.preview(scene)["items"]
    assert service.image(scene, 0).startswith(b"\xff\xd8")


def test_explicit_refresh_recovers_after_image_mtime_change(
    catalog: CourtDatasetCatalog,
) -> None:
    service = _service(catalog.project_root)
    scene = service.scenes("tennis_court_detector/train", limit=1)["items"][0]["id"]
    entry, record = service.datasets.sample(scene)
    before = record.image_path.stat()
    os.utime(record.image_path, ns=(before.st_atime_ns, before.st_mtime_ns + 1_000_000))

    with pytest.raises(ValueError, match="disk 上で変化"):
        service.preview(scene)
    service.catalog()
    assert service.scenes(entry.id)["items"][0]["id"] == scene
    assert service.preview(scene)["items"]


@pytest.mark.local_data
@pytest.mark.skipif(
    not (REAL_DATA_ROOT / "court" / "data_val.json").is_file()
    or not (
        REAL_DATA_ROOT
        / "synthetic_data_generation"
        / "scenes"
        / "B00"
        / "datasets"
        / "court"
        / "dataset.json"
    ).is_file(),
    reason="Court datasets are unavailable",
)
def test_real_dataset_smoke_for_both_sources() -> None:
    """Load the real sources once and review one ground-truth sample each."""
    service = _service(Path("/home/kamimura/projects/tennis-lab"))
    catalog = service.datasets

    for dataset_id in ("tennis_court_detector/train", "tennis_court_detector/val"):
        entry = catalog.entry(dataset_id)
        assert entry.available is True
        assert entry.count == _build_real_source(service, dataset_id)
    scenes = service.scenes("tennis_court_detector/val", limit=1)
    assert scenes["total"] > 0
    for dataset_id in ("tennis_court_detector/val", "synthetic_court/B00/val"):
        entry = catalog.entry(dataset_id)
        assert entry.published_schema is not None
        assert entry.count == _build_real_source(service, dataset_id)
        scene = service.scenes(dataset_id, limit=1)["items"][0]["id"]
        preview = service.preview(scene)
        assert preview["warnings"] == []
        assert preview["items"][0]["gt"]["rasters"]
        assert preview["width"] > 0 and preview["height"] > 0
        assert len(service.image(scene, 0)) > 0
