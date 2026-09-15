"""Fixture Court review project shared by the review and inference tests."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from src.tasks.court_detection.data.target_generation.materializer import (
    CourtTargetMaterializer,
)
from src.tasks.court_detection.visualization.review import datasets
from src.tasks.court_detection.visualization.review.datasets import (
    CourtDatasetCatalog,
)
from src.utils.schema.court import STANDARD_COURT_CONFIG, court_keypoints_3d
from tests.unit.tasks.court_detection.data.inputs.test_synthetic_court import (
    _write_v2_dataset,
)
from tests.unit.tasks.court_detection.data.inputs.test_tennis_court_detector import (
    _record,
    _write_source,
)

DENSE_KINDS = ("seg", "line", "semantic_line")
IMAGE_WIDTH = 32
IMAGE_HEIGHT = 24


def _planar_kp14() -> list[list[float]]:
    """Return a non-degenerate KP14 trapezoid scaled to the fixture image.

    The stock TennisCourtDetector fixture keypoints are collinear, which no
    court plane can be fit to, so dense ground-truth fixtures need a real
    projected court instead.
    """
    points = court_keypoints_3d(STANDARD_COURT_CONFIG)[:14, :2]
    image_points = torch.stack(
        (
            (points[:, 0] / 12.0 + 0.5) * float(IMAGE_WIDTH - 1),
            (0.5 - points[:, 1] / 26.0) * float(IMAGE_HEIGHT - 1),
        ),
        dim=1,
    )
    return [[float(x), float(y)] for x, y in image_points.tolist()]


@pytest.fixture
def review_project(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Build a tiny project with one real-schema Tennis and Synthetic dataset.

    The composed TennisCourtDetector preset is swapped for a fixture preset so
    the catalog is exercised through the same typed source contract (including
    its explicit ``excluded_sample_ids``) instead of a real-data-only path.
    """
    data_root = tmp_path / "data"
    _write_source(data_root / "court", _record(kps=_planar_kp14()))
    preset = tmp_path / "tennis_court_detector.yaml"
    preset.write_text(
        "\n".join(
            (
                "kind: tennis_court_detector",
                "root: court",
                "split_mapping:",
                "  train: train",
                "  val: val",
                "  test: null",
                "excluded_sample_ids: []",
                "",
            )
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(datasets, "_TENNIS_SOURCE_PRESET", preset)
    _write_v2_dataset(
        data_root / "synthetic_data_generation" / "scenes",
        schema="v3",
        court_order=("court-a", "court-b"),
        target_court_id="court-b",
    )
    return tmp_path


@pytest.fixture
def catalog(review_project: Path) -> CourtDatasetCatalog:
    data_root = review_project / "data"
    return CourtDatasetCatalog(
        project_root=review_project,
        data_root=data_root,
        checkpoint_root=review_project / "ckpt" / "court_detection",
        output_root=review_project / "outputs" / "court_detection",
    )


@pytest.fixture
def materialized_catalog(catalog: CourtDatasetCatalog) -> CourtDatasetCatalog:
    """Materialize the exact dense targets the canonical builders consume.

    Only the Tennis source is materialized: the synthetic fixture's hand-written
    camera-view UV layout is not a physical projection, so the rasterizer
    correctly refuses it.  Real synthetic dense ground truth (with the
    provenance-verified derived store) is covered by the ``local_data`` smoke
    test instead.
    """
    for entry in catalog.entries():
        if not entry.available or entry.source_kind != "tennis_court_detector":
            continue
        layer = catalog.input_for(entry)
        materializer = CourtTargetMaterializer(
            input_layer=layer,
            target_store=catalog.target_store,
        )
        materializer.materialize(splits=(entry.split,), target_kinds=DENSE_KINDS)
    return catalog


__all__ = ["DENSE_KINDS"]
