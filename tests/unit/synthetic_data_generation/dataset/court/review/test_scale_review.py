from types import SimpleNamespace

import numpy as np
import pytest
from fastapi.testclient import TestClient

from src.synthetic_data_generation.dataset.court.review.scale_service import (
    ScaleReviewService,
)
from src.synthetic_data_generation.dataset.court.review.web import create_app
from src.synthetic_data_generation.dataset.court.scale import publication
from src.synthetic_data_generation.dataset.court.scale.core import (
    GeometryRecipe,
    QualityPolicy,
    TrajectoryMethod,
)
from src.synthetic_data_generation.dataset.court.scale.planning import (
    build_candidate_plan,
)
from src.synthetic_data_generation.rendering.nht import NHTRenderArrays


@pytest.fixture
def versions(tmp_path, captured_cameras, multi_court_layout, monkeypatch):
    monkeypatch.setattr(
        publication.AlignmentResult,
        "from_dict",
        lambda _: SimpleNamespace(layout=multi_court_layout),
    )
    paths = []
    for method in (TrajectoryMethod.STRATIFIED_COUNTER, TrajectoryMethod.SOBOL):
        plan = build_candidate_plan(
            scene_id="B00",
            method=method,
            recipe=GeometryRecipe(width=80, height=48, base_step_m=8),
            seed=24,
            trajectory_count=3,
            density_level=0,
            cameras=captured_cameras,
            layout=multi_court_layout,
        )
        rejected_group = plan["trajectory_groups"][0]["trajectory_id"]

        def render(scene, records, rejected_group=rejected_group):
            for r in records:
                shape = (r["height"], r["width"], 1)
                yield (
                    r["sample_id"],
                    NHTRenderArrays(
                        rgb=np.full((*shape[:2], 3), 0.5, dtype=np.float32),
                        alpha=np.full(
                            shape,
                            float(r["trajectory_id"] != rejected_group),
                            dtype=np.float32,
                        ),
                        depth=np.ones(shape, dtype=np.float32),
                    ),
                )

        request = publication.freeze_request(
            tmp_path,
            plans=[plan],
            assets=[{"scene_id": "B00", "alignment": {}}],
            quality=QualityPolicy(),
            renderer={"schema": "fixture_renderer_v1"},
        )
        paths.append(publication.execute_request(tmp_path, request, render))
    return paths


def test_catalog_preserves_real_scene_and_selects_method_version(versions):
    service = ScaleReviewService(versions)
    scenes = service.scenes()
    assert len(scenes) == 2
    assert {s["scene_id"] for s in scenes} == {"B00"}
    assert len({s["id"] for s in scenes}) == 2
    assert {s["method"] for s in scenes} == {"stratified_counter", "sobol"}
    with TestClient(create_app(service)) as client:
        assert len(client.get("/api/catalog").json()) == 2
        for selected in scenes:
            key, revision = selected["id"], selected["revision"]
            summary = client.get(f"/api/scenes/{key}", params={"revision": revision})
            assert summary.status_code == 200, summary.text
            data = summary.json()
            assert data["scene_id"] == "B00"
            assert data["scale"]["method"] == selected["method"]
            empty = data["groups"][0]
            assert empty["samples"] == []
            assert len(empty["planned_points"]) == 257
            assert empty["rejected_points"]
            group = next(g for g in data["groups"] if g["samples"])
            sample = group["samples"][0]
            detail = client.get(
                f"/api/scenes/{key}/samples/{sample['id']}",
                params={"revision": revision},
            )
            assert detail.status_code == 200
            assert detail.json()["qa"] == sample["qa"]
            assert (
                detail.json()["target_binding"]["binding"]["court_instance_id"]
                == sample["target_court"]
            )
            image = client.get(
                f"/api/scenes/{key}/images/{sample['id']}",
                params={"revision": revision, "label_scope": "target"},
            )
            assert image.status_code == 200, image.text
            rejected = data["rejections"]["samples"][0]
            rejection = client.get(
                f"/api/scenes/{key}/rejections/{rejected['id']}",
                params={"revision": revision},
            )
            assert rejection.json()["image_state"] == "not_retained"
            assert (
                client.get(
                    f"/api/scenes/{key}/images/{rejected['id']}",
                    params={"revision": revision},
                ).status_code
                == 404
            )
        assert (
            client.get("/api/scenes/unknown", params={"revision": "x"}).status_code
            == 422
        )


def test_pinned_revision_changes_refuse_stale_scene_and_sample(versions):
    service = ScaleReviewService(versions)
    selected = service.scenes()[0]
    entry = service.entries[selected["id"]]
    path = entry["root"] / entry["scene"]["asset_manifest_path"]
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(RuntimeError, match="changed"):
        service.load(selected["id"], selected["revision"])
    with pytest.raises(ValueError, match="hash mismatch"):
        ScaleReviewService(versions)


def test_duplicate_explicit_version_is_rejected(versions):
    with pytest.raises(ValueError, match="Duplicate"):
        ScaleReviewService([versions[0], versions[0]])
