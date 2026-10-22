from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from src.synthetic_data_generation.dataset.court.sample_store import open_court_store
from src.synthetic_data_generation.dataset.court.scale import publication
from src.synthetic_data_generation.dataset.court.scale.core import (
    GeometryRecipe,
    QualityPolicy,
    TrajectoryMethod,
    canonical_bytes,
    digest,
)
from src.synthetic_data_generation.dataset.court.scale.planning import (
    build_candidate_plan,
)
from src.synthetic_data_generation.rendering.nht import (
    NHTRenderArrays,
    NHTRenderCommandRequest,
)


def plan(cameras, layout, n=2, k=0, recipe=None):
    return build_candidate_plan(
        scene_id="fixture",
        method=TrajectoryMethod.STRATIFIED_COUNTER,
        recipe=recipe or GeometryRecipe(width=80, height=48, base_step_m=8),
        seed=24,
        trajectory_count=n,
        density_level=k,
        cameras=cameras,
        layout=layout,
    )


def request(root, candidate, layout, monkeypatch, quality=None):
    monkeypatch.setattr(
        publication.AlignmentResult,
        "from_dict",
        lambda raw: SimpleNamespace(layout=layout),
    )
    return publication.freeze_request(
        root,
        plans=[candidate],
        assets=[{"scene_id": "fixture", "alignment": {"fixture": True}}],
        quality=quality or QualityPolicy(),
        renderer={"schema": "fixture_renderer_v1"},
    )


class Renderer:
    def __init__(self, *, reverse=False, visible=True):
        self.ids = []
        self.reverse, self.visible = reverse, visible

    def __call__(self, scene, records):
        for record in reversed(records) if self.reverse else records:
            self.ids.append(record["sample_id"])
            h, w = record["height"], record["width"]
            value = int(record["sample_id"][:2], 16) / 255
            yield (
                record["sample_id"],
                NHTRenderArrays(
                    rgb=np.full((h, w, 3), value, dtype=np.float32),
                    alpha=np.full(
                        (h, w, 1), 1 if self.visible else 0, dtype=np.float32
                    ),
                    depth=np.ones((h, w, 1), dtype=np.float32),
                ),
            )


def test_nested_versions_share_unique_render_and_strict_jpeg_stores(
    tmp_path, captured_cameras, multi_court_layout, monkeypatch
):
    coarse = plan(captured_cameras, multi_court_layout)
    first_request = request(tmp_path, coarse, multi_court_layout, monkeypatch)
    renderer = Renderer(reverse=True)
    first_path = publication.execute_request(tmp_path, first_request, renderer)
    first = publication.load_dataset_version(first_path, verify_images=True)
    assert first["scenes"][0]["counts"]["rendered_unique"] == len(renderer.ids)
    assert (
        publication.execute_request(tmp_path, first_request, Renderer()) == first_path
    )
    larger = plan(captured_cameras, multi_court_layout, 3, 1)
    second_request = request(tmp_path, larger, multi_court_layout, monkeypatch)
    more = Renderer()
    second_path = publication.execute_request(tmp_path, second_request, more)
    second = publication.load_dataset_version(second_path, verify_images=True)
    assert set(renderer.ids).isdisjoint(more.ids)
    assert second["scenes"][0]["counts"]["render_cache_reused"] == len(renderer.ids)
    assert set(s["path"] for s in first["scenes"][0]["stores"]) <= set(
        s["path"] for s in second["scenes"][0]["stores"]
    )

    def records(version):
        result = {}
        for entry in version["scenes"][0]["stores"]:
            store = open_court_store(tmp_path / entry["path"])
            result.update(
                {
                    store.record(i)["sample_id"]: (store.record(i), store.rgb(i))
                    for i in range(len(store))
                }
            )
        return result

    a, b = records(first), records(second)
    assert a.keys() <= b.keys()
    for sample_id, (record, rgb) in a.items():
        assert record == b[sample_id][0]
        assert np.array_equal(rgb, b[sample_id][1])
        assert record["split"] == "train"
        assert record["qa"]["roi_hole_fraction"] in (None, 0.0)
    # Reversed completion order yields identical sparse records and camera images.
    separate = tmp_path / "separate"
    same_request = request(separate, coarse, multi_court_layout, monkeypatch)
    other = publication.load_dataset_version(
        publication.execute_request(separate, same_request, Renderer())
    )
    assert first["scenes"][0]["stores"] == other["scenes"][0]["stores"]


def test_corrupt_versions_paths_counts_and_memberships_fail(
    tmp_path, captured_cameras, multi_court_layout, monkeypatch
):
    frozen = request(
        tmp_path,
        plan(captured_cameras, multi_court_layout),
        multi_court_layout,
        monkeypatch,
    )
    path = publication.execute_request(tmp_path, frozen, Renderer())
    original = publication.read_json(path)
    for mutate in (
        lambda v: v.update(split="validation"),
        lambda v: v["scenes"][0]["counts"].update(pending=1),
        lambda v: v["scenes"][0]["stores"][0].update(path="../escape"),
        lambda v: v["scenes"][0].update(accepted_ids_sha256="0" * 64),
    ):
        import copy

        value = copy.deepcopy(original)
        mutate(value)
        value["dataset_version"] = digest(
            {k: v for k, v in value.items() if k != "dataset_version"}
        )
        invalid = tmp_path / "invalid.dataset.json"
        invalid.write_bytes(canonical_bytes(value))
        with pytest.raises(ValueError):
            publication.load_dataset_version(invalid)


def test_no_quota_refill_and_renderer_failure_is_not_qa_rejection(
    tmp_path, captured_cameras, multi_court_layout, monkeypatch
):
    candidate = plan(captured_cameras, multi_court_layout)
    empty_root = tmp_path / "empty"
    frozen = request(empty_root, candidate, multi_court_layout, monkeypatch)
    renderer = Renderer(visible=False)
    with pytest.raises(ValueError, match="Empty accepted condition"):
        publication.execute_request(empty_root, frozen, renderer)
    eligible = [
        s
        for g in candidate["trajectory_groups"]
        for s in g["samples"]
        if not s["reasons"]
    ]
    assert set(renderer.ids) == {s["sample_id"] for s in eligible}
    assert not list(empty_root.glob("*.dataset.json"))
    failed_root = tmp_path / "failed"
    frozen = request(failed_root, candidate, multi_court_layout, monkeypatch)

    def fail(scene, records):
        raise RuntimeError("renderer crashed")

    with pytest.raises(RuntimeError, match="crashed"):
        publication.execute_request(failed_root, frozen, fail)
    failures = list(failed_root.glob(".staging/*/failure.json"))
    assert failures and publication.read_json(failures[0])["pending_ids"]
    assert not list(failed_root.glob("*.dataset.json"))


def test_qa_does_not_change_point_geometry_and_jitter_is_a_paired_comparison(
    captured_cameras, multi_court_layout
):
    recipe = GeometryRecipe()
    a = plan(captured_cameras, multi_court_layout, recipe=recipe)
    b = plan(
        captured_cameras, multi_court_layout, recipe=replace(recipe, jitter_radius_m=0)
    )
    for g, h in zip(a["trajectory_groups"], b["trajectory_groups"], strict=True):
        assert g["curve"] == h["curve"]
        assert g["trajectory_id"] == h["trajectory_id"]
        for p, q in zip(g["samples"], h["samples"], strict=True):
            assert p["scale"]["comparison_pair_id"] == q["scale"]["comparison_pair_id"]
            assert p["camera"]["intrinsics"] == q["camera"]["intrinsics"]
            assert p["sample_id"] != q["sample_id"]


def test_render_microbatch_is_explicit_without_changing_legacy_argv(tmp_path):
    export = tmp_path / "source/export"
    export.mkdir(parents=True)
    scene = export / "scene.json"
    scene.write_text("{}")
    legacy = NHTRenderCommandRequest(
        scene, tmp_path / "out", observed_camera_ids=("camera-0",)
    )
    assert "--batch-size" not in legacy.argv()
    assert replace(legacy, batch_size=1).argv()[-2:] == ("--batch-size", "1")
    with pytest.raises(ValueError, match="batch_size"):
        replace(legacy, batch_size=0)
