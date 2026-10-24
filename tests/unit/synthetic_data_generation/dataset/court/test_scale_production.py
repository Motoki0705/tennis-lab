import copy
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from src.synthetic_data_generation.dataset.court.sample_store import open_court_store
from src.synthetic_data_generation.dataset.court.scale import publication, quality
from src.synthetic_data_generation.dataset.court.scale.core import (
    D5_RULE_VERSION,
    QualityPolicy,
    canonical_bytes,
    digest,
)
from src.synthetic_data_generation.dataset.court.scale.planning import (
    build_candidate_plan,
)
from src.synthetic_data_generation.dataset.court.scale.recipes import ProductionRecipe
from src.synthetic_data_generation.rendering.nht import NHTRenderArrays
from src.synthetic_data_generation.scripts import generate_court_scale
from src.utils.data.image_record_store import file_sha256

RECIPE_PATH = Path(
    "src/synthetic_data_generation/configs/dataset/court/"
    "scale_stratified_counter_production_v1.json"
)


@pytest.fixture
def production():
    return ProductionRecipe.from_dict(publication.read_json(RECIPE_PATH))


def test_production_recipe_is_complete_and_keeps_all_values_in_its_hash(production):
    raw = production.to_dict()
    assert raw == publication.read_json(RECIPE_PATH)
    assert raw["method"] == "stratified_counter"
    assert raw["geometry"]["hull_expansion_percent"] == 0.0
    assert raw["geometry"]["hull_margin_m"] == 0.5
    assert raw["geometry"]["hfov_degrees"] == [75.0, 92.5, 110.0]
    assert raw["geometry"]["jitter_radius_m"] == 1.0
    assert raw["jitter"] == "stable_point_uniform_area_disk_v1"
    assert raw["quality"]["rule_version"] == D5_RULE_VERSION
    # Resolution/arc precision are also part of the recipe, not just D2/D5/D6.
    changed = replace(production, geometry=replace(production.geometry, width=80))
    assert digest(changed.to_dict()) != digest(raw)
    assert digest(raw) != digest(raw["geometry"])


@pytest.mark.parametrize(
    "section,key,value",
    [
        (None, "method", "sobol"),
        (None, "name", "stratified_counter_production_v2"),
        (None, "jitter", "index_randomness"),
        ("geometry", "hull_expansion_percent", 5.0),
        ("geometry", "hull_margin_m", 0.0),
        ("geometry", "hfov_degrees", [60, 90, 110]),
        ("geometry", "jitter_radius_m", 0.0),
        ("quality", "minimum_target_visible_kp", 3),
        ("quality", "maximum_roi_hole_fraction", 0.02),
        ("quality", "rule_version", "unknown_rule"),
        ("quality", "schema", "court_scale_quality_policy_v1"),
    ],
)
def test_production_recipe_rejects_policy_changes(production, section, key, value):
    raw = production.to_dict()
    (raw if section is None else raw[section])[key] = value
    with pytest.raises(ValueError):
        ProductionRecipe.from_dict(raw)


@pytest.mark.parametrize(
    "section,key",
    [(None, "method"), ("geometry", "width"), ("quality", "rule_version")],
)
def test_production_recipe_has_no_missing_field_defaults(production, section, key):
    raw = production.to_dict()
    del (raw if section is None else raw[section])[key]
    with pytest.raises(ValueError, match="requires|keys"):
        ProductionRecipe.from_dict(raw)


@pytest.mark.parametrize(
    "kp,holes,reasons",
    [
        (0, 0.0, ["target_visible_kp_below_explicit_threshold"]),
        (3, 0.0, ["target_visible_kp_below_explicit_threshold"]),
        (4, 0.0, []),
        (4, 0.01, []),
        (4, np.nextafter(0.01, 1.0), ["roi_holes_above_explicit_threshold_or_no_roi"]),
        (4, None, ["roi_holes_above_explicit_threshold_or_no_roi"]),
        (
            0,
            None,
            [
                "target_visible_kp_below_explicit_threshold",
                "roi_holes_above_explicit_threshold_or_no_roi",
            ],
        ),
    ],
)
def test_d5_boundaries_ignore_visible_points_from_other_courts(
    production, kp, holes, reasons
):
    qa = {
        "all_visible_kp": 14,
        "target_visible_kp": kp,
        "roi_hole_fraction": holes,
        "roi_alpha_lt_0_9": 1.0,
    }
    assert quality.quality_rejection_reasons(qa, production.quality) == reasons


def candidate(production, cameras, layout):
    return build_candidate_plan(
        scene_id="fixture",
        method=production.method,
        recipe=production.geometry,
        seed=24,
        trajectory_count=3,
        density_level=0,
        cameras=cameras,
        layout=layout,
    )


def test_empty_roi_is_rejected_after_renderer_visibility(
    production, captured_cameras, multi_court_layout, monkeypatch
):
    small = replace(
        production,
        geometry=replace(production.geometry, width=80, height=48, base_step_m=8),
    )
    plan = candidate(small, captured_cameras, multi_court_layout)
    record = next(
        s for g in plan["trajectory_groups"] for s in g["samples"] if not s["reasons"]
    )
    h, w = record["height"], record["width"]
    monkeypatch.setattr(quality, "roi_mask", lambda *_: np.zeros((h, w), dtype=bool))
    evaluated = quality.evaluate_render(
        record,
        NHTRenderArrays(
            rgb=np.ones((h, w, 3), dtype=np.float32),
            alpha=np.ones((h, w, 1), dtype=np.float32),
            depth=np.ones((h, w, 1), dtype=np.float32),
        ),
        layout=multi_court_layout,
        policy=production.quality,
        near_plane_m=production.geometry.near_plane_m,
    )
    assert evaluated["qa"]["roi_pixels"] == 0
    assert evaluated["qa"]["roi_hole_fraction"] is None
    assert "roi_holes_above_explicit_threshold_or_no_roi" in evaluated["reasons"]


def test_production_qa_membership_and_versioned_cache_cannot_reuse_legacy_qa(
    production, tmp_path, captured_cameras, multi_court_layout, monkeypatch
):
    monkeypatch.setattr(
        publication.AlignmentResult,
        "from_dict",
        lambda _: SimpleNamespace(layout=multi_court_layout),
    )
    production = replace(
        production,
        geometry=replace(production.geometry, width=80, height=48, base_step_m=8),
    )
    plan = candidate(production, captured_cameras, multi_court_layout)
    assets = [{"scene_id": "fixture", "alignment": {}}]
    calls = []

    def render(scene, records):
        for r in records:
            calls.append(r["sample_id"])
            h, w = r["height"], r["width"]
            # Low alpha is diagnostic only; a wholly empty render must fail D5.
            alpha = 0.0 if len(calls) == 1 else 0.8
            yield (
                r["sample_id"],
                NHTRenderArrays(
                    rgb=np.full((h, w, 3), 0.5, dtype=np.float32),
                    alpha=np.full((h, w, 1), alpha, dtype=np.float32),
                    depth=np.ones((h, w, 1), dtype=np.float32),
                ),
            )

    def freeze(policy, recipe=None):
        return publication.freeze_request(
            tmp_path,
            plans=[plan],
            assets=assets,
            renderer={"schema": "fixture_renderer_v1"},
            quality=policy,
            production_recipe=recipe,
        )

    legacy_path = publication.execute_request(tmp_path, freeze(QualityPolicy()), render)
    calls.clear()
    frozen = freeze(production.quality, production)
    published = publication.execute_request(tmp_path, frozen, render)
    version = publication.load_dataset_version(published, verify_images=True)
    assert version["schema"] == publication.PRODUCTION_VERSION_SCHEMA
    assert version["recipe_sha256"] == digest(production.to_dict())
    assert version["quality_policy_sha256"] == digest(production.quality.to_dict())
    assert version["quality_rule_version"] == D5_RULE_VERSION
    scene = version["scenes"][0]
    counts = scene["counts"]
    assert counts["render_cache_reused"] == 0
    assert counts["geometry_pass"] == len(calls)
    assert counts["qa_rejected"] >= 1
    assert counts["qa_accepted"] + counts["qa_rejected"] == len(calls)
    accepted = publication.read_json(tmp_path / scene["accepted_ids_path"])[
        "sample_ids"
    ]
    assert calls[0] not in accepted
    for entry in scene["stores"]:
        store = open_court_store(tmp_path / entry["path"])
        for i in range(len(store)):
            qa = store.record(i)["qa"]
            assert qa["target_visible_kp"] >= 4
            assert qa["roi_hole_fraction"] <= 0.01
            assert qa["roi_alpha_lt_0_9"] == 1.0
            assert qa["rule_version"] == D5_RULE_VERSION
    assert publication.execute_request(tmp_path, frozen, render) == published
    assert len(calls) == counts["geometry_pass"]
    legacy = publication.load_dataset_version(legacy_path)
    # Pinning a new policy to an old successful QA ledger cannot certify it.
    forged = copy.deepcopy(version)
    forged["scenes"] = legacy["scenes"]
    forged["dataset_version"] = digest(
        {k: v for k, v in forged.items() if k != "dataset_version"}
    )
    invalid = tmp_path / "legacy-qa-as-production.dataset.json"
    invalid.write_bytes(canonical_bytes(forged))
    with pytest.raises(ValueError, match="quality|QA"):
        publication.load_dataset_version(invalid)
    completion = tmp_path / scene["render_caches"][0]["path"]
    damaged = publication.read_json(completion)
    damaged["quality_rule_version"] = "unrecognized_rule"
    completion.write_bytes(canonical_bytes(damaged))
    scene["render_caches"][0]["sha256"] = file_sha256(completion)
    version["dataset_version"] = digest(
        {k: v for k, v in version.items() if k != "dataset_version"}
    )
    invalid.write_bytes(canonical_bytes(version))
    with pytest.raises(ValueError, match="renderer/QA version"):
        publication.load_dataset_version(invalid)


def test_production_cli_requires_explicit_selection_and_rejects_comparison_overrides(
    production, tmp_path, monkeypatch
):
    (tmp_path / "scenes").mkdir()
    comparison = tmp_path / "comparison.json"
    comparison.write_text("{}")
    common = [
        "generate_court_scale",
        "plan",
        "--scenes-root",
        str(tmp_path / "scenes"),
        "--scenes",
        "B00",
        "--output-root",
        str(tmp_path / "out"),
        "--generation-seed",
        "42",
        "--trajectory-count",
        "6",
        "--density-level",
        "0",
    ]
    monkeypatch.setattr(sys, "argv", common)
    with pytest.raises(SystemExit):
        generate_court_scale.main()
    monkeypatch.setattr(
        sys,
        "argv",
        [*common, "--production-recipe", str(RECIPE_PATH), "--methods", "sobol"],
    )
    with pytest.raises(SystemExit):
        generate_court_scale.main()
    monkeypatch.setattr(
        sys,
        "argv",
        [
            *common,
            "--production-recipe",
            str(RECIPE_PATH),
            "--quality",
            str(comparison),
        ],
    )
    with pytest.raises(ValueError, match="comparison overrides"):
        generate_court_scale.main()


def test_production_request_rejects_different_plan_or_quality(
    production, tmp_path, captured_cameras, multi_court_layout
):
    production = replace(
        production,
        geometry=replace(production.geometry, width=80, height=48, base_step_m=8),
    )
    plan = candidate(production, captured_cameras, multi_court_layout)
    for changed, policy in (
        ({**plan, "method": "sobol"}, production.quality),
        ({**plan, "recipe_sha256": "0" * 64}, production.quality),
        (plan, QualityPolicy()),
    ):
        with pytest.raises(ValueError, match="production recipe"):
            publication.freeze_request(
                tmp_path,
                plans=[changed],
                assets=[{"scene_id": "fixture", "alignment": {}}],
                renderer={"schema": "fixture_renderer_v1"},
                quality=policy,
                production_recipe=production,
            )


def test_production_cli_freezes_the_named_recipe(
    production, tmp_path, captured_cameras, multi_court_layout, monkeypatch
):
    (tmp_path / "scenes").mkdir()
    root = tmp_path / "out"
    monkeypatch.setattr(
        generate_court_scale,
        "renderer_fingerprint",
        lambda: {"schema": "fixture_renderer_v1"},
    )
    monkeypatch.setattr(
        generate_court_scale,
        "load_scene_inputs",
        lambda *a, **kw: {
            "scene_id": "fixture",
            "layout": multi_court_layout,
            "cameras": captured_cameras,
            "replay_configuration": None,
            "asset": {"scene_id": "fixture", "alignment": {}},
        },
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "generate_court_scale",
            "plan",
            "--scenes-root",
            str(tmp_path / "scenes"),
            "--scenes",
            "fixture",
            "--output-root",
            str(root),
            "--production-recipe",
            str(RECIPE_PATH),
            "--generation-seed",
            "42",
            "--trajectory-count",
            "1",
            "--density-level",
            "0",
        ],
    )
    generate_court_scale.main()
    frozen = publication.read_json(
        Path(publication.read_json(root / "planned-requests.json")["requests"][0])
    )
    assert frozen["schema"] == publication.PRODUCTION_REQUEST_SCHEMA
    assert frozen["recipe_sha256"] == digest(production.to_dict())
    assert frozen["production_recipe"] == production.to_dict()
