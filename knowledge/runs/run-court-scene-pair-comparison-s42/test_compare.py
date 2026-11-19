"""CPU contracts for the final comparison; expected contrasts are hand calculated."""

# mypy: disallow-untyped-decorators=False
from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
from typing import Any

import pytest

SPEC = importlib.util.spec_from_file_location(
    "scene_pair_compare", Path(__file__).with_name("compare.py")
)
assert SPEC is not None and SPEC.loader is not None
c = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(c)


def test_contrasts_keep_raw_sign_and_condition_medians() -> None:
    # Same-group medians 20, 40; cross-group 1, 3, 7, 13: delta=24.
    # Cross-group additivity: (1+13)-(3+7)=4. IoU uses its original sign.
    rows = {
        pair: {"kp_median_distance_px": median, "seg_miou": iou}
        for pair, median, iou in zip(
            ("P01", "P23", "P02", "P03", "P12", "P13"),
            (20, 40, 1, 3, 7, 13),
            (0.1, 0.3, 0.5, 0.7, 0.8, 0.6),
            strict=True,
        )
    }
    result = c.contrasts(rows, ("kp_median_distance_px", "seg_miou"))
    assert result[0] == {
        "metric": "kp_median_distance_px",
        "same_group_mean": 30.0,
        "cross_group_mean": 6.0,
        "delta": 24.0,
        "additivity": 4.0,
        "same_group_worse_direction": "positive",
    }
    assert result[1]["delta"] == pytest.approx(-0.45)
    assert result[1]["additivity"] == pytest.approx(-0.4)
    assert result[1]["same_group_worse_direction"] == "negative"


def test_scene_categories_include_unseen_same_group_and_absent_group() -> None:
    assert [c.category("scene-1", s) for s in c.SCENES] == [
        "seen",
        "unseen-with-same-group",
        "unseen-without-same-group",
        "unseen-without-same-group",
    ]
    assert [c.category("scene-2", s) for s in c.SCENES] == [
        "seen",
        "seen",
        "unseen-without-same-group",
        "unseen-without-same-group",
    ]
    assert [c.category("P02", s) for s in c.SCENES] == [
        "seen",
        "unseen-with-same-group",
        "seen",
        "unseen-with-same-group",
    ]
    assert [c.category("scene-3", s) for s in c.SCENES] == [
        "seen",
        "seen",
        "seen",
        "unseen-with-same-group",
    ]
    assert all(c.category("scene-4", s) == "seen" for s in c.SCENES)


@pytest.fixture(scope="module")
def reports() -> tuple[dict[str, Any], dict[str, Any]]:
    real_ids = [f"real-{i}" for i in range(2211)]
    syn_ids = [
        f"{s}:{i}"
        for s, n in zip(c.SCENES, (1056, 643, 462, 278), strict=True)
        for i in range(n)
    ]
    common = {
        "evaluation_manifest_sha256": "e" * 64,
        "real_ids_sha256": c.ordered_digest(real_ids),
        "synthetic_ids_sha256": c.ordered_digest(syn_ids),
    }
    result, pins = {}, {}
    for n, condition in enumerate(c.CONDITIONS):
        scene_values = {}
        for scene, value in zip(c.SCENES, (1.0, 3.0, 5.0, 15.0), strict=True):
            ids = [i for i in syn_ids if i.startswith(scene + ":")]
            scene_values[scene] = {
                "count": len(ids),
                "sample_ids": ids,
                "ordered_sample_ids_sha256": c.ordered_digest(ids),
                "metrics": {
                    m: value if m.startswith("kp_") else value / 20 for m in c.METRICS
                },
            }
        metrics = {m: float(n + 1) if m.startswith("kp_") else 0.25 for m in c.METRICS}
        data: dict[str, Any] = {
            "scene_images": {
                s: 5500 // len(c.MEMBERS[condition]) for s in c.MEMBERS[condition]
            },
            "accepted_images": 5500,
            "training_seed": 42,
            "synthetic_validation": {
                "images": 500,
                "scene_images": {
                    s: 500 // len(c.MEMBERS[condition]) for s in c.MEMBERS[condition]
                },
            },
        }
        if condition == "scene-3":
            data["scene_images"] = {"B00": 1833, "B01": 1833, "B02": 1834}
            data["synthetic_validation"]["scene_images"] = {
                "B00": 167,
                "B01": 167,
                "B02": 166,
            }
        config = {"run": {"seed": 42}, "data": {"batch_size": 8}}
        checkpoint = f"outputs/{condition}/update-00033160.ckpt"
        result[condition] = {
            "schema": "court_data_scaling_evaluation_v2",
            "series": "scene-pairs",
            "global_step": 33160,
            "checkpoint": "/content/tennis-lab/" + checkpoint,
            "checkpoint_sha256": str(n) * 64,
            "data_condition": data,
            "config": config,
            "synthetic_evaluation_manifest": {"sha256": "e" * 64},
            "splits": {
                "tennis_court_detector-val": {
                    "count": 2211,
                    "sample_ids": real_ids,
                    "ordered_sample_ids_sha256": common["real_ids_sha256"],
                    "batch_size": 8,
                    "pose_supervised": False,
                    "metrics": metrics,
                },
                "synthetic_court-val": {
                    "count": 2439,
                    "sample_ids": syn_ids,
                    "ordered_sample_ids_sha256": common["synthetic_ids_sha256"],
                    "batch_size": 8,
                    "pose_supervised": True,
                    "metrics": metrics,
                    "scene_order": list(c.SCENES),
                    "scenes": scene_values,
                    "scene_macro_mean": {
                        m: 6.0 if m.startswith("kp_") else 0.3 for m in c.METRICS
                    },
                },
            },
        }
        pins[condition] = {
            "checkpoint": checkpoint,
            "checkpoint_sha256": str(n) * 64,
            "config_digest": c.digest(config),
            "data_condition_digest": c.digest(data),
        }
    return result, {"common": common, "conditions": pins}


def test_unweighted_scene_mean_and_p01_alias(
    reports: tuple[dict[str, Any], dict[str, Any]],
) -> None:
    raw, _ = reports
    result = c.summarize(raw)
    assert [r["condition"] for r in result["real_pairs"]] == [
        "P01",
        "P02",
        "P03",
        "P12",
        "P13",
        "P23",
    ]
    assert result["real_pairs"][0]["kp_median_distance_px"] == 2.0  # scene-2, once
    assert [r["condition"] for r in result["b00_addition"]] == [
        "scene-1",
        "P01",
        "P02",
        "P03",
    ]
    assert len(result["synthetic_long"]) == 36
    assert (
        result["synthetic_matrix"]["kp_median_distance_px"][0]["four_scene_mean"] == 6.0
    )
    assert result["synthetic_matrix"]["kp_median_distance_px"][0][
        "four_scene_mean"
    ] != pytest.approx((1056 + 643 * 3 + 462 * 5 + 278 * 15) / 2439)


def test_original_python311_macro_rounding_is_preserved(
    reports: tuple[dict[str, Any], dict[str, Any]],
) -> None:
    raw, sources = reports
    report = copy.deepcopy(raw["P02"])
    synthetic = report["splits"]["synthetic_court-val"]
    # Left-to-right binary addition loses the three small positive terms.
    # Python 3.12's new sum instead keeps them, changing the recorded receipt.
    for scene, value in zip(c.SCENES, (1e16, 1.0, 1.0, 1.0), strict=True):
        synthetic["scenes"][scene]["metrics"]["kp_mean_distance_px"] = value
    synthetic["scene_macro_mean"]["kp_mean_distance_px"] = 2500000000000000.0
    c.validate_report("P02", report, sources)


@pytest.mark.parametrize(
    "fault",
    [
        "checkpoint",
        "step",
        "manifest",
        "config",
        "data",
        "order",
        "duplicate",
        "scene-order",
        "scene-count",
        "macro",
        "nonfinite",
        "batch",
    ],
)
def test_mismatched_reports_are_rejected(
    reports: tuple[dict[str, Any], dict[str, Any]], fault: str
) -> None:
    raw, sources = reports
    report = copy.deepcopy(raw["P02"])
    real = report["splits"]["tennis_court_detector-val"]
    syn = report["splits"]["synthetic_court-val"]
    if fault == "checkpoint":
        report["checkpoint_sha256"] = "f" * 64
    elif fault == "step":
        report["global_step"] = 29844
    elif fault == "manifest":
        report["synthetic_evaluation_manifest"]["sha256"] = "f" * 64
    elif fault == "config":
        report["config"]["run"]["seed"] = 43
    elif fault == "data":
        report["data_condition"]["accepted_images"] = 5499
    elif fault == "order":
        real["sample_ids"][0], real["sample_ids"][1] = (
            real["sample_ids"][1],
            real["sample_ids"][0],
        )
    elif fault == "duplicate":
        real["sample_ids"][0] = real["sample_ids"][1]
    elif fault == "scene-order":
        syn["scene_order"].reverse()
    elif fault == "scene-count":
        syn["scenes"]["B00"]["count"] -= 1
    elif fault == "macro":
        syn["scene_macro_mean"]["kp_median_distance_px"] = 5.0
    elif fault == "nonfinite":
        real["metrics"]["line_dice"] = float("nan")
    elif fault == "batch":
        real["batch_size"] = 16
    with pytest.raises(ValueError):
        c.validate_report("P02", report, sources)


def test_raw_bytes_hash_and_missing_condition_guard(
    tmp_path: Path, reports: tuple[dict[str, Any], dict[str, Any]]
) -> None:
    raw, original = reports
    sources = copy.deepcopy(original)
    for condition, report in raw.items():
        data = json.dumps(report).encode()
        (tmp_path / f"{condition}.json").write_bytes(data)
        sources["conditions"][condition]["evaluation_sha256"] = c.sha_bytes(data)
    loaded = c.load_reports(tmp_path, sources)
    assert len(loaded) == 9
    (tmp_path / "P02.json").write_text("{}")
    with pytest.raises(ValueError, match="hash"):
        c.load_reports(tmp_path, sources)
    (tmp_path / "P02.json").unlink()
    with pytest.raises(ValueError, match="inventory"):
        c.load_reports(tmp_path, sources)


@pytest.mark.parametrize("field", ["scene_images", "synthetic_validation"])
def test_budget_guard_rejects_reanchored_bad_allocation(
    reports: tuple[dict[str, Any], dict[str, Any]],
    field: str,
) -> None:
    raw, original = reports
    report, sources = copy.deepcopy(raw["P02"]), copy.deepcopy(original)
    if field == "scene_images":
        report["data_condition"][field] = {"B00": 2749, "B02": 2751}
    else:
        report["data_condition"][field]["scene_images"] = {"B00": 249, "B02": 251}
    # Even an internally self-consistent receipt must obey the agreed budget.
    sources["conditions"]["P02"]["data_condition_digest"] = c.digest(
        report["data_condition"]
    )
    with pytest.raises(ValueError, match="allocation"):
        c.validate_report("P02", report, sources)
