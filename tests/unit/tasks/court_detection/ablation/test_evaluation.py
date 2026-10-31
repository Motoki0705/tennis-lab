"""External synthetic membership and scene metrics, without images or a model."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import torch

from src.tasks.court_detection.ablation.evaluation import (
    _SceneMetrics,
    _synthetic_evaluation_records,
    evaluate_checkpoint,
)
from src.tasks.court_detection.configuration import (
    CourtTrainingConfig,
    SyntheticCourtDatasetSubsetSourceConfig,
)
from src.tasks.court_detection.data.contracts import (
    CourtSampleRecord,
    CourtTargetBundleSpec,
    CourtTargetKind,
    CourtTargetSpec,
)
from src.tasks.court_detection.scaling.schedule import has_step_schedule
from src.utils.data.image_record_store import file_sha256


def _record(scene: str, name: str, trajectory: str = "val") -> CourtSampleRecord:
    return CourtSampleRecord(
        sample_id=f"{scene}:{name}",
        split="train",
        image_path=Path("unused.jpg"),
        annotation_path=Path("unused.json"),
        payload={"scene_id": scene, "trajectory_group_id": trajectory},
    )


@pytest.fixture
def selection(tmp_path):
    source = tmp_path / "pool.dataset.json"
    source.write_text('{"fixture": "verified pool"}')
    records = (
        _record("B00", "train", "train"),
        _record("B00", "train-neighbor", "train"),
        _record("B00", "a"),
        _record("B01", "b", "train"),  # Same trajectory name in another scene is safe.
    )
    path = tmp_path / "evaluation.subset.json"
    path.write_text(
        json.dumps(
            {
                "schema": "court_dataset_subset_v1",
                "source_manifest": source.name,
                "source_manifest_sha256": file_sha256(source),
                "selection_seed": 42,
                "split": "val",
                "samples": [
                    {"scene_id": "B01", "sample_id": "b"},
                    {"scene_id": "B00", "sample_id": "a"},
                ],
            }
        )
    )
    return (
        path,
        records,
        {
            "source_manifest_sha256": file_sha256(source),
            "pool_records": records,
            "training_records": records[:1],
        },
    )


def test_external_manifest_preserves_order_and_remaps_only_selected_records(selection):
    path, records, arguments = selection
    selected = _synthetic_evaluation_records(path, **arguments)
    assert [record.sample_id for record in selected] == ["B01:b", "B00:a"]
    assert all(record.split == "val" for record in selected)
    assert all(record.split == "train" for record in records)


@pytest.mark.parametrize(
    "mutation, message",
    [
        ("sample", "training sample"),
        ("trajectory", "training trajectory"),
        ("unknown", "unknown"),
        ("duplicate", "Duplicate"),
        ("pool", "verified pool"),
    ],
)
def test_external_manifest_rejects_leakage_and_unverified_membership(
    selection, mutation, message
):
    path, _, arguments = selection
    document = json.loads(path.read_text())
    if mutation in {"sample", "trajectory", "unknown"}:
        sample = {
            "sample": "train",
            "trajectory": "train-neighbor",
            "unknown": "absent",
        }[mutation]
        document["samples"] = [{"scene_id": "B00", "sample_id": sample}]
    elif mutation == "duplicate":
        document["samples"].append(document["samples"][0])
    else:
        arguments["source_manifest_sha256"] = "different-checkpoint-pool"
    path.write_text(json.dumps(document))
    with pytest.raises(ValueError, match=message):
        _synthetic_evaluation_records(path, **arguments)


def _scene_fixture():
    # Interleaved scenes and a second B00 observation in another batch detect
    # sorting, tracker resets, pixel-vs-image weighting, and a mean-of-medians.
    records = tuple(
        _record(scene, name)
        for scene, name in (
            ("B02", "c"),
            ("B00", "a1"),
            ("B01", "b"),
            ("B00", "a0"),
            ("B03", "d"),
            ("B00", "a2"),
        )
    )
    channels: dict[CourtTargetKind, int] = {
        "kp": 2,
        "seg": 2,
        "line": 1,
        "semantic_line": 2,
    }
    bundle = CourtTargetBundleSpec(
        {
            kind: CourtTargetSpec(
                kind,
                f"fixture-{kind}",
                count,
                tuple(map(str, range(count))),
                torch.float32,
                False,
            )
            for kind, count in channels.items()
        }
    )
    truth = torch.tensor([[1, 1, 1, 1, 0, 0, 0, 0, 0]] * 6)[:, None, :]
    truth[2] = 1
    prediction = truth.clone()
    prediction[[2, 3, 5]] = 0
    prediction[4] = 1
    categorical = (
        torch.nn.functional.one_hot(prediction, 2).permute(0, 3, 1, 2).float() * 4
    )
    kp = torch.full((6, 2, 2, 10), -10.0)
    for sample, distance in enumerate((4, 0, 2, 6, 8, 0)):
        kp[sample, :, 0, distance] = 10.0
    kp[:, :, 1, 9] = 100.0  # Padded maxima must not affect KP metrics.
    visible = torch.zeros((6, 2, 1), dtype=torch.bool)
    visible[:, 0] = True
    visible[1, 1] = True
    targets = {
        "kp": {"points_xy": torch.zeros(6, 2, 1, 2), "point_visible": visible},
        "seg": torch.nn.functional.one_hot(truth, 2).permute(0, 3, 1, 2).float(),
        "semantic_line": truth,
        "line": truth[:, None].float(),
    }
    targets["line"][4] *= 0.5  # Preserve fractional GT coverage in Dice.
    logits = {
        "kp": kp,
        "seg": categorical,
        "semantic_line": -categorical,
        "line": prediction[:, None].float() * 20 - 10,
    }
    return records, bundle, logits, targets


def test_scene_metrics_match_hand_calculation_and_equal_scene_weighting():
    records, bundle, logits, targets = _scene_fixture()
    metrics = _SceneMetrics(records, bundle)
    ids = [record.sample_id for record in records]
    for first, last in ((0, 3), (3, 6)):
        selected_targets = {
            kind: {name: value[first:last] for name, value in target.items()}
            if isinstance(target, dict)
            else target[first:last]
            for kind, target in targets.items()
        }
        metrics.update(
            ids[first:last],
            {kind: value[first:last] for kind, value in logits.items()},
            selected_targets,
            torch.tensor([[1, 9]] * (last - first)),
        )
    result = metrics.compute()
    assert result["scene_order"] == ["B02", "B00", "B01", "B03"]
    scenes = result["scenes"]
    assert {scene: value["count"] for scene, value in scenes.items()} == {
        "B02": 1,
        "B00": 3,
        "B01": 1,
        "B03": 1,
    }
    assert scenes["B00"]["sample_ids"] == ["B00:a1", "B00:a0", "B00:a2"]
    expected_hash = hashlib.sha256(b'["B00:a1","B00:a0","B00:a2"]').hexdigest()
    assert scenes["B00"]["ordered_sample_ids_sha256"] == expected_hash
    assert scenes["B00"]["metrics"] == pytest.approx(
        {
            "kp_mean_distance_px": 1.5,  # Four visible points: 0, 0, 6, 0.
            "kp_median_distance_px": 0,
            "seg_miou": 34 / 69,  # Foreground 4/12, background 15/23.
            "line_dice": 7 / 15,  # Mean of 1, 1/5, 1/5.
            "semantic_line_miou": 4 / 27,
        }
    )
    assert result["scene_macro_mean"] == pytest.approx(
        {
            "kp_mean_distance_px": 3.875,
            "kp_median_distance_px": 3.5,
            "seg_miou": (34 / 69 + 1 + 2 / 9) / 4,
            "line_dice": 119 / 240,
            "semantic_line_miou": 25 / 108,
        }
    )


def test_scene_metrics_refuse_an_unconsumed_manifest():
    records, bundle, _, _ = _scene_fixture()
    with pytest.raises(ValueError, match="exact ID order"):
        _SceneMetrics(records, bundle).compute()


def test_api_rejects_conflicting_synthetic_selection_before_loading_checkpoint(
    tmp_path,
):
    with pytest.raises(ValueError, match="either"):
        evaluate_checkpoint(
            tmp_path / "absent.ckpt",
            tmp_path / "out",
            device="cpu",
            include_synthetic_val=True,
            synthetic_eval_manifest=tmp_path / "manifest.json",
        )


@pytest.mark.parametrize("scene_count", [1, 2, 3, 4])
def test_archived_scene_count_checkpoint_configuration_parses(
    scene_count,
):
    from omegaconf import OmegaConf

    path = Path(
        f"knowledge/runs/run-court-scene-diversity-colab-scene-{scene_count}-s42/evaluation.json"
    )
    receipt = json.loads(path.read_text())
    config = OmegaConf.create(receipt["config"])
    runtime = CourtTrainingConfig.from_config(config)
    assert has_step_schedule(config)
    assert isinstance(runtime.data.source, SyntheticCourtDatasetSubsetSourceConfig)
    assert runtime.data.source.manifest.name == f"scene-{scene_count}.train.subset.json"
    assert set(receipt["data_condition"]["dataset_subset"]["manifest_sha256"]) == {
        "train",
        "val",
    }
    if scene_count == 2:
        assert "/version_1/" in receipt["checkpoint"]
        assert config.run.resume.role == "artifact"
