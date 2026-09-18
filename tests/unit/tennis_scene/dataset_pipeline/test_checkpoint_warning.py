"""Warnings waive only declared checkpoint identity, with persistent evidence."""

import json
from pathlib import Path

import pytest

from src.tennis_scene.dataset_pipeline import checkpoint_integrity, people
from src.tennis_scene.dataset_pipeline.checkpoint_warning import (
    checkpoint_warning_policy,
    normalize_receipt,
)
from src.utils.checksum import FileIntegrityError

PINS = dict.fromkeys(("dino", "vitpose", "court", "plcs", "blcs", "dinov3"), "a" * 64)


def test_stage_strict_allowlist_and_provider_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(checkpoint_integrity, "dual_sha256", lambda _: "b" * 64)
    with pytest.raises(RuntimeError):
        checkpoint_integrity.verify_checkpoint_integrity({"dino": tmp_path}, PINS)
    log = tmp_path / "warnings.jsonl"
    with checkpoint_warning_policy(PINS, ["dino"], log):
        assert checkpoint_integrity.verify_checkpoint_integrity(
            {"dino": tmp_path}, PINS
        ) == {"dino": "b" * 64}
        with pytest.raises(RuntimeError):
            checkpoint_integrity.verify_checkpoint_integrity({"court": tmp_path}, PINS)

        def failed(_: Path) -> str:
            raise FileIntegrityError("provider disagreement", details={})

        monkeypatch.setattr(checkpoint_integrity, "dual_sha256", failed)
        with pytest.raises(FileIntegrityError):
            checkpoint_integrity.verify_checkpoint_integrity({"dino": tmp_path}, PINS)
    record = json.loads(log.read_text())
    assert record["observed_sha256"] == "b" * 64
    assert record["expected_sha256"] == PINS["dino"]


def test_cache_only_checkpoint_fields_are_waived(tmp_path: Path) -> None:
    receipt = tmp_path / "receipt.json"
    saved = {
        "detector_sha256": "b" * 64,
        "video_sha256": "video",
        "settings": {"stride": 5},
        "declared_checkpoint_sha256": {"dino": PINS["dino"]},
    }
    receipt.write_text(json.dumps(saved))
    expected = {**saved, "detector_sha256": "c" * 64}
    with checkpoint_warning_policy(PINS, ["dino"], tmp_path / "warnings.jsonl"):
        people._validate_cache_identity(
            receipt, expected, cache=tmp_path, prefix="cache"
        )
        for changed in ({"video_sha256": "changed"}, {"settings": {"stride": 6}}):
            with pytest.raises(ValueError, match="changed identity fields"):
                people._validate_cache_identity(
                    receipt, {**expected, **changed}, cache=tmp_path, prefix="cache"
                )
        with pytest.raises(ValueError, match="Different declared"):
            normalize_receipt(
                {**saved, "declared_checkpoint_sha256": {"dino": "d" * 64}},
                path=receipt,
                context="test",
            )
    assert json.loads(receipt.read_text()) == saved


def test_infer_people_and_raw_receipts_keep_observations(tmp_path: Path) -> None:
    saved = {
        "detector_sha256": "b" * 64,
        "pose_sha256": "c" * 64,
        "declared_checkpoint_sha256": {
            "dino": PINS["dino"],
            "vitpose": PINS["vitpose"],
        },
    }
    people_path = tmp_path / "cam_people.metadata.json"
    people_path.write_text(json.dumps(saved))
    raw_path = tmp_path / "cam_detections.metadata.json"
    raw_path.write_text(json.dumps({"checkpoint_sha256": "d" * 64}))
    with pytest.raises(FileIntegrityError):
        people.validate_people_receipts(["cam"], tmp_path, checkpoint_sha256=PINS)
    with checkpoint_warning_policy(
        PINS, ["dino", "vitpose"], tmp_path / "warnings.jsonl"
    ):
        people.validate_people_receipts(["cam"], tmp_path, checkpoint_sha256=PINS)
    assert json.loads(people_path.read_text()) == saved
    assert json.loads(raw_path.read_text())["checkpoint_sha256"] == "d" * 64


def test_quality_producers_use_declared_identity_and_reject_other_models(
    tmp_path: Path,
) -> None:
    from src.tennis_scene.dataset_pipeline.quality_report import observation_producers
    from src.tennis_scene.reference_pipeline.observations import sha256

    for camera, observed in (("a", "b" * 64), ("b", "c" * 64)):
        (tmp_path / f"{camera}_people.metadata.json").write_text(
            json.dumps(
                {
                    "schema_version": 4,
                    "policy": "same algorithm",
                    "settings": {},
                    "detector_sha256": observed,
                    "pose_sha256": PINS["vitpose"],
                    "declared_checkpoint_sha256": {
                        "dino": PINS["dino"],
                        "vitpose": PINS["vitpose"],
                    },
                }
            )
        )
    (tmp_path / "court.json").write_text(
        json.dumps({"identity": {"checkpoint_sha256": PINS["court"], "settings": {}}})
    )

    def hashes() -> dict[str, str]:
        return {path.name: sha256(path) for path in tmp_path.glob("*.json")}

    with pytest.raises(ValueError, match="Mixed provenance"):
        observation_producers(tmp_path, ["a", "b"], hashes())
    with checkpoint_warning_policy(
        PINS, ["dino", "vitpose"], tmp_path / "warnings.jsonl"
    ):
        result = observation_producers(tmp_path, ["a", "b"], hashes())
        assert result["people_producer"]["detector_declared_sha256"] == PINS["dino"]
        changed = tmp_path / "b_people.metadata.json"
        saved = json.loads(changed.read_text())
        saved["declared_checkpoint_sha256"]["dino"] = "e" * 64
        changed.write_text(json.dumps(saved))
        with pytest.raises(ValueError, match="Different declared"):
            observation_producers(tmp_path, ["a", "b"], hashes())


def test_quality_warning_audit_does_not_mutate_generation_run(tmp_path: Path) -> None:
    from src.tennis_scene.dataset_pipeline.quality_report import (
        audit_observation_producers,
    )
    from src.tennis_scene.reference_pipeline.observations import sha256

    run = tmp_path / "run"
    run.mkdir()
    (run / "recipe.json").write_text(
        json.dumps(
            {"checkpoint_sha256": PINS, "checkpoint_warning_roles": ["dino", "vitpose"]}
        )
    )
    (run / "checkpoint_warnings.jsonl").write_text('{"historical": true}\n')
    obs = tmp_path / "observations"
    obs.mkdir()
    (obs / "cam_people.metadata.json").write_text(
        json.dumps(
            {
                "schema_version": 4,
                "policy": "same",
                "settings": {},
                "detector_sha256": "b" * 64,
                "pose_sha256": "c" * 64,
            }
        )
    )
    (obs / "cam_detections.metadata.json").write_text(
        json.dumps({"checkpoint_sha256": "b" * 64})
    )
    (obs / "court.json").write_text(
        json.dumps({"identity": {"checkpoint_sha256": PINS["court"], "settings": {}}})
    )
    before = {
        str(path): path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()
    }
    producers, roles, warnings, source = audit_observation_producers(
        obs, ["cam"], {path.name: sha256(path) for path in obs.iterdir()}, run
    )
    assert roles == ["dino", "vitpose"] and warnings
    assert all("historical" not in warning for warning in warnings)
    assert source == run / "checkpoint_warnings.jsonl"
    assert (
        producers["observed_people_checkpoint_digests"]["cam"]["detector_sha256"]
        == "b" * 64
    )
    assert {
        str(path): path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()
    } == before
