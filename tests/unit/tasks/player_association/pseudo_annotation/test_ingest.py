from contextlib import ExitStack
from pathlib import Path

import numpy as np
import pytest

from src.tasks.person_tracking.evaluation import (
    score_units,
    stratified_scores,
    tracking_units,
)
from src.tasks.player_association.association.associate import CameraTracks
from src.tasks.player_association.evaluation.labels import PSEUDO_PROVENANCE, ClipLabels
from src.tasks.player_association.evaluation.metrics import CameraPrediction, evaluate
from src.tasks.player_association.pseudo_annotation import runner
from src.tasks.player_association.pseudo_annotation.job import (
    Packet,
    Response,
    ingest,
    read_json,
)
from src.utils.geometry.triangulation import PinholeCamera

_DIGEST = "a" * 64
_PLAYER = [10.0, 20.0, 30.0, 50.0]
_SPECTATORS = [[50.0, 20.0, 65.0, 45.0], [80.0, 35.0, 95.0, 65.0]]


def _packet() -> Packet:
    frames = []
    for frame in [0, 1, 4, 5]:
        panels = [
            {"camera": camera, "candidates": [{"box": box} for box in boxes]}
            for camera, boxes in (
                ("cam0", [_PLAYER, *_SPECTATORS] if frame == 0 else [_PLAYER]),
                ("cam3", [_PLAYER]),
            )
        ]
        if frame == 5:
            panels = [
                {"camera": camera, "candidates": []} for camera in ("cam0", "cam3")
            ]
        frames.append(
            {
                "index": frame,
                "strata": ["stopped_clip"],
                "image_sha256": _DIGEST,
                "panels": panels,
            }
        )
    packet: Packet = Packet.model_validate(
        {
            "schema_version": 1,
            "dataset_id": "meiji_4cam",
            "clip_id": "video_000/clip_020",
            "num_frames": 7,
            "start": 0,
            "stop": 6,
            "player_count": 4,
            "camera_ids": ["cam0", "cam3"],
            "show_prior_association": False,
            "input_snapshot_sha256": _DIGEST,
            "clip_manifest_sha256": _DIGEST,
            "media_sha256": dict.fromkeys(["cam0", "cam3"], _DIGEST),
            "frames": frames,
        }
    )
    return packet


def _response() -> Response:
    response: Response = Response.model_validate(
        {
            "input_sha256": _DIGEST,
            "player_descriptions": ["white shirt", "unseen", "unseen", "unseen"],
            "frames": [
                {
                    "frame": f.index,
                    "panels": [
                        {
                            "camera": p.camera,
                            "labels": [
                                "P1" if i == 0 else "non_player"
                                for i, _ in enumerate(p.candidates)
                            ],
                        }
                        for p in f.panels
                    ],
                }
                for f in _packet().frames
            ],
        }
    )
    return response


def test_ingest_v1_labels_score_only_annotated_spans(tmp_path: Path) -> None:
    labels = ingest(_packet(), _response(), input_sha256=_DIGEST, prompt_sha256=_DIGEST)
    path = tmp_path / "labels.json"
    labels.save(path)
    truth = ClipLabels.load(path)
    assert truth.annotated_frames == (0, 1, 4, 5)
    spans = truth.annotated_spans()
    assert spans is not None and spans.tolist() == [0, 0, -1, -1, 1, 1, -1]
    assert truth.provenance[PSEUDO_PROVENANCE]["model"] == "gpt-6.1-terra"
    assert truth.to_json()["schema"] == "player_association_labels_v1"
    assert truth.cameras["cam0"].boxes_xyxy.tolist()[:3] == [_PLAYER, *_SPECTATORS]
    boxes = np.repeat(
        np.array([_PLAYER, *_SPECTATORS], dtype=float)[:, None, :], 7, axis=1
    )
    cam0 = CameraPrediction(
        np.arange(3, dtype=np.int64),
        boxes,
        np.array(
            [
                [True, True, True, True, True, False, True],
                [True, False, True, False, False, False, False],
                [True, False, False, False, False, False, False],
            ]
        ),
        np.array([[1, 1, -1, 3, 2, -1, -1], [-1] * 7, [-1] * 7], dtype=np.int64),
    )
    cam3 = CameraPrediction(
        cam0.track_ids[:1], boxes[:1], cam0.observed[:1], cam0.player_ids[:1]
    )
    result = evaluate(truth, {"cam0": cam0, "cam3": cam3})
    assert result["pairs"]["tp"] == 3 and result["pairs"]["f1"] == 1
    assert result["exclusion"]["tp"] == 2
    assert result["coverage"]["cam0"]["unmatched_predicted_boxes"] == 0
    assert result["id_switch"]["predicted"] == 0
    camera = PinholeCamera("cam0", np.eye(3), np.eye(3), np.ones(3))
    tracks = CameraTracks(
        camera, (200, 100), np.arange(1, 4, dtype=np.int64), boxes, cam0.observed
    )
    units, unmatched = tracking_units(tracks, cam0.observed, truth)
    metrics = score_units(units)
    assert unmatched == 0 and metrics["nonplayer_units_kept"] == 2
    assert metrics["player_units_kept"] == 3
    assert any(
        row["camera"] == "cam3"
        for row in stratified_scores([{**units[0], "camera": "cam3"}])
    )


@pytest.mark.parametrize(
    ("fault", "message"),
    [
        ("hash", "does not match"),
        ("frame", "requested frames"),
        ("panel", "each camera"),
        ("missing_box", "expected 3 box labels"),
        ("unknown", "unknown labels"),
        ("roster", "every player"),
    ],
)
def test_ingest_rejects_bad_scope_and_labels(fault: str, message: str) -> None:
    item = _response().model_dump()
    if fault == "hash":
        item["input_sha256"] = "1" * 64
    elif fault == "frame":
        item["frames"][1] = item["frames"][0]
    elif fault == "panel":
        item["frames"][0]["panels"][1] = item["frames"][0]["panels"][0]
    elif fault == "missing_box":
        item["frames"][0]["panels"][0]["labels"].pop()
    elif fault == "unknown":
        item["frames"][0]["panels"][0]["labels"][1] = "P5"
    else:
        item["player_descriptions"].pop()
    with pytest.raises(ValueError, match=message):
        ingest(
            _packet(),
            Response.model_validate(item),
            input_sha256=_DIGEST,
            prompt_sha256=_DIGEST,
        )


def test_partial_provenance_rejects_rows_outside_scope_and_duplicate_json_keys(
    tmp_path: Path,
) -> None:
    document = ingest(
        _packet(), _response(), input_sha256=_DIGEST, prompt_sha256=_DIGEST
    ).to_json()
    document["provenance"][PSEUDO_PROVENANCE]["annotated_frames"] = [0, 1, 5]
    with pytest.raises(ValueError, match="outside"):
        ClipLabels.from_json(document)
    path = tmp_path / "response.json"
    path.write_text('{"frames": [], "frames": []}')
    with pytest.raises(ValueError, match="Duplicate JSON key"):
        read_json(path)


def test_annotation_slots_protect_clip_and_global_limit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(runner, "LOCK_ROOT", tmp_path)
    packet = _packet()
    with ExitStack() as active:
        active.enter_context(runner.annotation_slot(packet))
        with (
            pytest.raises(RuntimeError, match="already being annotated"),
            runner.annotation_slot(packet),
        ):
            pytest.fail("Duplicate clip was admitted")
        for clip in ["021", "022", "023"]:
            active.enter_context(
                runner.annotation_slot(
                    packet.model_copy(update={"clip_id": f"video_000/clip_{clip}"})
                )
            )
        fifth = packet.model_copy(update={"clip_id": "video_000/clip_024"})
        with (
            pytest.raises(RuntimeError, match="All four"),
            runner.annotation_slot(fifth),
        ):
            pytest.fail("A fifth annotator was admitted")
    with runner.annotation_slot(packet):
        pass
