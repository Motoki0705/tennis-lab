"""Original v3 -> immutable native images/teachers -> active-ball learning windows."""

from __future__ import annotations

import json
from fractions import Fraction
from pathlib import Path
from typing import Any, cast
from unittest.mock import Mock

import av
import cv2
import numpy as np
import pytest
from torch.utils.data import DataLoader, Subset

from src.tasks.ball_detection.annotation.video import read_annotation, write_annotation
from src.tasks.ball_detection.data.annotation_states import annotation_states
from src.tasks.ball_detection.data.store import POINT_KIND_CODES, BallFrameStore
from src.tasks.ball_detection.data.supervision import (
    OBSERVED_ONLY,
    resolve_frame_supervision,
)
from src.tasks.ball_detection.data.windows.active_ball import (
    ActiveBallWindowDataset,
    collate_active_ball_windows,
)
from src.tasks.ball_detection.generate_dataset.frame_store.publishing.contracts import (
    BallPublicationPolicy,
    LegacySourceGroup,
    SourceSplit,
)
from src.tasks.ball_detection.generate_dataset.frame_store.publishing.publisher import (
    BallPublisher,
)
from src.tasks.ball_detection.generate_dataset.frame_store.publishing.snapshot import (
    freeze_legacy_store,
    load_snapshot,
)
from src.tasks.base.annotation.artifacts import read_document, write_document
from src.tasks.base.annotation.contracts import PublicationReceipt, content_digest
from src.tasks.base.annotation.paths import AnnotationLayout
from tests.support.annotation_storage import assert_expired_but_protected
from tests.support.tasks.ball_detection.store import ball, frame, write_store_clip
from tests.support.tasks.ball_detection.video_publication import (
    incremental_case,
    prepare,
    scoped,
)
from tests.support.tasks.local_annotation import local_configuration


def test_publication_preserves_base_bytes_rows_and_sealed_review_masks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The legacy node with this name covers review seals; v3 intentionally uses
    # completed original judgments without adding an independent QA/adoption gate.
    layout = AnnotationLayout.testing(tmp_path / "annotation")
    data = local_configuration(layout)
    first, second = data.videos
    _, initial = prepare(
        layout,
        data,
        name="first",
        version="initial-active",
        videos=(scoped(first, 1, 3),),
    )
    base_receipt = read_document(
        layout, BallPublisher(layout).publish(initial), PublicationReceipt
    )
    base_snapshot, base = load_snapshot(layout, base_receipt.snapshot)
    before = {
        p: p.read_bytes()
        for p in (
            base.directory / "metadata.json",
            base.directory / "index.npz",
            base.image_path(base.clips[0]),
        )
    }
    _, request = prepare(
        layout,
        data,
        name="next",
        version="rally-snapshot",
        videos=(scoped(first, 3, 5), scoped(second, 0, 5)),
        base=base_receipt.snapshot,
    )
    # Original input scope spans multiple cameras and a related clip of cam0.
    submit, register = Mock(), Mock()
    monkeypatch.setattr(
        "src.tasks.base.annotation.execution.runner.StageRunner.submit", submit
    )
    monkeypatch.setattr(
        "src.tasks.base.annotation.learning.service.Learning.register", register
    )
    with monkeypatch.context() as patch:
        patch.setattr(
            cv2,
            "imencode",
            Mock(side_effect=AssertionError("re-encoding prepared JPEGs")),
        )
        reference = BallPublisher(layout).publish(request)
        assert BallPublisher(layout).publish(request) == reference
    receipt = read_document(layout, reference, PublicationReceipt)
    snapshot, store = load_snapshot(layout, receipt.snapshot)
    assert receipt.adoptions == ()
    assert set(receipt.annotations) == {
        saved.annotation for saved in request.annotations
    }
    assert len(receipt.added_clips) == 2 and len(receipt.reused_clips) == 1
    assert snapshot.clips[0].semantic_digest == base_snapshot.clips[0].semantic_digest
    assert store.metadata["version"] == "rally-snapshot"
    assert store.metadata["schema_version"] == "ball_detection_frames.v4"
    assert len(store.clips) == 3 and len(store) == 9
    assert {c.camera_id for c in store.clips} == {"cam0", "cam2"}
    assert {c.group_id for c in store.clips} == {content_digest("recording-a")}
    assert {c.split for c in store.clips} == {"val"}
    assert {c.source_id for c in snapshot.clips} == {content_digest("recording-a")}
    for path, payload in before.items():
        assert path.read_bytes() == payload
    assert (
        cast(list[Any], store.metadata["clips"])[0]
        == cast(list[Any], base.metadata["clips"])[0]
    )
    for name, values in (*base.frames.items(), *base.instances.items()):
        actual = store.frames[name] if name in base.frames else store.instances[name]
        np.testing.assert_array_equal(actual[: len(values)], values)
    assert store.teacher_masks is not None and base.teacher_masks is not None
    np.testing.assert_array_equal(
        store.teacher_masks.coordinate[:2], base.teacher_masks.coordinate
    )
    np.testing.assert_array_equal(
        store.teacher_masks.heatmap[:2], base.teacher_masks.heatmap
    )
    assert (
        store.image_path(store.clips[0]).stat().st_ino
        == base.image_path(base.clips[0]).stat().st_ino
    )
    assert_expired_but_protected(
        layout,
        (
            reference,
            receipt.snapshot,
            snapshot.index,
            snapshot.metadata,
            *(s.annotation for s in request.annotations),
            *(s.context for s in request.annotations),
        ),
        "publication",
    )

    # Independent oracle: original native frames and hand-written expected
    # coordinates/masks, including nonzero scope start and nonuniform PTS.
    native = {}
    for video in data.videos:
        with av.open(video.path) as container:
            frames = list(container.decode(video=0))
        assert [f.pts * Fraction(f.time_base) for f in frames] == [
            Fraction(t, 16000) for t in (0, 640, 1920, 2560, 4480)
        ]
        native[video.annotation.source.camera_id] = frames
    expected = {
        ("cam0", 1): ((9.0, 13.0), True, True, True, False, False),
        ("cam0", 2): ((22.5, 7.25), True, True, False, True, False),
        ("cam0", 3): (None, False, False, False, False, False),
        ("cam0", 4): (None, False, True, False, False, True),
        ("cam2", 0): ((61.5, 19.25), True, True, True, False, False),
        ("cam2", 1): ((49.25, 16.5), True, True, False, True, False),
        ("cam2", 2): ((31.75, 8.0), True, True, False, True, False),
        ("cam2", 3): (None, False, False, False, False, False),
        ("cam2", 4): (None, False, True, False, False, True),
    }
    dataset = ActiveBallWindowDataset(store, split="val", window_length=2, stride=2)
    assert len(dataset) == 5  # Two cam0 clips and three windows covering all of cam2.
    seen = set()
    for sample in dataset:
        assert sample["target_semantics"] == "active_ball"
        camera = sample["camera_id"]
        for i, frame_index in enumerate(sample["frame_indices"].tolist()):
            seen.add((camera, frame_index))
            xy, position, heatmap, observed, estimated, negative = expected[
                camera, frame_index
            ]
            assert bool(sample["position_valid"][i]) == position
            assert bool(sample["heatmap_valid"][i]) == heatmap
            assert bool(sample["direct_observation"][i]) == observed
            assert bool(sample["estimated"][i]) == estimated
            assert bool(sample["active_ball_negative"][i]) == negative
            assert sample["position_weight"][i].item() == float(position)
            assert sample["heatmap_weight"][i].item() == float(heatmap)
            if xy is not None:
                np.testing.assert_array_equal(sample["xy"][i].numpy(), xy)
                width, height = sample["source_size"].tolist()
                np.testing.assert_allclose(
                    sample["uv"][i].numpy(), np.asarray(xy) / (width - 1, height - 1)
                )
            original = native[camera][frame_index]
            assert sample["pts"][i].item() * Fraction(
                sample["time_base"]
            ) == original.pts * Fraction(original.time_base)
            ok, jpeg = cv2.imencode(
                ".jpg",
                original.to_ndarray(format="bgr24"),
                [cv2.IMWRITE_JPEG_QUALITY, 90],
            )
            assert ok
            decoded = cv2.imdecode(jpeg, cv2.IMREAD_COLOR)
            assert decoded is not None
            expected_rgb = decoded[..., ::-1].transpose(2, 0, 1)
            np.testing.assert_array_equal(sample["rgb"][i].numpy(), expected_rgb)
    assert seen == set(expected)
    # Real DataLoader batching must retain estimated teachers and null provenance.
    batch = next(
        iter(
            DataLoader(
                Subset(dataset, [0, 1]),
                batch_size=2,
                collate_fn=collate_active_ball_windows,
            )
        )
    )
    assert batch["position_valid"].tolist() == [[True, True], [False, False]]
    assert batch["heatmap_valid"].tolist() == [[True, True], [False, True]]
    assert batch["position_weight"].tolist() == [[1.0, 1.0], [0.0, 0.0]]
    assert batch["position_provenance"][1] == (None, None)
    with pytest.raises(ValueError, match="native shape"):
        collate_active_ball_windows([dataset[0], dataset[-1]])
    with pytest.raises(ValueError, match="target_semantics"):
        collate_active_ball_windows([dict(dataset[0], target_semantics="all_balls")])
    first_sample = dataset[0]
    assert first_sample["position_provenance"][1] == {
        "kind": "estimated",
        "support_frames": [1, 3],
    }
    estimated_clip = store.clips[0]
    assert store.instances_of(store.row_of(estimated_clip, 1)).point_kind.tolist() == [
        POINT_KIND_CODES["estimated"]
    ]
    for annotation in store.video_annotations.values():
        assert all(
            f.notes == "original camera context; no triangulation or smoothing"
            for f in annotation.frames
        )

    # Incompatible readers must fail explicitly, including the old window gate.
    with pytest.raises(ValueError, match="target_semantics"):
        resolve_frame_supervision(store, OBSERVED_ONLY)
    with pytest.raises(ValueError, match="target_semantics"):
        annotation_states(store, store.clips[0])
    with pytest.raises(ValueError, match="target_semantics"):
        ActiveBallWindowDataset(
            store, split="val", window_length=2, target_semantics="all_balls"
        )
    # The published learning reader does not depend on live annotation files.
    for saved in (*initial.annotations, *request.annotations):
        layout.artifact_path(saved.annotation).unlink()
    reopened = BallFrameStore(store.directory)
    standalone = ActiveBallWindowDataset(reopened, split="val", window_length=2)
    assert standalone[0]["estimated"].tolist() == [False, True]
    image_path = reopened.image_path(reopened.clips[0])
    payload = image_path.read_bytes()
    image_path.write_bytes(payload[:-1] + bytes([payload[-1] ^ 1]))
    with pytest.raises(ValueError, match="changed during dataset use"):
        standalone[0]
    submit.assert_not_called()
    register.assert_not_called()


@pytest.mark.parametrize(
    "violation",
    [
        "unprocessed",
        "input",
        "partial",
        "disabled",
        "policy",
        "legacy_base",
        "holdout",
        "duplicate",
        "correction",
    ],
)
def test_video_publication_refuses_incomplete_mismatched_and_repeated_teachers(
    tmp_path: Path, violation: str
) -> None:
    env, request, _ = incremental_case(tmp_path, include_related=True)
    layout = env.layout
    if violation == "partial":
        request = request.model_copy(update={"annotations": request.annotations[:1]})
        message = "every registered"
    elif violation == "disabled":
        plan = env.plan.model_copy(update={"publish_dataset": False})
        plan_ref = write_document(
            layout, layout.path("campaigns", "disabled-plan.json"), plan
        )
        request = request.model_copy(update={"plan": plan_ref})
        message = "explicitly select"
    elif violation == "policy":
        policy = BallPublicationPolicy(jpeg_quality=90, source_splits=())
        ref = write_document(
            layout, layout.path("campaigns", "legacy-policy.json"), policy
        )
        request = request.model_copy(
            update={
                "output": request.output.model_copy(update={"generation_policy": ref})
            }
        )
        message = "target_semantics"
    elif violation == "holdout":
        policy = BallPublicationPolicy(
            jpeg_quality=90,
            target_semantics="active_ball",
            source_splits=(
                SourceSplit(source_id=content_digest("recording-a"), split="train"),
            ),
        )
        ref = write_document(
            layout, layout.path("campaigns", "changed-holdout-policy.json"), policy
        )
        request = request.model_copy(
            update={
                "output": request.output.model_copy(update={"generation_policy": ref})
            }
        )
        message = "cannot change an existing source split"
    elif violation == "legacy_base":
        root = write_store_clip(
            tmp_path / "data/ball_detection/legacy",
            "old/clip",
            [frame(0, ball())],
            split="val",
        )
        ref = freeze_legacy_store(
            layout,
            root,
            (
                LegacySourceGroup(
                    source="tracknet",
                    group_id="old",
                    source_id=content_digest("recording-a"),
                ),
            ),
        )
        request = request.model_copy(
            update={"output": request.output.model_copy(update={"base_snapshot": ref})}
        )
        message = "target_semantics"
    else:
        if violation in {"duplicate", "correction"}:
            published = read_document(
                layout, BallPublisher(layout).publish(request), PublicationReceipt
            )
            request = request.model_copy(
                update={
                    "output": request.output.model_copy(
                        update={
                            "base_snapshot": published.snapshot,
                            "dataset_version": "another",
                        }
                    )
                }
            )
            message = "duplicate or change"
        else:
            message = "unprocessed" if violation == "unprocessed" else "expected input"
        if violation != "duplicate":
            saved = request.annotations[0]
            annotation = read_annotation(layout.artifact_path(saved.annotation))
            if violation == "input":
                annotation = annotation.model_copy(
                    update={
                        "source": annotation.source.model_copy(
                            update={"camera_id": "unregistered"}
                        )
                    }
                )
            else:
                row = annotation.frames[0]
                change = (
                    dict(
                        state="unprocessed",
                        position=None,
                        occlusion="unknown",
                        break_before=False,
                    )
                    if violation == "unprocessed"
                    else dict(notes="corrected annotation")
                )
                annotation = annotation.model_copy(
                    update={
                        "frames": (
                            row.model_copy(update=change),
                            *annotation.frames[1:],
                        )
                    }
                )
            ref = write_annotation(
                layout, layout.path("annotations", violation + ".json"), annotation
            )
            request = request.model_copy(
                update={
                    "annotations": (
                        saved.model_copy(update={"annotation": ref}),
                        *request.annotations[1:],
                    )
                }
            )
    with pytest.raises(ValueError, match=message):
        BallPublisher(layout).publish(request)
    target = (
        layout.repository
        / "data/ball_detection"
        / request.output.dataset_version
        / "frames"
    )
    if violation not in {"duplicate", "correction"}:
        assert not target.exists()
    assert all(
        layout.artifact_path(s.annotation).is_file() for s in request.annotations
    )


@pytest.mark.parametrize("violation", ["mask", "timeline", "weights", "recording"])
def test_active_ball_reader_rejects_contradictory_snapshot_contract(
    tmp_path: Path, violation: str
) -> None:
    env, request, base = incremental_case(tmp_path)
    # The initial v4 base already contains both observed and estimated positions.
    metadata_path = base / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    if violation == "mask":
        with np.load(base / "index.npz", allow_pickle=False) as archive:
            columns = dict(archive)
        columns["teacher_coordinate"][1] = 0  # A usable estimate may not be dropped.
        np.savez(base / "index.npz", **columns)
    else:
        if violation == "timeline":
            raw = next(iter(metadata["video_annotations"].values()))
            raw["frames"][1]["pts"] += 1
        elif violation == "weights":
            metadata["position_weights"]["estimated"] = 0.25
        else:
            metadata["clips"][0]["group_id"] = "another-recording"
        metadata_path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="active_ball"):
        BallFrameStore(base)
    assert request.output.base_snapshot is not None
    with pytest.raises(ValueError, match="hash changed|size changed"):
        load_snapshot(env.layout, request.output.base_snapshot)
