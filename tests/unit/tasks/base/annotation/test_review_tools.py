from __future__ import annotations

import base64
import json
from concurrent.futures import ThreadPoolExecutor
from fractions import Fraction
from pathlib import Path
from typing import Any, Literal

import av
import numpy as np
import pytest
from PIL import Image
from pydantic import ValidationError

from src.tasks.base.annotation.artifacts import (
    annotation_reference,
    read_document,
    write_asset,
    write_document,
)
from src.tasks.base.annotation.contracts import (
    AdoptionReceipt,
    ArtifactRef,
    ClipManifest,
    Document,
    FrameMap,
    FrameRange,
    ModelRef,
    ReviewEvidence,
    SourceIdentity,
    content_digest,
)
from src.tasks.base.annotation.editing import (
    ChangeResult,
    ChangeSummary,
    EditCommand,
    ReviewContext,
    Revision,
    RevisionConflict,
    RevisionStore,
)
from src.tasks.base.annotation.editing.contracts import DeltaItem, EditReceipt
from src.tasks.base.annotation.handoff import (
    Boundary,
    HandoffContent,
    SummaryRow,
    build_handoff,
    read_handoff,
)
from src.tasks.base.annotation.images import (
    image_to_native,
    native_to_image,
    presented_tiles,
    read_rendered,
    render_page,
)
from src.tasks.base.annotation.images.contracts import (
    Box,
    Crop,
    Point,
    RenderRequest,
    Tile,
)
from src.tasks.base.annotation.media import encode_video, probe_video
from src.tasks.base.annotation.paths import AnnotationLayout
from src.tasks.base.annotation.plan import CampaignPlan, save_plan
from src.tasks.base.annotation.predictions import PredictionContext, Predictions
from src.tasks.base.annotation.telemetry.capture import capture
from src.tasks.base.annotation.telemetry.contracts import RenderedImage
from src.tasks.base.annotation.workers.contracts import (
    ExitReceipt,
    Launch,
    PreparedWorker,
    WorkRequest,
)


class Payload(Document):
    values: tuple[int, ...]
    verified: tuple[bool, ...]
    notes: tuple[str, ...]


class Adapter:
    task = "fixture"

    def initialize(
        self, layout: AnnotationLayout, context: ReviewContext, manifest: ClipManifest
    ) -> Payload:
        count = len(manifest.frames)
        return Payload(
            values=tuple(range(count)),
            verified=tuple(i == 1 for i in range(count)),
            notes=("unresolved event",),
        )

    def validate(
        self,
        layout: AnnotationLayout,
        context: ReviewContext,
        manifest: ClipManifest,
        payload: Document,
    ) -> None:
        assert isinstance(payload, Payload)
        if (
            len(payload.values) != len(manifest.frames)
            or len(payload.verified) != len(manifest.frames)
            or any(v < 0 for v in payload.values)
        ):
            raise ValueError("invalid task rows")

    def apply(
        self,
        layout: AnnotationLayout,
        context: ReviewContext,
        manifest: ClipManifest,
        current_payload: ArtifactRef,
        command: EditCommand,
    ) -> ChangeResult:
        old = read_document(layout, current_payload, Payload)
        changes = json.loads(layout.artifact_path(command.changes).read_bytes())
        if (
            "derived_range" in changes
            and changes["derived_range"][0] < 3 < changes["derived_range"][1]
        ):
            raise ValueError("derived edit crosses a task event/ID boundary")
        values, verified = list(old.values), list(old.verified)
        examples = []
        changed = set()
        for update in changes["updates"]:
            i = update["frame"]
            if i in changed or not 0 <= i < len(values):
                raise ValueError("duplicate or invalid frame")
            if verified[i] and not update.get("acknowledged"):
                raise ValueError("cannot implicitly overwrite a verified observation")
            examples.append(
                DeltaItem(
                    frame_index=i,
                    row_id=str(i),
                    before=str(values[i]),
                    after=str(update["value"]),
                )
            )
            values[i], verified[i] = update["value"], True
            changed.add(i)
        payload = Payload(
            values=tuple(values),
            verified=tuple(verified),
            notes=() if changes.get("resolve_note") else old.notes,
        )
        return ChangeResult(
            payload=payload,
            summary=ChangeSummary(
                changed_ranges=tuple(
                    FrameRange(start=i, stop=i + 1) for i in sorted(changed)
                ),
                changed_rows=len(changed),
                examples=tuple(examples),
                unresolved_rows=verified.count(False),
            ),
        )

    def validate_adoption(
        self,
        layout: AnnotationLayout,
        context: ReviewContext,
        manifest: ClipManifest,
        revision: ArtifactRef,
        receipt: AdoptionReceipt,
    ) -> None:
        if (
            layout.artifact_path(receipt.teacher_mask).read_bytes()
            != b"fixture-only validated mask"
        ):
            raise ValueError("task teacher-mask validation failed")

    def retained_references(
        self,
        layout: AnnotationLayout,
        context: ReviewContext,
        revision: ArtifactRef,
        receipt: AdoptionReceipt,
    ) -> tuple[ArtifactRef, ...]:
        # Fixture labels contain scalars only; core lineage supplies their files.
        return ()

    def summarize(
        self,
        layout: AnnotationLayout,
        context: ReviewContext,
        manifest: ClipManifest,
        payload: ArtifactRef,
        scope: FrameRange,
        after_row: str | None,
    ) -> HandoffContent:
        data = read_document(layout, payload, Payload)
        rows = tuple(
            SummaryRow(
                frame_index=i,
                row_id=str(i),
                state="verified" if data.verified[i] else "unresolved",
                detail=str(data.values[i]),
            )
            for i in range(scope.start, scope.stop)
        )
        return HandoffContent(
            rows=rows,
            total_rows=len(rows),
            remaining_rows=0,
            next_row_cursor=None,
            reviewed_ranges=tuple(
                FrameRange(start=i, stop=i + 1)
                for i in range(scope.start, scope.stop)
                if data.verified[i]
            ),
            unresolved_ranges=tuple(
                FrameRange(start=i, stop=i + 1)
                for i in range(scope.start, scope.stop)
                if not data.verified[i]
            ),
            boundaries=(Boundary(frame_index=3, kind="fixture_event"),)
            if scope.start <= 3 <= scope.stop
            else (),
            next_actions=("inspect unresolved rows",),
        )


class PredictionAdapter:
    task = "fixture"

    def produce(
        self,
        layout: AnnotationLayout,
        context: PredictionContext,
        manifest: ClipManifest,
        directory: Path,
    ) -> ArtifactRef:
        return write_asset(
            layout,
            directory / "prediction.json",
            b'{"fixture":"unreviewed prediction"}',
        )

    def validate(
        self,
        layout: AnnotationLayout,
        context: PredictionContext,
        manifest: ClipManifest,
        result: ArtifactRef,
    ) -> None:
        assert layout.artifact_path(result).is_file()


class Environment:
    def __init__(
        self,
        root: Path,
        *,
        count: int = 8,
        real_video: bool = False,
        phase: Literal["preview", "production"] = "production",
    ) -> None:
        self.layout = AnnotationLayout.testing(root)
        media_path = self.layout.path("clips", "fixture", "media.mp4")
        media_path.parent.mkdir(parents=True)
        if real_video:
            frames = (
                (
                    av.VideoFrame.from_ndarray(
                        np.full((32, 64, 3), i * 20, dtype=np.uint8), format="rgb24"
                    ),
                    i * 3000,
                    3000,
                )
                for i in range(count)
            )
            encode_video(
                media_path,
                width=64,
                height=32,
                time_base=Fraction(1, 90000),
                rate=Fraction(30),
                frames=frames,
                crf=16,
                preset="ultrafast",
            )
            timeline = probe_video(media_path)
            pts, durations, time_base = (
                timeline.pts,
                timeline.durations,
                str(timeline.time_base),
            )
        else:
            media_path.write_bytes(b"synthetic video bytes")
            pts, durations, time_base = (
                tuple(i * 3000 for i in range(count)),
                (3000,) * count,
                "1/90000",
            )
        media = annotation_reference(self.layout, media_path)
        self.manifest = ClipManifest(
            source=SourceIdentity(
                provider="local",
                channel_id="fixture",
                video_id="fixture",
                media=media,
                origin_url="local:fixture",
            ),
            media=media,
            preparation_digest="a" * 64,
            width=64,
            height=32,
            time_base=time_base,
            nominal_fps="30",
            source_range=FrameRange(start=0, stop=count),
            target_range=FrameRange(start=1, stop=count - 1),
            frames=tuple(
                FrameMap(
                    frame_index=i,
                    source_frame_index=i,
                    source_pts=pts[i],
                    clip_pts=pts[i],
                    duration_pts=durations[i],
                )
                for i in range(count)
            ),
        )
        self.manifest_ref = write_document(
            self.layout, media_path.parent / "manifest.json", self.manifest
        )
        self.input = write_asset(
            self.layout,
            self.layout.path("campaigns", "input.txt"),
            b"fixture policy and model",
        )
        ref = self.input.model_dump()
        model = ModelRef(
            task="fixture",
            version="initial",
            training_weights=self.input,
            inference_weights=self.input,
            configuration=self.input,
        )
        self.plan = CampaignPlan.model_validate_json(
            json.dumps(
                {
                    "campaign_id": "review",
                    "phase": phase,
                    "annotation_root": str(root),
                    "counts": {
                        "channels": 1,
                        "videos_per_channel": 1,
                        "max_clips_per_video": 1,
                    },
                    "selection": ref,
                    "preparation": ref,
                    "code_commit": "a" * 40,
                    "code_diff_digest": "b" * 64,
                    "agent_model": "gpt-6.1-sol",
                    "agent_effort": "high",
                    "resources": {
                        "codex_workers": 1,
                        "additional_storage_bytes": 100_000_000,
                        "minimum_free_bytes": 1,
                        "total_pss_bytes": 1024**3,
                        "gpu_seconds": 60,
                        "stages": [
                            {
                                "stage": "review",
                                "kind": "codex",
                                "parallel": 1,
                                "threads_per_process": 1,
                            }
                        ],
                    },
                    "tasks": [
                        {
                            "task": "fixture",
                            "model": model.model_dump(),
                            "inference": ref,
                            "review": {
                                "policy": ref,
                                "preview_evidence": ref,
                                "decision": ref,
                            },
                            "learning": None,
                            "products": [
                                {
                                    "product": "frames",
                                    "dataset_task": "fixture",
                                    "dataset_version": "v1",
                                    "component": "frames",
                                    "base_snapshot": None,
                                    "generation_policy": ref,
                                }
                            ],
                        }
                    ],
                }
            )
        )
        self.plan_ref = annotation_reference(
            self.layout, save_plan(self.layout, self.plan)
        )
        predictions = Predictions(self.layout, self.plan)
        key = content_digest(["fixture", self.manifest.clip_id])
        with predictions.registry.transaction() as db:
            db.execute(
                "INSERT INTO sources(id,provider,channel_id,video_id,campaign_id,candidate,state) VALUES (?,?,?,?,?,?,?)",
                (
                    self.manifest.source.canonical_id,
                    "local",
                    "fixture",
                    "fixture",
                    "review",
                    "{}",
                    "prepared",
                ),
            )
            db.execute(
                "INSERT INTO clips VALUES (?,?,?,?,?)",
                (
                    self.manifest.clip_id,
                    self.manifest.source.canonical_id,
                    "review",
                    self.manifest_ref.canonical_bytes().decode(),
                    media.digest,
                ),
            )
            db.execute(
                "INSERT INTO annotation_tasks VALUES (?,?,?,?,?,?,?)",
                (
                    key,
                    "review",
                    "fixture",
                    self.manifest.clip_id,
                    model.sha256,
                    self.input.digest,
                    "pending",
                ),
            )
        self.prediction = predictions.run(self.manifest_ref, PredictionAdapter())
        self.context = ReviewContext(
            plan=self.plan_ref,
            task="fixture",
            manifest=self.manifest_ref,
            prediction=self.prediction,
        )
        self.adapter = Adapter()
        self.store = RevisionStore(self.layout)

    def initialize(self) -> tuple[ArtifactRef, Revision]:
        ref = self.store.initialize(self.context, self.adapter)
        return ref, read_document(self.layout, ref, Revision)

    def command(
        self, revision: ArtifactRef, updates: list[dict], **extras: Any
    ) -> EditCommand:
        previous = read_document(self.layout, revision, Revision)
        payload = json.dumps({"updates": updates, **extras}).encode()
        changes = write_asset(
            self.layout,
            self.layout.path(
                "campaigns", "changes", content_digest(payload.decode()) + ".json"
            ),
            payload,
        )
        return EditCommand(
            context=previous.context,
            expected_revision=revision,
            changes=changes,
            actor=self.input,
            reason="explicit fixture edit",
        )

    def image_request(
        self,
        *,
        annotation: ArtifactRef | None = None,
        tiles: tuple[Tile, ...] | None = None,
        drawing: str = "plain",
    ) -> RenderRequest:
        result: RenderRequest = RenderRequest.model_validate(
            {
                "manifest": self.manifest_ref,
                "annotation": annotation or self.prediction,
                "annotation_kind": "revision"
                if annotation is not None
                else "prediction",
                "drawing": drawing,
                "reason": "inspect local temporal context",
                "tiles": tiles
                or (
                    Tile(frame_index=2, crop=Crop(left=10, top=4, right=50, bottom=28)),
                ),
                "columns": 1,
                "tile_width": 160,
                "tile_height": 96,
            }
        )
        return result


@pytest.fixture
def env(tmp_path: Path) -> Environment:
    return Environment(tmp_path / "annotation")


def test_small_encoded_video_page_preserves_pts_crop_and_coordinate_conventions(
    tmp_path: Path,
) -> None:
    env = Environment(tmp_path / "annotation", real_video=True)
    request = env.image_request()
    rendered = render_page(env.layout, request)
    saved, receipt = read_rendered(env.layout, rendered)
    assert saved == request
    with Image.open(env.layout.artifact_path(rendered.image)) as image:
        assert image.size == (request.width, request.height)
    tile = receipt.tiles[0]
    assert (
        tile.clip_pts == env.manifest.frames[2].clip_pts
        and tile.time_base == env.manifest.time_base
    )
    edge = native_to_image(tile, 10.0, 4.0, coordinates="edge")
    center = native_to_image(tile, 10.0, 4.0, coordinates="pixel_center")
    assert edge == (tile.image_rect.left, tile.image_rect.top)
    assert center == (edge[0] + 1.5, edge[1] + 1.5)
    for kind in ("edge", "pixel_center"):
        point = native_to_image(tile, 23.25, 16.5, coordinates=kind)
        assert image_to_native(tile, *point, coordinates=kind) == pytest.approx(
            (23.25, 16.5)
        )
    with pytest.raises(ValueError, match="header, padding"):
        image_to_native(tile, 0.0, 0.0, coordinates="edge")
    assert render_page(env.layout, request) == rendered


def test_long_clip_only_converts_requested_frames_and_rejects_oversized_page(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from src.tasks.base.annotation.images import render

    env = Environment(tmp_path / "annotation", count=2500)
    converted: list[int] = []
    intervals = []

    class Frame:
        width, height, time_base = 64, 32, Fraction(1, 90000)

        def __init__(self, index: int) -> None:
            self.index, self.pts = index, index * 3000

        def to_ndarray(self, *, format: str) -> np.ndarray:
            assert format == "rgb24"
            converted.append(self.index)
            return np.full((32, 64, 3), self.index % 255, dtype=np.uint8)

    def decode(path: Path, timeline: Any, start: int, stop: int) -> Any:
        intervals.append((start, stop))
        yield from (Frame(i) for i in range(start, stop))

    monkeypatch.setattr(render, "decode_clip", decode)
    crop = Crop(left=0, top=0, right=64, bottom=32)
    request = env.image_request(
        tiles=tuple(Tile(frame_index=i, crop=crop) for i in (4, 100, 2300))
    )
    result = render_page(env.layout, request)
    assert converted == [4, 100, 2300] and intervals == [(4, 2301)]
    assert len(read_rendered(env.layout, result)[1].tiles) == 3
    with pytest.raises(ValidationError):
        RenderRequest.model_validate(
            {**request.model_dump(), "tiles": (request.tiles[0],) * 17}
        )
    with pytest.raises(ValidationError, match="pixel budget"):
        RenderRequest.model_validate(
            {
                **request.model_dump(),
                "tiles": (request.tiles[0],) * 16,
                "columns": 4,
                "tile_width": 1024,
                "tile_height": 1024,
            }
        )


def test_drawings_and_annotation_versions_are_distinct_inputs(tmp_path: Path) -> None:
    env = Environment(tmp_path / "annotation", real_video=True)
    initial, _ = env.initialize()
    raw = render_page(env.layout, env.image_request(annotation=initial))
    crop = Crop(left=0, top=0, right=64, bottom=32)
    tile = Tile(
        frame_index=2,
        crop=crop,
        drawings=(
            Point(x=20.0, y=10.0),
            Box(left=30.0, top=5.0, right=50.0, bottom=25.0),
        ),
    )
    marked = render_page(
        env.layout,
        env.image_request(annotation=initial, tiles=(tile,), drawing="revision"),
    )
    assert raw != marked and raw.image.digest != marked.image.digest
    with pytest.raises(ValidationError, match="markers"):
        env.image_request(tiles=(tile,))
    outside = Tile(frame_index=2, crop=Crop(left=0, top=0, right=65, bottom=32))
    with pytest.raises(ValueError, match="outside the native"):
        render_page(env.layout, env.image_request(tiles=(outside,)))
    env.layout.artifact_path(env.manifest.media).write_bytes(b"changed video")
    with pytest.raises(ValueError, match="changed"):
        render_page(env.layout, env.image_request(annotation=initial))


def test_edits_are_atomic_versioned_idempotent_and_keep_resolved_note_history(
    env: Environment,
) -> None:
    initial, revision = env.initialize()
    original = env.layout.artifact_path(revision.payload).read_bytes()
    command = env.command(
        initial,
        [{"frame": 2, "value": 20}, {"frame": 4, "value": 40}],
        resolve_note=True,
    )
    result = env.store.apply(command, env.adapter)
    receipt = read_document(env.layout, result, EditReceipt)
    current = read_document(env.layout, receipt.revision, Revision)
    assert current.sequence == 1 and current.previous == initial
    assert read_document(env.layout, current.payload, Payload).notes == ()
    assert env.layout.artifact_path(revision.payload).read_bytes() == original
    assert env.store.apply(command, env.adapter) == result
    assert env.store.initialize(env.context, env.adapter) == receipt.revision
    with pytest.raises(RevisionConflict, match="changed"):
        env.store.apply(env.command(initial, [{"frame": 5, "value": 50}]), env.adapter)


@pytest.mark.parametrize(
    "changes",
    [
        {"updates": [{"frame": 2, "value": 20}, {"frame": 1, "value": 10}]},
        {"updates": [{"frame": 2, "value": 20}, {"frame": 4, "value": -1}]},
        {"updates": [{"frame": 2, "value": 20}], "derived_range": [2, 5]},
    ],
)
def test_failed_batch_never_leaves_a_partial_revision(
    env: Environment, changes: dict
) -> None:
    initial, revision = env.initialize()
    updates = changes["updates"]
    command = env.command(
        initial, updates, **{k: v for k, v in changes.items() if k != "updates"}
    )
    with pytest.raises(ValueError):
        env.store.apply(command, env.adapter)
    assert env.store.current(revision.context, env.adapter) == (initial, None)
    assert read_document(env.layout, revision.payload, Payload).values[2] == 2


def test_parallel_edits_only_commit_one_expected_revision(env: Environment) -> None:
    initial, revision = env.initialize()
    commands = [env.command(initial, [{"frame": i, "value": i * 10}]) for i in (2, 4)]

    def edit(command: EditCommand) -> ArtifactRef | None:
        try:
            result: ArtifactRef = RevisionStore(env.layout).apply(command, env.adapter)
            return result
        except (RevisionConflict, BlockingIOError):
            return None

    with ThreadPoolExecutor(max_workers=2) as pool:
        outcomes = list(pool.map(edit, commands))
    assert sum(ref is not None for ref in outcomes) == 1
    head, _ = env.store.current(revision.context, env.adapter)
    assert read_document(env.layout, head, Revision).sequence == 1


def test_edit_commit_receipt_gap_is_recoverable(
    env: Environment, monkeypatch: pytest.MonkeyPatch
) -> None:
    from src.tasks.base.annotation.editing import store

    initial, revision = env.initialize()
    command = env.command(initial, [{"frame": 2, "value": 20}])
    original = store.write_document

    class Crash(BaseException):
        pass

    def crash(layout: AnnotationLayout, path: Path, document: Document) -> ArtifactRef:
        if path.name == "receipt.json":
            raise Crash("committed head; receipt not written")
        result: ArtifactRef = original(layout, path, document)
        return result

    monkeypatch.setattr(store, "write_document", crash)
    with pytest.raises(Crash):
        env.store.apply(command, env.adapter)
    head, _ = env.store.current(revision.context, env.adapter)
    assert read_document(env.layout, head, Revision).sequence == 1
    monkeypatch.setattr(store, "write_document", original)
    receipt = read_document(
        env.layout, RevisionStore(env.layout).apply(command, env.adapter), EditReceipt
    )
    assert receipt.revision == head
    assert env.store.current(revision.context, env.adapter)[0] == head


def test_adoption_requires_task_validation_then_seals_the_exact_revision(
    env: Environment,
) -> None:
    initial, revision = env.initialize()
    evidence = write_document(
        env.layout,
        env.layout.path("campaigns", "review-evidence.json"),
        ReviewEvidence(
            clip_id=env.manifest.clip_id,
            prediction=env.prediction,
            revision=initial,
            presented_ranges=(),
            reviewed_ranges=(),
            unresolved_ranges=(),
            evidence=(),
        ),
    )
    mask = write_asset(
        env.layout,
        env.layout.path("campaigns", "teacher-mask.bin"),
        b"fixture-only validated mask",
    )

    def adoption(teacher_mask: ArtifactRef) -> ArtifactRef:
        value = AdoptionReceipt(
            task="fixture",
            clip_id=env.manifest.clip_id,
            revision=initial,
            review_evidence=evidence,
            policy=env.input,
            teacher_mask=teacher_mask,
            reason="fixture adapter policy",
        )
        return write_document(
            env.layout, env.layout.path("campaigns", value.sha256 + ".json"), value
        )

    with pytest.raises(ValueError, match="teacher-mask validation failed"):
        env.store.seal(initial, adoption(env.input), env.adapter)
    accepted = adoption(mask)
    assert env.store.seal(initial, accepted, env.adapter) == accepted
    assert env.store.seal(initial, accepted, env.adapter) == accepted
    assert env.store.current(revision.context, env.adapter) == (initial, accepted)
    with pytest.raises(RevisionConflict, match="sealed"):
        env.store.apply(env.command(initial, [{"frame": 2, "value": 20}]), env.adapter)


def test_handoff_uses_current_bounded_rows_and_paged_history(env: Environment) -> None:
    initial, revision = env.initialize()
    first = read_document(
        env.layout,
        env.store.apply(
            env.command(initial, [{"frame": 2, "value": 20}], resolve_note=True),
            env.adapter,
        ),
        EditReceipt,
    )
    second = read_document(
        env.layout,
        env.store.apply(
            env.command(first.revision, [{"frame": 4, "value": 40}]), env.adapter
        ),
        EditReceipt,
    )
    handoff_ref = build_handoff(
        env.store,
        revision.context,
        env.adapter,
        FrameRange(start=1, stop=6),
        previous_revision=initial,
    )
    handoff = read_handoff(env.layout, handoff_ref)
    assert handoff.revision == second.revision and handoff.adoption is None
    rows = {row.frame_index: row for row in handoff.content.rows}
    assert (
        rows[2].state == rows[4].state == "verified" and rows[3].state == "unresolved"
    )
    assert rows[2].detail == "20" and len(rows) == 5
    assert len(handoff.commands) == 2 and handoff.history_cursor is None
    commands, cursor = env.store.changes_since(second.revision, initial, limit=1)
    assert len(commands) == 1 and cursor == first.revision
    assert env.store.changes_since(cursor, initial, limit=1)[1] is None
    assert read_document(env.layout, revision.payload, Payload).notes == (
        "unresolved event",
    )
    assert (
        build_handoff(
            env.store,
            revision.context,
            env.adapter,
            FrameRange(start=1, stop=6),
            previous_revision=initial,
        )
        == handoff_ref
    )
    env.layout.artifact_path(
        read_document(env.layout, second.revision, Revision).payload
    ).write_bytes(b"tampered")
    with pytest.raises(ValueError, match="changed"):
        read_handoff(env.layout, handoff_ref)


def test_preview_or_unregistered_prediction_cannot_start_editing(
    env: Environment,
) -> None:
    preview = env.plan.model_copy(update={"phase": "preview"})
    preview_ref = write_document(
        env.layout, env.layout.path("campaigns", "preview.json"), preview
    )
    with pytest.raises(ValueError, match="production task"):
        env.store.initialize(
            env.context.model_copy(update={"plan": preview_ref}), env.adapter
        )
    with env.store.registry.transaction() as db:
        db.execute("DELETE FROM annotation_predictions")
    with pytest.raises(ValueError, match="registered prediction"):
        env.store.initialize(env.context, env.adapter)


@pytest.mark.parametrize("registered_phase", ["production", "preview"])
def test_review_rejects_replacement_of_registered_campaign_plan(
    tmp_path: Path, registered_phase: Literal["preview", "production"]
) -> None:
    env = Environment(tmp_path / "annotation", phase=registered_phase)
    replacement = env.plan.model_copy(update={"phase": "production"})
    if registered_phase == "production":
        policy = write_asset(
            env.layout,
            env.layout.path("campaigns", "replacement-policy.txt"),
            b"different teacher adoption rules",
        )
        task = env.plan.tasks[0]
        assert task.review is not None
        replacement = replacement.model_copy(
            update={
                "tasks": (
                    task.model_copy(
                        update={"review": task.review.model_copy(update={"policy": policy})}
                    ),
                )
            }
        )
    replacement_ref = write_document(
        env.layout, env.layout.path("campaigns", "replacement-plan.json"), replacement
    )
    with pytest.raises(ValueError, match="registered campaign plan"):
        env.store.initialize(
            env.context.model_copy(update={"plan": replacement_ref}), env.adapter
        )
    with env.store.registry.transaction() as db:
        assert db.execute("SELECT * FROM annotation_revision_heads").fetchone() is None
        row = db.execute("SELECT digest FROM campaigns WHERE id='review'").fetchone()
        assert row["digest"] == env.plan.sha256


def test_handoff_final_row_page_can_finish_without_repeating_full_scope(
    env: Environment,
) -> None:
    _, revision = env.initialize()

    class PagedAdapter(Adapter):
        def summarize(
            self,
            layout: AnnotationLayout,
            context: ReviewContext,
            manifest: ClipManifest,
            payload: ArtifactRef,
            scope: FrameRange,
            after_row: str | None,
        ) -> HandoffContent:
            all_rows = super().summarize(
                layout, context, manifest, payload, scope, after_row
            )
            offset = int(after_row or "0")
            rows = all_rows.rows[offset : offset + 2]
            remaining = max(0, len(all_rows.rows) - offset - len(rows))
            result: HandoffContent = all_rows.model_copy(
                update={
                    "rows": rows,
                    "remaining_rows": remaining,
                    "next_row_cursor": str(offset + len(rows)) if remaining else None,
                }
            )
            return result

    adapter = PagedAdapter()
    first = read_handoff(
        env.layout,
        build_handoff(
            env.store, revision.context, adapter, FrameRange(start=1, stop=5)
        ),
    )
    assert first.content.total_rows == 4 and first.content.remaining_rows == 2
    assert first.content.next_row_cursor == "2"
    last = read_handoff(
        env.layout,
        build_handoff(
            env.store,
            revision.context,
            adapter,
            FrameRange(start=1, stop=5),
            after_row="2",
            expected_revision=first.revision,
        ),
    )
    assert (
        last.content.total_rows == 4
        and last.content.remaining_rows == 0
        and last.content.next_row_cursor is None
    )
    assert [row.frame_index for row in first.content.rows + last.content.rows] == [
        1,
        2,
        3,
        4,
    ]
    env.store.apply(
        env.command(first.revision, [{"frame": 4, "value": 40}]), env.adapter
    )
    with pytest.raises(RevisionConflict, match="handoff revision changed"):
        build_handoff(
            env.store,
            revision.context,
            adapter,
            FrameRange(start=1, stop=5),
            after_row="2",
            expected_revision=first.revision,
        )


def telemetry_for(
    env: Environment, rendered: RenderedImage, *, present: bool
) -> ArtifactRef:
    # Synthetic rollout through the real capture/archive path. No model API call.
    directory = env.layout.path(
        "campaigns", "review", "workers", "shown" if present else "unshown"
    )
    request = WorkRequest(
        adapter="fixture",
        adapter_version="v1",
        stage="review",
        subject_id=env.manifest.clip_id,
        payload=env.input,
        inputs=(env.manifest_ref,),
    )
    launch = Launch(
        work_id=request.work_id,
        attempt="shown" if present else "unshown",
        plan=env.plan_ref,
        request=request,
        prepared=PreparedWorker(prompt=env.input),
        directory=str(directory),
        code_root=str(env.layout.repository),
        codex_binary="codex",
        codex_home=str(env.layout.repository / "codex"),
        timeout_seconds=60.0,
        test_root=str(env.layout.root),
    )
    launch_ref = write_document(env.layout, directory / "launch.json", launch)
    execution = ExitReceipt(
        launch_digest=launch_ref.digest,
        exit_code=0,
        status="succeeded",
        started_at="2026-01-01T00:00:00Z",
        ended_at="2026-01-01T00:00:01Z",
        elapsed_seconds=1.0,
        events=env.input,
        errors=env.input,
        thread_id="11111111-1111-1111-1111-111111111111",
        reason=None,
    )
    execution_ref = write_document(env.layout, directory / "exit.json", execution)

    def event(kind: str, **payload: Any) -> dict:
        return {"type": kind, "timestamp": "2026-01-01T00:00:00Z", "payload": payload}

    events = [
        event(
            "session_meta",
            id=execution.thread_id,
            cwd=str(directory / "work"),
            cli_version="fixture",
            model_provider="openai",
        ),
        event("turn_context", model=env.plan.agent_model),
    ]
    if present:
        data = env.layout.artifact_path(rendered.image).read_bytes()
        events += [
            event(
                "response_item",
                type="custom_tool_call",
                call_id="view",
                name="view_image",
                input="rendered page",
            ),
            event(
                "response_item",
                type="custom_tool_call_output",
                call_id="view",
                output=[
                    {
                        "type": "input_image",
                        "detail": "original",
                        "image_url": "data:image/jpeg;base64,"
                        + base64.b64encode(data).decode(),
                    }
                ],
            ),
        ]
    events += [
        event(
            "token_usage_record",
            response_id="response-1",
            usage={"input_tokens": 100, "output_tokens": 10, "total_tokens": 110},
        )
    ]
    rollout = directory / "rollout.jsonl"
    rollout.write_text("".join(json.dumps(e) + "\n" for e in events))
    report: ArtifactRef = capture(
        env.layout, launch_ref, execution_ref, (rendered,), rollout=rollout
    )
    return report


def test_viewed_but_unresolved_frames_remain_distinct_in_handoff(
    env: Environment,
) -> None:
    _, revision = env.initialize()

    class ViewedUnresolved(Adapter):
        def summarize(
            self,
            layout: AnnotationLayout,
            context: ReviewContext,
            manifest: ClipManifest,
            payload: ArtifactRef,
            scope: FrameRange,
            after_row: str | None,
        ) -> HandoffContent:
            value = super().summarize(
                layout, context, manifest, payload, scope, after_row
            )
            result: HandoffContent = value.model_copy(
                update={"reviewed_ranges": (scope,), "unresolved_ranges": (scope,)}
            )
            return result

    handoff = read_handoff(
        env.layout,
        build_handoff(
            env.store, revision.context, ViewedUnresolved(), FrameRange(start=2, stop=3)
        ),
    )
    assert (
        handoff.content.reviewed_ranges
        == handoff.content.unresolved_ranges
        == (FrameRange(start=2, stop=3),)
    )
    assert handoff.content.rows[0].state == "unresolved" and handoff.adoption is None


def test_generated_and_presented_images_do_not_mark_reviewed_rows(
    tmp_path: Path,
) -> None:
    env = Environment(tmp_path / "annotation", real_video=True)
    initial, revision = env.initialize()
    before = env.layout.artifact_path(revision.payload).read_bytes()
    image = render_page(env.layout, env.image_request(annotation=initial))
    unshown = telemetry_for(env, image, present=False)
    with pytest.raises(ValueError, match="actual model response"):
        presented_tiles(
            env.layout, image, unshown, manifest=env.manifest_ref, annotation=initial
        )
    shown = telemetry_for(env, image, present=True)
    tiles = presented_tiles(
        env.layout, image, shown, manifest=env.manifest_ref, annotation=initial
    )
    assert [tile.frame_index for tile in tiles] == [2]
    with pytest.raises(ValueError, match="another clip or annotation"):
        presented_tiles(
            env.layout,
            image,
            shown,
            manifest=env.manifest_ref,
            annotation=env.prediction,
        )
    assert env.layout.artifact_path(revision.payload).read_bytes() == before
    assert env.store.current(revision.context, env.adapter) == (initial, None)
