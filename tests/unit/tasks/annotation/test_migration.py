from __future__ import annotations

import json
import zipfile
from fractions import Fraction
from pathlib import Path
from typing import Any

import av
import numpy as np
import pytest

from src.tasks.annotation.migration import LegacyReader, LegacySelection, LegacySource
from src.tasks.annotation.migration.legacy_format.contracts import (
    ClipManifest,
    FrameMap,
    FrameRange,
    Policies,
    SourceInfo,
    read_json,
    sha256_file,
)
from src.tasks.annotation.migration.legacy_format.layout import (
    done_video_path,
    video_path,
)
from src.tasks.annotation.migration.legacy_format.manifests import (
    load_prepared_manifests,
)
from src.tasks.ball_detection.generate_dataset.frame_store.sources.chat_annotation import (
    ChatAnnotationSourceConfig,
    collect_chat_annotation,
)
from src.tasks.base.annotation.media import decode_range, encode_video, probe_video
from src.tasks.base.annotation.paths import AnnotationLayout
from src.tasks.base.annotation.storage.migration import (
    MigrationStore,
    dry_run,
    imported_root,
)
from src.tasks.player_detection.generate_dataset.sources import collect_clip_sources
from tests.support.legacy_annotations import make_template, write_json


@pytest.fixture
def legacy(tmp_path: Path) -> tuple[Path, LegacyReader]:
    source = tmp_path / "synthetic.mp4"
    with av.open(str(source), "w") as container:
        stream = container.add_stream("libx264", rate=5)
        stream.width, stream.height, stream.pix_fmt = 96, 64, "yuv420p"
        for index in range(20):
            frame = av.VideoFrame.from_ndarray(
                np.full((64, 96, 3), index * 7, dtype=np.uint8), format="rgb24"
            )
            frame.pts, frame.time_base = index, Fraction(1, 5)
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    root = tmp_path / "legacy"
    # Retained v2 artifacts are fixtures, not output from a retired writer CLI.
    timeline = probe_video(source)
    info = SourceInfo(
        source_id="synthetic",
        youtube_id=None,
        url=None,
        title="Synthetic source",
        filename=source.name,
        sha256=sha256_file(source),
        bytes=source.stat().st_size,
        acquired_at="fixture",
    )
    record = root / "sources/synthetic/source_record.json"
    write_json(record, info.model_dump())
    (record.parent / source.name).write_bytes(source.read_bytes())
    (root / "project_kits").mkdir()
    run = root / "_preparation/synthetic/fixture"
    intervals = [FrameRange(start=0, stop=10), FrameRange(start=10, stop=20)]
    names = []
    for index, interval in enumerate(intervals):
        name = f"clip_{index}"
        names.append(name)
        media = root / "videos/synthetic" / f"synthetic__fixture__{name}.mp4"
        media.parent.mkdir(parents=True, exist_ok=True)
        frames = decode_range(source, timeline, interval.start, interval.stop)
        encode_video(
            media,
            width=timeline.width,
            height=timeline.height,
            time_base=timeline.time_base,
            rate=timeline.rate,
            crf=18,
            preset="ultrafast",
            frames=(
                (
                    frame,
                    timeline.pts[i] - timeline.pts[interval.start],
                    timeline.durations[i],
                )
                for i, frame in enumerate(frames, interval.start)
            ),
        )
        manifest = ClipManifest(
            schema_version="tennis_chat_clip.v1",
            kit_version="5.0.0",
            kit_id="c" * 64,
            clip_id=name,
            source=info,
            filename=media.name,
            sha256=sha256_file(media),
            bytes=media.stat().st_size,
            width=timeline.width,
            height=timeline.height,
            time_base=str(timeline.time_base),
            nominal_fps=str(timeline.rate),
            source_start_pts=timeline.pts[0],
            media_range=interval,
            target_range=interval,
            policies=Policies(ball_max_gap_seconds=0.2),
            frames=[
                FrameMap(
                    frame_index=i - interval.start,
                    source_frame_index=i,
                    source_pts=timeline.pts[i],
                    clip_pts=timeline.pts[i] - timeline.pts[interval.start],
                    duration_pts=timeline.durations[i],
                    is_target=True,
                )
                for i in range(interval.start, interval.stop)
            ],
        )
        path = run / "clips" / name / "clip_manifest.json"
        write_json(path, manifest.model_dump())
        write_json(
            run / "ready" / f"{name}.json",
            {
                "files": {
                    "clip_manifest.json": sha256_file(path),
                    media.name: manifest.sha256,
                }
            },
        )
    write_json(
        run / "prepared.json",
        {
            "schema_version": "tennis_chat_preparation.v2",
            "source_sha256": info.sha256,
            "kit_id": "c" * 64,
            "settings": {"sampling": {"max_clips_per_video": 2}, "max_bytes": 1000000},
            "requested_target_ranges": [
                interval.model_dump() for interval in intervals
            ],
            "selected_target_ranges": [interval.model_dump() for interval in intervals],
            "candidate_clip_count": 2,
            "clips": names,
            "source_frame_count": 20,
            "selected_frame_count": 20,
            "coverage_mode": "full",
        },
    )
    manifests = load_prepared_manifests(root)
    annotations = []
    for index, (clip_id, manifest) in enumerate(manifests.items()):
        if index == 0:
            completed = done_video_path(root, manifest)
            completed.parent.mkdir(parents=True)
            video_path(root, manifest).rename(completed)
        for target in ("ball", "player"):
            annotation = make_template(manifest, target).model_dump(mode="json")
            annotation["status"] = "completed"
            for row in annotation["frames"]:
                row["reviewed"] = True
            relative = f"annotated/processed/{target}/{clip_id}.json"
            write_json(root / relative, annotation)
            annotations.append(relative)
    raw = "annotated/raw/both-clips.zip"
    (root / raw).parent.mkdir(parents=True)
    with zipfile.ZipFile(root / raw, "w", zipfile.ZIP_DEFLATED) as archive:
        for annotation_path in annotations:
            if "/ball/" in annotation_path:
                archive.write(root / annotation_path, Path(annotation_path).name)
    record = next((root / "sources").glob("*/source_record.json"))
    original = record.parent / read_json(record)["filename"]
    reader = LegacyReader(
        LegacySelection(
            prepared_runs=(run.relative_to(root).as_posix(),),
            sources=(
                LegacySource(
                    record=record.relative_to(root).as_posix(),
                    video=original.relative_to(root).as_posix(),
                ),
            ),
            annotations=tuple(annotations),
            raw_zips=(raw,),
            max_clips=10,
            max_json_bytes=1024 * 1024,
        )
    )
    return root, reader


def test_full_bundle_preserves_both_clip_ids_and_existing_task_readers(
    tmp_path: Path, legacy: tuple[Path, LegacyReader]
) -> None:
    source, reader = legacy
    layout = AnnotationLayout.testing(tmp_path / "annotation")
    plan = dry_run(
        layout,
        reader,
        root_id="legacy-pilot",
        source_root=source,
        mode="copy",
        max_files=64,
    )
    MigrationStore(layout).apply(
        plan, reader, max_copy_bytes=plan.copy_payload_bytes, max_files=64
    )
    destination = imported_root(layout, plan.root_id, reader)
    old_manifests = load_prepared_manifests(source)
    assert load_prepared_manifests(destination) == old_manifests
    assert all(key != value.clip_id for key, value in old_manifests.items())
    for root in (source, destination):
        balls = collect_chat_annotation(
            ChatAnnotationSourceConfig(root, frozenset({"completed"}))
        )
        players = collect_clip_sources(root, allowed_statuses=frozenset({"completed"}))
        assert [
            spec.clip_id.removeprefix("chat_annotation/") for spec in balls
        ] == sorted(old_manifests)
        assert [item.clip_id for item in players] == sorted(old_manifests)
    for item in plan.bundle.files:
        assert (source / item.path).read_bytes() == (
            destination / item.path
        ).read_bytes()
    zips = [item for item in plan.bundle.files if item.path.endswith(".zip")]
    assert len(zips) == 1
    assert len(zips[0].dependencies) == 4  # Two complete clips; no ZIP splitting.
    assert (
        source / "project_kits"
    ).exists()  # Unselected legacy material remains untouched.


def test_existing_fixed_root_is_registered_without_copying(
    legacy: tuple[Path, LegacyReader],
) -> None:
    root, reader = legacy
    layout = AnnotationLayout.testing(root)
    plan = dry_run(
        layout,
        reader,
        root_id="existing",
        source_root=root,
        mode="in_place",
        max_files=64,
    )
    before = {item.path: (root / item.path).stat().st_ino for item in plan.bundle.files}
    MigrationStore(layout).apply(plan, reader, max_copy_bytes=0, max_files=64)
    assert imported_root(layout, "existing", reader) == root
    assert not layout.path("migration_archive").exists()
    assert before == {
        item.path: (root / item.path).stat().st_ino for item in plan.bundle.files
    }


@pytest.mark.parametrize(
    "change", ["summary", "receipt", "original", "clip", "annotation"]
)
def test_inconsistent_legacy_evidence_is_rejected(
    legacy: tuple[Path, LegacyReader], change: str
) -> None:
    root, reader = legacy
    reader.inspect(root)
    run = root / reader.selection.prepared_runs[0]
    if change == "summary":
        path = run / "prepared.json"
        value = read_json(path)
        value["selected_frame_count"] += 1
        write_json(path, value)
    elif change == "receipt":
        path = next((run / "ready").glob("*.json"))
        value = read_json(path)
        value["files"]["clip_manifest.json"] = "0" * 64
        write_json(path, value)
    elif change == "original":
        (root / reader.selection.sources[0].video).write_bytes(b"different original")
    elif change == "clip":
        manifest = next(iter(load_prepared_manifests(root).values()))
        done_video_path(root, manifest).write_bytes(b"different clip")
    else:
        path = root / reader.selection.annotations[0]
        value = read_json(path)
        value["frames"].pop()
        write_json(path, value)
    with pytest.raises(ValueError):
        reader.inspect(root)


def test_raw_zip_cannot_omit_referenced_clip(legacy: tuple[Path, LegacyReader]) -> None:
    root, reader = legacy
    value = read_json(root / reader.selection.annotations[0])
    value["clip_id"] = "another-clip"
    with zipfile.ZipFile(root / reader.selection.raw_zips[0], "w") as archive:
        archive.writestr("another.json", json.dumps(value))
    with pytest.raises(ValueError, match="outside the selected runs"):
        reader.inspect(root)


def test_symlinked_evidence_and_ambiguous_video_are_rejected(
    legacy: tuple[Path, LegacyReader],
) -> None:
    root, reader = legacy
    manifest = next(iter(load_prepared_manifests(root).values()))
    pending = video_path(root, manifest)
    pending.parent.mkdir(parents=True, exist_ok=True)
    pending.write_bytes(done_video_path(root, manifest).read_bytes())
    with pytest.raises(ValueError, match="both videos and done"):
        reader.inspect(root)
    pending.unlink()
    path = root / reader.selection.annotations[0]
    saved = path.with_suffix(".preserved")
    path.rename(saved)
    path.symlink_to(saved)
    with pytest.raises((ValueError, OSError)):
        reader.inspect(root)


def test_reader_limits_and_selection_identity_are_pinned(
    legacy: tuple[Path, LegacyReader],
) -> None:
    root, reader = legacy
    with pytest.raises(ValueError, match="clip limit"):
        LegacyReader(reader.selection.model_copy(update={"max_clips": 1})).inspect(root)
    with pytest.raises(ValueError, match="size limit"):
        LegacyReader(reader.selection.model_copy(update={"max_json_bytes": 1})).inspect(
            root
        )
    reduced = LegacyReader(
        reader.selection.model_copy(update={"annotations": (), "raw_zips": ()})
    )
    assert (
        reduced.inspect(root).selection_digest != reader.inspect(root).selection_digest
    )


def test_unlisted_clip_does_not_get_silently_dropped(
    legacy: tuple[Path, LegacyReader],
) -> None:
    root, reader = legacy
    (root / reader.selection.prepared_runs[0] / "clips" / "unexpected").mkdir()
    with pytest.raises(ValueError, match="unlisted or missing clips"):
        reader.inspect(root)


@pytest.mark.parametrize(
    "field,path",
    [
        ("prepared_runs", "../outside"),
        ("annotations", "raw/file.json"),
        ("raw_zips", "file.json"),
    ],
)
def test_selection_rejects_wrong_boundaries(
    legacy: tuple[Path, LegacyReader], field: str, path: str
) -> None:
    _, reader = legacy
    value: dict[str, Any] = reader.selection.model_dump()
    value[field] = (path,)
    with pytest.raises(ValueError):
        LegacySelection.model_validate(value)
