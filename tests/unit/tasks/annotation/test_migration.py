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
from src.tasks.ball_detection.generate_dataset.frame_store.sources.chat_annotation import (
    ChatAnnotationSourceConfig,
    collect_chat_annotation,
)
from src.tasks.base.annotation.paths import AnnotationLayout
from src.tasks.base.annotation.storage.migration import (
    MigrationStore,
    dry_run,
    imported_root,
)
from src.tasks.player_detection.generate_dataset.sources import collect_clip_sources
from src.tennis_scene.chat_annotation.configuration import PrepareConfig
from src.tennis_scene.chat_annotation.layout import done_video_path, video_path
from src.tennis_scene.chat_annotation.manifests import load_prepared_manifests
from src.tennis_scene.chat_annotation.preparation import prepare
from src.tennis_scene.chat_annotation.runtime.contracts import (
    Policies,
    make_template,
    read_json,
    write_json,
)


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
    run = prepare(
        PrepareConfig(
            url=None,
            urls=(),
            local_video=source,
            format_selector="unused",
            js_runtimes=None,
            remote_components=None,
            output=root,
            duration_seconds=2.0,
            context_seconds=0.2,
            max_bytes=1000000,
            crf=18,
            preset="ultrafast",
            policies=Policies(ball_max_gap_seconds=0.2),
            download_workers=1,
            max_clips_per_video=2,
            sampling_strategy="uniform_midpoints",
        )
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
        for path in annotations:
            if "/ball/" in path:
                archive.write(root / path, Path(path).name)
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
