from __future__ import annotations

import io
import json
import os
import zipfile
from pathlib import Path

import pytest

from src.tennis_scene.chat_annotation.artifacts.completion import sync_done
from src.tennis_scene.chat_annotation.artifacts.store import ArtifactStore, inspect_zip
from src.tennis_scene.chat_annotation.kit import build_kit
from src.tennis_scene.chat_annotation.layout import done_video_path, video_path
from src.tennis_scene.chat_annotation.preparation import _verify_published
from src.tennis_scene.chat_annotation.runtime.contracts import (
    ClipManifest,
    make_template,
    sha256_file,
    write_json,
)


def zip_bytes(members: dict[str, bytes]) -> bytes:
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, data in members.items():
            archive.writestr(name, data)
    return stream.getvalue()


def test_raw_is_immutable_idempotent_and_retains_distinct_submissions(
    tmp_path: Path,
) -> None:
    store = ArtifactStore(tmp_path)
    data = zip_bytes({"clip__ball.json": b'{"status": "partial"}'})
    receipt = store.save("batch.zip", data)
    assert (tmp_path / receipt["artifact_id"]).read_bytes() == data
    assert store.save("other.zip", data)["created"] is False
    second = store.save(
        "batch.zip", zip_bytes({"clip__ball.json": b'{"status":"completed"}'})
    )
    assert second["artifact_id"] != receipt["artifact_id"]
    assert len(store.list()["artifacts"]) == 2
    assert store.read(receipt["artifact_id"])["members"] == ["clip__ball.json"]
    (tmp_path / receipt["artifact_id"]).write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="corrupt"):
        store.save("batch.zip", data)
    with pytest.raises(ValueError, match="checksum"):
        store.read(receipt["artifact_id"])


@pytest.mark.parametrize(
    "members",
    [
        {},
        {"../ball.json": b"{}"},
        {"/ball.json": b"{}"},
        {"sub/ball.json": b"{}"},
        {"overlay.mp4": b"video"},
        {"ball.json": b'{"x":1,"x":2}'},
        {"ball.json": b'{"x": NaN}'},
        {"ball.json": b"[]"},
        {"ball.json": b"invalid"},
    ],
)
def test_bad_zip_members_never_publish(
    tmp_path: Path, members: dict[str, bytes]
) -> None:
    with pytest.raises(ValueError):
        ArtifactStore(tmp_path).save("batch.zip", zip_bytes(members))
    assert list(tmp_path.iterdir()) == []


def test_duplicate_and_symlink_members_rejected() -> None:
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w") as archive:
        archive.writestr("ball.json", "{}")
        with pytest.warns(UserWarning):
            archive.writestr("ball.json", "{}")
    with pytest.raises(ValueError, match="duplicate"):
        inspect_zip(stream.getvalue())
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w") as archive:
        info = zipfile.ZipInfo("ball.json")
        info.external_attr = 0o120777 << 16
        archive.writestr(info, "{}")
    with pytest.raises(ValueError, match="symlink"):
        inspect_zip(stream.getvalue())


def prepare(root: Path, manifest: ClipManifest) -> Path:
    video = video_path(root, manifest)
    video.parent.mkdir(parents=True)
    video.write_bytes(b"original clip")
    manifest.sha256 = sha256_file(video)
    manifest.bytes = video.stat().st_size
    directory: Path = (
        root / "_preparation" / "sample" / "run" / "clips" / manifest.clip_id
    )
    path = directory / "clip_manifest.json"
    write_json(path, manifest.model_dump())
    write_json(
        directory.parents[1] / "ready" / f"{manifest.clip_id}.json",
        {
            "files": {
                "clip_manifest.json": sha256_file(path),
                manifest.filename: manifest.sha256,
            }
        },
    )
    return directory


def publish(
    root: Path, manifest: ClipManifest, target: str, *, completed: bool = True
) -> Path:
    annotation = make_template(manifest, target)
    if completed:
        annotation.status = "completed"
        for frame in annotation.frames:
            frame.reviewed = True
    path = root / "annotated" / "processed" / target / f"{annotation.clip_id}.json"
    write_json(path, annotation.model_dump())
    return path


def test_completion_waits_for_both_targets_then_moves_and_reuses(
    tmp_path: Path, manifest: ClipManifest
) -> None:
    directory = prepare(tmp_path, manifest)
    publish(tmp_path, manifest, "ball")
    assert sync_done(tmp_path)["pending"] == ["sample"]
    publish(tmp_path, manifest, "player", completed=False)
    assert sync_done(tmp_path)["pending"] == ["sample"]
    publish(tmp_path, manifest, "player")
    assert sync_done(tmp_path, dry_run=True)["ready"] == ["sample"]
    assert video_path(tmp_path, manifest).is_file()
    assert sync_done(tmp_path)["moved"] == ["sample"]
    assert not video_path(tmp_path, manifest).exists()
    assert done_video_path(tmp_path, manifest).read_bytes() == b"original clip"
    assert sync_done(tmp_path)["already_done"] == ["sample"]
    assert _verify_published(directory) == manifest
    build_kit(tmp_path / "project_kits", manifest.policies)


@pytest.mark.parametrize(
    "problem",
    [
        "wrong_schema",
        "wrong_clip",
        "missing_frame",
        "corrupt_video",
        "conflict",
        "symlink",
    ],
)
def test_invalid_completion_keeps_video(
    tmp_path: Path, manifest: ClipManifest, problem: str
) -> None:
    prepare(tmp_path, manifest)
    ball = publish(tmp_path, manifest, "ball")
    publish(tmp_path, manifest, "player")
    if problem == "wrong_schema":
        value = make_template(manifest, "player").model_dump()
        write_json(ball, value)
    elif problem in {"wrong_clip", "missing_frame"}:
        value = json.loads(ball.read_text())
        if problem == "wrong_clip":
            value["clip_id"] = "other"
        else:
            value["frames"].pop()
        write_json(ball, value)
    elif problem == "corrupt_video":
        video_path(tmp_path, manifest).write_bytes(b"modified")
    elif problem == "conflict":
        dest = done_video_path(tmp_path, manifest)
        dest.parent.mkdir(parents=True)
        dest.write_bytes(b"do not overwrite")
    else:
        saved = tmp_path / "annotation.json"
        ball.rename(saved)
        ball.symlink_to(saved)
    result = sync_done(tmp_path)
    assert len(result["errors"]) == 1
    assert result["moved"] == []
    assert video_path(tmp_path, manifest).exists()


def test_interrupted_move_resumes_without_overwriting(
    tmp_path: Path, manifest: ClipManifest
) -> None:
    prepare(tmp_path, manifest)
    publish(tmp_path, manifest, "ball")
    publish(tmp_path, manifest, "player")
    dest = done_video_path(tmp_path, manifest)
    dest.parent.mkdir(parents=True)
    os.link(video_path(tmp_path, manifest), dest)
    assert sync_done(tmp_path)["moved"] == ["sample"]
    assert not video_path(tmp_path, manifest).exists()
