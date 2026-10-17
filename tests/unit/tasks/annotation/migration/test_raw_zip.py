from __future__ import annotations

import io
import zipfile

import pytest

from src.tasks.annotation.migration.legacy_format.raw_zip import inspect_zip


def zip_bytes(members: dict[str, bytes]) -> bytes:
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, data in members.items():
            archive.writestr(name, data)
    return stream.getvalue()

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
def test_bad_zip_members_are_rejected(members: dict[str, bytes]) -> None:
    with pytest.raises(ValueError):
        inspect_zip(zip_bytes(members))


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



def test_legacy_download_suffix_is_preserved() -> None:
    member = "annotation_clip(1).json"
    assert inspect_zip(zip_bytes({member: b'{"clip_id":"clip"}'})) == [member]
