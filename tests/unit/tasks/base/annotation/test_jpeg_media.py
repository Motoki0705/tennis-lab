from pathlib import Path

import pytest
from pydantic import ValidationError

from src.tasks.base.annotation.contracts import ClipManifest
from src.tasks.base.annotation.media import check_clip, decode_clip
from tests.support.tasks.annotation_predictions import example


def test_v1_manifest_identity_has_no_new_default_fields(tmp_path: Path) -> None:
    fixture = example(tmp_path)
    original = fixture.layout.artifact_path(fixture.reference).read_bytes().strip()
    assert b'"jpeg"' not in original
    assert ClipManifest.model_validate_json(original).canonical_bytes() == original


@pytest.mark.parametrize("change", ["missing_frame", "offset", "pts", "version"])
def test_jpeg_manifest_rejects_inconsistent_mapping(
    tmp_path: Path, change: str
) -> None:
    fixture = example(tmp_path, jpeg=True)
    values = fixture.manifest.model_dump(mode="json")
    if change == "missing_frame":
        values["jpeg"]["frames"].pop()
    elif change == "offset":
        values["jpeg"]["frames"][1]["offset"] += 1
    elif change == "pts":
        values["frames"][1]["clip_pts"] += 1
    else:
        values["schema_version"] = "annotation.clip.v1"
    import json

    with pytest.raises(ValidationError):
        ClipManifest.model_validate_json(json.dumps(values))


def test_wrong_dimensions_and_payload_corruption_fail_before_use(
    tmp_path: Path,
) -> None:
    fixture = example(tmp_path, jpeg=True)
    path = fixture.layout.artifact_path(fixture.manifest.media)
    wrong_size = fixture.manifest.model_copy(update={"width": 128})
    with pytest.raises(ValueError, match="dimensions"):
        check_clip(path, wrong_size)
    with path.open("r+b") as handle:
        handle.seek(10)
        old = handle.read(1)
        handle.seek(10)
        handle.write(bytes([old[0] ^ 1]))
    with pytest.raises(ValueError, match="hash"):
        check_clip(path, fixture.manifest)
    with pytest.raises(ValueError, match="frame bytes"):
        list(decode_clip(path, fixture.manifest, 0, 1))
