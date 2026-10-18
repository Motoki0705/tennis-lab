from pathlib import Path

import pytest

from src.tasks.base.annotation.artifacts import file_reference, write_asset
from src.tasks.base.annotation.paths import AnnotationLayout
from src.tasks.base.annotation.storage.references import distinct


def test_role_aliases_are_one_stable_protection_target(tmp_path: Path) -> None:
    layout = AnnotationLayout(tmp_path, tmp_path / "outputs", "test")
    original = write_asset(layout, layout.path("campaigns", "fixture.json"), b"fixture")
    alias = file_reference(layout, layout.artifact_path(original), role="artifact")
    assert distinct(layout, (alias, original, alias)) == (original,)
    assert distinct(layout, (original, alias)) == (original,)


@pytest.mark.parametrize("changed", [{"digest": "0" * 64}, {"size_bytes": 1}])
def test_a_physical_alias_cannot_hide_a_different_content_version(
    tmp_path: Path, changed: dict,
) -> None:
    layout = AnnotationLayout(tmp_path, tmp_path / "outputs", "test")
    original = write_asset(layout, layout.path("campaigns", "fixture.json"), b"fixture")
    alias = file_reference(layout, layout.artifact_path(original), role="artifact")
    with pytest.raises(ValueError, match="same physical path"):
        distinct(layout, (original, alias.model_copy(update=changed)))
