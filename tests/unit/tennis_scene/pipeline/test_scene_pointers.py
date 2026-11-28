"""An abandoned switch can only be finished or reverted by its own receipt."""

from pathlib import Path

import pytest

from src.tennis_scene.pipeline.storage import scene_pointers
from src.tennis_scene.pipeline.storage.clip_store import clip_store_lock
from src.tennis_scene.pipeline.storage.scene_pointers import (
    ScenePointerUpdateError,
    read_pending_scene_update,
    replace_scene_pointers,
    stable_scene_pointers,
)


def test_interruption_before_first_pointer_write_keeps_receipt_discoverable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    scene = tmp_path / "scene.json"
    scene.write_text("old scene")
    receipt = tmp_path / "receipt.json"
    other = tmp_path / "other.json"
    for path in (receipt, other):
        path.touch()

    def stop_before_rename(path: Path, content: str | None) -> None:
        raise OSError("stopped before the first rename")

    def switch(owner: Path, scene_text: str, annotation: str | None) -> None:
        with clip_store_lock(tmp_path):
            replace_scene_pointers(
                tmp_path,
                scene=scene_text,
                annotation=annotation,
                receipt=owner,
                recovery=f"restore using {owner.name}",
            )

    with (
        monkeypatch.context() as patch,
        pytest.raises(ScenePointerUpdateError, match="restore using receipt.json"),
        stable_scene_pointers(tmp_path),
    ):
        patch.setattr(scene_pointers, "_replace_text", stop_before_rename)
        with pytest.raises(OSError, match="stopped before the first rename"):
            switch(receipt, "new scene", "new annotation")
        assert scene.read_text() == "old scene"
        assert not (tmp_path / "annotation.json").exists()

    with pytest.raises(ScenePointerUpdateError, match="restore using receipt.json"):
        switch(other, "other scene", "other annotation")

    switch(receipt, "old scene", None)
    assert read_pending_scene_update(tmp_path) is None
    assert scene.read_text() == "old scene"
    assert not (tmp_path / "annotation.json").exists()
