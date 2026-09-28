"""The audit counts every frame and preserves split and absent-context states."""

import json
import subprocess
import sys

import pytest

from src.tasks.ball_detection.data.store import BallFrameStore
from src.tasks.ball_refiner.data.audit import audit_store
from tests.support.tasks.ball_detection.store import ball, frame, write_store_clip


def test_audit_cli_covers_unknowns_and_refuses_overwriting_report(tmp_path):
    store_path = tmp_path / "ball"
    write_store_clip(store_path, "meiji/video_000/clip_000/cam0", [frame(0, ball()), frame(1, ball("unresolved", None))], source="meiji", split="val")
    metadata = store_path / "metadata.json"
    doc = json.loads(metadata.read_text())
    doc["clips"][0].update(camera_id="cam0", group_id="video_000")
    metadata.write_text(json.dumps(doc))
    contexts = tmp_path / "contexts"
    contexts.mkdir()
    output = tmp_path / "audit"
    command = [sys.executable, "-m", "src.tasks.ball_refiner.scripts.audit_data", "--store", str(store_path),
               "--meiji-context-root", str(contexts), "--output", str(output)]
    result = subprocess.run(command, text=True, capture_output=True, check=False)
    assert result.returncode == 0, result.stderr
    report = json.loads((output / "audit.json").read_text())
    assert report["counts"]["meiji/val"]["frames"] == 2
    assert report["counts"]["meiji/val"]["presence_known"] == 1
    assert report["context_counts"]["meiji/val"] == {"pose_missing_scene": 1, "court_missing_scene": 1, "complete": 0}
    assert report["clips"][0]["clip"]["split"] == "val"
    assert len(report["store_sha256"]["metadata.json"]) == 64
    previous = (output / "audit.json").read_bytes()
    result = subprocess.run(command, text=True, capture_output=True, check=False)
    assert result.returncode != 0 and "already exists" in result.stderr
    assert (output / "audit.json").read_bytes() == previous


@pytest.mark.parametrize("argument,value", [
    ("--store", "relative/store"), ("--output", "relative/output"), ("--pose-threshold", "nan"),
])
def test_audit_cli_rejects_invalid_arguments_before_creating_output(tmp_path, argument, value):
    output = tmp_path / "audit"
    arguments = {"--store": str(tmp_path / "missing"), "--meiji-context-root": str(tmp_path),
                 "--output": str(output), "--pose-threshold": "0.5"}
    arguments[argument] = value
    result = subprocess.run(
        [sys.executable, "-m", "src.tasks.ball_refiner.scripts.audit_data",
         *(item for pair in arguments.items() for item in pair)],
        text=True, capture_output=True, check=False,
    )
    assert result.returncode != 0
    assert not output.exists()


def test_group_crossing_splits_fails_before_training(tmp_path):
    directory = tmp_path / "ball"
    write_store_clip(directory, "tracknet/game1/clip1", [frame(0, ball())], split="train")
    write_store_clip(directory, "tracknet/game1/clip2", [frame(0, ball())], split="test")
    context_root = tmp_path / "contexts"
    context_root.mkdir()
    with pytest.raises(ValueError, match="crosses splits"):
        audit_store(BallFrameStore(directory), meiji_context_root=context_root, pose_threshold=0.5)
