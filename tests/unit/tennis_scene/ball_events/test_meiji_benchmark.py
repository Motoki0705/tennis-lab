"""Fixed benchmark replay and split contracts without real video or model assets."""

from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path

import numpy as np
import pytest

from src.tennis_scene.ball_events.meiji_benchmark import (
    CLIP_IDS,
    SCORING_CLIPS,
    TUNING_CLIPS,
    BenchmarkInputs,
    make_splits,
)
from src.tennis_scene.pipeline.artifacts import json_value
from src.tennis_scene.pipeline.components.event_detection import EventDetectionConfig
from src.tennis_scene.pipeline.contracts import ClipSource, SourceVideo
from src.tennis_scene.pipeline.imports.scene_events import (
    replay_scene_events,
    verify_replayed_scene,
)
from src.tennis_scene.pipeline.storage.clip_store import ClipStore
from src.tennis_scene.pipeline.storage.codec import ArtifactCodec
from src.tennis_scene.pipeline.storage.scene_export import export_scene
from src.tennis_scene.schema import SceneResult
from src.tennis_scene.scripts import build_meiji_event_benchmark
from src.utils.configuration import PathContractError


@pytest.mark.parametrize(
    ("invalid_input", "message"),
    [("base", r"meiji_event_benchmark\.base"), ("annotations", r"annotations must be an existing directory")],
)
def test_cli_rejects_invalid_input_directories_before_replay(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, invalid_input: str, message: str,
) -> None:
    base = tmp_path / "base"
    annotations = tmp_path / "annotations"
    output = tmp_path / "output"
    if invalid_input != "base":
        base.mkdir()
    if invalid_input == "annotations":
        annotations.touch()
    else:
        annotations.mkdir()

    def unexpected_replay(*_args: Path) -> None:
        pytest.fail("Invalid input reached the benchmark builder")

    monkeypatch.setattr(build_meiji_event_benchmark, "build_meiji_benchmark", unexpected_replay)
    with pytest.raises(PathContractError, match=message):
        build_meiji_event_benchmark.main([
            "--base", str(base), "--annotations", str(annotations), "--output", str(output),
        ])
    assert not output.exists()


def test_stratification_partitions_clips_and_pins_scoring_set() -> None:
    # Four scoring clips have net contact. Some positive clips have two events.
    net = set(CLIP_IDS[:27]) | {SCORING_CLIPS[i] for i in (1, 3, 4, 5)} | {CLIP_IDS[35]}
    counts = {clip: (2 if clip == CLIP_IDS[35] else 1) if clip in net else 0 for clip in CLIP_IDS}
    splits = make_splits(net, counts)
    all_ids = [clip for split in ("train", "val", "test") for clip in splits[split]["clip_ids"]]
    assert sorted(all_ids) == list(CLIP_IDS)
    assert {key: (splits[key]["clips"], splits[key]["net_clips"]) for key in ("train", "val", "test")} == {
        "train": (47, 22), "val": (11, 5), "test": (11, 5),
    }
    assert set(SCORING_CLIPS) <= set(splits["test"]["clip_ids"])
    assert not set(TUNING_CLIPS) & set(splits["test"]["clip_ids"])
    assert sum(splits[split]["net_events"] for split in ("train", "val", "test")) == 33


def test_saved_scene_replay_shares_inputs_and_publishes_only_v3_index(tmp_path: Path, v2_scene: SceneResult) -> None:
    old = tmp_path / "v2"
    new = tmp_path / "v3"
    clip = "video_000/clip_000"
    source = ClipSource(clip, tuple(
        SourceVideo(camera, tmp_path / f"{camera}.mp4", "a" * 64, 3, 30.0, 640, 360)
        for camera in ("near", "far")
    ))
    v2_scene.metadata["dataset_clip_id"] = clip
    # Negative zero would pass a value-only comparison.
    assert v2_scene.ball_3d is not None
    v2_scene.ball_3d[0, 0] = -0.0
    base = ClipStore(old, json_value(source))
    ref = base.publish("scene_assembly", v2_scene, ArtifactCodec(SceneResult), schema="scene_result", version=2,
        identity={"assembly": "past checkpoint"}, dependencies={}, provenance={"origin": "component"})
    archive = export_scene(base, v2_scene, ref)
    original = base.index_path.read_bytes()
    files = {path.relative_to(old): sha256(path.read_bytes()).hexdigest() for path in old.rglob("*") if path.is_file()}
    shared = BenchmarkInputs(old, {}, files).stage(new, {Path("scene.json")})
    result, path, index = replay_scene_events(new, EventDetectionConfig("rule"), code_commit="c" * 40, code_identity="d" * 64)
    assert result.schema_version == 3 and base.index_path.read_bytes() == original
    assert index["artifacts"]["scene_assembly"] == json_value(ref)
    event = index["artifacts"]["event_detection"]
    assert index["exports"]["scene"]["inputs"] == {"event_detection": event}
    descriptor = json.loads((new / event["path"]).read_text())
    assert descriptor["dependencies"] == {"scene": json_value(ref)}
    assert descriptor["identity"]["base_export_sha256"] == files[archive.relative_to(old)]
    assert (new / archive.relative_to(old)).samefile(archive)
    assert all((new / file).samefile(old / file) for file in shared)
    v2_descriptor = base.descriptor(ref)
    before = (old / ref.path).parent / v2_descriptor["payload"]["ball_3d"]["array"]
    after = (new / event["path"]).parent / descriptor["payload"]["ball_3d"]["array"]
    assert before.samefile(after)
    assert path.is_file()
    saved = verify_replayed_scene(base.index_path, new / "scene.json")
    assert saved.ball_3d is not None and np.signbit(saved.ball_3d[0, 0])


def test_saved_v3_scene_replay_produces_v4_with_bit_identical_v2_arrays(
    tmp_path: Path, v2_scene: SceneResult, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src.tennis_scene.ball_events import annotation as meiji_annotation

    old = tmp_path / "v2"
    v3_dir = tmp_path / "v3"
    v4_dir = tmp_path / "v4"
    clip = "video_000/clip_000"
    source = ClipSource(clip, tuple(
        SourceVideo(camera, tmp_path / f"{camera}.mp4", digit * 64, 3, 30.0, 640, 360)
        for camera, digit in (("cam0", "a"), ("cam1", "b"))
    ))
    v2_scene.metadata["dataset_clip_id"] = clip
    assert v2_scene.ball_3d is not None
    v2_scene.ball_3d[0, 0] = -0.0
    base = ClipStore(old, json_value(source))
    ref = base.publish(
        "scene_assembly", v2_scene, ArtifactCodec(SceneResult),
        schema="scene_result", version=2,
        identity={"assembly": "past checkpoint"}, dependencies={}, provenance={"origin": "component"},
    )
    export_scene(base, v2_scene, ref)

    # Stage v2 -> v3
    v2_files = {p.relative_to(old): sha256(p.read_bytes()).hexdigest() for p in old.rglob("*") if p.is_file()}
    BenchmarkInputs(old, {}, v2_files).stage(v3_dir, {Path("scene.json")})
    v3_result, _, _ = replay_scene_events(
        v3_dir, EventDetectionConfig("rule"), code_commit="c" * 40, code_identity="d" * 64,
    )
    assert v3_result.schema_version == 3

    # Build minimal v2 annotation + pinned v3 manifest, then replay v3 -> v4
    ann_dir = tmp_path / "ann_v2"
    ann_dir.mkdir()
    (ann_dir / meiji_annotation.ANNOTATION_SCHEMA_FILENAME_V2).write_text("{}")
    v1_sha = "f" * 64
    v3_manifest = tmp_path / "v3_manifest.json"
    v3_manifest.write_text(json.dumps({
        "benchmark_id": "meiji_4cam/multiview-v3",
        "clips": [{
            "clip_id": clip,
            "fps": 30.0,
            "frames": 3,
            "cameras": [
                {"camera_id": "cam0", "sha256": "a" * 64},
                {"camera_id": "cam1", "sha256": "b" * 64},
            ],
            "ball_events": {"annotation_sha256": v1_sha},
        }],
    }))
    monkeypatch.setattr(meiji_annotation, "MEIJI_V3_MANIFEST_SHA256", sha256(v3_manifest.read_bytes()).hexdigest())
    (ann_dir / "clip_000_ball_events.json").write_text(json.dumps({
        "schema": "meiji_event_labels.v2",
        "benchmark": "multiview-v3",
        "clip_id": clip,
        "fps": 30.0,
        "frames": 3,
        "source_v1_annotation_sha256": v1_sha,
        "labeler": "Codex gpt-6.2-sol (xhigh)",
        "unlabelable": [],
        "events": [
            {
                "id": "e001", "source_event_id": "e001", "kind": "bounce", "subtype": "bounce",
                "frame": 1, "frame_uncertainty": 1, "status": "certain",
                "cameras": ["cam0"], "evidence": "Bounce.", "review_source": "mechanical_bounce",
            },
        ],
        "excluded_events": [],
    }))

    v3_files = {p.relative_to(v3_dir): sha256(p.read_bytes()).hexdigest() for p in v3_dir.rglob("*") if p.is_file()}
    BenchmarkInputs(v3_dir, {}, v3_files).stage(v4_dir, {Path("scene.json")})
    v4_cfg = EventDetectionConfig(
        "annotation", annotation_directory=ann_dir, annotation_benchmark_manifest=v3_manifest,
    )
    v4_result, v4_path, v4_index = replay_scene_events(
        v4_dir, v4_cfg, code_commit="e" * 40, code_identity="1" * 64,
    )
    assert v4_result.schema_version == 4
    assert v4_result.ball_event_in_point is None
    assert v4_index["artifacts"]["event_detection"]["version"] == 4
    assert v4_path.is_file()
    saved_v4 = verify_replayed_scene(v3_dir / "scene.json", v4_dir / "scene.json", expected_version=4)
    assert saved_v4.schema_version == 4
    assert saved_v4.ball_3d is not None and np.signbit(saved_v4.ball_3d[0, 0])
