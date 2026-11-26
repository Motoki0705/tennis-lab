"""Review regressions at the saved-batch and published-generation CLI boundaries."""

from __future__ import annotations

import csv
import json
import sys
from contextlib import nullcontext
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from src.tennis_scene.multiview_scene import __main__ as scene_cli
from src.tennis_scene.multiview_scene.configuration import MultiviewSceneConfig
from src.tennis_scene.multiview_scene.inputs import INPUT_NODE, RECIPE
from src.tennis_scene.multiview_scene.publication import (
    open_legacy_store,
    read_pointers,
)
from src.tennis_scene.pipeline.artifacts import json_value
from src.tennis_scene.pipeline.contracts import ClipSource, SourceVideo
from src.tennis_scene.pipeline.storage.clip_store import ArtifactRef, ClipStore
from src.tennis_scene.pipeline.storage.codec import ArtifactCodec
from src.tennis_scene.pipeline.storage.scene_export import export_scene
from src.tennis_scene.schema import SceneResult
from src.tennis_scene.visualization.pose_comparison import __main__ as compare_cli
from src.tennis_scene.visualization.pose_comparison.inputs import ClipPlan, _read_scene
from src.tennis_scene.visualization.review.store import StoreReader
from src.utils.checksum import dual_sha256
from tests.support.tennis_scene.video_dataset import add_clip
from tests.unit.tennis_scene.conftest import v2_scene as v2_scene

CLIP_ID = "video_000/clip_000"


def publish(
    store: ClipStore,
    node: str,
    value: dict,
    schema: str,
    *,
    identity: dict | None = None,
    deps: dict[str, ArtifactRef] | None = None,
) -> ArtifactRef:
    return store.publish(
        node,
        value,
        ArtifactCodec(dict),
        schema=schema,
        version=2 if schema == "scene_result" else 1,
        identity={"node": node, **(identity or {})},
        dependencies=deps or {},
        provenance={},
    )


def invoke(monkeypatch: pytest.MonkeyPatch, entry: str, *args: object) -> None:
    monkeypatch.setattr(sys, "argv", [entry, *map(str, args)])


@pytest.fixture
def adopted_clip(tmp_path: Path, v2_scene: SceneResult):
    dataset = tmp_path / "dataset"
    clip = add_clip(dataset, CLIP_ID, frames=3, cameras=2)
    paths = sorted(clip.glob("*.mp4"))
    source = ClipSource(
        CLIP_ID,
        tuple(
            SourceVideo(path.stem, path, dual_sha256(path), 3, 30.0, 1280, 720)
            for path in paths
        ),
    )
    old = ClipStore(clip / "annotations/tennis_scene", json_value(source))
    alignment = publish(
        old,
        "external_ball_camera_geometry",
        {
            "geometry": {
                "cameras": [
                    {
                        "camera_id": name,
                        "intrinsic": np.eye(3),
                        "rotation": np.eye(3),
                        "translation": np.array([0.0, 0.0, 1.0]),
                    }
                    for name in source.camera_ids
                ]
            }
        },
        "aligned_cameras",
    )
    ball = publish(
        old,
        "ball_triangulation",
        {},
        "external_ball_trajectory",
        deps={"alignment": alignment},
    )
    metadata = {
        "scene_schema_version": 2,
        "camera_ids": list(source.camera_ids),
        "dataset_clip_id": CLIP_ID,
        "status": "ok",
        "validity_statistics": {},
        "track_ids": [],
        "court_reference": {"camera_ids": list(source.camera_ids)},
    }
    empty_players = {
        name: value[:0]
        for name in SceneResult.__dataclass_fields__
        if (
            name.startswith(("player_", "smpl_", "human_kp"))
            and isinstance(value := getattr(v2_scene, name), np.ndarray)
        )
    }
    legacy = replace(
        v2_scene, width=1280, height=720, metadata=metadata, **empty_players
    )
    legacy_ref = publish(
        old,
        "scene_assembly",
        asdict(legacy),
        "scene_result",
        deps={"alignment": alignment, "ball": ball},
    )
    original_export = export_scene(old, legacy, legacy_ref)
    (old.root / "annotation.json").write_text(
        json.dumps(
            {
                "scene_index": "scene.json",
                "scene_result": str(original_export.relative_to(old.root)),
            }
        )
    )
    output = tmp_path / "new"
    new = ClipStore(output / CLIP_ID, json_value(source))
    frozen = publish(
        new,
        INPUT_NODE,
        {},
        INPUT_NODE,
        identity={
            "pins": {
                "recipe": RECIPE,
                "legacy_index_sha256": dual_sha256(old.index_path),
            }
        },
    )
    replacement = replace(
        v2_scene,
        width=1280,
        height=720,
        player_track_ids=np.array([47], np.int32),
        metadata={
            **metadata,
            "track_ids": [47],
            "player_reconstruction_method": RECIPE,
            "reused_ball": json_value(ball),
        },
    )
    new_ref = publish(
        new,
        "scene_assembly",
        asdict(replacement),
        "scene_result",
        deps={"tracked": frozen},
    )
    export_scene(new, replacement, new_ref)
    progress = tmp_path / "progress.json"
    another = "video_000/clip_001"
    progress.write_text(
        json.dumps(
            {
                "dataset_id": "batch-fixture",
                "clips": {key: {"status": "generated"} for key in (CLIP_ID, another)},
                "nodes": {key: {} for key in (CLIP_ID, another)},
                "targets": [
                    {
                        "record": {
                            "clip_id": key,
                            "path": f"videos/video_000/clips/{key.split('/')[-1]}",
                            "num_frames": 3,
                        }
                    }
                    for key in (CLIP_ID, another)
                ],
            }
        )
    )
    return dataset, progress, source, old, legacy_ref, output, new_ref


def test_report_and_comparison_read_the_saved_legacy_generation_after_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    adopted_clip,
) -> None:
    dataset, progress, source, old, legacy_ref, output, new_ref = adopted_clip
    before = read_pointers(old.root)
    common = ("--dataset", dataset, "--progress", progress, "--clip", CLIP_ID)
    scene_args = (*common, "--players-per-side", 1)
    invoke(monkeypatch, "multiview_scene", "publish", *scene_args, "--output", output)
    assert scene_cli.main() == 0
    published = read_pointers(old.root)
    assert (
        published != before
        and StoreReader(old.index_path).references["scene_assembly"]["artifact_id"]
        == new_ref.artifact_id
    )

    report = tmp_path / "reports"
    invoke(monkeypatch, "multiview_scene", "report", *scene_args, "--output", report)
    with pytest.raises(ValueError, match="choose --legacy-generation before-multiview"):
        scene_cli.main()
    history = ("--legacy-generation", "before-multiview")
    invoke(
        monkeypatch,
        "multiview_scene",
        "report",
        *scene_args,
        "--output",
        report,
        *history,
    )
    assert scene_cli.main() == 0
    with (report / "legacy-completion.csv").open(newline="") as stream:
        (row,) = csv.DictReader(stream)
    assert (row["clip_id"], row["joint_player_frames"], row["ball_3d_frames"]) == (
        CLIP_ID,
        "0",
        "1",
    )

    # Keep the real old archive and lineage checks. Supply the candidate and
    # encoding results because their contracts have dedicated tests.
    def load(old_store: StoreReader, _new: StoreReader, _plan: ClipPlan, _record: dict):
        viewed = _read_scene(old_store, source)
        assert viewed is not None and viewed.player_ids == ()
        return SimpleNamespace(
            provenance={"legacy_scene_assembly": old_store.references["scene_assembly"]}
        )

    def render(_data, _plan, bundle: Path, *_settings):
        (bundle / "stage1").mkdir(parents=True)
        return {"stage1": {"video_frames": 1, "video_bytes": 0}}

    monkeypatch.setattr(compare_cli, "load_comparison", load)
    monkeypatch.setattr(compare_cli, "_render_clip", render)
    monkeypatch.setattr(compare_cli, "clip_metrics", lambda *_: {})
    candidates = tmp_path / "candidates"
    ClipStore(candidates / CLIP_ID, json_value(source))
    pose = tmp_path / "pose.json"
    pose.write_text(json.dumps({"clips": {CLIP_ID: {"reconstruction_seconds": 0}}}))
    page = tmp_path / "page"
    invoke(
        monkeypatch,
        "pose_comparison",
        *common,
        "--candidates",
        candidates,
        "--pose-record",
        pose,
        "--output",
        page,
    )
    with pytest.raises(ValueError, match="choose --legacy-generation before-multiview"):
        compare_cli.main()
    invoke(
        monkeypatch,
        "pose_comparison",
        *common,
        "--candidates",
        candidates,
        "--pose-record",
        pose,
        "--output",
        page,
        *history,
    )
    compare_cli.main()
    manifest = json.loads((page / "manifest.json").read_text())
    (entry,) = manifest["stages"][0]["clips"]
    assert manifest["legacy_progress"]["scene_generation"] == "before-multiview"
    assert entry["provenance"]["legacy_scene_assembly"] == json_value(legacy_ref)
    assert read_pointers(old.root) == published

    journal = old.root / "updates" / "multiview_scene" / f"{new_ref.artifact_id}.json"
    assert journal.is_file()
    journal.rename(journal.with_suffix(".recovery-test"))
    with pytest.raises(
        FileNotFoundError, match="Missing multiview publication receipt"
    ):
        open_legacy_store(old.root, "before-multiview")


@pytest.mark.parametrize("phase", ["prepare", "run"])
def test_saved_batch_without_events_reaches_multiview_inputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    adopted_clip,
    phase: str,
) -> None:
    dataset, progress, source, _, _, _, _ = adopted_clip
    record = json.loads(progress.read_text())
    fixture = Path(__file__).with_name("fixtures") / "multiview_saved_batch.yaml"
    record["pipeline_config_yaml"] = fixture.read_text().replace(
        "TEST_PROJECT_ROOT", str(tmp_path)
    )
    cfg = MultiviewSceneConfig.from_batch_yaml(record["pipeline_config_yaml"])
    assert cfg.sampling_max_frames == 2048
    assert cfg.people.runtime.hmr2.batch_size == 16
    assert cfg.people.runtime.static_cam
    assert cfg.player_placement.max_nfev == 200
    progress.write_text(json.dumps(record))
    tracks, candidates = tmp_path / "tracks", tmp_path / "candidates"
    ClipStore(tracks / CLIP_ID, json_value(source))
    candidates.mkdir()

    class ReachedSavedInputs(RuntimeError):
        pass

    def bind(clip, old, track_store, candidate, *, players_per_side, threshold, crops):
        assert (
            clip.record.clip_id
            == old.source["clip_id"]
            == track_store.source["clip_id"]
            == CLIP_ID
        )
        assert candidate == candidates / CLIP_ID
        assert (players_per_side, threshold, crops.bbox_enlarge) == (1, 0.15, 1.2)
        raise ReachedSavedInputs

    monkeypatch.setattr(scene_cli, "read_inputs", bind)
    monkeypatch.setattr(
        scene_cli, "InferenceRecorder", lambda *_a, **_kw: nullcontext()
    )
    monkeypatch.setenv("TENNIS_RUN_ID", "fixture")
    monkeypatch.setenv("TENNIS_GPU_RESOURCE", "fixture")
    invoke(
        monkeypatch,
        "multiview_scene",
        phase,
        "--dataset",
        dataset,
        "--progress",
        progress,
        "--clip",
        CLIP_ID,
        "--tracks",
        tracks,
        "--candidates",
        candidates,
        "--output",
        tmp_path / "recipe",
        "--players-per-side",
        1,
    )
    with pytest.raises(ReachedSavedInputs):
        scene_cli.main()
