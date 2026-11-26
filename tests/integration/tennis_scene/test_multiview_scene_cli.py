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

from src.tasks.slcs.data.annotation import load_slcs_annotation
from src.tennis_scene.generate_dataset.manifest import ClipManifest
from src.tennis_scene.multiview_scene import __main__ as scene_cli
from src.tennis_scene.multiview_scene.configuration import MultiviewSceneConfig
from src.tennis_scene.multiview_scene.inputs import INPUT_NODE, RECIPE
from src.tennis_scene.multiview_scene.publication import (
    open_legacy_store,
    read_pointers,
)
from src.tennis_scene.pipeline.artifacts import json_value
from src.tennis_scene.pipeline.contracts import ClipSource, SourceVideo
from src.tennis_scene.pipeline.dataset_publication import (
    ANNOTATION_SCHEMA_VERSION,
    scene_array_manifest,
)
from src.tennis_scene.pipeline.storage import scene_pointers
from src.tennis_scene.pipeline.storage.clip_store import ArtifactRef, ClipStore
from src.tennis_scene.pipeline.storage.codec import ArtifactCodec
from src.tennis_scene.pipeline.storage.scene_export import export_scene
from src.tennis_scene.pipeline.storage.scene_index import indexed_scene_path
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
                "version": ANNOTATION_SCHEMA_VERSION,
                "clip_manifest_sha256": ClipManifest.load(clip).digest(),
                "scene_index": "scene.json",
                "scene_result": str(original_export.relative_to(old.root)),
                "scene_schema_version": legacy.schema_version,
                "arrays": scene_array_manifest(legacy),
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
    clip = ClipManifest.load(old.root.parents[1])
    journal = old.root / "updates" / "multiview_scene" / f"{new_ref.artifact_id}.json"
    common = ("--dataset", dataset, "--progress", progress, "--clip", CLIP_ID)
    scene_args = (*common, "--players-per-side", 1)
    history = ("--legacy-generation", "before-multiview")

    def change(phase: str) -> None:
        invoke(monkeypatch, "multiview_scene", phase, *scene_args, "--output", output)
        assert scene_cli.main() == 0

    def report(directory: Path, *options: str) -> None:
        invoke(
            monkeypatch,
            "multiview_scene",
            "report",
            *scene_args,
            "--output",
            directory,
            *options,
        )
        assert scene_cli.main() == 0

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

    def compare(directory: Path, *options: str) -> None:
        invoke(
            monkeypatch,
            "pose_comparison",
            *common,
            "--candidates",
            candidates,
            "--pose-record",
            pose,
            "--output",
            directory,
            *options,
        )
        compare_cli.main()

    replace_text = scene_pointers._replace_text

    def interrupt_annotation(path: Path, content: str | None) -> None:
        if path.name == "annotation.json":
            raise OSError("simulated stop after the scene index rename")
        replace_text(path, content)

    def assert_recovery_needed() -> None:
        pending = scene_pointers.read_pending_scene_update(old.root)
        assert pending is not None and pending.receipt == journal
        message = r"pending .*updates/multiview_scene/.*multiview_scene publish.*multiview_scene restore"
        with pytest.raises(scene_pointers.ScenePointerUpdateError, match=message):
            load_slcs_annotation(clip)
        with pytest.raises(scene_pointers.ScenePointerUpdateError, match=message):
            indexed_scene_path(old.index_path)
        pending_page = tmp_path / "pending-page"
        with pytest.raises(scene_pointers.ScenePointerUpdateError, match=message):
            compare(pending_page, *history)
        pending_report = tmp_path / "pending-report"
        with pytest.raises(scene_pointers.ScenePointerUpdateError, match=message):
            report(pending_report, *history)
        assert not pending_page.exists()
        assert not (pending_report / "legacy-completion.csv").exists()

    with monkeypatch.context() as patch:
        patch.setattr(scene_pointers, "_replace_text", interrupt_annotation)
        with pytest.raises(OSError, match="simulated stop after"):
            change("publish")
    interrupted = read_pointers(old.root)
    assert interrupted["scene.json"] != before["scene.json"]
    assert interrupted["annotation.json"] == before["annotation.json"]
    assert_recovery_needed()

    change("publish")
    published = read_pointers(old.root)
    assert published["scene.json"] == interrupted["scene.json"]
    assert published["annotation.json"] != before["annotation.json"]
    assert scene_pointers.read_pending_scene_update(old.root) is None
    assert (
        indexed_scene_path(old.index_path)
        == old.root / "exports" / new_ref.artifact_id / "scene.npz"
    )
    assert load_slcs_annotation(clip).player_track_ids.tolist() == [47]
    assert json.loads(published["annotation.json"])["scene_schema_version"] == 2

    reports = tmp_path / "reports"
    with pytest.raises(ValueError, match="choose --legacy-generation before-multiview"):
        report(reports)
    report(reports, *history)
    with (reports / "legacy-completion.csv").open(newline="") as stream:
        (row,) = csv.DictReader(stream)
    assert (row["clip_id"], row["joint_player_frames"], row["ball_3d_frames"]) == (
        CLIP_ID,
        "0",
        "1",
    )

    page = tmp_path / "page"
    with pytest.raises(ValueError, match="choose --legacy-generation before-multiview"):
        compare(page)
    compare(page, *history)
    manifest = json.loads((page / "manifest.json").read_text())
    (entry,) = manifest["stages"][0]["clips"]
    assert manifest["legacy_progress"]["scene_generation"] == "before-multiview"
    assert entry["provenance"]["legacy_scene_assembly"] == json_value(legacy_ref)
    assert read_pointers(old.root) == published

    moved = journal.rename(journal.with_suffix(".recovery-test"))
    with pytest.raises(
        FileNotFoundError, match="Missing multiview publication receipt"
    ):
        open_legacy_store(old.root, "before-multiview")
    moved.rename(journal)

    with monkeypatch.context() as patch:
        patch.setattr(scene_pointers, "_replace_text", interrupt_annotation)
        with pytest.raises(OSError, match="simulated stop after"):
            change("restore")
    assert read_pointers(old.root) == {
        "scene.json": before["scene.json"],
        "annotation.json": published["annotation.json"],
    }
    assert_recovery_needed()

    change("restore")
    assert read_pointers(old.root) == before
    assert scene_pointers.read_pending_scene_update(old.root) is None
    assert (
        indexed_scene_path(old.index_path)
        == old.root / json.loads(before["annotation.json"])["scene_result"]
    )
    assert load_slcs_annotation(clip).player_track_ids.size == 0
    report(reports)
    compare(tmp_path / "restored-page")


def test_restore_uses_pending_receipt_before_the_first_pointer_change(
    monkeypatch: pytest.MonkeyPatch,
    adopted_clip,
) -> None:
    dataset, progress, _, old, _, output, new_ref = adopted_clip
    before = read_pointers(old.root)
    common = (
        "--dataset",
        dataset,
        "--progress",
        progress,
        "--clip",
        CLIP_ID,
        "--players-per-side",
        1,
        "--output",
        output,
    )

    def interrupt_first(path: Path, content: str | None) -> None:
        assert path == old.index_path
        raise OSError("simulated stop before the first rename")

    with monkeypatch.context() as patch:
        patch.setattr(scene_pointers, "_replace_text", interrupt_first)
        invoke(monkeypatch, "multiview_scene", "publish", *common)
        with pytest.raises(OSError, match="simulated stop before"):
            scene_cli.main()
    assert read_pointers(old.root) == before
    pending = scene_pointers.read_pending_scene_update(old.root)
    assert pending is not None
    assert pending.receipt.name == f"{new_ref.artifact_id}.json"

    invoke(monkeypatch, "multiview_scene", "restore", *common)
    assert scene_cli.main() == 0
    assert read_pointers(old.root) == before
    assert scene_pointers.read_pending_scene_update(old.root) is None


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
