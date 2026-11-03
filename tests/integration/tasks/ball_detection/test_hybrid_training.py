import copy
import json
from collections import Counter
from pathlib import Path

import pytest
import torch

from src.tasks.ball_detection.checkpoints.hybrid import initialize_hybrid_from_query
from src.tasks.ball_detection.checkpoints.hybrid_training import (
    initialize_hybrid,
    restore_hybrid_model,
)
from src.tasks.ball_detection.checkpoints.provenance import data_identity
from src.tasks.ball_detection.data.store import BallFrameStore
from src.tasks.ball_detection.data.windows.dataset import FrozenBallWindowDataset
from src.tasks.ball_detection.data.windows.hybrid import (
    AbsencePolicy,
    build_hybrid_manifest,
)
from src.tasks.ball_detection.data.windows.intervals import PlayIntervalConfig
from src.tasks.ball_detection.data.windows.manifest import (
    build_coordinate_manifest,
    clip_semantic_digest,
)
from src.tasks.ball_detection.data.windows.temporal import TemporalSamplingConfig
from src.tasks.ball_detection.evaluation import hybrid_cli
from src.tasks.ball_detection.model_io.mdd_hybrid import load_mdd_hybrid_detector
from src.tasks.ball_detection.models.mdd import DeepMDDHybridDetector, MDDHybridConfig
from src.tasks.ball_detection.training.hybrid.runner import train_hybrid
from src.tasks.ball_detection.training.sampling import HybridWindowSampler
from src.utils.checksum import dual_sha256
from tests.integration.tasks.ball_detection.mdd_training_contracts import (
    assert_replay,
    diagnostic_prefix,
    invoke_main,
    read_json,
    tiny_hybrid_recipe,
    weights,
)
from tests.support.tasks.ball_detection.store import ball, frame, write_store_clip

pytestmark = pytest.mark.integration


def make_manifest(root: Path) -> Path:
    store_path = root / "store"
    for split in ("train", "val", "test"):
        frames = []
        for i in range(320):
            if i < 128:
                frames.append(frame(i, ball()) if i % 16 != 8 else frame(i))
            elif i < 256:
                frames.append(frame(i, ball("out_of_frame", xy=None)) if i % 2 else frame(i))
            elif i == 280:
                frames.append(frame(i, ball()))
            elif i % 8 == 0:
                frames.append(frame(i, ball("out_of_frame", xy=None)))
            else:
                frames.append(frame(i, annotated=False))
        write_store_clip(store_path, f"{split}/main", frames, split=split, size=(32, 32))
    write_store_clip(store_path, "val/negative", [frame(i) for i in range(128)], source="meiji", split="val", size=(32, 32))
    return write_manifest(store_path, root / "windows.json")


def write_manifest(store_path: Path, path: Path) -> Path:
    store = BallFrameStore(store_path)
    clips = [c for c in store.clips if c.clip_id.endswith(("main", "pose"))]
    common = dict(clip_ids=[c.clip_id for c in clips],
                  clip_semantic_sha256={c.clip_id: clip_semantic_digest(store, c) for c in clips})
    raw = build_coordinate_manifest(store, PlayIntervalConfig(), TemporalSamplingConfig((1,)),
                                    input_kind="mdd_only", common_reference=common)
    path.write_text(json.dumps(raw))
    return path


def test_negative_sources_are_explicit_and_mixed_deterministically(tmp_path: Path) -> None:
    base = make_manifest(tmp_path)
    policy = AbsencePolicy(("reviewed_empty", "out_of_frame"), dict(play=.5, absent=.25, nonplay=.25))
    raw = build_hybrid_manifest(base, policy)
    manifest = tmp_path / "hybrid.json"
    manifest.write_text(json.dumps(raw))
    dataset = FrozenBallWindowDataset(manifest, split="train", requires_pose=False)
    assert set(dataset.window_sources) == {"play", "absent", "nonplay"}
    sampler = HybridWindowSampler(dataset, windows_per_epoch=12, seed=42)
    indices = list(sampler)
    assert Counter(dataset.window_sources[i] for i in indices) == dict(play=6, absent=3, nonplay=3)
    assert indices == list(sampler)
    sampler.set_epoch(1)
    assert indices != list(sampler)
    for source in ("absent", "nonplay"):
        sample = dataset[dataset.window_sources.index(source)]
        if source == "absent":
            assert not sample["position_valid"].any() and sample["heatmap_valid"].all()
        else:
            assert not sample["heatmap_valid"].all() # unknown is not labelled absent
    with pytest.raises(ValueError, match="budget too small"):
        HybridWindowSampler(dataset, windows_per_epoch=1, seed=0)
    # Disable one kind: out_of_frame now stays ignored, never becomes an implicit empty label.
    only_empty = AbsencePolicy(("reviewed_empty",), dict(play=.5, absent=0., nonplay=.5))
    changed = build_hybrid_manifest(base, only_empty)
    manifest.write_text(json.dumps(changed))
    empty_data = FrozenBallWindowDataset(manifest, split="train", requires_pose=False)
    assert "absent" not in empty_data.window_sources # alternating empty/out_of_frame
    for i, source in enumerate(empty_data.window_sources):
        if source == "nonplay":
            sample = empty_data[i]
            assert not sample["heatmap_valid"].all()
    # Enabled bucket with no data is rejected, without silently changing the mix.
    changed["absence"]["window_weights"] = dict(play=.5, absent=.5, nonplay=0.)
    manifest.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="No training windows"):
        HybridWindowSampler(FrozenBallWindowDataset(manifest, split="train", requires_pose=False), windows_per_epoch=4, seed=0)


def test_chat_noted_empty_requires_verified_original_and_frozen_mask(tmp_path: Path) -> None:
    frames = [frame(i) if i == 16 or i >= 64 else frame(i, ball()) for i in range(160)]
    root = write_store_clip(tmp_path / "chat", "chat_annotation/example", frames, source="chat_annotation", size=(32, 32))
    annotation = tmp_path / "annotation.json"
    raw: dict = dict(schema_version="tennis_chat_ball_annotation.v1", clip_id="example", frame_count=len(frames),
               frames=[dict(frame_index=i, reviewed=True, balls=[] if i == 16 or i >= 64 else [{}],
                            notes="position unresolved" if i in (16, 80) else "") for i in range(len(frames))])
    annotation.write_text(json.dumps(raw))
    metadata = json.loads((root / "metadata.json").read_text())
    metadata["clips"][0].update(annotation_path=str(annotation), annotation_sha256=dual_sha256(annotation))
    (root / "metadata.json").write_text(json.dumps(metadata))
    base = write_manifest(root, tmp_path / "base.json")
    policy = AbsencePolicy(("reviewed_empty", "out_of_frame"), dict(play=.5, absent=.5, nonplay=0.))
    frozen = build_hybrid_manifest(base, policy)
    record = frozen["clips"][0]
    assert 16 not in record["reviewed_empty_frames"] and 80 not in record["reviewed_empty_frames"]
    assert 81 in record["reviewed_empty_frames"]
    assert record["negative_evidence"]["excluded_noted_empty_frames"] == 2
    dpt = FrozenBallWindowDataset(base, split="train", requires_pose=False, verify_empty_annotations=True)
    assert not dpt[0]["heatmap_valid"][16]
    saved_manifest = tmp_path / "frozen.json"
    saved_manifest.write_text(json.dumps(frozen))
    annotation.rename(tmp_path / "removed-original.json")
    dataset = FrozenBallWindowDataset(saved_manifest, split="train", requires_pose=False)
    assert not dataset[0]["heatmap_valid"][16]
    with pytest.raises(ValueError, match="missing or checksum"):
        build_hybrid_manifest(base, policy)
    raw["frames"][16]["notes"] = ""
    annotation.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="checksum"):
        build_hybrid_manifest(base, policy)
    # Disabling empty-frame supervision removes this external evidence dependency.
    build_hybrid_manifest(base, AbsencePolicy(("out_of_frame",), dict(play=1., absent=0., nonplay=0.)))


def test_query_hybrid_resume_evaluate_and_cumulative_snapshot(tmp_path: Path) -> None:
    cfg = tiny_hybrid_recipe(tmp_path)
    manifest = Path(cfg.paths.artifact_root) / cfg.manifest
    parent_path = Path(cfg.paths.checkpoint_root) / cfg.initialization.checkpoint
    parent = weights(parent_path)
    spec = MDDHybridConfig(**parent["model_config"])
    torch.manual_seed(42)
    fresh = DeepMDDHybridDetector(spec)
    torch.manual_seed(42)
    transferred, receipt = initialize_hybrid_from_query(parent_path, spec)
    for name in receipt["initialized_parameters"]:
        assert torch.equal(transferred.state_dict()[name], fresh.state_dict()[name])
    for name in receipt["transferred_parameters"]:
        assert torch.equal(transferred.state_dict()[name], parent["state_dict"][name])

    full = tmp_path / cfg.run.output_dir
    train_hybrid(cfg)
    frozen = weights(full / "epoch-000.pt")
    assert all(torch.equal(frozen["state_dict"][k], parent["state_dict"][k]) for k in receipt["transferred_parameters"])
    assert any(not torch.equal(frozen["state_dict"][k], fresh.state_dict()[k]) for k in receipt["initialized_parameters"])
    final = weights(full / "epoch-001.pt")
    assert any(not torch.equal(final["state_dict"][k], parent["state_dict"][k]) for k in receipt["transferred_parameters"])

    # Restore a committed frozen epoch and perform just the next joint update.
    resumed = tmp_path / "hybrid_resume"
    diagnostic_prefix(full, resumed)
    cfg.run.output_dir = resumed.name
    cfg.resume = f"{resumed.name}/epoch-000.pt"
    changed = copy.deepcopy(cfg)
    changed.presence_threshold = .6
    with pytest.raises(ValueError, match="identical recipe"):
        train_hybrid(changed)
    original_bytes = (resumed / "epoch-000.pt").read_bytes()
    broken = copy.deepcopy(frozen)
    broken["code"]["source_sha256"] = {"changed.py": "0" * 64}
    torch.save(broken, resumed / "epoch-000.pt")
    with pytest.raises(ValueError, match="source code changed"):
        train_hybrid(cfg)
    (resumed / "epoch-000.pt").write_bytes(original_bytes)
    train_hybrid(cfg)
    assert_replay(full / "epoch-001.pt", resumed / "epoch-001.pt")
    assert read_json(full / "COMPLETED.json") == read_json(resumed / "COMPLETED.json")
    with pytest.raises(ValueError, match="latest completed"):
        train_hybrid(cfg)

    evaluated = tmp_path / "evaluation.json"
    invoke_main(hybrid_cli.main, [
        "--checkpoint", str(full / "epoch-001.pt"), "--manifest", str(full / "data_manifest.json"),
        "--output", str(evaluated), "--split", "val", "--device", "cpu",
    ])
    assert read_json(evaluated)["scopes"] == final["validation"]["scopes"]
    bound = load_mdd_hybrid_detector(full / "epoch-001.pt")
    assert bound.model.config.input_channels == 5
    assert not bound.model.training
    tampered = copy.deepcopy(final)
    tampered["recipe"]["input_contract"]["channels"] = 2
    with pytest.raises(ValueError):
        restore_hybrid_model(tampered)
    missing_channels = copy.deepcopy(final)
    missing_channels["model_config"].pop("input_channels")
    missing_channels["recipe"]["model"].pop("input_channels")
    with pytest.raises(TypeError, match="input_channels"):
        restore_hybrid_model(missing_channels)

    cfg.resume = None
    cfg.paths.checkpoint_root = str(full)
    cfg.initialization = dict(mode="continual", checkpoint="epoch-001.pt")
    cfg.run.output_dir = "continued"
    with pytest.raises(ValueError, match="new snapshot"):
        train_hybrid(cfg)

    import shutil

    new_store = tmp_path / "cumulative_store"
    shutil.copytree(Path(read_json(manifest)["ball_store"]["directory"]), new_store)
    write_store_clip(new_store, "train/new", [frame(i, ball()) for i in range(32)], size=(32, 32))
    new_manifest = write_manifest(new_store, tmp_path / "cumulative.json")
    # write_manifest's common subset is the original cohort, not new train data.
    moved = data_identity(new_manifest)
    validation_clip = next(clip for clip, split in moved["splits"].items() if split == "val")
    moved["splits"][validation_clip] = "train"
    with pytest.raises(ValueError, match="retain old clips and their splits"):
        initialize_hybrid(full / cfg.initialization.checkpoint, spec, mode="continual",
                          data=moved, image_decode=final["recipe"]["image_decode"])
    cfg.paths.artifact_root, cfg.manifest = str(new_manifest.parent), new_manifest.name
    cfg.epochs, cfg.freeze.encoder_epochs, cfg.freeze.query_epochs = 1, 0, 0
    cfg.warmup_updates = 0
    restored, continuation = initialize_hybrid(full / cfg.initialization.checkpoint, spec, mode="continual",
        data=data_identity(new_manifest), image_decode=final["recipe"]["image_decode"])
    assert all(torch.equal(v, restored.state_dict()[k]) for k, v in final["state_dict"].items())
    assert continuation["initialized_parameters"] == []
    receipt = train_hybrid(cfg)
    assert receipt["complete"] and receipt["global_step"] == 1
    continued = weights(tmp_path / cfg.run.output_dir / "epoch-000.pt")
    assert continued["recipe"]["parent"]["sha256"] != continued["recipe"]["parent"]["ancestry"]["parent"]["sha256"]
    assert continued["recipe"]["data_identity"]["snapshot"] != final["recipe"]["data_identity"]["snapshot"]
    assert continued["validation"]["scopes"]["full"]["presence_frame_fps_pairs"] > 0
