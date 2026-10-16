import copy
import json
import os
import subprocess
import sys
from collections import Counter
from dataclasses import asdict
from pathlib import Path

import pytest
import torch
import yaml
from omegaconf import OmegaConf

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
from src.tasks.ball_detection.model_io.mdd_hybrid import load_mdd_hybrid_detector
from src.tasks.ball_detection.models.mdd import DeepMDDHybridDetector, MDDHybridConfig
from src.tasks.ball_detection.training.hybrid.runner import train_hybrid
from src.tasks.ball_detection.training.sampling import HybridWindowSampler
from src.utils.checksum import dual_sha256
from tests.integration.tasks.ball_detection.test_rgb_mdd_contract import config
from tests.support.tasks.ball_detection.hybrid import hybrid_recipe
from tests.support.tasks.ball_detection.store import ball, frame, write_store_clip


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
    clips = [c for c in store.clips if c.clip_id.endswith("main")]
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


def run_cli(args: list[str]) -> None:
    result = subprocess.run([sys.executable, "-m", "src.tasks.ball_detection.scripts.train", *args],
        env=dict(os.environ, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", PYTHONDONTWRITEBYTECODE="1"),
        text=True, capture_output=True, timeout=180)
    assert result.returncode == 0, result.stdout + result.stderr


def weights(path: Path) -> dict:
    saved: dict = torch.load(path, map_location="cpu", weights_only=True)
    return saved


def assert_replay(full: Path, resumed: Path) -> None:
    a, b = weights(full / "epoch-001.pt"), weights(resumed / "epoch-001.pt")
    assert a["recipe"] == b["recipe"] and a["global_step"] == b["global_step"]
    assert torch.equal(a["torch_rng"], b["torch_rng"])
    assert all(torch.equal(value, b["state_dict"][name]) for name, value in a["state_dict"].items())
    assert a["validation"] == b["validation"]


def test_three_five_channel_stages_resume_transfer_evaluate_and_continue(tmp_path: Path) -> None:
    manifest = make_manifest(tmp_path / "original")
    model_file = tmp_path / "dpt.yaml"
    model_file.write_text(yaml.safe_dump(dict(name="mdd_dpt_pretrain", **asdict(config()))))
    common = ["--manifest", str(manifest), "--epochs", "2", "--windows-per-epoch", "2", "--learning-rate", ".001",
              "--warmup-updates", "1", "--seed", "42", "--device", "cpu", "--precision", "fp32",
              "--jpeg-decoder", "opencv", "--input-verification", "lazy", "--num-workers", "0",
              "--compile-mode", "off", "--preview-clips", "0", "--log-every", "1"]
    pre, pre_resume = tmp_path / "pre", tmp_path / "pre_resume"
    first = ["--stage", "dpt_pretrain", *common, "--model-config", str(model_file)]
    run_cli([*first, "--output", str(pre)])
    run_cli([*first, "--output", str(pre_resume), "--stop-after-epoch", "0"])
    run_cli([*first, "--output", str(pre_resume), "--resume", str(pre_resume / "epoch-000.pt")])
    assert_replay(pre, pre_resume)
    aug = Path(__file__).resolve().parents[4] / "src/tasks/ball_detection/configs/augmentation/mdd_posttraining.yaml"
    post, post_resume = tmp_path / "post", tmp_path / "post_resume"
    second = ["--stage", "coordinates", "--initialize", "encoder", *common, "--pretraining-run", str(pre),
              "--freeze-epochs", "1", "--augmentation-config", str(aug)]
    run_cli([*second, "--output", str(post)])
    run_cli([*second, "--output", str(post_resume), "--stop-after-epoch", "0"])
    run_cli([*second, "--output", str(post_resume), "--resume", str(post_resume / "epoch-000.pt")])
    assert_replay(post, post_resume)
    parent_path = post / json.loads((post / "best.json").read_text())["checkpoint"]
    parent = weights(parent_path)
    initial = weights(pre / json.loads((pre / "best.json").read_text())["checkpoint"])
    frozen = weights(post / "epoch-000.pt")
    assert all(torch.equal(v, frozen["state_dict"][k]) for k, v in initial["state_dict"].items() if k.startswith("encoder."))
    assert parent["recipe"]["data_identity"] == initial["recipe"]["data_identity"]
    assert parent["recipe"]["parent"]["ancestry"]["data"] == data_identity(manifest)
    # Stage 3 must not open stage 1's DPT, even when its path was valid during stage 2.
    pre.rename(tmp_path / "unavailable_stage_one")
    cfg = hybrid_recipe(manifest, parent_path, tmp_path / "hybrid")
    torch.manual_seed(42)
    fresh = DeepMDDHybridDetector(MDDHybridConfig(**asdict(config())))
    torch.manual_seed(42)
    transferred, receipt = initialize_hybrid_from_query(parent_path, fresh.config)
    for name in receipt["initialized_parameters"]:
        assert torch.equal(transferred.state_dict()[name], fresh.state_dict()[name])
    assert all(torch.equal(transferred.state_dict()[name], parent["state_dict"][name]) for name in receipt["transferred_parameters"])
    # Exercise the public Hydra stage entry as well as the adapter-callable function.
    cli_recipe = OmegaConf.to_container(cfg, resolve=True)
    assert isinstance(cli_recipe, dict)
    cli_recipe["defaults"] = [{"override hydra/job_logging": "disabled"}, {"override hydra/hydra_logging": "disabled"}]
    cli_recipe["hydra"] = dict(run=dict(dir="."), output_subdir=None, job=dict(chdir=False))
    (tmp_path / "recipe.yaml").write_text(yaml.safe_dump(cli_recipe))
    run_cli(["--stage", "hybrid", "--config-path", str(tmp_path), "--config-name", "recipe"])
    full = Path(cfg.output)
    cfg.output = str(tmp_path / "hybrid_resume")
    cfg.stop_after_epoch = 0
    train_hybrid(cfg)
    resumed = Path(cfg.output)
    checkpoint = weights(resumed / "epoch-000.pt")
    assert all(torch.equal(checkpoint["state_dict"][k], parent["state_dict"][k]) for k in receipt["transferred_parameters"])
    assert any(not torch.equal(checkpoint["state_dict"][k], fresh.state_dict()[k]) for k in receipt["initialized_parameters"])
    # A changed threshold, source or freeze policy cannot be a resume.
    cfg.resume, cfg.stop_after_epoch = str(resumed / "epoch-000.pt"), None
    changed = copy.deepcopy(cfg)
    changed.presence_threshold = .6
    with pytest.raises(ValueError, match="identical recipe"):
        train_hybrid(changed)
    original_bytes = (resumed / "epoch-000.pt").read_bytes()
    broken = copy.deepcopy(checkpoint)
    broken["code"]["source_sha256"] = {"changed.py": "0" * 64}
    torch.save(broken, resumed / "epoch-000.pt")
    with pytest.raises(ValueError, match="source code changed"):
        train_hybrid(cfg)
    (resumed / "epoch-000.pt").write_bytes(original_bytes)
    train_hybrid(cfg)
    assert_replay(full, resumed)
    final = weights(full / "epoch-001.pt")
    assert any(not torch.equal(final["state_dict"][k], parent["state_dict"][k]) for k in receipt["transferred_parameters"])
    with pytest.raises(ValueError, match="latest completed"):
        train_hybrid(cfg)
    evaluated = tmp_path / "evaluation.json"
    result = subprocess.run([sys.executable, "-m", "src.tasks.ball_detection.scripts.evaluate", "--mode", "hybrid",
        "--checkpoint", str(full / "epoch-001.pt"), "--manifest", str(full / "data_manifest.json"),
        "--output", str(evaluated), "--split", "val", "--device", "cpu"],
        text=True, capture_output=True, timeout=180)
    assert result.returncode == 0, result.stdout + result.stderr
    assert json.loads(evaluated.read_text())["scopes"] == final["validation"]["scopes"]
    bound = load_mdd_hybrid_detector(full / "epoch-001.pt")
    assert bound.model.config.input_channels == 5
    assert not bound.model.training
    tampered = copy.deepcopy(final)
    tampered["recipe"]["input_contract"]["channels"] = 2
    with pytest.raises(ValueError):
        restore_hybrid_model(tampered)
    # A new output on the same snapshot is not cumulative learning.
    cfg.resume = None
    cfg.initialization = dict(mode="continual", checkpoint=str(full / "epoch-001.pt"))
    cfg.output = str(tmp_path / "continued")
    with pytest.raises(ValueError, match="new snapshot"):
        train_hybrid(cfg)
    # Retain every old split, then add reviewed cumulative training frames.
    import shutil

    new_store = tmp_path / "cumulative_store"
    shutil.copytree(manifest.parent / "store", new_store)
    write_store_clip(new_store, "train/new", [frame(i, ball()) for i in range(64)], size=(32, 32))
    new_manifest = write_manifest(new_store, tmp_path / "cumulative.json")
    moved = data_identity(new_manifest)
    moved["splits"]["val/main"] = "train"
    with pytest.raises(ValueError, match="retain old clips and their splits"):
        initialize_hybrid(Path(cfg.initialization.checkpoint), fresh.config, mode="continual",
                          data=moved, image_decode=final["recipe"]["image_decode"])
    cfg.manifest = str(new_manifest)
    cfg.epochs, cfg.freeze.encoder_epochs, cfg.freeze.query_epochs = 1, 0, 0
    restored, continuation = initialize_hybrid(Path(cfg.initialization.checkpoint), fresh.config, mode="continual",
        data=data_identity(new_manifest), image_decode=final["recipe"]["image_decode"])
    assert all(torch.equal(v, restored.state_dict()[k]) for k, v in final["state_dict"].items())
    assert continuation["initialized_parameters"] == []
    receipt = train_hybrid(cfg)
    assert receipt["complete"] and receipt["global_step"] == 4
    continued = weights(Path(cfg.output) / "epoch-000.pt")
    assert continued["recipe"]["parent"]["sha256"] != continued["recipe"]["parent"]["ancestry"]["parent"]["sha256"]
    assert continued["recipe"]["data_identity"]["snapshot"] != final["recipe"]["data_identity"]["snapshot"]
    assert continued["validation"]["scopes"]["full"]["presence_frame_fps_pairs"] > 0
