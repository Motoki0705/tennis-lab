"""Separate portable stores must prove source correspondence and validation provenance."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pytest
import torch

from src.tasks.ball_detection.checkpoints.temporal_heatmap import save_json
from src.tasks.ball_detection.data.distribution import prepare_distribution
from src.tasks.ball_detection.data.distribution.manifest import distribution_files
from src.tasks.ball_detection.data.store import BallFrameStore
from src.tasks.ball_detection.data.windows.all_frame_preparation import (
    prepare_all_frame_training,
)
from src.tasks.ball_detection.data.windows.all_frames import AllFrameWindowConfig
from src.tasks.ball_detection.evaluation.temporal_heatmap import (
    EVALUATION_SCHEMA,
    LEGACY_EVALUATION_SCHEMA,
    evaluate_temporal_heatmap,
    validate_calibration,
)
from src.tasks.ball_detection.evaluation.temporal_heatmap_contract import (
    SeparateTestInputs,
    build_evaluation_contract,
)
from src.tasks.ball_detection.model_io.mdd_heatmap import restore_heatmap_model
from src.tasks.ball_detection.training.runtime.execution import CoordinateRuntime
from src.tasks.ball_detection.training.temporal_heatmap.runner import (
    train_temporal_heatmap,
)
from src.tasks.ball_detection.visualization.presentation.temporal_heatmap import (
    render_temporal_heatmap_comparison,
)
from src.utils.checksum import dual_sha256
from tests.support.tasks.ball_detection.store import ball, frame, write_store_clip
from tests.support.tasks.ball_detection.temporal_heatmap import (
    prepare_fixture,
    small_model,
    update_budget,
)

RUNTIME = CoordinateRuntime(cpu_threads=1)
RUNTIME_ARGS = ["--device", "cpu", "--precision", "fp32", "--jpeg-decoder", "opencv", "--compile-mode", "off",
                "--input-verification", "upfront", "--cpu-threads", "1"]


def read(path: Path) -> dict[str, Any]:
    result: dict[str, Any] = json.loads(path.read_text())
    return result


def prepare_stores(root: Path) -> dict[str, Path]:
    prepare_fixture(root)
    source = root / "store"
    # Notes are original evidence; a self-consistent transfer manifest must not
    # turn a noted empty frame into a confirmed negative.
    write_store_clip(source, "chat_annotation/test/notes", [
        frame(i, ball(xy=(10., 12.))) if i < 72 else frame(i) for i in range(144)
    ], source="chat_annotation", split="test", size=(24, 24))
    annotation = root / "original-chat.json"
    save_json(annotation, dict(schema_version="tennis_chat_ball_annotation.v1", clip_id="test/notes", frame_count=144,
        frames=[dict(frame_index=i, reviewed=True, notes="uncertain" if i == 80 else "", balls=[{}] if i < 72 else [])
                for i in range(144)]))
    metadata = read(source / "metadata.json")
    metadata["clips"][-1].update(annotation_path=str(annotation), annotation_sha256=dual_sha256(annotation))
    save_json(source / "metadata.json", metadata)
    store = BallFrameStore(source)
    result = dict(source=source)
    configs = [AllFrameWindowConfig(t) for t in (32, 64, 128)]
    for name, splits in (("training", {"train", "val"}), ("test", {"test"})):
        stage, destination = root / "stage" / name, root / "elsewhere" / name
        prepare_distribution(source, stage, version=f"separate-{name}-v1",
                             clip_ids=[c.clip_id for c in store.clips if c.split in splits], inputs={})
        for relative, path in distribution_files(stage, source_directory=source).items():
            target = destination / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target)
        shutil.copyfile(stage / "distribution.json", destination / "distribution.json")
        prepared = root / f"{name}-manifests"
        prepare_all_frame_training(destination, prepared, configs=configs)
        result[f"{name}_root"] = destination
        result[f"{name}_manifests"] = prepared
        result[f"{name}_distribution"] = destination / "distribution.json"
    contract = build_evaluation_contract(source_store=source, training_distribution=result["training_distribution"],
        test_distribution=result["test_distribution"],
        validation_manifests=[result["training_manifests"] / f"evaluation_T{t}.json" for t in (32, 64, 128)],
        test_manifests=[result["test_manifests"] / f"evaluation_T{t}.json" for t in (32, 64, 128)])
    result["contract"] = root / "contract.json"
    save_json(result["contract"], contract)
    return result


def train_fixture(root: Path, *, architecture: str = "dpt", frames: int = 32) -> dict[str, Path]:
    paths = prepare_stores(root)
    pretrain = root / "pretraining"
    options = dict(architecture=architecture, phase="pretrain", model_config=small_model(architecture),
                   manifest=paths["training_manifests"] / "train_T32.json",
                   evaluation_manifest=paths["training_manifests"] / "evaluation_T32.json", output=pretrain,
                   budget=update_budget(updates=1, warmup_updates=0, evaluation_interval=1), runtime=RUNTIME,
                   device=torch.device("cpu"), augmentation=None, distance_px=20.)
    train_temporal_heatmap(**options)
    output = pretrain
    if frames != 32:
        parent = pretrain / "weights-update-00000001.pt"
        output = root / "finetuning"
        options.update(phase="finetune", model_config=small_model(architecture, frames=frames),
                       manifest=paths["training_manifests"] / f"train_T{frames}.json",
                       evaluation_manifest=paths["training_manifests"] / f"evaluation_T{frames}.json", output=output,
                       initial_artifact=parent, initial_artifact_sha256=dual_sha256(parent), parent_update=1)
        train_temporal_heatmap(**options)
    paths.update(artifact=output / "weights-update-00000001.pt", run=output / "run.json",
                 validation=output / "evaluation_manifest.json", calibration=output / "validation-update-00000001.json",
                 test=paths["test_manifests"] / f"evaluation_T{frames}.json")
    # The final VM only needs this small file from the training distribution.
    directory = root / "training-artifacts"
    directory.mkdir()
    shutil.copyfile(paths["training_distribution"], directory / "distribution.json")
    paths["training_distribution"] = directory / "distribution.json"
    return paths


def inputs(paths: dict[str, Path]) -> SeparateTestInputs:
    return SeparateTestInputs(artifact=paths["artifact"], training_run_record=paths["run"],
        validation_manifest=paths["validation"], training_distribution=paths["training_distribution"],
        test_distribution=paths["test_distribution"], evaluation_contract=paths["contract"],
        evaluation_contract_sha256=dual_sha256(paths["contract"]), calibration=paths["calibration"],
        calibration_sha256=dual_sha256(paths["calibration"]))


def evaluate(paths: dict[str, Path], **kwargs: Any) -> dict[str, Any]:
    model = restore_heatmap_model(torch.load(paths["artifact"], map_location="cpu", weights_only=True))
    options = dict(split="test", runtime=RUNTIME, device=torch.device("cpu"),
                   artifact_sha256=dual_sha256(paths["artifact"]), distance_px=20., separate_test_inputs=inputs(paths))
    options.update(kwargs)
    result: dict[str, Any] = evaluate_temporal_heatmap(model, paths["test"], **options)
    return result


def cli(paths: dict[str, Path], output: Path, *, split: str) -> subprocess.CompletedProcess[str]:
    command = [sys.executable, "-m", "src.tasks.ball_detection.scripts.evaluate", "--mode", "temporal_heatmap",
               "--artifact", str(paths["artifact"]), "--manifest", str(paths["validation"] if split == "val" else paths["test"]),
               "--split", split, "--output", str(output), "--distance-px", "20", *RUNTIME_ARGS]
    if split == "test":
        extra = dict(calibration=paths["calibration"], calibration_sha256=dual_sha256(paths["calibration"]),
                     training_run_record=paths["run"], validation_manifest=paths["validation"],
                     training_distribution=paths["training_distribution"], test_distribution=paths["test_distribution"],
                     evaluation_contract=paths["contract"], evaluation_contract_sha256=dual_sha256(paths["contract"]))
        command += [item for name, value in extra.items() for item in ("--" + name.replace("_", "-"), str(value))]
    return subprocess.run(command, capture_output=True, text=True, timeout=90,
                          env=dict(os.environ, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1"))


@pytest.fixture(scope="module")
def fixed(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    return train_fixture(tmp_path_factory.mktemp("separate-test"))


@pytest.mark.parametrize("architecture", ["dpt", "hybrid"])
@pytest.mark.parametrize("frames", [32, 64, 128])
def test_real_cli_uses_saved_validation_calibration_with_only_relocated_test_payloads(
    tmp_path: Path, architecture: str, frames: int,
) -> None:
    paths = train_fixture(tmp_path, architecture=architecture, frames=frames)
    paths["calibration"] = tmp_path / "validation-output" / "calibration.json"
    result = cli(paths, paths["calibration"], split="val")
    assert result.returncode == 0, result.stdout + result.stderr
    validation = read(paths["calibration"])
    assert validation["schema"] == EVALUATION_SCHEMA
    # Relocate after pinning; do not require source/train/val payloads on the test VM.
    relocated = tmp_path / "test-vm" / "dataset"
    relocated.parent.mkdir()
    shutil.move(paths["test_root"], relocated)
    manifest = read(paths["test"])
    manifest["ball_store"]["directory"] = str(relocated)
    paths["test"] = tmp_path / "test-vm" / "evaluation.json"
    save_json(paths["test"], manifest)
    paths["test_distribution"] = relocated / "distribution.json"
    shutil.rmtree(paths["source"])
    shutil.rmtree(paths["training_root"])
    output = tmp_path / "test-output" / "evaluation.json"
    result = cli(paths, output, split="test")
    assert result.returncode == 0, result.stdout + result.stderr
    report = read(output)
    assert report["calibration"] == validation["calibration"]
    assert report["provenance"]["validation_manifest_sha256"] == dual_sha256(paths["validation"])
    assert report["provenance"]["calibration_report_sha256"] == dual_sha256(paths["calibration"])
    assert report["provenance"]["evaluation_contract_sha256"] == dual_sha256(paths["contract"])
    assert report["provenance"]["mode"] == "separate_distributions"
    assert report["predictions"]["frames"] == manifest["summaries"]["test"]["counts"]["frames"]
    assert manifest["summaries"]["val"]["counts"]["frames"] == 0
    assert report["evaluation_code"]["source_sha256"]
    assert cli(paths, output, split="test").returncode != 0


@pytest.mark.parametrize("change", ["mapping", "missing_clip", "extra_clip", "source_snapshot", "split",
                                   "gt", "pts", "teacher", "rgb", "owners", "common_frames"])
def test_fixed_contract_rejects_corrupted_correspondence_and_payloads(
    fixed: dict[str, Path], tmp_path: Path, change: str,
) -> None:
    paths = fixed.copy()
    root = tmp_path / "test-store"
    shutil.copytree(fixed["test_root"], root)
    paths["test_distribution"] = root / "distribution.json"
    paths["test"] = tmp_path / "test.json"
    manifest = read(fixed["test"])
    manifest["ball_store"]["directory"] = str(root)
    distribution = read(paths["test_distribution"])
    if change == "mapping":
        distribution["clips"][0]["source_index"] = 999
    elif change == "source_snapshot":
        distribution["source"]["hashes"]["index.npz"] = "0" * 64
    elif change in {"missing_clip", "extra_clip"}:
        if change == "missing_clip":
            manifest["clips"].pop()
        else:
            manifest["clips"].append(manifest["clips"][0].copy())
    elif change == "split":
        manifest["clips"][0]["clip"]["split"] = "train"
    elif change in {"gt", "pts", "teacher"}:
        with np.load(root / "index.npz") as archive:
            columns = {k: archive[k] for k in archive.files}
        if change == "gt":
            columns["xy"][0, 0] += 1
        elif change == "pts":
            columns["pts"][1] += 1
        else:
            columns["annotated"][0] = False
        np.savez(root / "index.npz", **columns)
    elif change == "rgb":
        shard = root / distribution["clips"][0]["payload"]
        payload = bytearray(shard.read_bytes())
        payload[len(payload) // 2] ^= 1
        shard.write_bytes(payload)
    elif change == "owners":
        manifest["clips"][0]["windows"][0]["owner_range"][1] -= 1
    else:
        manifest["clips"][0]["common_intervals"][0][1] -= 1
    save_json(paths["test_distribution"], distribution)
    save_json(paths["test"], manifest)
    with pytest.raises(ValueError):
        evaluate(paths, output=tmp_path / "result.json")
    assert not (tmp_path / "result.json").exists()


@pytest.mark.parametrize("change", ["mapping", "payload_mapping", "source_snapshot", "split", "gt", "pts", "teacher", "annotation", "rgb", "missing_clip", "extra_clip", "owners"])
def test_contract_generation_cannot_pin_a_false_source_correspondence(
    fixed: dict[str, Path], tmp_path: Path, change: str,
) -> None:
    root = tmp_path / "test-store"
    shutil.copytree(fixed["test_root"], root)
    distribution = read(root / "distribution.json")
    if change == "mapping":
        distribution["clips"][0]["source_index"] = 999
    elif change == "payload_mapping":
        original = distribution["clips"][0]["payload"]
        distribution["clips"][0]["payload"] = "shards/wrong-index.bin"
        for entry in distribution["files"]:
            if entry["path"] == original:
                entry["path"] = "shards/wrong-index.bin"
    elif change == "source_snapshot":
        distribution["source"]["hashes"]["metadata.json"] = "0" * 64
    elif change == "split":
        metadata = read(root / "metadata.json")
        metadata["clips"][0]["split"] = "train"
        save_json(root / "metadata.json", metadata)
    elif change == "annotation":
        clip = next(c for c in read(root / "metadata.json")["clips"] if c["source"] == "chat_annotation")
        path = root / clip["annotation_path"]
        original = read(path)
        original["frames"][80]["notes"] = ""
        save_json(path, original)
        entry = next(e for e in distribution["files"] if e["path"] == clip["annotation_path"])
        entry.update(sha256=dual_sha256(path), size_bytes=path.stat().st_size)
    elif change in {"gt", "pts", "teacher"}:
        with np.load(root / "index.npz") as archive:
            columns = {k: archive[k] for k in archive.files}
        if change == "gt":
            columns["xy"][0, 0] += 1
        elif change == "pts":
            columns["pts"] += 1
        else:
            columns["annotated"][0] = False
        np.savez(root / "index.npz", **columns)
    elif change == "rgb":
        distribution["files"][-1]["sha256"] = "0" * 64
    for entry in distribution["files"]:
        if entry["path"] in {"index.npz", "metadata.json"}:
            entry.update(sha256=dual_sha256(root / entry["path"]), size_bytes=(root / entry["path"]).stat().st_size)
    save_json(root / "distribution.json", distribution)
    manifests = []
    for length in (32, 64, 128):
        manifest = read(fixed["test_manifests"] / f"evaluation_T{length}.json")
        manifest["ball_store"]["directory"] = str(root)
        manifest["ball_store"]["hashes"] = {n: dual_sha256(root / n) for n in ("index.npz", "metadata.json")}
        if change == "missing_clip":
            manifest["clips"].clear()
        elif change == "extra_clip":
            manifest["clips"].append(manifest["clips"][0].copy())
        elif change == "owners":
            manifest["clips"][0]["windows"][0]["owner_range"][1] -= 1
        path = tmp_path / f"evaluation_T{length}.json"
        save_json(path, manifest)
        manifests.append(path)
    with pytest.raises(ValueError):
        build_evaluation_contract(source_store=fixed["source"],
            training_distribution=fixed["training_root"] / "distribution.json", test_distribution=root / "distribution.json",
            validation_manifests=[fixed["training_manifests"] / f"evaluation_T{t}.json" for t in (32, 64, 128)],
            test_manifests=manifests)


@pytest.mark.parametrize("field", ["schema", "report_split", "calibration_split", "artifact", "manifest", "calibration_manifest",
                                  "calibration_artifact", "frames", "kind", "decoder", "precision", "decoding", "teacher",
                                  "scored", "target", "observed", "negatives", "confusion", "method", "threshold"])
def test_calibration_provenance_is_checked_against_the_explicit_validation_manifest(
    fixed: dict[str, Path], tmp_path: Path, field: str,
) -> None:
    paths = fixed.copy()
    report = read(fixed["calibration"])
    cal = report["calibration"]
    if field == "schema":
        report["schema"] = "unknown.v1"
    elif field == "report_split":
        report["split"] = "test"
    elif field == "calibration_split":
        cal["split"] = "train"
    elif field == "artifact":
        report["artifact_sha256"] = "0" * 64
    elif field == "manifest":
        report["manifest_sha256"] = "0" * 64
    elif field == "calibration_manifest":
        cal["manifest_sha256"] = "0" * 64
    elif field == "calibration_artifact":
        cal["artifact_sha256"] = "0" * 64
    elif field == "frames":
        report["model_config"]["frames"] = 64
    elif field == "kind":
        report["model_kind"] = "mdd_heatmap_hybrid"
    elif field == "decoder":
        report["image_decode"]["version"] = "different"
    elif field == "precision":
        report["runtime"]["precision"] = "bf16"
    elif field == "decoding":
        report["decoding"] = "argmax_without_subpixel"
    elif field in {"teacher", "scored"}:
        cal["teacher_identity_sha256" if field == "teacher" else "scored_frame_sha256"] = "0" * 64
    elif field == "target":
        report["identities"]["target_identity_sha256"] = "0" * 64
    elif field in {"observed", "negatives"}:
        cal[field] = 0
    elif field == "confusion":
        cal["false_negative"] = -1
    elif field == "method":
        cal["method"] = "test_optimized"
    else:
        cal.update(mode="threshold", threshold=float("inf"))
    paths["calibration"] = tmp_path / "calibration.json"
    paths["calibration"].write_text(json.dumps(report))
    with pytest.raises(ValueError):
        evaluate(paths)


@pytest.mark.parametrize("which", ["observed", "negative"])
def test_empty_validation_support_cannot_be_rescued_by_test(
    fixed: dict[str, Path], which: str,
) -> None:
    validation = read(fixed["validation"])
    counts = validation["summaries"]["val"]["counts"]
    if which == "observed":
        counts["observed"] = 0
    else:
        counts["reviewed_empty"] = counts["out_of_frame"] = 0
    model = restore_heatmap_model(torch.load(fixed["artifact"], weights_only=True))
    with pytest.raises(ValueError, match="both observed and confirmed-negative"):
        validate_calibration(read(fixed["calibration"]), validation, manifest_sha256=dual_sha256(fixed["validation"]),
            model=model, artifact_sha256=dual_sha256(fixed["artifact"]), runtime=RUNTIME)


def test_explicit_hashes_run_record_and_api_have_no_alternate_source_fallback(
    fixed: dict[str, Path], tmp_path: Path,
) -> None:
    original = inputs(fixed)
    for changed in (replace(original, evaluation_contract_sha256="0" * 64),
                    replace(original, calibration_sha256="0" * 64)):
        with pytest.raises(ValueError, match="SHA-256 mismatch"):
            evaluate(fixed, separate_test_inputs=changed)
    run = read(fixed["run"])
    run["recipe"]["budget"]["seed"] += 1
    path = tmp_path / "run.json"
    save_json(path, run)
    with pytest.raises(ValueError, match="run record/recipe"):
        evaluate(fixed, separate_test_inputs=replace(original, training_run_record=path))
    with pytest.raises(ValueError, match="both observed"):
        evaluate(fixed, separate_test_inputs=None, calibration_report=read(fixed["calibration"]))
    with pytest.raises(ValueError, match="single explicit calibration"):
        evaluate(fixed, calibration_report=read(fixed["calibration"]))


def test_api_checks_the_actual_weights_and_exact_validation_manifest_bytes(
    fixed: dict[str, Path], tmp_path: Path,
) -> None:
    model = restore_heatmap_model(torch.load(fixed["artifact"], map_location="cpu", weights_only=True))
    with torch.no_grad():
        next(model.parameters()).add_(1)
    with pytest.raises(ValueError, match="model weights differ"):
        evaluate_temporal_heatmap(model, fixed["test"], split="test", runtime=RUNTIME, device=torch.device("cpu"),
            artifact_sha256=dual_sha256(fixed["artifact"]), distance_px=20., separate_test_inputs=inputs(fixed))
    changed = tmp_path / "validation.json"
    changed.write_bytes(fixed["validation"].read_bytes() + b"\n")
    with pytest.raises(ValueError, match="Validation manifest differs"):
        evaluate(fixed, separate_test_inputs=replace(inputs(fixed), validation_manifest=changed))
    contract = read(fixed["contract"])
    contract["schema"] = "unknown.v1"
    changed = tmp_path / "contract.json"
    save_json(changed, contract)
    with pytest.raises(ValueError, match="Unknown separate-test"):
        evaluate(fixed, separate_test_inputs=replace(inputs(fixed), evaluation_contract=changed,
                                                   evaluation_contract_sha256=dual_sha256(changed)))


def test_evaluation_source_change_is_rejected_before_publishing_results(
    fixed: dict[str, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src.tasks.ball_detection.evaluation import temporal_heatmap as evaluation

    code = evaluation.source_identity()
    changed = {**code, "source_sha256": {**code["source_sha256"], "changed.py": "0" * 64}}
    observations = iter([code, changed])
    monkeypatch.setattr(evaluation, "source_identity", lambda: next(observations))
    output = tmp_path / "result.json"
    with pytest.raises(ValueError, match="changed during evaluation"):
        evaluate(fixed, output=output)
    assert not output.exists() and not output.with_suffix(".frames.jsonl").exists()


def test_evaluation_accepts_resume_with_unchanged_code_content_and_new_checkout_metadata(
    fixed: dict[str, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src.tasks.ball_detection.training.temporal_heatmap import runner

    output = tmp_path / "resumed"
    options: dict[str, Any] = dict(architecture="dpt", phase="pretrain", model_config=small_model("dpt"),
        manifest=fixed["training_manifests"] / "train_T32.json",
        evaluation_manifest=fixed["training_manifests"] / "evaluation_T32.json", output=output,
        budget=update_budget(updates=2, warmup_updates=0, evaluation_interval=2), runtime=RUNTIME,
        device=torch.device("cpu"), augmentation=None, distance_px=20.)
    train_temporal_heatmap(**options, stop_after_update=1)
    original = read(output / "run.json")
    relocated_code = {**runner.source_identity(), "base_commit": "f" * 40, "directory": "/separate-source-checkout"}
    monkeypatch.setattr(runner, "source_identity", lambda: relocated_code)
    assert train_temporal_heatmap(**options, resume=output / "checkpoint-update-00000001.pt")["complete"]
    paths = {**fixed, "artifact": output / "weights-update-00000002.pt", "run": output / "run.json",
             "calibration": output / "validation-update-00000002.json", "validation": output / "evaluation_manifest.json"}
    report = evaluate(paths)
    assert report["calibration"] == read(paths["calibration"])["calibration"]
    assert report["provenance"]["training_code"] == relocated_code
    assert read(paths["run"])["code"] == original["code"]


def test_v2_video_cli_uses_explicit_relocated_snapshot_and_keeps_legacy_reports(
    fixed: dict[str, Path], tmp_path: Path,
) -> None:
    first = tmp_path / "first.json"
    evaluate(fixed, output=first)
    second = tmp_path / "second.json"
    shutil.copyfile(first, second)
    relocated = tmp_path / "relocated"
    shutil.copytree(fixed["test_root"], relocated)
    video = tmp_path / "comparison.mp4"
    command = [sys.executable, "-m", "src.tasks.ball_detection.scripts.visualize_temporal_heatmaps",
               "--evaluations", str(first), str(second), "--clip-id", "tracknet/test/dense", "--start", "0", "--stop", "8",
               "--snapshot-root", str(relocated), "--output", str(video), "--panel-width", "160"]
    result = subprocess.run(command, capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
    capture = cv2.VideoCapture(str(video))
    assert capture.isOpened() and int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) == 8
    capture.release()
    for mutation in ("missing_root", "wrong_root", "unknown_schema", "prediction_hash", "rgb"):
        report = read(first)
        root: Path | None = relocated
        if mutation == "missing_root":
            root = None
        elif mutation == "wrong_root":
            root = fixed["training_root"]
        elif mutation == "unknown_schema":
            report["schema"] = "unknown.v1"
        elif mutation == "prediction_hash":
            report["predictions"]["sha256"] = "0" * 64
        else:
            report["rgb_shards_sha256"]["tracknet/test/dense"] = "0" * 64
        changed = tmp_path / "changed.json"
        save_json(changed, report)
        with pytest.raises(ValueError):
            render_temporal_heatmap_comparison([changed, second], clip_id="tracknet/test/dense", start=0, stop=8,
                output=tmp_path / f"{mutation}.mp4", snapshot_root=root)
    report = read(first)
    report["schema"] = LEGACY_EVALUATION_SCHEMA
    report.pop("provenance")
    report.pop("evaluation_code")
    save_json(first, report)
    before = first.read_bytes()
    render_temporal_heatmap_comparison([first, second], clip_id="tracknet/test/dense", start=0, stop=8,
                                       output=tmp_path / "legacy.mp4", snapshot_root=relocated)
    with pytest.raises(ValueError, match="Never overwrite"):
        evaluate(fixed, output=first)
    assert first.read_bytes() == before
