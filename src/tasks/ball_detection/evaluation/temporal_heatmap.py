"""Owner-manifest evaluation for variable-length heatmap-only models."""

from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

from src.tasks.ball_detection.checkpoints.provenance import source_identity
from src.tasks.ball_detection.checkpoints.temporal_heatmap import json_bytes, save_json
from src.tasks.ball_detection.data.windows.all_frames import LABEL_KINDS
from src.tasks.ball_detection.data.windows.dataset import FrozenBallWindowDataset
from src.tasks.ball_detection.evaluation.temporal_heatmap_contract import (
    SeparateTestInputs,
    digest,
    validate_separate_test,
)
from src.tasks.ball_detection.evaluation.temporal_heatmap_metrics import (
    CALIBRATION_METHOD,
    calibrate,
    metrics_report,
)
from src.tasks.ball_detection.model_io.mdd_heatmap import (
    HeatmapModel,
    MDDHeatmapAdapter,
    heatmap_model_kind,
)
from src.tasks.ball_detection.model_io.mdd_query import MDDQueryInput
from src.tasks.ball_detection.models.mdd import MDDHeatmapConfig
from src.tasks.ball_detection.training.runtime.compilation import (
    coordinate_compile_scope,
)
from src.tasks.ball_detection.training.runtime.execution import CoordinateRuntime
from src.tasks.ball_detection.training.runtime.images import (
    coordinate_batches,
    image_input,
    jpeg_decoder_contract,
)
from src.utils.checksum import dual_sha256
from src.utils.data.heatmaps import heatmaps_to_argmax, refine_peaks_log_parabolic

LEGACY_EVALUATION_SCHEMA = "ball_temporal_heatmap_evaluation.v1"
EVALUATION_SCHEMA = "ball_temporal_heatmap_evaluation.v2"
DECODING = "global_argmax_log_parabolic_subpixel; max_native_sigmoid_score"


def validate_calibration(
    report: dict[str, Any], validation: dict[str, Any], *, manifest_sha256: str,
    model: HeatmapModel, artifact_sha256: str, runtime: CoordinateRuntime,
) -> dict[str, Any]:
    summary = validation["summaries"]["val"]
    counts = summary["counts"]
    negatives = counts["reviewed_empty"] + counts["out_of_frame"]
    if counts["observed"] <= 0 or negatives <= 0:
        raise ValueError("Validation needs both observed and confirmed-negative frames for calibration")
    calibration: dict[str, Any] = report["calibration"]
    identities = {k: summary[k] for k in ("target_identity_sha256", "teacher_identity_sha256", "scored_frame_sha256")}
    if (report.get("schema") not in {EVALUATION_SCHEMA, LEGACY_EVALUATION_SCHEMA} or report.get("split") != "val"
            or report["artifact_sha256"] != artifact_sha256 or report["manifest_sha256"] != manifest_sha256
            or report["snapshot"] != validation["ball_store"] or report["identities"] != identities
            or report["rgb_shards_sha256"] != {r["clip"]["clip_id"]: r["rgb_shard_sha256"] for r in validation["clips"]}
            or digest(report["model_config"]) != digest(asdict(model.config)) or report["model_kind"] != heatmap_model_kind(model.config)
            or report["decoding"] != DECODING or report["image_decode"] != jpeg_decoder_contract(runtime.jpeg_decoder)
            or report["runtime"]["precision"] != runtime.precision
            or calibration["split"] != "val" or calibration["method"] != CALIBRATION_METHOD
            or calibration["artifact_sha256"] != artifact_sha256 or calibration["manifest_sha256"] != manifest_sha256
            or calibration["scored_frame_sha256"] != summary["scored_frame_sha256"]
            or calibration["teacher_identity_sha256"] != summary["teacher_identity_sha256"]
            or calibration["observed"] != counts["observed"] or calibration["negatives"] != negatives):
        raise ValueError("Validation calibration does not match this artifact, manifest, T, target set or decoding runtime")
    for total, a, b in ((counts["observed"], "true_positive", "false_negative"),
                        (negatives, "true_negative", "false_positive")):
        if (any(type(calibration[k]) is not int or calibration[k] < 0 for k in (a, b))
                or calibration[a] + calibration[b] != total):
            raise ValueError("Invalid validation calibration confusion counts")
    mode, threshold = calibration["mode"], calibration["threshold"]
    if (mode not in {"threshold", "all_present", "all_absent"}
            or mode == "threshold" and (not isinstance(threshold, (int, float)) or not math.isfinite(threshold) or not 0 <= threshold <= 1)
            or mode != "threshold" and threshold is not None):
        raise ValueError("Invalid saved validation threshold/mode")
    return calibration


def collect_predictions(model: HeatmapModel, dataset: FrozenBallWindowDataset, *, runtime: CoordinateRuntime,
                        device: torch.device) -> list[dict[str, Any]]:
    if (not dataset.all_frames or dataset.manifest["mode"] != "evaluation"
            or dataset.window_length != model.config.frames):
        raise ValueError("Evaluation requires an owner manifest at the saved model T")
    if not isinstance(model.config, MDDHeatmapConfig):
        raise ValueError("Expected an explicit variable-length heatmap model")
    adapter = MDDHeatmapAdapter(model.config)
    records = []
    previous: tuple[str, int, int, str] | None = None
    model.eval()
    loader = runtime.loader(dataset, batch_size=1)
    with torch.inference_mode(), coordinate_compile_scope(model):
        for batch in coordinate_batches(loader, device, prefetch=runtime.image_prefetch):
            rgb = image_input(batch, device)
            call = adapter.build_call(MDDQueryInput(rgb, batch["timestamps"].to(device)))
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=runtime.precision == "bf16"):
                logits = model(*call.args)
            if not bool(torch.isfinite(logits).all()):
                raise ValueError("Nonfinite evaluation logits")
            probability = logits.float().sigmoid()
            uv, score = heatmaps_to_argmax(probability)
            uv = refine_peaks_log_parabolic(probability, uv).cpu()
            score = score.cpu()
            errors = ((uv - batch["uv"]) * (batch["source_size"][:, None] - 1)).norm(dim=-1)
            clip_id = batch["clip_id"][0]
            clip = dataset.store.clip_by_id(clip_id)
            for offset in torch.nonzero(batch["owner_mask"][0], as_tuple=False).flatten().tolist():
                frame = int(batch["frame_indices"][0, offset])
                key = clip_id, frame, 1, batch["native_fps"][0]
                if previous is not None and key <= previous:
                    raise ValueError("Evaluation owner frames must be unique and in manifest order")
                previous = key
                kind = LABEL_KINDS[int(batch["label_kind"][0, offset])]
                row = dataset.store.row_of(clip, frame)
                natural = None
                if kind == "observed":
                    natural = bool(dataset.store.instances["occluded"][dataset.store.frames["inst_start"][row]])
                records.append(dict(
                    clip_id=clip_id, frame_index=frame, frame_step=1, native_fps=key[-1], kind=kind,
                    source=clip.source, source_group=f"{clip.source}/{clip.group_id}", natural_occlusion=natural,
                    pred_uv=uv[0, offset].tolist(), score=float(score[0, offset]),
                    error_px=float(errors[0, offset]) if kind == "observed" else None,
                    pts=int(batch["pts"][0, offset]), time_base=batch["time_base"][0],
                    owner_start=batch["start"][0], owner_offset=offset,
                    context_left=offset, context_right=model.config.frames - 1 - offset, mdd_start=offset == 0,
                    edge_distance=int(batch["edge_distance"][0, offset]), edge_bin=int(batch["edge_bin"][0, offset]),
                    edge_distance_seconds=float(batch["edge_distance_seconds"][0, offset]),
                ))
    for record in records:
        if (not all(math.isfinite(v) for v in [*record["pred_uv"], record["score"]])
                or record["error_px"] is not None and not math.isfinite(record["error_px"])):
            raise ValueError("Nonfinite decoded evaluation output")
    return records


def prediction_identities(records: list[dict[str, Any]]) -> dict[str, Any]:
    digests = {name: hashlib.sha256() for name in ("target_identity", "teacher_identity", "scored_frame")}
    for r in records:
        key = [r["clip_id"], r["frame_index"], 1, r["native_fps"]]
        # This is the manifest's compact JSONL identity, not the report's JSON encoding.
        line = (json.dumps(key, separators=(",", ":")) + "\n").encode()
        digests["target_identity"].update(line)
        code = LABEL_KINDS.index(r["kind"])
        digests["teacher_identity"].update((json.dumps([*key, code], separators=(",", ":")) + "\n").encode())
        if code:
            digests["scored_frame"].update(line)
    return {f"{name}_sha256": digest.hexdigest() for name, digest in digests.items()}


def evaluate_temporal_heatmap(
    model: HeatmapModel, manifest: Path, *, split: str, runtime: CoordinateRuntime, device: torch.device,
    artifact_sha256: str, distance_px: float, calibration_report: dict[str, Any] | None = None,
    output: Path | None = None, separate_test_inputs: SeparateTestInputs | None = None,
) -> dict[str, Any]:
    if not math.isfinite(distance_px) or distance_px <= 0:
        raise ValueError("Distance threshold must be positive source pixels")
    if split not in {"val", "test"} or split == "test" and calibration_report is None and separate_test_inputs is None:
        raise ValueError("Use validation for calibration; test requires its saved validation report")
    if separate_test_inputs is not None and (split != "test" or calibration_report is not None):
        raise ValueError("Separate-test inputs require test and their single explicit calibration file")
    if output is not None and output.exists() and json.loads(output.read_text()).get("schema") != EVALUATION_SCHEMA:
        raise ValueError("Never overwrite or reinterpret a legacy evaluation report")
    started = time.perf_counter()
    evaluation_code = source_identity()
    raw = json.loads(manifest.read_text())
    provenance: dict[str, Any] = dict(mode="same_snapshot")
    validation = raw
    validation_hash = dual_sha256(manifest)
    if separate_test_inputs is not None:
        if output is not None and (output.exists() or output.with_suffix(".frames.jsonl").exists()):
            raise ValueError("Separate-test evaluation requires new output files")
        validation, calibration_report, provenance = validate_separate_test(
            separate_test_inputs, model=model, artifact_sha256=artifact_sha256, test_manifest=raw)
        validation_hash = dual_sha256(separate_test_inputs.validation_manifest)
    calibration = None
    if calibration_report is not None:
        calibration = validate_calibration(calibration_report, validation, manifest_sha256=validation_hash,
                                          model=model, artifact_sha256=artifact_sha256, runtime=runtime)
        provenance["calibration_source"] = dict(manifest_sha256=validation_hash, split="val",
                                                report_identity_sha256=digest(calibration_report))
    dataset = FrozenBallWindowDataset(manifest, split=split, requires_pose=False, jpeg_decoder=runtime.jpeg_decoder)
    records = collect_predictions(model, dataset, runtime=runtime, device=device)
    identities = prediction_identities(records)
    expected = dataset.manifest["summaries"][split]
    if any(expected[key] != value for key, value in identities.items()) or len(records) != expected["counts"]["frames"]:
        raise ValueError("Actual scored/owner/teacher frame set differs from the frozen manifest")
    if calibration is None:
        calibration = calibrate(records, split=split)
        calibration.update(artifact_sha256=artifact_sha256, manifest_sha256=dual_sha256(manifest),
                           scored_frame_sha256=identities["scored_frame_sha256"],
                           teacher_identity_sha256=identities["teacher_identity_sha256"])
    final_code = source_identity()
    if any(final_code[key] != evaluation_code[key] for key in ("source_sha256", "torch_version")):
        raise ValueError("Evaluation code or torch version changed during evaluation")
    report = dict(schema=EVALUATION_SCHEMA, split=split, artifact_sha256=artifact_sha256,
                  manifest_sha256=dual_sha256(manifest), snapshot=dataset.manifest["ball_store"],
                  rgb_shards_sha256={r["clip"]["clip_id"]: r["rgb_shard_sha256"] for r in dataset.manifest["clips"]},
                  model_config=asdict(model.config), model_kind=heatmap_model_kind(model.config),
                  runtime=asdict(runtime), image_decode=jpeg_decoder_contract(runtime.jpeg_decoder),
                  calibration=calibration, identities=identities,
                  decoding=DECODING, provenance=provenance, evaluation_code=evaluation_code,
                  metrics=metrics_report(records, calibration, distance_px=distance_px), seconds=time.perf_counter() - started)
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        predictions = output.with_suffix(".frames.jsonl")
        temporary = predictions.with_suffix(".tmp")
        with temporary.open("wb") as stream:
            for record in records:
                stream.write(json_bytes(record))
        temporary.replace(predictions)
        report["predictions"] = dict(file=predictions.name, sha256=dual_sha256(predictions), frames=len(records))
        save_json(output, report)
    return report
