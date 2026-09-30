"""CPU-only cam2 media intervention through the existing pipeline detector/refiner.

Only the reader's returned BGR pixels change. No model/gate/default is edited.
Run each mode separately (about five minutes with two CPU threads).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import resource
import time
from collections.abc import Iterator
from dataclasses import asdict, fields, replace
from pathlib import Path
from typing import Any
from unittest.mock import patch

import cv2
import numpy as np
import torch
from audit_frames import CACHE, PLAN, ROOT, STORE, Inputs, read, source_evidence, write
from audit_pixels import encode_store_frame
from omegaconf import OmegaConf

from src.tasks.ball_detection.data.store import BallFrameStore, shard_name
from src.tasks.ball_detection.inference.checkpoint import load_ball_checkpoint
from src.tasks.ball_detection.inference.predictor import BallDetectionPredictor
from src.tasks.ball_detection.model_io.contracts import (
    BallCandidateConfig,
    BallCandidates,
    BallPrediction,
)
from src.tasks.ball_refiner.data.evidence_cache import EvidenceCache
from src.tasks.ball_refiner.data.inputs import detector_only_sequence_input
from src.tasks.ball_refiner.data.temporal import window_starts
from src.tasks.ball_refiner.deployment import load_inference_bundle
from src.tasks.ball_refiner.inference import predict_sequence
from src.tasks.ball_refiner.pipeline_options import select_ball_path
from src.tennis_scene.configuration import build_ball_detection_config
from src.tennis_scene.pipeline.components.ball_detection import BallDetectionModule
from src.tennis_scene.pipeline.contracts import SourceVideo
from src.utils.configuration import PathResolver, RuntimePathRoots
from src.utils.resource_guard import available_ram_bytes
from src.utils.video import BgrToTensorTransform, FramePacket, OpenCVVideoFrameReader


def main(mode: str) -> None:
    if torch.cuda.is_initialized():
        raise RuntimeError("This experiment must not initialize CUDA")
    torch.set_num_threads(2)
    cv2.setNumThreads(1)
    out = ROOT / "isolation" / "cpu-v3" / f"cam2-{mode}"
    out.mkdir(parents=True, exist_ok=False)
    inputs = Inputs()
    plan = read(inputs.pin(PLAN))
    bundle = load_inference_bundle(Path(plan["bundle"]))
    for name in ("manifest.json", "weights.pt"):
        inputs.pin(bundle.directory / name)
    inputs.pin(Path(plan["detector_checkpoint"]), bundle.detector.checkpoint_sha256)
    meta = read(inputs.pin(STORE / "metadata.json"))
    inputs.pin(STORE / "index.npz")
    inputs.pin(CACHE / "manifest.json")
    store = BallFrameStore(STORE)
    clip = store.clip_by_id("meiji/video_000/clip_010/cam2")
    cache = EvidenceCache(CACHE, store)
    history = next(r for r in cache.manifest["clips"] if r["clip"]["clip_id"] == clip.clip_id)
    inputs.pin(STORE / "shards" / shard_name(clip.index), history["jpeg_shard_sha256"])
    inputs.pin(CACHE / history["file"], history["sha256"])
    cached = cache.load(clip.clip_id)
    row = plan["cameras"][2]
    video_path = inputs.pin(Path(row["video"]), clip.media_sha256)
    video_info = read(Path(plan["report"]) / "cam2/execute.json")["source"]["videos"][0]
    video = SourceVideo("cam2", video_path, clip.media_sha256, 270, video_info["fps"], clip.source_width, clip.source_height)
    saved, settings = source_evidence(plan, "cam2", inputs)
    project = ROOT.parents[2]
    roots = RuntimePathRoots(project_root=project, data_root=STORE.parent.parent,
        checkpoint_root=Path(plan["detector_checkpoint"]).parent, cache_root=out,
        artifact_root=out, output_root=out, external_asset_root=project / "third_party")
    config = dict(settings["config"])
    del config["device"]  # build_ball_detection_config takes this as its separate explicit argument.
    config["enabled"] = True  # Runtime dataclass serialization omits the YAML stage flag.
    config["checkpoint"] = Path(plan["detector_checkpoint"]).name
    config["pin_memory"] = False
    config["prefetch_batches"] = 1
    loaded = load_ball_checkpoint(Path(plan["detector_checkpoint"]), strict=True, weights_only=False)
    if loaded.image_normalization != bundle.detector.normalization:
        raise ValueError("Checkpoint normalization changed")
    transform = BgrToTensorTransform(image_size=bundle.detector.image_size_hw, normalize_imagenet=False)
    starts = window_starts(270, 8, 4)
    batch_records: list[dict[str, Any]] = []
    start = time.monotonic()
    minimum_available = available_ram_bytes()

    class AuditedPredictor(BallDetectionPredictor):
        offset = 0

        def predict(self, images: torch.Tensor, *, candidate_config: BallCandidateConfig) -> BallPrediction:
            nonlocal minimum_available
            selected = starts[self.offset:self.offset + len(images)]
            expected = torch.stack([torch.stack([transform(store.read_bgr(store.row_of(clip, i)))
                                                for i in range(s, s + 8)]) for s in selected])
            identical = torch.equal(images, expected)
            if mode == "reencoded" and not identical:
                raise ValueError("Reencoded input does not exactly reproduce the cached-path RGB batch")
            actual_call = self.adapter.prepare_model_call(images, image_normalization=self.image_normalization)
            expected_call = self.adapter.prepare_model_call(expected, image_normalization=self.image_normalization)
            model_delta = actual_call.model_input - expected_call.model_input
            batch_records.append({"window_starts": list(selected), "rgb_batch_equal_cache": identical,
                "rgb_sha256": hashlib.sha256(images.numpy().tobytes()).hexdigest(),
                "cache_rgb_sha256": hashlib.sha256(expected.numpy().tobytes()).hexdigest(),
                "model_input_shape": list(actual_call.model_input.shape), "model_input_dtype": str(actual_call.model_input.dtype),
                "model_input_abs_mean_delta_cache": float(model_delta.abs().mean()),
                "model_input_max_delta_cache": float(model_delta.abs().max())})
            self.offset += len(images)
            minimum_available = min(minimum_available, available_ram_bytes())
            if minimum_available < 6 * 1024**3:
                raise MemoryError("Host MemAvailable below 6 GiB")
            prediction = super().predict(images, candidate_config=candidate_config)
            print(json.dumps({"mode": mode, "windows": self.offset, "total_windows": len(starts),
                              "seconds": time.monotonic() - start, "available_ram": minimum_available}), flush=True)
            return prediction

    predictor = AuditedPredictor(loaded.model_io, torch.device("cpu"), subpixel_refine=True,
                                 image_normalization=loaded.image_normalization)
    module = BallDetectionModule(build_ball_detection_config(config, PathResolver(roots), device="cpu"))
    module._pipeline = predictor

    def reader(path: Path, *, max_frames: int) -> Iterator[FramePacket]:
        for packet in OpenCVVideoFrameReader(path, max_frames=max_frames):
            if mode == "raw":
                yield packet
                continue
            if mode == "resize_only":
                frame = cv2.resize(packet.frame, (clip.width, clip.height), interpolation=cv2.INTER_AREA)
            else:
                encoded, frame = encode_store_frame(packet.frame, (clip.width, clip.height), meta["jpeg_quality"])
                if not np.array_equal(encoded.reshape(-1), store.read_jpeg(store.row_of(clip, packet.index))):
                    raise ValueError("JPEG intervention must reproduce the frozen store byte-for-byte")
            yield replace(packet, frame=frame)

    with patch("src.tennis_scene.pipeline.components.ball_detection.OpenCVVideoFrameReader", reader):
        _, _, evidence = module._predict_video(video)
    np.testing.assert_array_equal(evidence.selected_window_start, cached.window_start)
    np.testing.assert_array_equal(evidence.selected_time_index, cached.time_index)
    if predictor.offset != len(starts):
        raise ValueError("Incomplete detector windows")
    pair = bundle.load_model()
    pair.model.eval()
    calibration_request = select_ball_path(plan["ball_path"], bundle, Path(plan["calibration_artifact"]))
    if calibration_request is None:
        raise ValueError("The same fixed calibration is required")
    inputs.pin(Path(plan["calibration_artifact"]), calibration_request.sha256)
    calibration = calibration_request.load()
    scale = np.asarray([clip.source_width - 1, clip.source_height - 1], np.float32)
    factor = np.asarray([clip.width - 1, clip.height - 1], np.float32) / (clip.scale * scale)
    raw_candidates = BallCandidates(
        coords=torch.from_numpy(evidence.candidate_uv_px / scale)[None],
        scores=torch.from_numpy(evidence.candidate_scores)[None], valid=torch.from_numpy(evidence.candidate_valid)[None],
        cells=torch.from_numpy(evidence.candidate_cells)[None], patches=torch.from_numpy(evidence.patches)[None],
        patch_valid=torch.from_numpy(evidence.patch_valid)[None], config=evidence.config)
    variants = {"pipeline": raw_candidates,
                "store_coordinate_convention": replace(raw_candidates, coords=raw_candidates.coords * torch.from_numpy(factor)),
                "cached_evidence_replay": cached.candidates}
    arrays: dict[str, Any] = {f"detector_{f.name}": getattr(evidence, f.name) for f in fields(evidence)
                             if isinstance(getattr(evidence, f.name), np.ndarray) and f.name != "heatmaps"}
    for name, cand in variants.items():
        seq = detector_only_sequence_input(cand, torch.from_numpy(cached.timestamps_seconds)[None], bundle.model_config)
        result = predict_sequence(pair, seq, window_length=33, stride=16, batch_size=4, device=torch.device("cpu"))
        prediction = calibration.apply(result.distribution)
        arrays.update({f"{name}_{f.name}": getattr(prediction, f.name)[0].numpy() for f in fields(prediction)})
        arrays[f"{name}_refiner_window_start"] = result.window_start
    np.savez_compressed(out / "predictions.npz", **arrays)
    # Detector differences to the existing CUDA evidence: same-grid and scores, with endpoint correction explicit.
    comparison = {}
    for name, target in (("source_cuda", saved), ("cache_cuda", {
            "candidate_uv_px": cached.candidates.coords[0].numpy() * scale,
            "candidate_scores": cached.candidates.scores[0].numpy(),
            "candidate_cells": cached.candidates.cells[0].numpy()})):
        coords = evidence.candidate_uv_px * factor if name == "cache_cuda" else evidence.candidate_uv_px
        comparison[name] = {"all_rank_native_cells_equal_frames": int(np.all(evidence.candidate_cells == target["candidate_cells"], axis=(1, 2)).sum()),
            "same_rank_native_cells": int(np.all(evidence.candidate_cells == target["candidate_cells"], axis=-1).sum()),
            "score_max_abs": float(np.abs(evidence.candidate_scores - target["candidate_scores"]).max()),
            "same_rank_position_max_abs_px": float(np.abs(coords - target["candidate_uv_px"]).max())}
    inputs.finish(out / "input_sha256.json")
    write(out / "summary.json", {"status": "complete", "mode": mode, "camera": "cam2", "frames": 270,
        "seconds": time.monotonic() - start, "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        "minimum_host_available_bytes": minimum_available, "cuda_initialized": torch.cuda.is_initialized(),
        "normalization": asdict(loaded.image_normalization), "checkpoint_model": OmegaConf.to_container(loaded.config.model, resolve=True),
        "comparison": comparison, "store_coordinate_factor": factor.tolist(), "batches": batch_records})
    print(json.dumps({"complete": mode, "comparison": comparison, "seconds": time.monotonic() - start}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", required=True, choices=("reencoded", "raw", "resize_only"))
    main(parser.parse_args().mode)
