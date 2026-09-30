"""Capture fixed CPU val windows before the fix, then require byte identity."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from omegaconf import OmegaConf

from src.tasks.ball_detection.data.store import BallFrameStore
from src.tasks.ball_refiner.data.evidence_cache import EvidenceCache
from src.tasks.ball_refiner.data.gaps import fixed_gap_mask, mask_detector_evidence
from src.tasks.ball_refiner.data.inputs import detector_only_input
from src.tasks.ball_refiner.evaluation.detector_selection import validation_clips
from src.tasks.ball_refiner.refiner_2d import build_ball_refiner_2d
from src.tasks.ball_refiner.training.configuration import PilotConfig
from src.utils.checksum import dual_sha256

BUNDLE = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("before", "after"), required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    plan = json.loads((BUNDLE / "plan.json").read_text())
    anchored = next(v for v in plan["variants"] if v["name"] == "anchored_12k")
    runs = {"r18": Path(plan["baseline"]),
            "anchored_12k": Path("/home/kamimura/projects/tennis-lab/outputs") / OmegaConf.load(anchored["training_config"]).run.output_dir}
    arrays: dict[str, Any] = {}
    records = []
    model_hashes = {}
    for name, run in runs.items():
        cfg = PilotConfig.from_config(OmegaConf.load(run / "config.yaml"))
        best = json.loads((run / "best.json").read_text())
        path = run / best["checkpoint"]
        assert dual_sha256(path) == best["checkpoint_sha256"]
        model_hashes[name] = best["checkpoint_sha256"]
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        pair = build_ball_refiner_2d(cfg.model)
        pair.model.load_state_dict(checkpoint["state_dict"], strict=True)
        pair.model.eval()
        store = BallFrameStore(cfg.store)
        cache = EvidenceCache(cfg.evidence, store)
        clips = validation_clips(store)
        # Fixed independent of model errors: clip010 all cameras + lexical first
        # TrackNet/chat val clips, first/middle/last complete 33-frame windows.
        ids = [f"meiji/video_000/clip_010/cam{i}" for i in range(3)]
        ids += [min(r.clip_id for r in clips if r.source == source) for source in ("tracknet", "chat_annotation")]
        for clip_id in ids:
            clip = store.clip_by_id(clip_id)
            assert clip.split == "val"
            evidence = cache.load(clip_id)
            gap = fixed_gap_mask(clip.frame_count, clip_id=clip_id, block_length=cfg.window_length,
                                 lengths=cfg.training.gap_lengths, seed=cfg.partition_seed)
            for start in (0, (clip.frame_count - cfg.window_length) // 2, clip.frame_count - cfg.window_length):
                for condition in ("observed", "evidence_gap"):
                    batch = detector_only_input(evidence, start, cfg.window_length, cfg.model)
                    if condition == "evidence_gap":
                        batch = mask_detector_evidence(batch, torch.from_numpy(gap[start:start + cfg.window_length])[None])
                    call = pair.build_call(batch)
                    ident = f"{name}/{clip_id}/{start}/{condition}"
                    input_hashes = [hashlib.sha256(a.contiguous().numpy().tobytes()).hexdigest() for a in call.args]
                    with torch.inference_mode():
                        raw = pair.model(*call.args)
                        result = pair.decode_output(raw)
                    values = {"raw": raw, **{field: getattr(result, field) for field in
                              ("means", "scale_tril", "mixture_logits", "presence_logits", "covariance", "weights", "presence_probability")}}
                    for field, value in values.items():
                        arrays[f"{ident}/{field}"] = value.numpy()
                    records.append({"id": ident, "frames": cfg.window_length, "input_sha256": input_hashes,
                                    "frame_index": evidence.frame_index[start:start + cfg.window_length].tolist(),
                                    "pts": evidence.pts[start:start + cfg.window_length].tolist()})
    data_path = BUNDLE / "cpu-before.npz"
    meta_path = BUNDLE / "cpu-before.json"
    metadata = {"torch": torch.__version__, "device": "cpu", "threads": 1, "mode": "eager float32 eval/inference_mode",
                "checkpoint_sha256": model_hashes, "windows": records,
                "model_source_sha256": dual_sha256(Path("src/tasks/ball_refiner/refiner_2d/model.py")),
                "selection_rule": "clip010/cam0-2 and lexical first TrackNet/chat val clips; first/middle/last 33 frames; observed/fixed gaps",
                "output_tensor_sha256": {k: hashlib.sha256(v.tobytes()).hexdigest() for k, v in arrays.items()}}
    if args.phase == "before":
        assert not data_path.exists() and not meta_path.exists()
        np.savez_compressed(data_path, **arrays)
        metadata["snapshot_sha256"] = dual_sha256(data_path)
        meta_path.write_text(json.dumps(metadata, indent=2) + "\n")
    else:
        before = json.loads(meta_path.read_text())
        assert dual_sha256(data_path) == before["snapshot_sha256"]
        for key in ("torch", "device", "threads", "mode", "checkpoint_sha256", "windows", "output_tensor_sha256"):
            assert metadata[key] == before[key], key
        with np.load(data_path, allow_pickle=False) as saved:
            assert set(saved.files) == set(arrays)
            for key, value in arrays.items():
                assert value.shape == saved[key].shape and value.dtype == saved[key].dtype
                assert value.tobytes() == saved[key].tobytes(), key
        metadata.update(status="bit_identical", before_model_source_sha256=before["model_source_sha256"],
                        snapshot_sha256=before["snapshot_sha256"], tensors_compared=len(arrays), max_abs_error=0.0,
                        scope="60 windows / 1,980 model-frame evaluations / 480 tensors; CPU only, no CUDA claim")
        (BUNDLE / "cpu-identity.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps({"phase": args.phase, "windows": len(records), "tensors": len(arrays), "checkpoint_sha256": model_hashes}))


if __name__ == "__main__":
    main()

