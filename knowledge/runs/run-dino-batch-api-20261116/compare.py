"""Fixed C7 acceptance comparison; run only through the shared GPU queue."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path.cwd()))

from src.submodules.models import (
    DinoPersonDetector,
    PersonDetectionRequest,
    validate_dino_extension,
)
from src.tasks.person_tracking.duplicate_boxes import merge_person_boxes


def sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def main() -> None:
    root = Path(sys.argv[1]).resolve()
    protocol = json.loads((root / "protocol.json").read_text())
    assert not (root / "results.json").exists()
    torch.set_num_threads(4)
    torch.cuda.set_per_process_memory_fraction(protocol["allocated_fraction_cap"])
    extension = validate_dino_extension()
    with np.load(root / "inputs.npz") as archive:
        requests = [PersonDetectionRequest(archive[key]) for key in archive.files]
    inputs = json.loads((root / "inputs.json").read_text())["frames"]
    for request, item in zip(requests, inputs, strict=True):
        assert hashlib.sha256(request.frame_bgr.tobytes()).hexdigest() == item["bgr_sha256"]

    checkpoint = Path("ckpt/dino/checkpoint0029_4scale_swin.pth").resolve()
    detector = DinoPersonDetector(
        checkpoint=checkpoint, repository=Path("third_party/DINO").resolve(),
        device="cuda", confidence=protocol["confidence"],
        short_side=protocol["short_side"], max_long_side=protocol["max_long_side"],
    )
    start = time.perf_counter()
    detector.load()
    torch.cuda.synchronize()
    load_seconds = time.perf_counter() - start
    assert detector._model is not None
    model = detector._model
    spec = importlib.util.spec_from_file_location("dino_scalar_reference", root / "scalar_reference.py")
    assert spec is not None and spec.loader is not None
    reference_module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = reference_module
    spec.loader.exec_module(reference_module)
    reference = reference_module.DinoPersonDetector(
        checkpoint=checkpoint, repository=Path("third_party/DINO").resolve(),
        device="cuda", confidence=protocol["confidence"],
        short_side=protocol["short_side"], max_long_side=protocol["max_long_side"],
    )
    reference._model, reference._loaded = model, True

    captured = []
    def capture(_module, _args, output):
        captured.append({key: output[key].detach().cpu() for key in ("pred_logits", "pred_boxes")})

    hook = model.register_forward_hook(capture)
    baseline = [reference.predict(reference_module.PersonDetectionRequest(req.frame_bgr)) for req in requests]
    baseline_raw = {key: torch.cat([out[key] for out in captured]) for key in captured[0]}
    hook.remove()
    captured.clear()
    baseline_mask = baseline_raw["pred_logits"][:, :, 1].sigmoid() >= protocol["confidence"]

    report = {
        "protocol": protocol, "protocol_sha256": sha256(root / "protocol.json"),
        "inputs_sha256": sha256(root / "inputs.json"), "load_seconds": load_seconds,
        "checkpoint_sha256": sha256(checkpoint), "extension": str(extension),
        "extension_sha256": sha256(extension), "scalar_reference_sha256": sha256(root / "scalar_reference.py"),
        "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "diff_sha256": hashlib.sha256(subprocess.check_output(["git", "diff", "HEAD"])).hexdigest(),
        "dino_commit": subprocess.check_output(["git", "-C", "third_party/DINO", "rev-parse", "HEAD"], text=True).strip(),
        "torch": torch.__version__, "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(), "vram_bytes": torch.cuda.get_device_properties(0).total_memory,
        "allow_tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
        "allow_tf32_cudnn": torch.backends.cudnn.allow_tf32,
        "matmul_precision": torch.get_float32_matmul_precision(), "queue_run": os.environ.get("TENNIS_RUN_ID"),
        "comparisons": {}, "timing": {},
    }
    arrays = {f"baseline_{i}_{key}": getattr(result, key) for i, result in enumerate(baseline)
              for key in ("boxes_xyxy", "scores")}
    all_passed = True
    try:
        for batch_size in protocol["batch_sizes"]:
            captured.clear()
            hook = model.register_forward_hook(capture)
            actual = detector.predict_batch(requests, batch_size=batch_size)
            hook.remove()
            raw = {key: torch.cat([out[key] for out in captured]) for key in captured[0]}
            raw_mask = raw["pred_logits"][:, :, 1].sigmoid() >= protocol["confidence"]
            raw_selection_same = bool(torch.equal(raw_mask, baseline_mask))
            max_score, max_box = 0., 0.
            rows = []
            for i, (old, new) in enumerate(zip(baseline, actual, strict=True)):
                arrays.update({f"batch{batch_size}_{i}_{key}": getattr(new, key) for key in ("boxes_xyxy", "scores")})
                shapes_same = old.boxes_xyxy.shape == new.boxes_xyxy.shape
                score_diff = float(np.max(np.abs(old.scores - new.scores), initial=0)) if shapes_same else float("inf")
                box_diff = float(np.max(np.abs(old.boxes_xyxy - new.boxes_xyxy), initial=0)) if shapes_same else float("inf")
                max_score, max_box = max(max_score, score_diff), max(max_box, box_diff)
                old_queries = torch.where(baseline_mask[i])[0]
                new_queries = torch.where(raw_mask[i])[0]
                old_order = baseline_raw["pred_logits"][i, old_queries, 1].sigmoid().argsort(descending=True)
                new_order = raw["pred_logits"][i, new_queries, 1].sigmoid().argsort(descending=True)
                query_order_same = bool(torch.equal(old_queries[old_order], new_queries[new_order]))
                merge_checks = {}
                for enabled in (False, True):
                    old_keep, old_drops = merge_person_boxes(i, np.arange(len(old.scores), dtype=np.int64), old.boxes_xyxy, old.scores, enabled=enabled)
                    new_keep, new_drops = merge_person_boxes(i, np.arange(len(new.scores), dtype=np.int64), new.boxes_xyxy, new.scores, enabled=enabled)
                    old_pairs = [(d.kept_row, d.dropped_row) for d in old_drops]
                    new_pairs = [(d.kept_row, d.dropped_row) for d in new_drops]
                    merge_checks[str(enabled)] = {"same": bool(np.array_equal(old_keep, new_keep) and old_pairs == new_pairs),
                        "kept_rows": new_keep.tolist(), "dropped_rows": new_pairs}
                passed = (shapes_same and query_order_same and score_diff <= protocol["score_atol"]
                          and box_diff <= protocol["box_atol_original_pixels"] and all(v["same"] for v in merge_checks.values()))
                rows.append({"input": inputs[i]["key"], "detections": len(new.scores), "same_shape": shapes_same,
                    "query_order_same": query_order_same, "score_max_abs_diff": score_diff,
                    "box_max_abs_diff_original_pixels": box_diff, "merge": merge_checks, "passed": passed})
            passed = all(row["passed"] for row in rows) and raw_selection_same
            all_passed &= passed
            report["comparisons"][str(batch_size)] = {"passed": passed, "raw_threshold_mask_same": raw_selection_same,
                "score_max_abs_diff": max_score, "box_max_abs_diff_original_pixels": max_box, "frames": rows}
            torch.cuda.empty_cache()
            for _ in range(protocol["warmups_per_batch_size"]):
                detector.predict_batch(requests, batch_size=batch_size)
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            seconds = []
            for _ in range(protocol["timed_repeats"]):
                torch.cuda.synchronize()
                start = time.perf_counter()
                detector.predict_batch(requests, batch_size=batch_size)
                torch.cuda.synchronize()
                seconds.append(time.perf_counter() - start)
            report["timing"][str(batch_size)] = {"seconds": seconds, "median_seconds": statistics.median(seconds),
                "frames_per_second": len(requests) / statistics.median(seconds),
                "peak_allocated_bytes": torch.cuda.max_memory_allocated(), "peak_reserved_bytes": torch.cuda.max_memory_reserved()}
            print(json.dumps({"batch_size": batch_size, "passed": passed,
                "max_score_diff": max_score, "max_box_diff": max_box, "timing": report["timing"][str(batch_size)]}), flush=True)
        report["passed"] = all_passed
    finally:
        np.savez_compressed(root / "predictions.npz", **arrays)
        (root / "results.json").write_text(json.dumps(report, indent=2) + "\n")
        reference._model = None
        detector.unload()
    if not all_passed:
        raise SystemExit("DINO batch parity failed; inspect results.json")


if __name__ == "__main__":
    main()
