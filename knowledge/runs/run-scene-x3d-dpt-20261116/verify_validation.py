"""Compare the scene predictor with frozen BF16 training validation owner records.

GPU execution must be submitted to the shared training queue. Inputs, tolerances
and window owners are frozen in validation-plan.json before execution.
"""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time

import numpy as np
import torch
from torchvision.io import ImageReadMode, decode_jpeg

from src.tasks.ball_detection.inference.x3d_dpt import X3DDPTPredictor


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--store", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--compile-mode", choices=("off", "default"), required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    # Exclusive queue reservation plus a cap below physical VRAM; never change
    # precision, input size or window batch in response to memory pressure.
    torch.cuda.set_per_process_memory_fraction(.70)
    plan = json.loads(args.plan.read_text())
    if sha(args.store / "metadata.json") != "7a804cc4e3b3006d0128fbd953b0696712d45acee9dcef8a96c02ec46379037c":
        raise ValueError("Metadata identity differs from frozen training input")
    if sha(args.store / "index.npz") != "0bb00facd10f9961081f9c47af1b62f5c3ca4f5d349a2ac011573d9e510a9feb":
        raise ValueError("Frame ranges differ from frozen training input")
    with np.load(args.store / "index.npz") as index:
        columns = {key: index[key] for key in ("clip", "offset", "length", "frame_index")}
    shards = {}
    for window in plan["windows"]:
        number = window["metadata"]["index"]
        if number in shards:
            continue
        path = args.store / "shards" / f"clip-{number:05d}.bin"
        if sha(path) != window["shard_sha256"]:
            raise ValueError("RGB shard differs from training validation")
        shards[number] = path
    report = dict(schema="scene_dpt_validation_parity.v1", plan_sha256=sha(args.plan),
                  checkpoint_sha256=sha(args.checkpoint), commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                  torch=torch.__version__, gpu=torch.cuda.get_device_name(), memory_fraction=.70,
                  compile_mode=args.compile_mode,
                  input_decoder="torchvision.io.decode_jpeg/nvjpeg (same encoded JPEG bytes)", by_precision={})
    for precision in ("bf16", "fp32"):
        predictor = X3DDPTPredictor.load_from_checkpoint(args.checkpoint, device="cuda", image_size=(720, 1280),
                    frames=32, input_channels=5, normalize_imagenet=False, precision=precision, subpixel_refine=True,
                    compile_mode=args.compile_mode)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        differences = []
        tolerance = plan["tolerances"][precision]
        for window in plan["windows"]:
            number = window["metadata"]["index"]
            rows = np.flatnonzero(columns["clip"] == number)
            rows = rows[window["start"]:window["start"] + 32]
            if columns["frame_index"][rows].tolist() != list(range(window["start"], window["start"] + 32)):
                raise ValueError("Input frame order mismatch")
            encoded = []
            with shards[number].open("rb") as stream:
                for row in rows:
                    stream.seek(int(columns["offset"][row]))
                    encoded.append(torch.from_numpy(np.frombuffer(stream.read(int(columns["length"][row])), np.uint8).copy()))
            rgb = torch.stack(decode_jpeg(encoded, mode=ImageReadMode.RGB, device="cuda")).unsqueeze(0)
            prediction = predictor.predict(rgb)
            scale = np.array([window["metadata"]["source_width"] - 1, window["metadata"]["source_height"] - 1])
            for original in window["rows"]:
                offset = original["owner_offset"]
                uv = prediction.coords[0, offset].numpy()
                score = float(prediction.confidence[0, offset])
                pixel_difference = float(np.abs((uv - original["pred_uv"]) * scale).max())
                score_difference = abs(score - original["score"])
                differences.append(dict(clip_id=original["clip_id"], frame_index=original["frame_index"],
                    kind=original["kind"], source_px_difference=pixel_difference, score_difference=score_difference,
                    passed=pixel_difference <= tolerance["source_px_atol"] and score_difference <= tolerance["score_atol"]))
            print(json.dumps(dict(precision=precision, clip_id=window["clip_id"], start=window["start"], phase="complete")), flush=True)
        torch.cuda.synchronize()
        report["by_precision"][precision] = dict(frames=len(differences),
            max_source_px_difference=max(r["source_px_difference"] for r in differences),
            max_score_difference=max(r["score_difference"] for r in differences),
            mismatches=[r for r in differences if not r["passed"]], all_differences=differences,
            tolerance=tolerance, seconds=time.perf_counter() - started,
            peak_allocated_bytes=torch.cuda.max_memory_allocated(), peak_reserved_bytes=torch.cuda.max_memory_reserved())
        del predictor, rgb, prediction
        torch.cuda.empty_cache()
    report["passed"] = all(not item["mismatches"] for item in report["by_precision"].values())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: {k: v for k, v in value.items() if k not in {"all_differences", "mismatches"}}
                      for key, value in report["by_precision"].items()}), flush=True)
    if not report["passed"]:
        raise ValueError("Validation parity exceeded a predeclared tolerance; see the saved mismatches")


if __name__ == "__main__":
    main()
