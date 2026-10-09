"""Measure nvJPEG decode and its difference from the saved OpenCV RGB contract."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torchvision.io import ImageReadMode, decode_jpeg

from src.tasks.ball_detection.data.coordinate_dataset import CoordinateWindowDataset
from src.tasks.ball_detection.preprocessing import RGBToMDD


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Choose a fresh report")
    torch.set_num_threads(2)
    data = CoordinateWindowDataset(args.manifest, split="train", requires_pose=False)
    chosen: dict[tuple[str, int], int] = {}
    for i, (record, window) in enumerate(data.windows):
        chosen.setdefault((data.records[record]["clip"]["source"], window.frame_step), i)
    result: dict[str, Any] = dict(status="running", rows=[], torch=str(torch.__version__))
    for (source, step), i in sorted(chosen.items()):
        record_index, window = data.windows[i]
        record = data.records[record_index]
        clip = data.store.clip_by_id(record["clip"]["clip_id"])
        data._verify_clip(record, clip)
        jpegs = [torch.from_numpy(np.frombuffer(data.store.read_jpeg(data.store.row_of(clip, int(f))), np.uint8).copy())
                 for f in window.indices()]
        reference = data[i]["rgb"]
        decoded = torch.stack(decode_jpeg(jpegs, mode=ImageReadMode.RGB, device="cuda"))
        delta = (decoded.cpu().to(torch.int16) - reference.to(torch.int16)).abs().float()
        rgb_reference = reference.cuda()
        transform = RGBToMDD().cuda()
        with torch.no_grad():
            diff = (transform(decoded[None]) - transform(rgb_reference[None])).abs()
            mdd_mean, mdd_max = float(diff.mean()), float(diff.max())
        del rgb_reference, diff
        torch.cuda.synchronize()
        begin = time.perf_counter()
        for _ in range(10):
            decoded = torch.stack(decode_jpeg(jpegs, mode=ImageReadMode.RGB, device="cuda"))
        torch.cuda.synchronize()
        seconds = (time.perf_counter() - begin) / 10
        row = dict(source=source, frame_step=step, clip_id=clip.clip_id,
                   rgb_mean_abs=float(delta.mean()), rgb_max_abs=float(delta.max()),
                   rgb_equal_fraction=float((delta == 0).float().mean()),
                   rgb_gt2_fraction=float((delta > 2).float().mean()),
                   mdd_mean_abs=mdd_mean, mdd_max_abs=mdd_max, decode_seconds=seconds,
                   jpeg_bytes=sum(x.numel() for x in jpegs))
        result["rows"].append(row)
        print(json.dumps(row), flush=True)
    result["status"] = "ok"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
