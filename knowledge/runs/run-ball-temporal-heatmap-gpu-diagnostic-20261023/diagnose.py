"""Reproduce the small BF16 path diagnostic, through the public training CLI.

Run this script only through the shared training queue. Synthetic 64px JPEG
fixtures exercise I/O and masking; their metrics do not measure tennis accuracy.
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
from pathlib import Path

import torch
import yaml

from src.tasks.ball_detection.data.windows.all_frame_preparation import (
    prepare_all_frame_training,
)
from src.tasks.ball_detection.data.windows.all_frames import AllFrameWindowConfig
from src.tasks.ball_detection.visualization.presentation.temporal_heatmap import (
    render_temporal_heatmap_comparison,
)
from tests.support.tasks.ball_detection.store import ball, frame, write_store_clip


def run(command: list[str]) -> None:
    print(json.dumps(dict(command=command)), flush=True)
    subprocess.run(command, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise ValueError("Use a new diagnostic directory")
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise ValueError("Diagnostic requires CUDA BF16; no fallback")
    free, total = torch.cuda.mem_get_info()
    if free < 8 * 1024**3:
        raise ValueError("Diagnostic requires at least 8 GiB free before its small runs")
    output.mkdir(parents=True)
    (output / "environment.json").write_text(json.dumps(dict(torch=str(torch.__version__), cuda=torch.version.cuda,
        gpu=torch.cuda.get_device_name(), free_vram_bytes=free, total_vram_bytes=total), indent=2))
    store = output / "store"
    for split in ("train", "val", "test"):
        labels = [frame(i, ball(xy=(20., 30.))) if i < 48 or i >= 96 else
                  frame(i) if i < 72 else frame(i, ball("out_of_frame", None)) for i in range(128)]
        write_store_clip(store, f"tracknet/{split}/diagnostic", labels, size=(64, 64), split=split)
    prepared = output / "prepared"
    prepare_all_frame_training(store, prepared, configs=[AllFrameWindowConfig(t) for t in (32, 64)])
    reports = {}
    evaluations = []
    for architecture in ("dpt", "hybrid"):
        template = "mdd_dpt_heatmap" if architecture == "dpt" else "mdd_heatmap_hybrid"
        model = yaml.safe_load(Path(f"src/tasks/ball_detection/configs/model/{template}.yaml").read_text())
        model.update(frame_chunk_size=8, activation_checkpointing=True)
        model_path = output / f"{architecture}.yaml"
        model_path.write_text(yaml.safe_dump(model))
        directory = output / architecture
        command = [sys.executable, "-m", "src.tasks.ball_detection.scripts.train", "--stage", "temporal_heatmap",
            "--architecture", architecture, "--phase", "pretrain", "--manifest", str(prepared / "train_T32.json"),
            "--evaluation-manifest", str(prepared / "evaluation_T32.json"), "--model-config", str(model_path),
            "--output", str(directory), "--updates", "30", "--input-frames", "128", "--microbatch-windows", "1",
            "--learning-rate", ".0002", "--warmup-updates", "2", "--minimum-lr-ratio", ".1",
            "--weight-decay", ".01", "--gradient-clip", "1", "--evaluation-interval", "30", "--checkpoint-interval", "10",
            "--seed", "42", "--sigma-ratio", ".012", "--gamma", "2", "--distance-px", "20", "--augmentation", "none",
            "--device", "cuda", "--precision", "bf16", "--jpeg-decoder", "nvjpeg", "--image-prefetch", "--pin-memory",
            "--compile-mode", "off", "--input-verification", "upfront", "--cpu-threads", "1", "--num-workers", "0"]
        # Preserve the 30-update schedule while first checking two small updates.
        run([*command, "--stop-after-update", "2"])
        initial = [json.loads(line) for line in (directory / "train.jsonl").read_text().splitlines()]
        if max(row["peak_reserved_vram_bytes"] for row in initial) > 4 * 1024**3:
            raise ValueError("Small diagnostic exceeded its 4 GiB reservation guard")
        run([*command, "--resume", str(directory / "checkpoint-update-00000002.pt")])
        rows = [json.loads(line) for line in (directory / "train.jsonl").read_text().splitlines()]
        if [row["update"] for row in rows] != list(range(1, 31)) or any(row["input_frames"] != 128 for row in rows):
            raise ValueError("GPU diagnostic update/frame accounting failed")
        reports[architecture] = dict(updates=len(rows), input_frames=sum(row["input_frames"] for row in rows),
            first_loss=rows[0]["loss"], last_loss=rows[-1]["loss"],
            peak_allocated_vram_bytes=max(row["peak_vram_bytes"] for row in rows),
            peak_reserved_vram_bytes=max(row["peak_reserved_vram_bytes"] for row in rows),
            median_update_seconds=statistics.median(row["seconds"] for row in rows),
            median_frames_per_second=statistics.median(row["frames_per_second"] for row in rows),
            sample_positions=[rows[0]["sample_position"], rows[-1]["sample_position"]],
            zero_teacher_child_possible=False, checkpoint_resume_at_update=2)
        evaluations.append(directory / "validation-update-00000030.json")
    first, second = [[json.loads(line) for line in (output / kind / "train.jsonl").read_text().splitlines()]
                     for kind in ("dpt", "hybrid")]
    for a, b in zip(first, second, strict=True):
        for key in ("update", "pack_ids", "pack_indices", "input_frames", "teachers_by_kind", "augmentation_sha256"):
            if a[key] != b[key]:
                raise ValueError(f"Architectures disagree on shared budget field {key}")
    render_temporal_heatmap_comparison(evaluations, clip_id="tracknet/val/diagnostic", start=0, stop=64,
                                      output=output / "comparison.mp4")
    (output / "summary.json").write_text(json.dumps(dict(models=reports, environment=json.loads((output / "environment.json").read_text()),
        budget_logs_match=True, image_size=[64, 64], precision="bf16", frames=32,
        decoder="nvjpeg", frame_chunk_size=8, activation_checkpointing=True, compile_mode="off",
        data="synthetic_JPEG_fixture; not_accuracy_evidence", output=str(output)), indent=2))


if __name__ == "__main__":
    main()
