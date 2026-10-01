"""Evaluate deployable DINO-format checkpoints exactly as the pipeline runs them."""

from __future__ import annotations

import json
from pathlib import Path
from typing import cast

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

from src.submodules.models.dino.architecture import (
    build_dino,
    dino_4scale_swin_args,
    load_dino_state_dict,
)
from src.tasks.player_detection.configuration import EvaluateConfig
from src.tasks.player_detection.data.detection_dataset import (
    DetectionBatch,
    DetectionSample,
    PlayerDetectionDataset,
    collate_detection,
    select_detection_frames,
)
from src.tasks.player_detection.data.store import PlayerFrameStore, split_names
from src.tasks.player_detection.evaluation.metrics import PlayerDetectionMetrics
from src.tasks.player_detection.models.dino_detector import (
    FrameDetections,
    decode_player_detections,
)


def draw_overlay(
    bgr: np.ndarray, targets: np.ndarray, detections: FrameDetections, *, threshold: float, title: str
) -> np.ndarray:
    """GT in green, detections above ``threshold`` in red with scores."""
    canvas = bgr.copy()
    for x1, y1, x2, y2 in targets.round().astype(int):
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 220, 0), 3)
    keep = detections.scores >= threshold
    for (x1, y1, x2, y2), score in zip(
        detections.boxes_xyxy[keep].round().int().tolist(), detections.scores[keep].tolist(), strict=True
    ):
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(canvas, f"{score:.2f}", (x1, max(y1 - 6, 14)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    cv2.putText(canvas, title, (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    return canvas


def evaluate_checkpoints(config: EvaluateConfig) -> dict[str, dict[str, float]]:
    # Hydra may already have created ``<output_dir>/hydra``; results must not exist.
    if (config.output_dir / "summary.json").exists():
        raise FileExistsError(f"Evaluation output already exists: {config.output_dir}")
    device = torch.device(config.device)
    if device.type != "cuda":
        raise RuntimeError("DINO deformable attention requires a CUDA device")
    store = PlayerFrameStore(config.dataset_dir)
    (split,) = split_names([config.split])
    selection = select_detection_frames(store, split, config.selection, frame_stride=config.frame_stride)
    dataset = PlayerDetectionDataset(store, selection, input_size=config.input_size, augmentation=None)
    overlay_positions = (
        set(np.linspace(0, len(dataset) - 1, config.overlay_frames).round().astype(int).tolist())
        if config.overlay_frames
        else set()
    )
    config.output_dir.mkdir(parents=True, exist_ok=True)
    summary: dict[str, dict[str, float]] = {}
    for name, checkpoint in config.checkpoints.items():
        model, _ = build_dino(config.repository, dino_4scale_swin_args(device, use_checkpoint=False))
        model.load_state_dict(load_dino_state_dict(checkpoint), strict=True)
        model = model.to(device).eval()
        metrics = PlayerDetectionMetrics(config.evaluation)
        loader: DataLoader[DetectionSample] = DataLoader(
            dataset, batch_size=1, shuffle=False, num_workers=config.num_workers, collate_fn=collate_detection
        )
        boxes_out: list[torch.Tensor] = []
        scores_out: list[torch.Tensor] = []
        overlay_dir = config.output_dir / name / "overlays"
        overlay_dir.mkdir(parents=True)
        with torch.no_grad():
            for position, item in enumerate(loader):
                batch = cast(DetectionBatch, item)  # collate_detection output
                outputs = model([image.to(device) for image in batch.images])
                detections = decode_player_detections(
                    outputs, batch.original_sizes, max_detections=config.evaluation.max_detections
                )
                metrics.update(detections, batch.boxes_xyxy_px)
                boxes_out.append(detections[0].boxes_xyxy)
                scores_out.append(detections[0].scores)
                if position in overlay_positions:
                    frame = batch.frames[0]
                    cv2.imwrite(
                        str(overlay_dir / f"{position:06d}.jpg"),
                        draw_overlay(
                            store.read_bgr(frame),
                            batch.boxes_xyxy_px[0].numpy(),
                            detections[0],
                            threshold=config.evaluation.score_threshold,
                            title=f"{name} {store.frame_key(frame)}",
                        ),
                    )
        headline, diagnostics = metrics.compute()
        summary[name] = headline
        np.savez_compressed(
            config.output_dir / name / "predictions.npz",
            boxes_xyxy=torch.stack(boxes_out).numpy(),
            scores=torch.stack(scores_out).numpy(),
            frame_keys=np.asarray(dataset.prediction_ids),
        )
        (config.output_dir / name / "metrics.json").write_text(
            json.dumps(
                {"checkpoint": str(checkpoint), "headline": headline, "diagnostics": diagnostics},
                indent=2,
            ),
            encoding="utf-8",
        )
        del model
        torch.cuda.empty_cache()
    (config.output_dir / "summary.json").write_text(
        json.dumps(
            {
                "dataset": str(config.dataset_dir),
                "split": config.split,
                "frame_stride": config.frame_stride,
                "selection_stats": selection.stats,
                "metrics": summary,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return summary


def write_comparison_markdown(summary: dict[str, dict[str, float]], path: Path) -> None:
    names = list(next(iter(summary.values())))
    lines = ["| checkpoint | " + " | ".join(names) + " |", "|---|" + "---|" * len(names)]
    lines += [
        f"| {checkpoint} | " + " | ".join(f"{metrics[name]:.4f}" for name in names) + " |"
        for checkpoint, metrics in summary.items()
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
