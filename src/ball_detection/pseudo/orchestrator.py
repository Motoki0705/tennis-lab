"""End-to-end pseudo-label generation orchestrator."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torchvision import transforms

from src.ball_detection.data.io.annotation_merger import merge_annotation_records
from src.ball_detection.data.io.layout import discover_clip_layouts
from src.ball_detection.data.io.label_writer import write_label_csv
from src.ball_detection.data.io.metadata_writer import write_metadata_json
from src.ball_detection.data.type import DetectionRecord, PathPolicy
from src.ball_detection.inference.ensemble_predictor import BallEnsemblePredictor
from src.ball_detection.pseudo.components.clip_sampler import ClipSampler
from src.ball_detection.pseudo.components.confidence_scorer import ConfidenceScorer
from src.ball_detection.pseudo.components.event_tagger import EventTagger
from src.ball_detection.pseudo.components.quality_filter import QualityFilter
from src.ball_detection.pseudo.components.trajectory_refiner import TrajectoryRefiner


@dataclass(frozen=True)
class PseudoOrchestratorConfig:
    labeled_root_dir: str = "data/tennis"
    pseudo_output_dir: str = "outputs/ball_detection/pseudo"
    ensemble_checkpoints: tuple[str, ...] = ()
    ensemble_weights: tuple[float, ...] | None = None
    trajectory_checkpoint: str | None = None
    event_checkpoint: str | None = None
    confidence_threshold: float = 0.3
    min_clip_length: int = 16
    max_gap: int = 4
    image_h: int = 288
    image_w: int = 512
    allow_overwrite: bool = False
    device: str = "cpu"


class PseudoLabelOrchestrator:
    """Pipeline orchestrator for unlabeled inference -> refinement -> pseudo labels."""

    def __init__(self, config: PseudoOrchestratorConfig) -> None:
        if not config.ensemble_checkpoints:
            raise ValueError("ensemble_checkpoints must be non-empty")

        self.config = config
        self.ensemble = BallEnsemblePredictor.from_checkpoints(
            list(config.ensemble_checkpoints),
            device=config.device,
            weights=list(config.ensemble_weights) if config.ensemble_weights is not None else None,
        )
        self.clip_sampler = ClipSampler(min_length=config.min_clip_length, max_gap=config.max_gap)
        self.trajectory_refiner = TrajectoryRefiner(config.trajectory_checkpoint, device=config.device)
        self.event_tagger = EventTagger(config.event_checkpoint, device=config.device)
        self.confidence_scorer = ConfidenceScorer()
        self.quality_filter = QualityFilter(min_confidence=config.confidence_threshold)
        self.transform = transforms.Compose(
            [transforms.Resize((config.image_h, config.image_w)), transforms.ToTensor()]
        )

    def run(self) -> dict[str, Any]:
        """Generate pseudo labels for discovered clips under configured root."""
        layouts = discover_clip_layouts(self.config.labeled_root_dir)
        policy = PathPolicy(root_dir=Path(self.config.pseudo_output_dir), allow_overwrite=self.config.allow_overwrite)

        processed = 0
        for layout in layouts:
            if not layout.frames:
                continue

            frames = []
            for fr in layout.frames:
                with Image.open(fr.frame_path) as img:
                    frames.append(self.transform(img.convert("RGB")))
            frames_t = torch.stack(frames, dim=0)

            pred = self.ensemble.predict(frames_t)
            uv = pred["ball_uv"]
            score = pred["score"].view(-1)
            visibility = pred["visibility"].view(-1)

            sampled_windows = self.clip_sampler.sample([bool(v.item()) for v in visibility])
            if not sampled_windows:
                continue

            # Minimal court placeholders for compatibility with downstream APIs.
            t = uv.shape[0]
            court_kp = torch.zeros((1, 20, 2), dtype=torch.float32)
            ball_uv = uv.view(1, t, 2)
            ball_vis = visibility.view(1, t)
            ball_mask = torch.ones((1, t), dtype=torch.float32)
            court_vis = torch.ones((1, 20), dtype=torch.float32)

            refined = self.trajectory_refiner.refine(
                ball_uv,
                court_kp,
                ball_vis=ball_vis,
                ball_mask=ball_mask,
                court_vis=court_vis,
            )
            events = self.event_tagger.tag(
                refined,
                court_kp,
                ball_vis=ball_vis,
                ball_mask=ball_mask,
                court_vis=court_vis,
            )

            detector_records: dict[int, DetectionRecord] = {}
            refined_xy: dict[int, tuple[float, float]] = {}
            for i, fr in enumerate(layout.frames):
                with Image.open(fr.frame_path) as img:
                    w, h = img.size
                detector_records[i] = DetectionRecord(
                    frame_index=i,
                    x=float(uv[i, 0]) * max(w - 1, 1),
                    y=float(uv[i, 1]) * max(h - 1, 1),
                    score=float(score[i]),
                    visible=bool(visibility[i] >= 0.5),
                )
                refined_xy[i] = (
                    float(refined[0, i, 0]) * max(w - 1, 1),
                    float(refined[0, i, 1]) * max(h - 1, 1),
                )

            confidence = self.confidence_scorer.score(
                visibility=[bool(v.item()) for v in visibility],
                detector_scores=[float(s.item()) for s in score],
                events=events,
            )
            keep = self.quality_filter.keep_indices(confidence)
            confidence_filtered = {i: c for i, c in confidence.items() if i in keep}

            merged = merge_annotation_records(
                file_names=[f.file_name for f in layout.frames],
                detections=detector_records,
                refined_xy=refined_xy,
                confidence=confidence_filtered,
                events=events,
                confidence_threshold=self.config.confidence_threshold,
            )

            clip_out_dir = Path(self.config.pseudo_output_dir) / layout.game_name / layout.clip_name
            clip_out_dir.mkdir(parents=True, exist_ok=True)
            for fr in layout.frames:
                dst = clip_out_dir / fr.file_name
                if not dst.exists():
                    dst.symlink_to(fr.frame_path.resolve())

            write_label_csv(clip_out_dir / "Label.csv", merged, policy=policy)
            write_metadata_json(
                clip_out_dir / "pseudo_meta.json",
                {
                    "game": layout.game_name,
                    "clip": layout.clip_name,
                    "frames": len(layout.frames),
                    "sampled_windows": [{"start": w.start, "end": w.end} for w in sampled_windows],
                    "confidence_threshold": self.config.confidence_threshold,
                },
                policy=policy,
            )
            processed += 1

        return {
            "processed_clips": processed,
            "root": self.config.pseudo_output_dir,
        }
