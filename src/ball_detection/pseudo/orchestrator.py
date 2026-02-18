"""End-to-end pseudo-label generation orchestrator."""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from src.ball_detection.data.io.annotation_merger import merge_annotation_records
from src.ball_detection.data.io.layout import discover_video_layouts
from src.ball_detection.data.io.label_writer import write_label_csv
from src.ball_detection.data.io.metadata_writer import write_metadata_json
from src.ball_detection.data.type import DetectionRecord, PathPolicy
from src.ball_detection.inference.ensemble_predictor import BallEnsemblePredictor
from src.ball_detection.pseudo.components.clip_sampler import ClipSampler
from src.ball_detection.pseudo.components.confidence_scorer import ConfidenceScorer
from src.ball_detection.pseudo.components.event_tagger import EventTagger
from src.ball_detection.pseudo.components.quality_filter import QualityFilter
from src.ball_detection.pseudo.components.trajectory_refiner import TrajectoryRefiner
from src.wasb.utils.video_extractor import VideoExtractor


@dataclass(frozen=True)
class PseudoOrchestratorConfig:
    video_root_dir: str = "data/videos"
    video_extensions: tuple[str, ...] = (".mp4", ".mov", ".avi", ".mkv")
    pseudo_output_dir: str = "outputs/ball_detection/pseudo"
    ensemble_checkpoints: tuple[str, ...] = ()
    ensemble_weights: tuple[float, ...] | None = None
    ensemble_model_config_paths: tuple[str | None, ...] | None = None
    visibility_threshold: float = 0.5
    trajectory_checkpoint: str | None = None
    event_checkpoint: str | None = None
    confidence_threshold: float = 0.3
    min_clip_length: int = 16
    max_gap: int = 4
    image_h: int = 288
    image_w: int = 512
    batch_size: int = 32
    max_frames: int | None = None
    jpeg_quality: int = 95
    allow_overwrite: bool = False
    device: str = "cuda"


class PseudoLabelOrchestrator:
    """Pipeline orchestrator for video inference -> windowing -> pseudo labels."""

    def __init__(self, config: PseudoOrchestratorConfig) -> None:
        if not config.ensemble_checkpoints:
            raise ValueError("ensemble_checkpoints must be non-empty")

        self.config = config
        self.ensemble = BallEnsemblePredictor.from_checkpoints(
            list(config.ensemble_checkpoints),
            device=config.device,
            weights=list(config.ensemble_weights) if config.ensemble_weights is not None else None,
            model_config_paths=(
                list(config.ensemble_model_config_paths)
                if config.ensemble_model_config_paths is not None
                else None
            ),
            visibility_threshold=float(config.visibility_threshold),
        )
        self.clip_sampler = ClipSampler(min_length=config.min_clip_length, max_gap=config.max_gap)
        self.trajectory_refiner = TrajectoryRefiner(config.trajectory_checkpoint, device=config.device)
        self.event_tagger = EventTagger(config.event_checkpoint, device=config.device)
        self.confidence_scorer = ConfidenceScorer()
        self.quality_filter = QualityFilter(min_confidence=config.confidence_threshold)

    def _video_to_tensor(self, frames_rgb) -> torch.Tensor:
        frames_t = torch.from_numpy(frames_rgb).permute(0, 3, 1, 2).contiguous().float() / 255.0
        if frames_t.shape[-2:] != (self.config.image_h, self.config.image_w):
            frames_t = F.interpolate(
                frames_t,
                size=(self.config.image_h, self.config.image_w),
                mode="bilinear",
                align_corners=False,
            )
        return frames_t

    def _predict_video(self, extractor: VideoExtractor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        uv_chunks: list[torch.Tensor] = []
        score_chunks: list[torch.Tensor] = []
        vis_chunks: list[torch.Tensor] = []

        total_processed = 0
        reset_state = True

        for frames_rgb, _start_idx in extractor.iter_batches(batch_size=max(1, int(self.config.batch_size))):
            if self.config.max_frames is not None and total_processed >= int(self.config.max_frames):
                break

            if self.config.max_frames is not None:
                remain = int(self.config.max_frames) - total_processed
                if remain <= 0:
                    break
                if int(frames_rgb.shape[0]) > remain:
                    frames_rgb = frames_rgb[:remain]

            if int(frames_rgb.shape[0]) == 0:
                continue

            frames_t = self._video_to_tensor(frames_rgb)
            pred = self.ensemble.predict(frames_t, reset_state=reset_state)
            reset_state = False

            uv_chunks.append(pred["ball_uv"].detach().cpu())
            score_chunks.append(pred["score"].view(-1).detach().cpu())
            vis_chunks.append(pred["visibility"].view(-1).detach().cpu())
            total_processed += int(frames_rgb.shape[0])

        if not uv_chunks:
            empty_uv = torch.zeros((0, 2), dtype=torch.float32)
            empty_score = torch.zeros((0,), dtype=torch.float32)
            empty_vis = torch.zeros((0,), dtype=torch.float32)
            return empty_uv, empty_score, empty_vis

        return (
            torch.cat(uv_chunks, dim=0),
            torch.cat(score_chunks, dim=0),
            torch.cat(vis_chunks, dim=0),
        )

    def _prepare_clip_dir(self, clip_out_dir: Path) -> None:
        if clip_out_dir.exists():
            if not self.config.allow_overwrite:
                raise FileExistsError(f"Output already exists: {clip_out_dir}")
            shutil.rmtree(clip_out_dir)
        clip_out_dir.mkdir(parents=True, exist_ok=False)

    def _process_window(
        self,
        *,
        extractor: VideoExtractor,
        game_name: str,
        clip_name: str,
        start: int,
        end: int,
        uv: torch.Tensor,
        score: torch.Tensor,
        visibility: torch.Tensor,
        policy: PathPolicy,
        source_video: Path,
    ) -> bool:
        width = int(extractor.width)
        height = int(extractor.height)
        t = end - start + 1
        if t <= 0:
            return False

        window_uv = uv[start : end + 1]
        window_score = score[start : end + 1].view(-1)
        window_visibility = visibility[start : end + 1].view(-1)

        court_kp = torch.zeros((1, 20, 2), dtype=torch.float32)
        ball_uv = window_uv.view(1, t, 2)
        ball_vis = window_visibility.view(1, t)
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

        clip_out_dir = Path(self.config.pseudo_output_dir) / game_name / clip_name
        self._prepare_clip_dir(clip_out_dir)
        saved_files = extractor.extract_segment(
            start_frame=start,
            end_frame=end + 1,
            output_dir=clip_out_dir,
            frame_format="frame_{:06d}.jpg",
            jpeg_quality=int(self.config.jpeg_quality),
        )
        valid_len = min(len(saved_files), t)
        if valid_len <= 0:
            shutil.rmtree(clip_out_dir, ignore_errors=True)
            return False

        detector_records: dict[int, DetectionRecord] = {}
        refined_xy: dict[int, tuple[float, float]] = {}
        for local_idx in range(valid_len):
            detector_records[local_idx] = DetectionRecord(
                frame_index=local_idx,
                x=float(window_uv[local_idx, 0]) * max(width - 1, 1),
                y=float(window_uv[local_idx, 1]) * max(height - 1, 1),
                score=float(window_score[local_idx]),
                visible=bool(window_visibility[local_idx] >= 0.5),
            )
            refined_xy[local_idx] = (
                float(refined[0, local_idx, 0]) * max(width - 1, 1),
                float(refined[0, local_idx, 1]) * max(height - 1, 1),
            )

        visibility_list = [bool(v.item()) for v in window_visibility[:valid_len]]
        detector_scores = [float(s.item()) for s in window_score[:valid_len]]
        trimmed_events = {i: evt for i, evt in events.items() if i < valid_len}
        confidence = self.confidence_scorer.score(
            visibility=visibility_list,
            detector_scores=detector_scores,
            events=trimmed_events,
        )
        keep = self.quality_filter.keep_indices(confidence)
        confidence_filtered = {i: c for i, c in confidence.items() if i in keep}

        merged = merge_annotation_records(
            file_names=list(saved_files[:valid_len]),
            detections=detector_records,
            refined_xy=refined_xy,
            confidence=confidence_filtered,
            events=trimmed_events,
            confidence_threshold=self.config.confidence_threshold,
        )

        write_label_csv(clip_out_dir / "Label.csv", merged, policy=policy)
        write_metadata_json(
            clip_out_dir / "pseudo_meta.json",
            {
                "game": game_name,
                "clip": clip_name,
                "source_video": str(source_video),
                "video_fps": float(extractor.fps),
                "start_frame": int(start),
                "end_frame": int(start + valid_len - 1),
                "frames": int(valid_len),
                "confidence_threshold": float(self.config.confidence_threshold),
            },
            policy=policy,
        )
        return True

    def run(self) -> dict[str, Any]:
        """Generate pseudo labels from discovered videos under configured root."""
        layouts = discover_video_layouts(
            self.config.video_root_dir,
            extensions=self.config.video_extensions,
        )
        policy = PathPolicy(root_dir=Path(self.config.pseudo_output_dir), allow_overwrite=self.config.allow_overwrite)

        processed = 0
        processed_videos = 0
        for layout in layouts:
            extractor = VideoExtractor(layout.video_path)
            uv, score, visibility = self._predict_video(extractor)
            if int(uv.shape[0]) == 0:
                continue

            processed_videos += 1

            sampled_windows = self.clip_sampler.sample([bool(v.item()) for v in visibility])
            if not sampled_windows:
                continue

            for window_idx, window in enumerate(sampled_windows, start=1):
                clip_name = f"Clip{window_idx:04d}"
                kept = self._process_window(
                    extractor=extractor,
                    game_name=layout.game_name,
                    clip_name=clip_name,
                    start=window.start,
                    end=window.end,
                    uv=uv,
                    score=score,
                    visibility=visibility,
                    policy=policy,
                    source_video=layout.video_path,
                )
                if kept:
                    processed += 1

        return {
            "processed_videos": processed_videos,
            "processed_clips": processed,
            "root": self.config.pseudo_output_dir,
        }
