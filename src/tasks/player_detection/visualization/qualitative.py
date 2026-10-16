"""Render validation samples' original, continuous clips with the training model."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

from src.submodules.models.dino.architecture import preprocess_frame
from src.tasks.base.training.qualitative_saving import save_qualitative_clip
from src.tasks.player_detection.configuration import PlayerTrainingConfig
from src.tasks.player_detection.data.detection_dataset import DetectionBatch
from src.tasks.player_detection.data.store import PlayerFrameStore
from src.tasks.player_detection.models.dino_detector import (
    PlayerDinoModel,
    decode_player_detections,
)
from src.tasks.player_detection.visualization.overlays import draw_overlay
from src.tennis_scene.chat_annotation.layout import published_video_path
from src.tennis_scene.chat_annotation.manifests import load_prepared_manifests
from src.tennis_scene.chat_annotation.runtime.media import check_clip, decode_range


class PlayerClipRenderer:
    def __init__(self, runtime: PlayerTrainingConfig) -> None:
        if runtime.qualitative is None:
            raise ValueError("Player clip rendering requires qualitative configuration.")
        self.config = runtime.qualitative
        self.input_size = runtime.data.input_size
        self.evaluation = runtime.evaluation
        self.store = PlayerFrameStore(runtime.data.dataset_dir)

    def render(
        self,
        model: PlayerDinoModel,
        batches: Sequence[DetectionBatch],
        artifact_dir: Path,
        tb_writer: Any | None,
        global_step: int,
    ) -> None:
        # Match other task hooks: one source sample per selected validation batch.
        # A repeated clip is rendered once; unrelated clips are never concatenated.
        clips = {}
        for batch in batches:
            if not batch.frames:
                raise ValueError("Player qualitative batch has no validation samples.")
            clip = self.store.clip_of(batch.frames[0])
            if clip.split != "val":
                raise ValueError(f"Player qualitative clip is not in validation: {clip.clip_id}")
            clips[clip.clip_id] = clip
        manifests = load_prepared_manifests(self.config.annotation_root)
        device = next(model.parameters()).device
        was_training = model.training
        model.eval()
        try:
            with torch.inference_mode():
                for clip in clips.values():
                    if clip.clip_id not in manifests:
                        raise FileNotFoundError(f"Missing prepared manifest for {clip.clip_id}")
                    manifest = manifests[clip.clip_id]
                    if clip.video_sha256 != manifest.sha256:
                        raise ValueError(f"Store/manifest video hash mismatch: {clip.clip_id}")
                    video = published_video_path(self.config.annotation_root, manifest)
                    timeline = check_clip(video, manifest)
                    if (clip.frame_count, clip.width, clip.height) != (
                        len(timeline.pts), timeline.width, timeline.height,
                    ):
                        raise ValueError(f"Store/video geometry or frame count mismatch: {clip.clip_id}")
                    stop = min(clip.frame_count, self.config.max_frames * self.config.frame_stride)
                    if len(range(0, stop, self.config.frame_stride)) < 2:
                        raise ValueError(f"Player qualitative GIF requires at least two frames: {clip.clip_id}")
                    rendered: list[np.ndarray] = []
                    pending: list[tuple[int, np.ndarray]] = []
                    for index, frame in enumerate(decode_range(video, timeline, 0, stop)):
                        if index % self.config.frame_stride:
                            continue
                        pending.append((index, frame.to_ndarray(format="bgr24")))
                        if len(pending) == self.config.batch_size:
                            rendered.extend(self._render_frames(model, device, clip.clip_id, pending))
                            pending.clear()
                    if pending:
                        rendered.extend(self._render_frames(model, device, clip.clip_id, pending))
                    save_qualitative_clip(
                        frames_rgb=rendered,
                        artifact_dir=artifact_dir,
                        name=f"player_clip{clip.index:04d}",
                        tb_writer=tb_writer,
                        tag=f"qualitative/player_detection/{clip.clip_id}",
                        global_step=global_step,
                        fps=float(timeline.rate) / self.config.frame_stride,
                    )
        finally:
            model.train(was_training)

    def _render_frames(
        self, model: PlayerDinoModel, device: torch.device,
        clip_id: str, frames: list[tuple[int, np.ndarray]],
    ) -> list[np.ndarray]:
        images = [
            preprocess_frame(
                bgr, short_side=self.input_size.short_side,
                max_long_side=self.input_size.max_long_side,
            ).to(device)
            for _, bgr in frames
        ]
        detections = decode_player_detections(
            model(images, None),
            [(bgr.shape[0], bgr.shape[1]) for _, bgr in frames],
            max_detections=self.evaluation.max_detections,
        )
        result: list[np.ndarray] = []
        for (index, bgr), detection in zip(frames, detections, strict=True):
            overlay = draw_overlay(
                bgr, np.empty((0, 4), dtype=np.float32), detection,
                threshold=self.evaluation.score_threshold,
                title=f"{clip_id} | frame {index}",
            )
            if overlay.shape[1] > self.config.display_width:
                height = max(1, round(overlay.shape[0] * self.config.display_width / overlay.shape[1]))
                overlay = cv2.resize(overlay, (self.config.display_width, height), interpolation=cv2.INTER_AREA)
            result.append(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
        return result
