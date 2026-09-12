"""Clip-grouped multiview video annotations connected to BallDetectionDataset."""

from __future__ import annotations

import shutil
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import cached_property
from pathlib import Path
from typing import TYPE_CHECKING, Any

import cv2
from torch.utils.data import DataLoader

from src.tasks.ball_detection.configuration import BallRuntimePaths
from src.tasks.ball_detection.data.components.multiview import (
    AnnotatedVideo,
    child_path,
    frame_cache_path,
    prepare_frame_cache,
    read_annotated_video,
    read_object,
    validate_frame_cache,
)
from src.tasks.ball_detection.data.dataset import BallDetectionDataset
from src.tasks.ball_detection.data.tracknet_datamodule import TrackNetDataModule
from src.tasks.ball_detection.data.types import ClipWindow, FrameLabel

if TYPE_CHECKING:
    from omegaconf import DictConfig


def _initialize_image_worker(worker_id: int) -> None:
    """Avoid eight DataLoader workers each starting an OpenCV CPU thread pool."""
    cv2.setNumThreads(1)


class MultiviewBallDataModule(TrackNetDataModule):
    """Prepare camera frames once and build fully supervised contiguous windows.

    Split entries are ``recording/clip`` IDs from dataset.json. All cameras of a
    clip stay together; the three split files must partition the full manifest.
    """

    def __init__(self, config: DictConfig) -> None:
        super().__init__(config)
        self.frame_cache_dir = BallRuntimePaths.from_config(config).cache(
            str(config.data.frame_cache_dir)
        )
        self.image_size = (
            int(config.data.image_size[0]),
            int(config.data.image_size[1]),
        )
        self.accepted_statuses = frozenset(
            str(s) for s in config.data.accepted_statuses
        )
        self.prepare_workers = int(config.data.prepare_workers)
        self.prefetch_factor = int(config.data.prefetch_factor)
        self.preparation_report: dict[str, Any] = {}

    @cached_property
    def split_entries(self) -> dict[str, list[str]]:
        result = {
            name: self._read_split_entries(self._resolve_split_file(path))
            for name, path in {
                "train": self.train_split_file,
                "val": self.val_split_file,
                "test": self.test_split_file,
            }.items()
        }
        flattened = [entry for entries in result.values() for entry in entries]
        if len(set(flattened)) != len(flattened):
            raise ValueError(
                "Duplicate clip IDs within/across multiview splits (camera leakage)."
            )
        manifest = read_object(self.data_dir / "dataset.json")
        if manifest["version"] != 1:
            raise ValueError("Unsupported multiview dataset version")
        ids = [clip["clip_id"] for clip in manifest["clips"]]
        if len(set(ids)) != len(ids) or set(ids) != set(flattened):
            raise ValueError(
                "Multiview splits must partition all dataset.json clip IDs exactly."
            )
        return result

    @cached_property
    def streams(self) -> dict[str, list[AnnotatedVideo]]:
        _ = self.split_entries
        result: dict[str, list[AnnotatedVideo]] = {}
        for entry in read_object(self.data_dir / "dataset.json")["clips"]:
            clip_id = entry["clip_id"]
            # Also prevents ambiguous common Dataset window IDs after caching.
            if len(clip_id.split("/")) != 2 or any(
                "__" in part for part in clip_id.split("/")
            ):
                raise ValueError(
                    f"Expected recording/clip identifier without '__': {clip_id}"
                )
            child_path(self.data_dir, clip_id)
            clip_dir = child_path(self.data_dir, entry["path"])
            clip = read_object(clip_dir / "clip.json")
            for key in ("clip_id", "num_frames", "width", "height"):
                if clip[key] != entry[key]:
                    raise ValueError(
                        f"dataset.json and clip.json disagree on {key}: {clip_dir}"
                    )
            cameras = clip["cameras"]
            ids = [camera["camera_id"] for camera in cameras]
            if (
                len(set(ids)) != len(ids)
                or ids != clip["camera_ids"]
                or len(ids) != entry["num_cameras"]
            ):
                raise ValueError(f"Camera inventory mismatch: {clip_dir}")
            result[clip_id] = []
            for camera in cameras:
                camera_id = camera["camera_id"]
                if Path(camera_id).name != camera_id or camera_id in {".", ".."}:
                    raise ValueError(f"Invalid camera_id: {camera_id!r}")
                result[clip_id].append(
                    read_annotated_video(
                        clip_id=clip_id,
                        camera_id=camera_id,
                        video_path=child_path(clip_dir, camera["video"]),
                        annotation_path=clip_dir
                        / "outsource"
                        / f"{camera_id}_annotations.json",
                        original_size=(clip["width"], clip["height"]),
                        num_frames=clip["num_frames"],
                        accepted_statuses=self.accepted_statuses,
                    )
                )
        return result

    def prepare_data(self) -> None:
        """Fully expand all videos before training, using CPU and local disk only."""
        started = time.monotonic()
        streams = [stream for group in self.streams.values() for stream in group]
        caches = [
            frame_cache_path(self.frame_cache_dir, stream, self.image_size)
            for stream in streams
        ]
        self.frame_cache_dir.mkdir(parents=True, exist_ok=True)
        height, width = self.image_size
        bytes_per_frame = 54 + ((width * 3 + 3) // 4 * 4) * height
        missing = sum(
            len(stream.labels)
            for stream, cache in zip(streams, caches, strict=True)
            if not cache.exists()
        )
        required_bytes = missing * bytes_per_frame
        free_bytes = shutil.disk_usage(self.frame_cache_dir).free
        if required_bytes and free_bytes < required_bytes + 1024**3:
            raise OSError(
                f"Insufficient disk space for BMP preprocessing: {required_bytes} bytes "
                f"plus 1 GiB reserve required, {free_bytes} available at {self.frame_cache_dir}"
            )
        print(
            f"[multiview] preprocess {len(streams)} videos; {missing} new BMP frames; "
            f"workers={self.prepare_workers}; cache={self.frame_cache_dir}",
            flush=True,
        )
        previous_threads = cv2.getNumThreads()
        cv2.setNumThreads(1)
        try:
            with ThreadPoolExecutor(max_workers=self.prepare_workers) as executor:
                futures = {
                    executor.submit(prepare_frame_cache, cache, stream, self.image_size): stream
                    for stream, cache in zip(streams, caches, strict=True)
                }
                for completed, future in enumerate(as_completed(futures), start=1):
                    future.result()
                    stream = futures[future]
                    print(
                        f"[multiview] prepared {completed}/{len(streams)} "
                        f"{stream.clip_id}/{stream.camera_id}",
                        flush=True,
                    )
        finally:
            cv2.setNumThreads(previous_threads)
        self.preparation_report = {
            "cache_root": str(self.frame_cache_dir),
            "format": "bmp",
            "size_hw": list(self.image_size),
            "videos": len(streams),
            "frames": sum(len(stream.labels) for stream in streams),
            "new_frames": missing,
            "frame_bytes": sum(len(stream.labels) for stream in streams) * bytes_per_frame,
            "prepare_workers": self.prepare_workers,
            "elapsed_seconds": time.monotonic() - started,
        }
        print(f"[multiview] preprocess complete: {self.preparation_report}", flush=True)

    def create_windows(
        self, *, split_name: str, split_file: str | Path
    ) -> list[ClipWindow]:
        """Keep physical frame indices; never join frames across missing labels."""
        windows: list[ClipWindow] = []
        for clip_id in self.split_entries[split_name]:
            for stream in self.streams[clip_id]:
                cache = frame_cache_path(self.frame_cache_dir, stream, self.image_size)
                validate_frame_cache(cache, stream, self.image_size)
                names = tuple(f"{index:06d}.bmp" for index in range(len(stream.labels)))
                labels: dict[str, tuple[FrameLabel, ...]] = {
                    name: (label,)
                    for name, label in zip(names, stream.labels, strict=True)
                    if label is not None
                }
                windows.extend(
                    ClipWindow(cache, names, labels, stream.original_size, start)
                    for start in stream.valid_starts(
                        self.num_frames, self.sample_stride
                    )
                )
        if not windows:
            raise RuntimeError(
                f"No supervised multiview windows for split={split_name}"
            )
        return windows

    def _dataloader(
        self, dataset: BallDetectionDataset | None, *, training: bool
    ) -> DataLoader:
        if dataset is None:
            raise RuntimeError("setup() must run before requesting a dataloader.")
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=training,
            drop_last=training,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.num_workers > 0,
            prefetch_factor=self.prefetch_factor if self.num_workers > 0 else None,
            worker_init_fn=_initialize_image_worker,
        )

    def train_dataloader(self) -> DataLoader:
        """Retain workers between epochs, with bounded prefetch memory."""
        return self._dataloader(self.train_dataset, training=True)

    def val_dataloader(self) -> DataLoader:
        """Use the same loader settings for validation and batch calibration."""
        return self._dataloader(self.val_dataset, training=False)

    def test_dataloader(self) -> DataLoader:
        """Read held-out windows in order."""
        return self._dataloader(self.test_dataset, training=False)

    def summary(self) -> dict[str, Any]:
        """Summarize all splits without decoding videos or loading model weights."""
        return {
            name: {
                "clip_ids": entries,
                "cameras": sum(len(self.streams[entry]) for entry in entries),
                "windows": sum(
                    len(stream.valid_starts(self.num_frames, self.sample_stride))
                    for entry in entries
                    for stream in self.streams[entry]
                ),
                "statuses": {
                    status: sum(
                        stream.status_counts.get(status, 0)
                        for entry in entries
                        for stream in self.streams[entry]
                    )
                    for status in (
                        "observed",
                        "interpolated",
                        "occlusion_estimated",
                        "unresolved",
                    )
                },
            }
            for name, entries in self.split_entries.items()
        }


__all__ = ["MultiviewBallDataModule"]
