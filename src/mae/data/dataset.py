"""Video frame dataset for MAE pre-training.

This module provides datasets for loading frames from tennis videos
with variable resolution support. The dataset is designed for
self-supervised MAE pre-training.

Features:
- Variable resolution: Samples images within a configurable resolution range
- Random crop and flip augmentations
- Efficient video frame extraction using decord or OpenCV
- Caching of frame indices for fast loading
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Callable, Optional, Sequence

import torch
from torch import Tensor
from torch.utils.data import Dataset

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

try:
    from decord import VideoReader, cpu
    HAS_DECORD = True
except ImportError:
    HAS_DECORD = False


class VideoFrameDataset(Dataset):
    """Dataset for loading random frames from video files.

    Extracts random frames from video files for MAE pre-training.
    Supports variable resolution training.

    Attributes:
        video_paths: List of paths to video files.
        min_resolution: Minimum output resolution.
        max_resolution: Maximum output resolution.
        frames_per_video: Number of frames to sample per video.
        transform: Optional transform to apply to frames.

    """

    def __init__(
        self,
        video_paths: Sequence[str | Path],
        min_resolution: int = 160,
        max_resolution: int = 320,
        frames_per_video: int = 100,
        patch_size: int = 16,
        transform: Optional[Callable[[Tensor], Tensor]] = None,
        use_decord: bool = True,
        balanced_sampling: bool = True,
        sampling_ratio: float = 0.3,
    ) -> None:
        """Initialize video frame dataset.

        Args:
            video_paths: List of paths to video files.
            min_resolution: Minimum resolution (both H and W).
            max_resolution: Maximum resolution (both H and W).
            frames_per_video: Approximate number of frames per video.
            patch_size: Patch size (resolution will be rounded to multiple).
            transform: Optional transform for augmentation.
            use_decord: Use decord for video reading (faster if available).
            balanced_sampling: Whether to use balanced per-video sampling.
            sampling_ratio: Fraction of frames to sample from each video (e.g., 0.3 for 30%).

        """
        super().__init__()
        self.video_paths = [Path(p) for p in video_paths]
        self.min_resolution = min_resolution
        self.max_resolution = max_resolution
        self.frames_per_video = frames_per_video
        self.patch_size = patch_size
        self.transform = transform
        self.use_decord = use_decord and HAS_DECORD
        self.balanced_sampling = balanced_sampling
        self.sampling_ratio = sampling_ratio

        # Build frame index
        self._build_index()

    def _build_index(self) -> None:
        """Build index of (video_idx, frame_idx) pairs.
        
        If balanced_sampling=True, samples indices uniformly across the
        sampling_ratio range per video per epoch.
        If balanced_sampling=False, uses fixed frame stepping (legacy).
        """
        self.frame_indices: list[tuple[int, int]] = []
        self.video_info: list[dict] = []

        for vid_idx, video_path in enumerate(self.video_paths):
            if not video_path.exists():
                continue

            # Get video info
            try:
                if self.use_decord:
                    vr = VideoReader(str(video_path), ctx=cpu(0))
                    num_frames = len(vr)
                    height, width = vr[0].shape[:2]
                else:
                    cap = cv2.VideoCapture(str(video_path))
                    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    cap.release()
            except Exception:
                continue

            if num_frames < 10:
                continue

            self.video_info.append({
                "path": video_path,
                "num_frames": num_frames,
                "width": width,
                "height": height,
            })

            # Sample frame indices
            if self.balanced_sampling:
                # Balanced sampling: store video_idx only, sample frames at runtime
                # Add placeholder entry for this video
                self.frame_indices.append((len(self.video_info) - 1, -1))  # -1 = sample at runtime
            else:
                # Legacy fixed-step sampling
                step = max(1, num_frames // self.frames_per_video)
                for frame_idx in range(0, num_frames - 1, step):
                    self.frame_indices.append((len(self.video_info) - 1, frame_idx))

    def __len__(self) -> int:
        """Return number of samples."""
        return len(self.frame_indices)

    def _sample_resolution(self) -> int:
        """Sample a resolution within the range, rounded to patch_size."""
        res = random.randint(self.min_resolution, self.max_resolution)
        res = (res // self.patch_size) * self.patch_size
        return max(res, self.patch_size)

    def _load_frame(self, video_idx: int, frame_idx: int) -> Tensor:
        """Load a single frame from video.

        Args:
            video_idx: Index into video_info.
            frame_idx: Frame number to load.

        Returns:
            Frame tensor, shape (C, H, W), values in [0, 1].

        """
        info = self.video_info[video_idx]
        video_path = info["path"]

        try:
            if self.use_decord:
                vr = VideoReader(str(video_path), ctx=cpu(0))
                frame = vr[frame_idx].asnumpy()  # (H, W, C), uint8
            else:
                cap = cv2.VideoCapture(str(video_path))
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                cap.release()
                if not ret:
                    raise ValueError(f"Failed to read frame {frame_idx}")
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        except Exception:
            # Return random noise on error
            res = self._sample_resolution()
            return torch.rand(3, res, res)

        # Convert to tensor: (H, W, C) -> (C, H, W), float32 [0, 1]
        frame = torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0

        return frame

    def _random_crop_resize(
        self,
        frame: Tensor,
        target_size: int,
    ) -> Tensor:
        """Random crop and resize to target size.

        Args:
            frame: Input frame, shape (C, H, W).
            target_size: Target resolution (square).

        Returns:
            Cropped and resized frame, shape (C, target_size, target_size).

        """
        C, H, W = frame.shape

        # Random crop (maintain aspect ratio, crop to square first)
        min_dim = min(H, W)
        crop_size = random.randint(int(min_dim * 0.5), min_dim)

        top = random.randint(0, H - crop_size)
        left = random.randint(0, W - crop_size)
        frame = frame[:, top:top + crop_size, left:left + crop_size]

        # Resize to target
        frame = torch.nn.functional.interpolate(
            frame.unsqueeze(0),
            size=(target_size, target_size),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)

        return frame

    def __getitem__(self, idx: int) -> dict:
        """Get a single sample.

        Args:
            idx: Sample index.

        Returns:
            Dictionary with:
                - image: Frame tensor, shape (C, H, W).
                - resolution: Sampled resolution.
                - video_idx: Source video index.
                - frame_idx: Source frame index.

        """
        video_idx, frame_idx = self.frame_indices[idx]

        # If using balanced sampling, randomly sample frame at runtime
        if self.balanced_sampling and frame_idx == -1:
            num_frames = self.video_info[video_idx]["num_frames"]
            # Sample within the sampling_ratio range
            sampling_range = int(num_frames * self.sampling_ratio)
            start_idx = max(0, num_frames - sampling_range)
            frame_idx = random.randint(start_idx, num_frames - 1)

        # Load frame
        frame = self._load_frame(video_idx, frame_idx)

        # Sample target resolution
        resolution = self._sample_resolution()

        # Random crop and resize
        frame = self._random_crop_resize(frame, resolution)

        # Random horizontal flip
        if random.random() > 0.5:
            frame = torch.flip(frame, dims=[-1])

        # Apply additional transforms
        if self.transform is not None:
            frame = self.transform(frame)

        return {
            "image": frame,
            "resolution": resolution,
            "video_idx": video_idx,
            "frame_idx": frame_idx,
        }


class TennisVideoDataset(VideoFrameDataset):
    """Dataset for tennis videos from data/tennis/raw/videos/.

    Automatically discovers video files in the tennis video directory.
    Supports common video formats (mp4, avi, mkv, mov).
    """

    SUPPORTED_EXTENSIONS = {".mp4", ".avi", ".mkv", ".mov", ".webm"}

    def __init__(
        self,
        video_dir: str | Path = "data/tennis/raw/videos",
        min_resolution: int = 160,
        max_resolution: int = 320,
        frames_per_video: int = 100,
        patch_size: int = 16,
        transform: Optional[Callable[[Tensor], Tensor]] = None,
        use_decord: bool = True,
        balanced_sampling: bool = True,
        sampling_ratio: float = 0.3,
    ) -> None:
        """Initialize tennis video dataset.

        Args:
            video_dir: Directory containing tennis videos.
            min_resolution: Minimum output resolution.
            max_resolution: Maximum output resolution.
            frames_per_video: Frames to sample per video.
            patch_size: Patch size for resolution rounding.
            transform: Optional transform.
            use_decord: Use decord for video reading.
            balanced_sampling: Whether to use balanced per-video sampling.
            sampling_ratio: Fraction of frames to sample from each video.

        """
        video_dir = Path(video_dir)
        if not video_dir.exists():
            raise ValueError(f"Video directory does not exist: {video_dir}")

        # Discover video files
        video_paths = []
        for ext in self.SUPPORTED_EXTENSIONS:
            video_paths.extend(video_dir.glob(f"*{ext}"))
            video_paths.extend(video_dir.glob(f"*{ext.upper()}"))

        if not video_paths:
            raise ValueError(f"No video files found in {video_dir}")

        super().__init__(
            video_paths=video_paths,
            min_resolution=min_resolution,
            max_resolution=max_resolution,
            frames_per_video=frames_per_video,
            patch_size=patch_size,
            transform=transform,
            use_decord=use_decord,
            balanced_sampling=balanced_sampling,
            sampling_ratio=sampling_ratio,
        )
