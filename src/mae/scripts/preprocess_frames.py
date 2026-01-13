"""Video frame extraction script for MAE pre-training.

Extracts frames from tennis videos and saves them to structured directories.
Supports efficient batch processing with decord or OpenCV.

Usage:
    python -m src.mae.scripts.preprocess_frames \\
        --video-dir data/tennis/raw/videos \\
        --output-dir data/tennis/raw/frames \\
        --sample-rate 10
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import torch
from tqdm import tqdm

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


class FrameExtractor:
    """Extract frames from videos with configurable sampling."""

    def __init__(
        self,
        video_dir: Path | str,
        output_dir: Path | str,
        sample_rate: int = 10,
        use_decord: bool = True,
        max_resolution: Optional[int] = None,
        min_resolution: Optional[int] = None,
    ):
        """Initialize frame extractor.

        Args:
            video_dir: Directory containing video files.
            output_dir: Directory to save extracted frames.
            sample_rate: Sample every Nth frame (e.g., 10 = every 10th frame).
            use_decord: Use decord for video reading (faster if available).
            max_resolution: Maximum resolution for downsampling (optional).
            min_resolution: Minimum resolution (aspect ratio preserved).

        """
        self.video_dir = Path(video_dir)
        self.output_dir = Path(output_dir)
        self.sample_rate = sample_rate
        self.use_decord = use_decord and HAS_DECORD
        self.max_resolution = max_resolution
        self.min_resolution = min_resolution

        if not self.video_dir.exists():
            raise ValueError(f"Video directory not found: {video_dir}")

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _get_video_files(self) -> list[Path]:
        """Find all video files in video_dir."""
        extensions = {".mp4", ".avi", ".mkv", ".mov", ".webm"}
        videos = []
        for ext in extensions:
            videos.extend(self.video_dir.glob(f"*{ext}"))
            videos.extend(self.video_dir.glob(f"*{ext.upper()}"))
        return sorted(set(videos))

    def _resize_frame(
        self,
        frame: torch.Tensor,
        max_res: Optional[int] = None,
    ) -> torch.Tensor:
        """Resize frame if needed.

        Args:
            frame: Frame tensor, shape (C, H, W).
            max_res: Maximum resolution.

        Returns:
            Resized frame.

        """
        if max_res is None:
            return frame

        C, H, W = frame.shape
        if max(H, W) <= max_res:
            return frame

        # Calculate new dimensions
        scale = max_res / max(H, W)
        new_h, new_w = int(H * scale), int(W * scale)

        # Resize
        frame = torch.nn.functional.interpolate(
            frame.unsqueeze(0),
            size=(new_h, new_w),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)

        return frame

    def extract_frames(self, video_path: Path) -> dict:
        """Extract frames from a single video.

        Args:
            video_path: Path to video file.

        Returns:
            Dictionary with extraction metadata.

        """
        video_name = video_path.stem
        output_subdir = self.output_dir / video_name
        output_subdir.mkdir(parents=True, exist_ok=True)

        try:
            # Get video info
            if self.use_decord:
                vr = VideoReader(str(video_path), ctx=cpu(0))
                num_frames = len(vr)
                frame_h, frame_w = vr[0].shape[:2]
            else:
                cap = cv2.VideoCapture(str(video_path))
                num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                cap.release()

            if num_frames == 0:
                return {"video": video_name, "status": "error", "reason": "No frames"}

            # Extract frames
            extracted_frames = []
            frame_indices = list(range(0, num_frames, self.sample_rate))

            for frame_idx in tqdm(
                frame_indices,
                desc=f"Extracting {video_name}",
                leave=False,
            ):
                try:
                    if self.use_decord:
                        vr = VideoReader(str(video_path), ctx=cpu(0))
                        frame_np = vr[frame_idx].asnumpy()  # (H, W, C), uint8
                    else:
                        cap = cv2.VideoCapture(str(video_path))
                        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                        ret, frame_np = cap.read()
                        cap.release()
                        if not ret:
                            continue
                        frame_np = cv2.cvtColor(frame_np, cv2.COLOR_BGR2RGB)

                    # Convert to tensor
                    frame = torch.from_numpy(frame_np).permute(2, 0, 1).float() / 255.0

                    # Resize if needed
                    if self.max_resolution is not None:
                        frame = self._resize_frame(frame, self.max_resolution)

                    # Save frame
                    frame_path = output_subdir / f"frame_{frame_idx:06d}.pt"
                    torch.save(frame, frame_path)
                    extracted_frames.append(frame_idx)

                except Exception as e:
                    print(f"  Error extracting frame {frame_idx}: {e}")
                    continue

            return {
                "video": video_name,
                "status": "success",
                "num_frames_extracted": len(extracted_frames),
                "frame_indices": extracted_frames,
                "original_h": frame_h,
                "original_w": frame_w,
            }

        except Exception as e:
            return {"video": video_name, "status": "error", "reason": str(e)}

    def process_all(self) -> None:
        """Extract frames from all videos."""
        videos = self._get_video_files()
        if not videos:
            print(f"No video files found in {self.video_dir}")
            return

        print(f"Found {len(videos)} video files")

        results = []
        for video_path in tqdm(videos, desc="Processing videos"):
            result = self.extract_frames(video_path)
            results.append(result)
            status = result["status"]
            if status == "success":
                print(f"✓ {result['video']}: {result['num_frames_extracted']} frames")
            else:
                print(f"✗ {result['video']}: {result['reason']}")

        # Save metadata
        metadata_path = self.output_dir / "metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(results, f, indent=2)

        print(f"\nExtraction complete! Results saved to {metadata_path}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Extract frames from tennis videos"
    )
    parser.add_argument(
        "--video-dir",
        type=str,
        default="data/tennis/raw/videos",
        help="Directory containing video files",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/tennis/raw/frames",
        help="Directory to save extracted frames",
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=10,
        help="Sample every Nth frame (default: 10)",
    )
    parser.add_argument(
        "--max-resolution",
        type=int,
        default=None,
        help="Maximum resolution for resizing (optional)",
    )
    parser.add_argument(
        "--min-resolution",
        type=int,
        default=None,
        help="Minimum resolution (aspect ratio preserved, optional)",
    )
    parser.add_argument(
        "--use-opencv",
        action="store_true",
        help="Use OpenCV instead of decord for video reading",
    )

    args = parser.parse_args()

    extractor = FrameExtractor(
        video_dir=args.video_dir,
        output_dir=args.output_dir,
        sample_rate=args.sample_rate,
        use_decord=not args.use_opencv,
        max_resolution=args.max_resolution,
        min_resolution=args.min_resolution,
    )

    extractor.process_all()


if __name__ == "__main__":
    main()
