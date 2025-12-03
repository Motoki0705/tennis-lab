"""BLCS inference pipeline combining WASB detection and BLCS prediction."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from src.blcs.inference import BLCSPredictor, TrajectoryVisualizer
from src.wasb.inference import WASBPredictor

if TYPE_CHECKING:
    from numpy.typing import NDArray


class BLCSPipeline:
    """End-to-end pipeline for ball trajectory estimation.

    Combines WASB ball detection with BLCS 3D trajectory prediction.

    Example:
        >>> pipeline = BLCSPipeline(
        ...     wasb_checkpoint="wasb.pth.tar",
        ...     blcs_checkpoint="blcs.ckpt",
        ... )
        >>> results = pipeline.run(frames, court_kp)
        >>> print(results["ball_3d"].shape)  # (T, 3)

    """

    def __init__(
        self,
        wasb_checkpoint: str | Path,
        blcs_checkpoint: str | Path,
        device: str = "cuda",
    ) -> None:
        """Initialize the pipeline.

        Args:
            wasb_checkpoint: Path to WASB model checkpoint (.pth.tar).
            blcs_checkpoint: Path to BLCS model checkpoint (.ckpt).
            device: Device for inference. WASB requires CUDA.

        """
        self._device = device

        # Initialize predictors
        print("Loading WASB model...")
        self._wasb = WASBPredictor.load_from_checkpoint(wasb_checkpoint, device="cuda")

        print("Loading BLCS model...")
        self._blcs = BLCSPredictor.load_from_checkpoint(blcs_checkpoint, device=device)

        # Initialize visualizer
        self._visualizer = TrajectoryVisualizer()

        print("Pipeline ready.")

    def run(
        self,
        frames: NDArray[np.uint8],
        court_kp: NDArray[np.float32],
        visualize: bool = True,
    ) -> dict[str, Any]:
        """Run full inference pipeline.

        Args:
            frames: Video frames, shape (T, H, W, 3), RGB uint8.
            court_kp: Court keypoints, shape (20, 2), normalized [0, 1].
            visualize: Whether to generate visualization figures.

        Returns:
            Results dictionary:
                - ball_uv: 2D detections (T, 2), normalized
                - ball_3d: 3D trajectory (T, 3), meters
                - visibility: Detection visibility (T,)
                - score: Detection scores (T,)
                - fig_3d: 3D trajectory figure (if visualize=True)
                - fig_2d: 2D top-down figure (if visualize=True)

        """
        # Step 1: WASB ball detection
        print(f"Running WASB detection on {len(frames)} frames...")
        wasb_out = self._wasb.predict(frames)

        ball_uv = wasb_out["ball_uv"]  # (T, 2)
        visibility = wasb_out["visibility"]  # (T,)
        score = wasb_out["score"]  # (T,)

        print(f"  Detected {visibility.sum()}/{len(visibility)} visible frames")

        # Step 2: BLCS 3D prediction
        print("Running BLCS 3D prediction...")
        ball_uv_t = torch.from_numpy(ball_uv).float()
        court_kp_t = torch.from_numpy(court_kp).float()
        ball_mask_t = torch.from_numpy(visibility.astype(np.float32))

        blcs_out = self._blcs.predict(
            ball_uv=ball_uv_t,
            court_kp=court_kp_t,
            ball_mask=ball_mask_t,
            denormalize=True,
        )

        ball_3d = blcs_out["position"].squeeze(0).cpu().numpy()  # (T, 3)
        print(f"  Predicted 3D trajectory shape: {ball_3d.shape}")

        # Build results
        results: dict[str, Any] = {
            "ball_uv": ball_uv,
            "ball_3d": ball_3d,
            "visibility": visibility,
            "score": score,
        }

        # Step 3: Visualization
        if visualize:
            print("Generating visualizations...")
            results["fig_3d"] = self._visualizer.plot_trajectory_3d(
                ball_3d, title="BLCS 3D Ball Trajectory"
            )
            results["fig_2d"] = self._visualizer.plot_trajectory_2d(
                ball_3d, title="Ball Trajectory (Top View)"
            )
            results["fig_uv"] = self._visualizer.plot_uv_trajectory(
                ball_uv, visibility=visibility, title="2D Detection (UV)"
            )

        return results

    def run_blcs_only(
        self,
        ball_uv: NDArray[np.float32],
        court_kp: NDArray[np.float32],
        ball_mask: NDArray[np.bool_] | None = None,
        visualize: bool = True,
    ) -> dict[str, Any]:
        """Run BLCS prediction only (skip WASB detection).

        Useful when 2D ball positions are already available.

        Args:
            ball_uv: Ball 2D positions, shape (T, 2), normalized [0, 1].
            court_kp: Court keypoints, shape (20, 2), normalized [0, 1].
            ball_mask: Visibility mask, shape (T,).
            visualize: Whether to generate figures.

        Returns:
            Results dictionary with ball_3d and optional figures.

        """
        ball_uv_t = torch.from_numpy(ball_uv).float()
        court_kp_t = torch.from_numpy(court_kp).float()

        ball_mask_t = None
        if ball_mask is not None:
            ball_mask_t = torch.from_numpy(ball_mask.astype(np.float32))

        blcs_out = self._blcs.predict(
            ball_uv=ball_uv_t,
            court_kp=court_kp_t,
            ball_mask=ball_mask_t,
            denormalize=True,
        )

        ball_3d = blcs_out["position"].squeeze(0).cpu().numpy()

        results: dict[str, Any] = {
            "ball_uv": ball_uv,
            "ball_3d": ball_3d,
        }

        if visualize:
            results["fig_3d"] = self._visualizer.plot_trajectory_3d(ball_3d)
            results["fig_2d"] = self._visualizer.plot_trajectory_2d(ball_3d)

        return results

    def _run_blcs_from_uv(
        self,
        ball_uv: NDArray[np.float32],
        visibility: NDArray[np.bool_],
        score: NDArray[np.floating[Any]] | None,
        court_kp: NDArray[np.float32],
        visualize: bool,
        shot_frames: list[int] | None,
    ) -> dict[str, Any]:
        """Run BLCS and visualization given 2D detections.

        This is shared by streaming (online WASB) and offline WASB modes.
        """
        print(f"  Detected {visibility.sum()}/{len(visibility)} visible frames")

        # Step 2: BLCS 3D prediction (optionally segment-based, chunked by 60 frames)
        blcs_seq_len = 60  # Process in 60-frame chunks
        court_kp_t = torch.from_numpy(court_kp).float()

        # Prepare output array for full-length 3D trajectory
        ball_3d_full = np.zeros((len(ball_uv), 3), dtype=np.float32)

        # Build segments from shot frames if provided
        segments: list[tuple[int, int]] = []
        if shot_frames is not None and len(shot_frames) >= 2:
            frames_sorted = sorted(int(f) for f in shot_frames if 0 <= f < len(ball_uv))
            # First timestamp is the first start_i
            for i in range(len(frames_sorted) - 1):
                start_i = frames_sorted[i]
                end_i = frames_sorted[i + 1]
                if end_i - start_i >= 3:  # require at least a few frames
                    segments.append((start_i, end_i))
        else:
            # Fallback: single segment over the whole video
            segments.append((0, len(ball_uv)))

        print("Running BLCS 3D prediction (chunk_size=60) on segments:")
        for si, (start_i, end_i) in enumerate(segments):
            print(f"  Segment {si}: [{start_i}, {end_i}) len={end_i - start_i}")

        for seg_idx, (start_i, end_i) in enumerate(segments):
            seg_uv = ball_uv[start_i:end_i]
            seg_vis = visibility[start_i:end_i]

            if len(seg_uv) <= 0:
                continue

            # Chunk the segment into at most 60-frame pieces
            num_chunks = (len(seg_uv) + blcs_seq_len - 1) // blcs_seq_len
            seg_ball_3d_parts: list[NDArray[np.floating[Any]]] = []

            for chunk_idx in range(num_chunks):
                c_start = chunk_idx * blcs_seq_len
                c_end = min(c_start + blcs_seq_len, len(seg_uv))

                chunk_uv = seg_uv[c_start:c_end]
                chunk_vis = seg_vis[c_start:c_end]

                ball_uv_t = torch.from_numpy(chunk_uv).float()
                ball_mask_t = torch.from_numpy(chunk_vis.astype(np.float32))

                blcs_out = self._blcs.predict(
                    ball_uv=ball_uv_t,
                    court_kp=court_kp_t,
                    ball_mask=ball_mask_t,
                    denormalize=True,
                )

                chunk_3d = blcs_out["position"].squeeze(0).cpu().numpy()
                seg_ball_3d_parts.append(chunk_3d)

                print(
                    f"  Segment {seg_idx} chunk {chunk_idx + 1}/{num_chunks} "
                    f"({c_end - c_start} frames)",
                    end="\r",
                )

            print()  # newline after last chunk in segment
            seg_ball_3d = np.concatenate(seg_ball_3d_parts, axis=0)

            # Place segment prediction into full trajectory
            length = min(len(seg_ball_3d), end_i - start_i)
            ball_3d_full[start_i : start_i + length] = seg_ball_3d[:length]

        print(f"  Predicted 3D trajectory shape: {ball_3d_full.shape}")

        # Build results
        results: dict[str, Any] = {
            "ball_uv": ball_uv,
            "ball_3d": ball_3d_full,
            "visibility": visibility,
            "score": score,
        }

        # Step 3: Visualization
        if visualize:
            print("Generating visualizations...")
            results["fig_3d"] = self._visualizer.plot_trajectory_3d(
                ball_3d_full, title="BLCS 3D Ball Trajectory"
            )
            results["fig_2d"] = self._visualizer.plot_trajectory_2d(
                ball_3d_full, title="Ball Trajectory (Top View)"
            )
            results["fig_uv"] = self._visualizer.plot_uv_trajectory(
                ball_uv, visibility=visibility, title="2D Detection (UV)"
            )

        return results

    def run_streaming(
        self,
        video_processor,
        court_kp: NDArray[np.float32],
        batch_size: int = 32,
        visualize: bool = True,
        shot_frames: list[int] | None = None,
    ) -> dict[str, Any]:
        """Run inference with streaming video frames (memory efficient).

        Args:
            video_processor: VideoProcessor with video already opened.
            court_kp: Court keypoints, shape (20, 2), normalized [0, 1].
            batch_size: Frames to process at once.
            visualize: Whether to generate figures.
            shot_frames: Optional list of user-annotated shot frame indices.
                The earliest timestamp becomes the first segment start, and
                segments are defined between consecutive shots.

        Returns:
            Results dictionary with ball_uv, ball_3d, visibility, score.

        """
        total_frames = video_processor.total_frames

        # Collect all WASB results
        all_ball_uv = []
        all_visibility = []
        all_score = []

        # WASB requires at least 3 frames (frames_in parameter)
        min_frames = 3

        print(
            f"Running WASB detection on {total_frames} frames (batch_size={batch_size})..."
        )

        processed = 0
        pending_frames = None
        pending_indices = None

        for batch_frames, batch_indices in video_processor.iterate_frames(
            batch_size=batch_size
        ):
            # If we have pending frames from last iteration, prepend them
            if pending_frames is not None:
                batch_frames = np.concatenate([pending_frames, batch_frames], axis=0)
                batch_indices = pending_indices + batch_indices
                pending_frames = None
                pending_indices = None

            # If batch is too small, save for next iteration or handle at end
            if len(batch_frames) < min_frames:
                pending_frames = batch_frames
                pending_indices = batch_indices
                continue

            wasb_out = self._wasb.predict(batch_frames)
            all_ball_uv.append(wasb_out["ball_uv"])
            all_visibility.append(wasb_out["visibility"])
            all_score.append(wasb_out["score"])

            processed += len(batch_indices)
            print(f"  Processed {processed}/{total_frames} frames", end="\r")

        # Handle remaining frames by padding with previous frames
        if pending_frames is not None and len(pending_frames) > 0:
            # Get extra frames from video to meet minimum requirement
            need_extra = min_frames - len(pending_frames)
            start_idx = pending_indices[0] - need_extra
            if start_idx >= 0:
                extra_frames = []
                for idx in range(start_idx, pending_indices[0]):
                    frame = video_processor.get_single_frame(idx)
                    extra_frames.append(frame)
                if extra_frames:
                    extra_frames = np.stack(extra_frames, axis=0)
                    batch_frames = np.concatenate(
                        [extra_frames, pending_frames], axis=0
                    )
                    wasb_out = self._wasb.predict(batch_frames)
                    # Only keep results for the pending frames (skip the extra padding)
                    offset = need_extra
                    all_ball_uv.append(wasb_out["ball_uv"][offset:])
                    all_visibility.append(wasb_out["visibility"][offset:])
                    all_score.append(wasb_out["score"][offset:])
                    processed += len(pending_indices)
                    print(f"  Processed {processed}/{total_frames} frames", end="\r")

        print()  # Newline after progress

        # Concatenate WASB results
        ball_uv = np.concatenate(all_ball_uv, axis=0)
        visibility = np.concatenate(all_visibility, axis=0)
        score = np.concatenate(all_score, axis=0)

        return self._run_blcs_from_uv(
            ball_uv=ball_uv,
            visibility=visibility,
            score=score,
            court_kp=court_kp,
            visualize=visualize,
            shot_frames=shot_frames,
        )

    def run_from_wasb_results(
        self,
        ball_uv: NDArray[np.float32],
        visibility: NDArray[np.bool_],
        score: NDArray[np.floating[Any]] | None,
        court_kp: NDArray[np.float32],
        visualize: bool = True,
        shot_frames: list[int] | None = None,
    ) -> dict[str, Any]:
        """Run BLCS given pre-computed WASB outputs.

        Args:
            ball_uv: 2D detections, shape (T, 2).
            visibility: Visibility flags, shape (T,).
            score: Detection scores, shape (T,) or None.
            court_kp: Court keypoints, shape (20, 2).
            visualize: Whether to generate figures.
            shot_frames: Optional list of shot frame indices.

        """
        return self._run_blcs_from_uv(
            ball_uv=ball_uv,
            visibility=visibility,
            score=score,
            court_kp=court_kp,
            visualize=visualize,
            shot_frames=shot_frames,
        )

    @property
    def visualizer(self) -> TrajectoryVisualizer:
        """Get the trajectory visualizer."""
        return self._visualizer


class BLCSPipelineOffline:
    """Offline pipeline that only uses BLCS (no WASB/CUDA required).

    Useful for testing or when 2D detections are pre-computed.

    """

    def __init__(
        self,
        blcs_checkpoint: str | Path,
        device: str = "cpu",
    ) -> None:
        """Initialize offline pipeline.

        Args:
            blcs_checkpoint: Path to BLCS checkpoint.
            device: Inference device.

        """
        print("Loading BLCS model...")
        self._blcs = BLCSPredictor.load_from_checkpoint(blcs_checkpoint, device=device)
        self._visualizer = TrajectoryVisualizer()
        print("Offline pipeline ready.")

    def run(
        self,
        ball_uv: NDArray[np.float32],
        court_kp: NDArray[np.float32],
        ball_mask: NDArray[np.bool_] | None = None,
        visualize: bool = True,
    ) -> dict[str, Any]:
        """Run BLCS prediction.

        Args:
            ball_uv: Ball 2D positions (T, 2).
            court_kp: Court keypoints (20, 2).
            ball_mask: Visibility mask (T,).
            visualize: Generate figures.

        Returns:
            Results with ball_3d and figures.

        """
        ball_uv_t = torch.from_numpy(ball_uv).float()
        court_kp_t = torch.from_numpy(court_kp).float()

        ball_mask_t = None
        if ball_mask is not None:
            ball_mask_t = torch.from_numpy(ball_mask.astype(np.float32))

        blcs_out = self._blcs.predict(
            ball_uv=ball_uv_t,
            court_kp=court_kp_t,
            ball_mask=ball_mask_t,
            denormalize=True,
        )

        ball_3d = blcs_out["position"].squeeze(0).cpu().numpy()

        results: dict[str, Any] = {
            "ball_uv": ball_uv,
            "ball_3d": ball_3d,
        }

        if visualize:
            results["fig_3d"] = self._visualizer.plot_trajectory_3d(ball_3d)
            results["fig_2d"] = self._visualizer.plot_trajectory_2d(ball_3d)

        return results

    @property
    def visualizer(self) -> TrajectoryVisualizer:
        """Get the visualizer."""
        return self._visualizer
