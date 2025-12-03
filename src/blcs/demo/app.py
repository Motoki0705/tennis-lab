#!/usr/bin/env python
"""BLCS demo application with matplotlib-based UI.

This demo allows users to:
1. Load a video file
2. Annotate court keypoints on a reference frame
3. Run WASB + BLCS inference
4. Visualize the estimated 3D ball trajectory

Usage:
    python -m src.blcs.demo.app --video input.mp4 --wasb-checkpoint wasb.pth.tar --blcs-checkpoint blcs.ckpt

"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Button, Slider

from src.blcs.demo.court_annotator import CourtAnnotator, QuickAnnotator
from src.blcs.demo.pipeline import BLCSPipeline, BLCSPipelineOffline
from src.blcs.demo.video_processor import VideoProcessor

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.backend_bases import MouseEvent
    from matplotlib.figure import Figure
    from numpy.typing import NDArray


class BLCSDemoApp:
    """Interactive matplotlib-based demo application.

    Provides a multi-step workflow:
    1. Video loading and frame selection
    2. Court keypoint annotation
    3. Inference execution
    4. Results visualization

    """

    def __init__(
        self,
        video_path: str | Path,
        wasb_checkpoint: str | Path | None = None,
        blcs_checkpoint: str | Path | None = None,
        device: str = "cuda",
    ) -> None:
        """Initialize demo application.

        Args:
            video_path: Path to input video.
            wasb_checkpoint: Path to WASB checkpoint. If None, uses offline mode.
            blcs_checkpoint: Path to BLCS checkpoint.
            device: Inference device.

        """
        self._video_path = Path(video_path)
        self._wasb_checkpoint = wasb_checkpoint
        self._blcs_checkpoint = blcs_checkpoint
        self._device = device

        # State
        self._first_frame: NDArray[np.uint8] | None = (
            None  # Only store first frame for annotation
        )
        self._fps: float = 30.0
        self._total_frames: int = 0
        self._court_kp: NDArray[np.float32] | None = None
        self._results: dict | None = None
        self._current_frame: int = 0
        self._shots: list[int] = []
        self._wasb_results_path: Path | None = None

        # UI elements
        self._fig: Figure | None = None
        self._axes: dict[str, Axes] = {}
        self._buttons: list[Button] = []  # Keep references to prevent GC
        self._slider = None

        # Initialize video processor
        self._video_processor = VideoProcessor()

    def run(self) -> None:
        """Start the demo application."""
        print("=" * 60)
        print("BLCS Ball Trajectory Demo")
        print("=" * 60)

        # Step 1: Load video
        self._load_video()

        # Step 2: Show main UI
        self._show_main_ui()

    def _load_video(self) -> None:
        """Load video file (lazy - only first frame loaded)."""
        print(f"\nLoading video: {self._video_path}")
        info = self._video_processor.open_video(self._video_path)
        self._fps = info["fps"]
        self._total_frames = info["total_frames"]
        # Only load first frame for annotation (memory efficient)
        self._first_frame = self._video_processor.get_single_frame(0)
        print(f"  Video: {self._total_frames} frames at {self._fps:.1f} fps")
        print(f"  Resolution: {info['width']}x{info['height']}")
        print("  (Only first frame loaded for annotation)")

    def _show_main_ui(self) -> None:
        """Show the main interactive UI."""
        # Create figure with subplots
        self._fig = plt.figure(figsize=(16, 10))
        self._fig.suptitle("BLCS Ball Trajectory Demo", fontsize=14, fontweight="bold")

        # Layout: 2x3 grid
        # Row 1: Video frame | Court KP status | Instructions
        # Row 2: Controls (slider, buttons)

        # Video frame display
        ax_video = self._fig.add_axes([0.05, 0.3, 0.4, 0.6])
        ax_video.set_title("Video Frame")
        self._axes["video"] = ax_video

        # Court KP status
        ax_status = self._fig.add_axes([0.5, 0.3, 0.2, 0.6])
        ax_status.set_title("Court Keypoints")
        ax_status.axis("off")
        self._axes["status"] = ax_status

        # Instructions
        ax_info = self._fig.add_axes([0.72, 0.3, 0.25, 0.6])
        ax_info.set_title("Instructions")
        ax_info.axis("off")
        self._axes["info"] = ax_info

        # Frame slider for shot annotation
        ax_slider = self._fig.add_axes([0.1, 0.18, 0.35, 0.03])
        self._slider = Slider(
            ax_slider,
            "Frame",
            0,
            self._total_frames - 1,
            valinit=0,
            valstep=1,
        )
        self._slider.on_changed(self._on_frame_change)

        # Frame info text
        ax_frame_info = self._fig.add_axes([0.5, 0.18, 0.2, 0.03])
        ax_frame_info.axis("off")
        self._axes["frame_info"] = ax_frame_info
        self._update_frame_info()

        # Buttons
        ax_btn_annotate = self._fig.add_axes([0.05, 0.08, 0.12, 0.05])
        ax_btn_quick = self._fig.add_axes([0.18, 0.08, 0.12, 0.05])
        ax_btn_load_kp = self._fig.add_axes([0.31, 0.08, 0.12, 0.05])
        ax_btn_run = self._fig.add_axes([0.5, 0.08, 0.15, 0.05])
        ax_btn_export = self._fig.add_axes([0.66, 0.08, 0.12, 0.05])
        ax_btn_add_shot = self._fig.add_axes([0.82, 0.08, 0.12, 0.05])
        ax_btn_clear_shot = self._fig.add_axes([0.82, 0.02, 0.12, 0.05])

        btn_annotate = Button(ax_btn_annotate, "Annotate (20pt)")
        btn_quick = Button(ax_btn_quick, "Quick (4pt)")
        btn_load_kp = Button(ax_btn_load_kp, "Load KP")
        btn_run = Button(ax_btn_run, "Run Inference")
        btn_export = Button(ax_btn_export, "Export")
        btn_add_shot = Button(ax_btn_add_shot, "Add Shot")
        btn_clear_shot = Button(ax_btn_clear_shot, "Clear Shots")

        btn_annotate.on_clicked(self._on_annotate_full)
        btn_quick.on_clicked(self._on_annotate_quick)
        btn_load_kp.on_clicked(self._on_load_keypoints)
        btn_run.on_clicked(self._on_run_inference)
        btn_export.on_clicked(self._on_export)
        btn_add_shot.on_clicked(self._on_add_shot)
        btn_clear_shot.on_clicked(self._on_clear_shots)

        # Store button references to prevent garbage collection
        self._buttons = [
            btn_annotate,
            btn_quick,
            btn_load_kp,
            btn_run,
            btn_export,
            btn_add_shot,
            btn_clear_shot,
        ]

        # Initial display
        self._update_video_display()
        self._update_status_display()
        self._update_info_display()

        plt.show()

    def _on_frame_change(self, val: float) -> None:
        self._current_frame = int(val)
        self._update_video_display()
        if "frame_info" in self._axes:
            self._update_frame_info()

    def _update_video_display(self) -> None:
        """Update video frame display."""
        ax = self._axes["video"]
        ax.clear()
        frame = self._video_processor.get_single_frame(self._current_frame)
        ax.imshow(frame)
        ax.set_title(f"Frame {self._current_frame} (reference=0)")
        ax.axis("off")

        # Draw court keypoints if available
        if self._court_kp is not None:
            h, w = frame.shape[:2]
            valid_mask = self._court_kp[:, 0] >= 0
            xs = self._court_kp[valid_mask, 0] * w
            ys = self._court_kp[valid_mask, 1] * h
            ax.scatter(xs, ys, c="lime", s=50, marker="o", edgecolors="black")

        self._fig.canvas.draw_idle()

    def _update_frame_info(self) -> None:
        ax = self._axes.get("frame_info")
        if ax is None:
            return
        ax.clear()
        ax.axis("off")
        t = self._current_frame / self._fps if self._fps > 0 else 0.0
        ax.text(
            0.5,
            0.5,
            f"Frame: {self._current_frame}/{self._total_frames - 1}  t={t:.2f}s",
            ha="center",
            va="center",
            fontsize=10,
        )
        self._fig.canvas.draw_idle()

    def _update_status_display(self) -> None:
        """Update court keypoint status display."""
        ax = self._axes["status"]
        ax.clear()
        ax.axis("off")
        ax.set_title("Court Keypoints")

        if self._court_kp is None:
            status_text = "Not annotated\n\nClick 'Annotate' or\n'Quick' to start"
            color = "red"
        else:
            valid = (self._court_kp[:, 0] >= 0).sum()
            status_text = f"Annotated: {valid}/20 points"
            color = "green" if valid >= 4 else "orange"

        if self._shots:
            status_text += "\n\nShots:\n"
            for i, frame in enumerate(sorted(self._shots)):
                t = frame / self._fps if self._fps > 0 else 0.0
                status_text += f"#{i}: f={frame}, t={t:.2f}s\n"

        ax.text(
            0.5,
            0.5,
            status_text,
            ha="center",
            va="center",
            fontsize=12,
            color=color,
            transform=ax.transAxes,
        )

        self._fig.canvas.draw_idle()

    def _update_info_display(self) -> None:
        """Update instructions display."""
        ax = self._axes["info"]
        ax.clear()
        ax.axis("off")
        ax.set_title("Instructions")

        instructions = """
1. Click 'Annotate' for
   full 20-point annotation
   OR 'Quick' for 4-corner
   (uses frame 0)

2. Click 'Run Inference'
   to estimate 3D trajectory

3. View results in new
   windows

4. 'Export' to save results

Note: For static camera,
frame 0 is used for court
keypoint annotation.
"""
        ax.text(
            0.1,
            0.9,
            instructions,
            ha="left",
            va="top",
            fontsize=10,
            transform=ax.transAxes,
            family="monospace",
        )

    def _on_annotate_full(self, event: MouseEvent) -> None:
        """Handle full annotation button."""
        print("\nOpening full annotation window (using frame 0)...")
        annotator = CourtAnnotator()
        # Use first frame for static camera
        self._court_kp = annotator.annotate(self._first_frame, self._court_kp)

        valid = (self._court_kp[:, 0] >= 0).sum()
        print(f"Annotation complete: {valid}/20 points")

        self._update_video_display()
        self._update_status_display()

    def _on_annotate_quick(self, event: MouseEvent) -> None:
        """Handle quick annotation button."""
        print("\nOpening quick annotation window (4 corners, using frame 0)...")
        annotator = QuickAnnotator()
        # Use first frame for static camera
        self._court_kp = annotator.annotate(self._first_frame)

        valid = (self._court_kp[:, 0] >= 0).sum()
        print(f"Quick annotation complete: {valid}/20 points (interpolated)")

        self._update_video_display()
        self._update_status_display()

    def _on_load_keypoints(self, event: MouseEvent) -> None:
        """Handle load keypoints button."""
        # Simple file dialog using input
        print("\nEnter path to keypoints JSON file:")
        # For simplicity, use a hardcoded path or prompt
        # In a real app, you'd use tkinter filedialog
        try:
            import tkinter as tk
            from tkinter import filedialog

            root = tk.Tk()
            root.withdraw()
            filepath = filedialog.askopenfilename(
                title="Select Keypoints JSON",
                filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
            )
            root.destroy()

            if filepath:
                self._court_kp = CourtAnnotator.load_keypoints(filepath)
                print(f"Loaded keypoints from: {filepath}")
                self._update_video_display()
                self._update_status_display()
        except ImportError:
            print("tkinter not available. Please provide path manually.")

    def _on_add_shot(self, event: MouseEvent) -> None:
        frame = int(self._current_frame)
        if frame < 0 or frame >= self._total_frames:
            return
        if frame not in self._shots:
            self._shots.append(frame)
            self._shots.sort()
        t = frame / self._fps if self._fps > 0 else 0.0
        print(f"Added shot at frame {frame} (t={t:.3f}s)")
        self._update_status_display()

    def _on_clear_shots(self, event: MouseEvent) -> None:
        if not self._shots:
            return
        self._shots = []
        print("Cleared all shot annotations.")
        self._update_status_display()

    def _save_shots_json(self, path: Path) -> None:
        """Save shot annotations to a JSON file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "video_path": str(self._video_path),
            "fps": float(self._fps),
            "total_frames": int(self._total_frames),
            "frames": sorted(int(f) for f in self._shots),
        }
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        print(f"Shot annotations exported to: {path}")

    def load_shots_from_json(self, path: str | Path) -> None:
        """Load shot annotations from a JSON file."""
        p = Path(path)
        if not p.exists():
            print(f"Shot annotation file not found: {p}")
            return

        with p.open("r", encoding="utf-8") as f:
            data = json.load(f)

        frames = data.get("frames") or data.get("shots") or []
        self._shots = sorted(int(f) for f in frames if 0 <= int(f) < self._total_frames)
        print(f"Loaded {len(self._shots)} shots from: {p}")
        self._update_status_display()

    def _on_run_inference(self, event: MouseEvent) -> None:
        """Handle run inference button."""
        if self._court_kp is None:
            print("\nError: Please annotate court keypoints first!")
            return

        valid = (self._court_kp[:, 0] >= 0).sum()
        if valid < 4:
            print(f"\nError: Need at least 4 keypoints, got {valid}")
            return

        print("\n" + "=" * 40)
        print("Running inference...")
        print("=" * 40)

        try:
            # Initialize pipeline
            if self._wasb_checkpoint is not None and self._blcs_checkpoint is not None:
                pipeline = BLCSPipeline(
                    wasb_checkpoint=self._wasb_checkpoint,
                    blcs_checkpoint=self._blcs_checkpoint,
                    device=self._device,
                )

                # If precomputed WASB results are available, skip WASB inference
                if (
                    self._wasb_results_path is not None
                    and self._wasb_results_path.exists()
                ):
                    print(
                        f"Using precomputed WASB results from: {self._wasb_results_path}"
                    )
                    data = np.load(self._wasb_results_path)
                    ball_uv = data["ball_uv"].astype(np.float32)
                    visibility = data["visibility"].astype(bool)
                    score = (
                        data["score"].astype(np.float32)
                        if "score" in data.files
                        else None
                    )

                    self._results = pipeline.run_from_wasb_results(
                        ball_uv=ball_uv,
                        visibility=visibility,
                        score=score,
                        court_kp=self._court_kp,
                        visualize=True,
                        shot_frames=self._shots if self._shots else None,
                    )
                else:
                    # Use streaming inference to avoid OOM
                    self._results = pipeline.run_streaming(
                        self._video_processor,
                        self._court_kp,
                        shot_frames=self._shots if self._shots else None,
                    )
            elif self._blcs_checkpoint is not None:
                # Offline mode - need pre-computed ball_uv
                print("Warning: Running in offline mode (no WASB)")
                print("Using dummy ball positions for demo")

                # Create dummy ball positions for demo
                T = self._total_frames
                ball_uv = np.random.rand(T, 2).astype(np.float32) * 0.5 + 0.25
                ball_mask = np.ones(T, dtype=bool)

                pipeline = BLCSPipelineOffline(
                    blcs_checkpoint=self._blcs_checkpoint,
                    device=self._device,
                )
                self._results = pipeline.run(ball_uv, self._court_kp, ball_mask)
            else:
                print("Error: No checkpoint provided!")
                return

            print("\nInference complete!")
            print(f"  3D trajectory shape: {self._results['ball_3d'].shape}")

            # Show results
            self._show_results()

        except Exception as e:
            print(f"\nError during inference: {e}")
            import traceback

            traceback.print_exc()

    def _show_results(self) -> None:
        """Show inference results in new windows."""
        if self._results is None:
            return

        # Show 3D trajectory
        if "fig_3d" in self._results:
            self._results["fig_3d"].show()

        # Show 2D trajectory
        if "fig_2d" in self._results:
            self._results["fig_2d"].show()

        # Show UV trajectory
        if "fig_uv" in self._results:
            self._results["fig_uv"].show()

        plt.show(block=False)

    def _on_export(self, event: MouseEvent) -> None:
        """Handle export button."""
        if self._results is None:
            print("\nNo results to export. Run inference first.")
            return

        # Prepare output directory: data/demo/{video_stem}/
        base_dir = Path("data") / "demo" / self._video_path.stem
        base_dir.mkdir(parents=True, exist_ok=True)

        ball_uv = self._results.get("ball_uv")
        visibility = self._results.get("visibility")
        score = self._results.get("score")

        # Export WASB-only results (2D detections)
        if ball_uv is not None and visibility is not None:
            wasb_path = base_dir / "wasb_results.npz"
            np.savez(
                wasb_path,
                ball_uv=ball_uv,
                visibility=visibility,
                score=score,
            )
            print(f"\nWASB results exported to: {wasb_path}")

        # Export BLCS results (3D trajectory)
        blcs_path = base_dir / "blcs_results.npz"
        np.savez(
            blcs_path,
            ball_uv=ball_uv,
            ball_3d=self._results["ball_3d"],
            visibility=visibility,
            score=score,
            court_kp=self._court_kp,
        )
        print(f"BLCS results exported to: {blcs_path}")

        # Export court keypoints
        kp_path = base_dir / "court_kp.json"
        annotator = CourtAnnotator()
        h, w = self._first_frame.shape[:2]
        annotator._keypoints = [
            (kp[0] * w, kp[1] * h) for kp in self._court_kp if kp[0] >= 0
        ]
        annotator._image_shape = (w, h)
        annotator.save_keypoints(kp_path)
        print(f"Keypoints exported to: {kp_path}")

        # Export shot annotations
        shots_path = base_dir / "shots.json"
        self._save_shots_json(shots_path)


def run_demo(
    video_path: str,
    wasb_checkpoint: str | None = None,
    blcs_checkpoint: str | None = None,
    keypoints_path: str | None = None,
    shots_path: str | None = None,
    wasb_results_path: str | None = None,
    device: str = "cuda",
) -> None:
    """Run the BLCS demo application.

    Args:
        video_path: Path to input video file.
        wasb_checkpoint: Path to WASB checkpoint.
        blcs_checkpoint: Path to BLCS checkpoint.
        keypoints_path: Path to pre-annotated keypoints JSON.
        shots_path: Path to shot annotations JSON.
        wasb_results_path: Path to precomputed WASB 2D detections (.npz).
        device: Inference device.

    """
    app = BLCSDemoApp(
        video_path=video_path,
        wasb_checkpoint=wasb_checkpoint,
        blcs_checkpoint=blcs_checkpoint,
        device=device,
    )

    demo_dir = Path("data") / "demo" / app._video_path.stem

    # Load pre-annotated keypoints
    kp_source: Path | str | None = None
    if keypoints_path is not None:
        app._court_kp = CourtAnnotator.load_keypoints(keypoints_path)
        kp_source = keypoints_path
    else:
        default_kp = demo_dir / "court_kp.json"
        if default_kp.exists():
            app._court_kp = CourtAnnotator.load_keypoints(default_kp)
            kp_source = default_kp

    if app._court_kp is not None and kp_source is not None:
        print(f"Loaded keypoints from: {kp_source}")
        valid = (app._court_kp[:, 0] >= 0).sum()
        print(f"  {valid}/20 points loaded")

    # Load shot annotations
    if shots_path is not None:
        app.load_shots_from_json(shots_path)
    else:
        default_shots = demo_dir / "shots.json"
        if default_shots.exists():
            app.load_shots_from_json(default_shots)

    # Set precomputed WASB results path (if any)
    if wasb_results_path is not None:
        p = Path(wasb_results_path)
        if p.exists():
            app._wasb_results_path = p
        else:
            print(f"Warning: Specified WASB results not found: {p}")
    else:
        default_wasb = demo_dir / "wasb_results.npz"
        if default_wasb.exists():
            app._wasb_results_path = default_wasb

    app.run()


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="BLCS Ball Trajectory Demo",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--video",
        type=str,
        required=True,
        help="Path to input video file",
    )
    parser.add_argument(
        "--wasb-checkpoint",
        type=str,
        default=None,
        help="Path to WASB model checkpoint (.pth.tar)",
    )
    parser.add_argument(
        "--blcs-checkpoint",
        type=str,
        default=None,
        help="Path to BLCS model checkpoint (.ckpt)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Inference device",
    )
    parser.add_argument(
        "--keypoints",
        type=str,
        default=None,
        help="Path to pre-annotated court keypoints JSON file",
    )
    parser.add_argument(
        "--shots",
        type=str,
        default=None,
        help="Path to shot annotations JSON file",
    )
    parser.add_argument(
        "--wasb-results",
        type=str,
        default=None,
        help=(
            "Path to precomputed WASB 2D results (.npz). "
            "If omitted, will try data/demo/{video_stem}/wasb_results.npz."
        ),
    )
    return parser.parse_args()


def main() -> int:
    """Main entry point."""
    args = parse_args()

    if args.wasb_checkpoint is None and args.blcs_checkpoint is None:
        print("Warning: No checkpoints provided. Demo will run in limited mode.")

    run_demo(
        video_path=args.video,
        wasb_checkpoint=args.wasb_checkpoint,
        blcs_checkpoint=args.blcs_checkpoint,
        keypoints_path=args.keypoints,
        shots_path=args.shots,
        wasb_results_path=args.wasb_results,
        device=args.device,
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
