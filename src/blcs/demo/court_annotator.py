"""Interactive court keypoint annotator using matplotlib."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Button

from src.utils.geometry.constants import COURT_KP_NAMES, NUM_COURT_KP

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.backend_bases import MouseEvent
    from matplotlib.figure import Figure
    from numpy.typing import NDArray


# Court keypoint names (20 points) using CourtKP20 definition
# Convert to human-readable format for UI display
def _format_name_for_display(name: str) -> str:
    """Convert snake_case keypoint name to human-readable format."""
    return name.replace("_", " ")


COURT_KEYPOINT_NAMES: list[str] = [
    _format_name_for_display(name) for name in COURT_KP_NAMES
]

NUM_COURT_KEYPOINTS = NUM_COURT_KP


class CourtAnnotator:
    """Interactive matplotlib-based court keypoint annotator.

    Click on the image to add keypoints in order. Points are added
    sequentially following the BLCS court keypoint definition.

    Example:
        >>> annotator = CourtAnnotator()
        >>> keypoints = annotator.annotate(reference_frame)
        >>> print(keypoints.shape)  # (20, 2)

    """

    def __init__(self) -> None:
        """Initialize the annotator."""
        self._keypoints: list[tuple[float, float]] = []
        self._current_index: int = 0
        self._fig: Figure | None = None
        self._ax: Axes | None = None
        self._image_shape: tuple[int, int] | None = None
        self._scatter = None
        self._texts: list = []
        self._done: bool = False
        self._on_complete: Callable[[NDArray[np.float32]], None] | None = None
        # Keep references to buttons to prevent garbage collection
        self._buttons: list[Button] = []

    def annotate(
        self,
        image: NDArray[np.uint8],
        existing_keypoints: NDArray[np.float32] | None = None,
    ) -> NDArray[np.float32]:
        """Open interactive annotation window.

        Args:
            image: Reference image to annotate, shape (H, W, 3).
            existing_keypoints: Pre-existing keypoints to edit, shape (N, 2).

        Returns:
            Annotated keypoints, shape (20, 2), normalized to [0, 1].

        """
        self._image_shape = (image.shape[1], image.shape[0])  # (W, H)
        self._keypoints = []
        self._current_index = 0
        self._done = False

        # Load existing keypoints if provided
        if existing_keypoints is not None:
            for i, kp in enumerate(existing_keypoints):
                if i < NUM_COURT_KEYPOINTS:
                    # Convert from normalized to pixel coordinates
                    x = kp[0] * self._image_shape[0]
                    y = kp[1] * self._image_shape[1]
                    self._keypoints.append((x, y))
            self._current_index = len(self._keypoints)

        # Create figure
        self._fig, self._ax = plt.subplots(figsize=(14, 10))
        self._ax.imshow(image)
        self._ax.set_title(self._get_title())

        # Add buttons (keep references to prevent garbage collection)
        ax_undo = plt.axes([0.7, 0.02, 0.1, 0.04])
        ax_reset = plt.axes([0.81, 0.02, 0.1, 0.04])
        ax_done = plt.axes([0.59, 0.02, 0.1, 0.04])

        btn_undo = Button(ax_undo, "Undo")
        btn_reset = Button(ax_reset, "Reset")
        btn_done = Button(ax_done, "Done")

        btn_undo.on_clicked(self._on_undo)
        btn_reset.on_clicked(self._on_reset)
        btn_done.on_clicked(self._on_done)

        # Store button references to prevent garbage collection
        self._buttons = [btn_undo, btn_reset, btn_done]

        # Connect mouse event
        self._cid = self._fig.canvas.mpl_connect("button_press_event", self._on_click)

        # Draw existing keypoints
        self._update_display()

        # Use canvas event loop instead of plt.show() to work with existing windows
        self._fig.canvas.draw()
        self._fig.canvas.start_event_loop()

        # Return normalized keypoints
        return self._get_normalized_keypoints()

    def _get_title(self) -> str:
        """Get current title with instruction."""
        if self._current_index < NUM_COURT_KEYPOINTS:
            name = COURT_KEYPOINT_NAMES[self._current_index]
            return f"Click point {self._current_index}: {name} ({self._current_index + 1}/{NUM_COURT_KEYPOINTS})"
        return f"All {NUM_COURT_KEYPOINTS} points annotated. Click 'Done' to finish."

    def _on_click(self, event: MouseEvent) -> None:
        """Handle mouse click event."""
        # Only handle left mouse button
        # event.button can be int (1) or MouseButton.LEFT depending on matplotlib version
        if event.button not in (1, "left") and str(event.button) != "MouseButton.LEFT":
            return
        # Only handle clicks on the image axes (not buttons)
        if event.inaxes is None or event.inaxes != self._ax:
            return
        if self._current_index >= NUM_COURT_KEYPOINTS:
            return
        # Check for valid coordinates
        if event.xdata is None or event.ydata is None:
            return

        x, y = event.xdata, event.ydata
        self._keypoints.append((x, y))
        self._current_index += 1
        print(f"  Added point {self._current_index}: ({x:.1f}, {y:.1f})")

        self._update_display()

    def _on_undo(self, event: MouseEvent) -> None:
        """Handle undo button click."""
        if self._keypoints:
            self._keypoints.pop()
            self._current_index = max(0, self._current_index - 1)
            self._update_display()

    def _on_reset(self, event: MouseEvent) -> None:
        """Handle reset button click."""
        self._keypoints = []
        self._current_index = 0
        self._update_display()

    def _on_done(self, event: MouseEvent) -> None:
        """Handle done button click."""
        self._done = True
        # Stop the event loop first, then close
        self._fig.canvas.stop_event_loop()
        plt.close(self._fig)

        if self._on_complete is not None:
            self._on_complete(self._get_normalized_keypoints())

    def _update_display(self) -> None:
        """Update the display with current keypoints."""
        # Remove old scatter and texts
        if self._scatter is not None:
            self._scatter.remove()
            self._scatter = None

        for txt in self._texts:
            txt.remove()
        self._texts = []

        # Draw keypoints
        if self._keypoints:
            xs = [kp[0] for kp in self._keypoints]
            ys = [kp[1] for kp in self._keypoints]
            self._scatter = self._ax.scatter(
                xs,
                ys,
                c="lime",
                s=100,
                marker="o",
                edgecolors="black",
                linewidths=2,
                zorder=10,
            )

            # Add labels
            for i, (x, y) in enumerate(self._keypoints):
                txt = self._ax.text(
                    x + 5,
                    y - 5,
                    str(i),
                    color="yellow",
                    fontsize=10,
                    fontweight="bold",
                    zorder=11,
                )
                self._texts.append(txt)

        # Update title
        self._ax.set_title(self._get_title())
        self._fig.canvas.draw_idle()

    def _get_normalized_keypoints(self) -> NDArray[np.float32]:
        """Get keypoints normalized to [0, 1].

        Returns:
            Normalized keypoints, shape (20, 2).
            Missing points are filled with (-1, -1).

        """
        result = np.full((NUM_COURT_KEYPOINTS, 2), -1.0, dtype=np.float32)

        if self._image_shape is None:
            return result

        w, h = self._image_shape
        for i, (x, y) in enumerate(self._keypoints):
            if i < NUM_COURT_KEYPOINTS:
                result[i, 0] = x / w
                result[i, 1] = y / h

        return result

    def save_keypoints(self, path: str | Path) -> None:
        """Save keypoints to JSON file.

        Args:
            path: Output JSON file path.

        """
        keypoints = self._get_normalized_keypoints()
        data = {
            "keypoints": keypoints.tolist(),
            "names": COURT_KEYPOINT_NAMES,
            "num_annotated": len(self._keypoints),
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    @staticmethod
    def load_keypoints(path: str | Path) -> NDArray[np.float32]:
        """Load keypoints from JSON file.

        Args:
            path: JSON file path.

        Returns:
            Keypoints array, shape (20, 2).

        """
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return np.array(data["keypoints"], dtype=np.float32)

    def is_complete(self) -> bool:
        """Check if all keypoints are annotated.

        Returns:
            True if all 20 keypoints are annotated.

        """
        return len(self._keypoints) >= NUM_COURT_KEYPOINTS


class QuickAnnotator:
    """Simplified annotator for essential court points only.

    Only annotates the 4 corner points, then interpolates the rest.
    Faster for quick demos but less accurate.

    """

    # Essential points: 4 doubles corners
    ESSENTIAL_INDICES = [0, 1, 2, 3]  # far/near doubles corners
    ESSENTIAL_NAMES = [
        "far doubles corner left",
        "far doubles corner right",
        "near doubles corner left",
        "near doubles corner right",
    ]

    def __init__(self) -> None:
        """Initialize quick annotator."""
        self._keypoints: list[tuple[float, float]] = []
        self._current_index: int = 0
        self._fig: Figure | None = None
        self._ax: Axes | None = None
        self._image_shape: tuple[int, int] | None = None
        self._buttons: list[Button] = []

    def annotate(self, image: NDArray[np.uint8]) -> NDArray[np.float32]:
        """Annotate essential corner points.

        Args:
            image: Reference image.

        Returns:
            All 20 keypoints (interpolated), shape (20, 2).

        """
        self._image_shape = (image.shape[1], image.shape[0])
        self._keypoints = []
        self._current_index = 0

        self._fig, self._ax = plt.subplots(figsize=(14, 10))
        self._ax.imshow(image)
        self._ax.set_title(self._get_title())

        ax_done = plt.axes([0.8, 0.02, 0.1, 0.04])
        btn_done = Button(ax_done, "Done")
        btn_done.on_clicked(self._on_done)

        # Store button reference to prevent garbage collection
        self._buttons = [btn_done]

        self._cid = self._fig.canvas.mpl_connect("button_press_event", self._on_click)

        # Use canvas event loop instead of plt.show()
        self._fig.canvas.draw()
        self._fig.canvas.start_event_loop()

        return self._interpolate_all_keypoints()

    def _on_done(self, event: MouseEvent) -> None:
        """Handle done button click."""
        self._fig.canvas.stop_event_loop()
        plt.close(self._fig)

    def _get_title(self) -> str:
        """Get current title."""
        if self._current_index < len(self.ESSENTIAL_INDICES):
            name = self.ESSENTIAL_NAMES[self._current_index]
            return f"Click: {name} ({self._current_index + 1}/4)"
        return "4 corners annotated. Click 'Done'."

    def _on_click(self, event: MouseEvent) -> None:
        """Handle click."""
        # Only handle left mouse button
        if event.button not in (1, "left") and str(event.button) != "MouseButton.LEFT":
            return
        if event.inaxes is None or event.inaxes != self._ax:
            return
        if self._current_index >= len(self.ESSENTIAL_INDICES):
            return
        # Check for valid coordinates
        if event.xdata is None or event.ydata is None:
            return

        self._keypoints.append((event.xdata, event.ydata))
        self._current_index += 1
        print(
            f"  Added corner {self._current_index}: ({event.xdata:.1f}, {event.ydata:.1f})"
        )

        # Draw point
        self._ax.scatter(
            [event.xdata],
            [event.ydata],
            c="lime",
            s=150,
            marker="o",
            edgecolors="black",
            linewidths=2,
        )
        self._ax.text(
            event.xdata + 5,
            event.ydata - 5,
            self.ESSENTIAL_NAMES[self._current_index - 1][:10],
            color="yellow",
            fontsize=9,
        )
        self._ax.set_title(self._get_title())
        self._fig.canvas.draw_idle()

    def _interpolate_all_keypoints(self) -> NDArray[np.float32]:
        """Interpolate all 20 keypoints from 4 corners.

        This is a simplified interpolation assuming a standard court layout.

        Returns:
            All 20 keypoints, shape (20, 2).

        """
        result = np.full((NUM_COURT_KEYPOINTS, 2), -1.0, dtype=np.float32)

        if len(self._keypoints) < 4 or self._image_shape is None:
            return result

        w, h = self._image_shape

        # Normalize corner points
        corners = np.array(self._keypoints[:4], dtype=np.float32)
        corners[:, 0] /= w
        corners[:, 1] /= h

        # Assign corners
        result[0] = corners[0]  # far left
        result[1] = corners[1]  # far right
        result[2] = corners[2]  # near left
        result[3] = corners[3]  # near right

        # Interpolate singles corners (inside doubles)
        # Ratio: singles width / doubles width ≈ 0.75
        ratio = 0.75
        center_far = (corners[0] + corners[1]) / 2
        center_near = (corners[2] + corners[3]) / 2

        result[4] = center_far + ratio * (corners[0] - center_far)  # far singles left
        result[6] = center_far + ratio * (corners[1] - center_far)  # far singles right
        result[5] = center_near + ratio * (
            corners[2] - center_near
        )  # near singles left
        result[7] = center_near + ratio * (
            corners[3] - center_near
        )  # near singles right

        # Service lines (roughly 54% from net to baseline)
        service_ratio = 0.54
        result[8] = result[4] + service_ratio * (
            center_far - result[4]
        )  # far service left
        result[9] = result[6] + service_ratio * (
            center_far - result[6]
        )  # far service right
        result[10] = result[5] + service_ratio * (
            center_near - result[5]
        )  # near service left
        result[11] = result[7] + service_ratio * (
            center_near - result[7]
        )  # near service right

        # Service T
        net_center = (center_far + center_near) / 2
        result[12] = (result[8] + result[9]) / 2  # far T
        result[13] = (result[10] + result[11]) / 2  # near T

        # Net points
        result[14] = net_center  # net center ground
        result[15] = net_center + 0.6 * (corners[0] - center_far)  # left post base
        result[16] = result[15]  # left post top (same UV)
        result[17] = net_center + 0.6 * (corners[1] - center_far)  # right post base
        result[18] = result[17]  # right post top (same UV)
        result[19] = net_center  # center strap

        return result
