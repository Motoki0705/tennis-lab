"""Visualization utilities for BLCS.

This module provides visualization tools for ball trajectory prediction,
using the shared rendering components from src.utils.rendering.

Example:
    >>> from src.blcs.inference.visualization import TrajectoryVisualizer
    >>> visualizer = TrajectoryVisualizer()
    >>> fig = visualizer.plot_trajectory_3d(trajectory)

"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from torch import Tensor

from src.utils.rendering import BallRenderer, CourtRenderer
from src.utils.rendering.ball_renderer import BallStyle

if TYPE_CHECKING:
    import matplotlib.pyplot as plt


class TrajectoryVisualizer:
    """Visualize ball trajectories in 2D and 3D.

    Uses shared rendering components for consistent visualization
    across the codebase.

    Example:
        >>> visualizer = TrajectoryVisualizer()
        >>> fig = visualizer.plot_trajectory_3d(trajectory)
        >>> fig = visualizer.plot_trajectory_2d(trajectory, gt_trajectory)

    """

    def __init__(self) -> None:
        """Initialize the visualizer with shared renderers."""
        self.court_renderer = CourtRenderer()
        self.ball_renderer = BallRenderer()

    def plot_trajectory_3d(
        self,
        trajectory: Tensor | np.ndarray,
        gt_trajectory: Tensor | np.ndarray | None = None,
        title: str = "Ball Trajectory",
        show_court: bool = True,
        figsize: tuple[int, int] = (12, 8),
    ) -> plt.Figure:
        """Plot 3D ball trajectory.

        Args:
            trajectory: Predicted trajectory, shape (T, 3) in meters.
            gt_trajectory: Ground truth trajectory (optional).
            title: Plot title.
            show_court: Whether to draw court.
            figsize: Figure size.

        Returns:
            matplotlib Figure.

        """
        import matplotlib.pyplot as plt

        trajectory_np = self._to_numpy(trajectory)
        gt_np = self._to_numpy(gt_trajectory) if gt_trajectory is not None else None

        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection="3d")

        # Draw court if requested
        if show_court:
            self.court_renderer.render_3d(ax, show_net=True)

        # Plot predicted trajectory
        pred_style = BallStyle(trajectory_color="#4444FF", ball_color="#4444FF")
        self.ball_renderer.render_trajectory_3d(
            ax, trajectory_np, show_start_end=True, style_override=pred_style
        )

        # Plot ground truth if provided
        if gt_np is not None:
            gt_style = BallStyle(trajectory_color="#44AA44", ball_color="#44AA44")
            self.ball_renderer.render_trajectory_3d(
                ax, gt_np, show_start_end=False, style_override=gt_style
            )

        ax.set_title(title)
        ax.legend(["Predicted", "Ground Truth"] if gt_np is not None else ["Predicted"])

        return fig

    def plot_trajectory_2d(
        self,
        trajectory: Tensor | np.ndarray,
        gt_trajectory: Tensor | np.ndarray | None = None,
        title: str = "Ball Trajectory (Top View)",
        show_court: bool = True,
        figsize: tuple[int, int] = (10, 12),
    ) -> plt.Figure:
        """Plot 2D bird's eye view of trajectory.

        Args:
            trajectory: Predicted trajectory, shape (T, 3) in meters.
            gt_trajectory: Ground truth trajectory (optional).
            title: Plot title.
            show_court: Whether to draw court.
            figsize: Figure size.

        Returns:
            matplotlib Figure.

        """
        import matplotlib.pyplot as plt

        trajectory_np = self._to_numpy(trajectory)
        gt_np = self._to_numpy(gt_trajectory) if gt_trajectory is not None else None

        fig, ax = plt.subplots(figsize=figsize)

        # Draw court if requested
        if show_court:
            self.court_renderer.render_2d(ax, show_fence=True)

        # Plot predicted trajectory
        pred_style = BallStyle(
            trajectory_color="#4444FF",
            ball_color="#4444FF",
            use_height_colormap=True,
        )
        self.ball_renderer.render_trajectory_2d(
            ax, trajectory_np, show_start_end=True, style_override=pred_style
        )

        # Plot ground truth if provided
        if gt_np is not None:
            gt_style = BallStyle(trajectory_color="#44AA44", ball_color="#44AA44")
            self.ball_renderer.render_trajectory_2d(
                ax, gt_np, show_start_end=False, style_override=gt_style
            )

        ax.set_title(title)
        ax.legend(["Predicted", "Ground Truth"] if gt_np is not None else ["Predicted"])

        return fig

    def plot_uv_trajectory(
        self,
        ball_uv: Tensor | np.ndarray,
        visibility: np.ndarray | None = None,
        title: str = "Ball 2D Trajectory",
        figsize: tuple[int, int] = (10, 6),
    ) -> plt.Figure:
        """Plot 2D ball trajectory in UV coordinates.

        Args:
            ball_uv: Ball UV trajectory, shape (T, 2).
            visibility: Visibility mask, shape (T,).
            title: Plot title.
            figsize: Figure size.

        Returns:
            matplotlib Figure.

        """
        import matplotlib.pyplot as plt

        uv_np = self._to_numpy(ball_uv)

        fig, ax = plt.subplots(figsize=figsize)

        self.ball_renderer.render_trajectory_uv(
            ax, uv_np, visibility=visibility, show_start_end=True
        )

        ax.set_title(title)

        return fig

    @staticmethod
    def _to_numpy(data: Tensor | np.ndarray | None) -> np.ndarray | None:
        """Convert tensor to numpy array.

        Args:
            data: Input tensor or array.

        Returns:
            Numpy array or None.

        """
        if data is None:
            return None
        if isinstance(data, Tensor):
            return data.cpu().numpy()
        return np.asarray(data)
