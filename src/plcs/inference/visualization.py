"""Visualization utilities for PLCS predictions."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

from src.rendering import CourtRenderer

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
    from torch import Tensor


# Shared court renderer instance
_court_renderer = CourtRenderer()


def visualize_prediction(
    position: Tensor | tuple[float, float, float],
    yaw: float,
    ax: Axes | None = None,
    show_court: bool = True,
    player_color: str = "red",
    arrow_length: float = 1.5,
) -> Figure:
    """Visualize a PLCS prediction on a court diagram.

    Args:
        position: Player position (x, y, z) in meters.
        yaw: Player yaw angle in radians.
        ax: Optional matplotlib axes. Creates new figure if None.
        show_court: Whether to draw the court lines.
        player_color: Color for player marker and arrow.
        arrow_length: Length of direction arrow in meters.

    Returns:
        Figure: Matplotlib figure with the visualization.

    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(10, 12))
    else:
        fig = ax.get_figure()

    if show_court:
        _court_renderer.render_2d(ax, show_fence=False)

    # Convert position to numpy
    if hasattr(position, "numpy"):
        position = position.numpy()
    x, y, z = position[0], position[1], position[2]

    # Plot player position
    ax.plot(x, y, "o", color=player_color, markersize=12, zorder=10)

    # Draw direction arrow
    dx = arrow_length * math.sin(yaw)
    dy = arrow_length * math.cos(yaw)
    ax.arrow(
        x,
        y,
        dx,
        dy,
        head_width=0.3,
        head_length=0.2,
        fc=player_color,
        ec=player_color,
        zorder=11,
    )

    # Add position text
    ax.text(
        x + 0.5,
        y + 0.5,
        f"({x:.2f}, {y:.2f})",
        fontsize=9,
        color="white",
        zorder=12,
    )

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title("PLCS Prediction")

    return fig


def visualize_batch(
    positions: Tensor,
    yaws: Tensor,
    figsize: tuple[int, int] = (10, 12),
    max_players: int = 10,
) -> Figure:
    """Visualize multiple predictions on a single court.

    Args:
        positions: Player positions (N, 3) in meters.
        yaws: Player yaw angles (N,) in radians.
        figsize: Figure size.
        max_players: Maximum number of players to display.

    Returns:
        Figure: Matplotlib figure with all players.

    """
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    _court_renderer.render_2d(ax, show_fence=False)

    # Color map for different players
    colors = plt.cm.tab10(np.linspace(0, 1, min(len(positions), max_players)))

    for i, (pos, yaw) in enumerate(zip(positions[:max_players], yaws[:max_players])):
        if hasattr(pos, "numpy"):
            pos = pos.numpy()
        x, y = pos[0], pos[1]
        yaw_val = yaw.item() if hasattr(yaw, "item") else yaw

        # Plot player
        ax.plot(x, y, "o", color=colors[i], markersize=10, zorder=10)

        # Direction arrow
        dx = 1.2 * math.sin(yaw_val)
        dy = 1.2 * math.cos(yaw_val)
        ax.arrow(
            x,
            y,
            dx,
            dy,
            head_width=0.2,
            head_length=0.15,
            fc=colors[i],
            ec=colors[i],
            zorder=11,
        )

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title(f"PLCS Predictions (n={len(positions)})")

    return fig


def visualize_sequence_trajectory(
    positions: Tensor | np.ndarray,
    yaws: Tensor | np.ndarray | None = None,
    max_arrows: int = 5,
) -> Figure:
    """Visualize a sequence trajectory on the court.

    Args:
        positions: Player positions over time, shape (T, 3) in meters.
        yaws: Optional yaw angles over time, shape (T,) in radians.
        max_arrows: Maximum number of orientation arrows to draw along trajectory.

    Returns:
        Figure: Matplotlib figure with the trajectory visualization.

    """
    from torch import (
        Tensor as TorchTensor,  # local import to avoid hard dep in type hints
    )

    if isinstance(positions, TorchTensor):
        positions_np = positions.cpu().numpy()
    else:
        positions_np = np.asarray(positions)

    T = positions_np.shape[0]
    fig, ax = plt.subplots(1, 1, figsize=(10, 12))
    _court_renderer.render_2d(ax, show_fence=False)

    # Plot trajectory in XY plane
    xs = positions_np[:, 0]
    ys = positions_np[:, 1]
    ax.plot(xs, ys, "-o", color="yellow", markersize=4, linewidth=2, zorder=10)

    # Draw orientation arrows at a few time steps
    if yaws is not None:
        if isinstance(yaws, TorchTensor):
            yaws_np = yaws.cpu().numpy()
        else:
            yaws_np = np.asarray(yaws)

        num_arrows = min(max_arrows, T)
        if num_arrows > 0:
            indices = np.linspace(0, T - 1, num_arrows, dtype=int)
            for idx in indices:
                x = xs[idx]
                y = ys[idx]
                yaw_val = float(yaws_np[idx])
                dx = 1.2 * math.sin(yaw_val)
                dy = 1.2 * math.cos(yaw_val)
                ax.arrow(
                    x,
                    y,
                    dx,
                    dy,
                    head_width=0.25,
                    head_length=0.2,
                    fc="orange",
                    ec="orange",
                    zorder=11,
                )

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title("PLCS Sequence Trajectory")

    return fig


def visualize_sequence_batch(
    positions_batch: Tensor | np.ndarray,
    yaws_batch: Tensor | np.ndarray | None = None,
    max_sequences: int = 5,
) -> Figure:
    """Visualize multiple sequence trajectories on a single court.

    Args:
        positions_batch: Player positions (B, T, 3) in meters.
        yaws_batch: Optional yaw angles (B, T) in radians.
        max_sequences: Maximum number of sequences to display.

    Returns:
        Figure: Matplotlib figure with all trajectories.

    """
    from torch import (
        Tensor as TorchTensor,  # local import to avoid hard dep in type hints
    )

    if isinstance(positions_batch, TorchTensor):
        positions_np = positions_batch.cpu().numpy()
    else:
        positions_np = np.asarray(positions_batch)

    if yaws_batch is not None:
        if isinstance(yaws_batch, TorchTensor):
            yaws_np = yaws_batch.cpu().numpy()
        else:
            yaws_np = np.asarray(yaws_batch)
    else:
        yaws_np = None

    B = positions_np.shape[0]
    fig, ax = plt.subplots(1, 1, figsize=(10, 12))
    _court_renderer.render_2d(ax, show_fence=False)

    num_seq = min(B, max_sequences)
    colors = plt.cm.tab10(np.linspace(0, 1, num_seq))

    for i in range(num_seq):
        seq_pos = positions_np[i]
        xs = seq_pos[:, 0]
        ys = seq_pos[:, 1]
        ax.plot(xs, ys, "-o", color=colors[i], markersize=3, linewidth=2, zorder=10)

        if yaws_np is not None:
            yaws_seq = yaws_np[i]
            # Draw arrow at final frame
            x = xs[-1]
            y = ys[-1]
            yaw_val = float(yaws_seq[-1])
            dx = 1.2 * math.sin(yaw_val)
            dy = 1.2 * math.cos(yaw_val)
            ax.arrow(
                x,
                y,
                dx,
                dy,
                head_width=0.25,
                head_length=0.2,
                fc=colors[i],
                ec=colors[i],
                zorder=11,
            )

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title(f"PLCS Sequence Predictions (n={num_seq})")

    return fig


def plot_prediction_errors(
    pred_positions: Tensor,
    true_positions: Tensor,
    figsize: tuple[int, int] = (12, 5),
) -> Figure:
    """Plot prediction error distributions.

    Args:
        pred_positions: Predicted positions (N, 3) in meters.
        true_positions: Ground truth positions (N, 3) in meters.
        figsize: Figure size.

    Returns:
        Figure: Matplotlib figure with error histograms.

    """
    if hasattr(pred_positions, "numpy"):
        pred_positions = pred_positions.numpy()
    if hasattr(true_positions, "numpy"):
        true_positions = true_positions.numpy()

    errors = pred_positions - true_positions
    euclidean_errors = np.linalg.norm(errors, axis=1)

    fig, axes = plt.subplots(1, 4, figsize=figsize)

    # X error
    axes[0].hist(errors[:, 0], bins=30, color="blue", alpha=0.7)
    axes[0].set_xlabel("X Error (m)")
    axes[0].set_ylabel("Count")
    axes[0].axvline(0, color="red", linestyle="--")

    # Y error
    axes[1].hist(errors[:, 1], bins=30, color="green", alpha=0.7)
    axes[1].set_xlabel("Y Error (m)")
    axes[1].axvline(0, color="red", linestyle="--")

    # Z error
    axes[2].hist(errors[:, 2], bins=30, color="orange", alpha=0.7)
    axes[2].set_xlabel("Z Error (m)")
    axes[2].axvline(0, color="red", linestyle="--")

    # Euclidean error
    axes[3].hist(euclidean_errors, bins=30, color="purple", alpha=0.7)
    axes[3].set_xlabel("Euclidean Error (m)")
    mean_err = euclidean_errors.mean()
    axes[3].axvline(
        mean_err, color="red", linestyle="--", label=f"Mean: {mean_err:.3f}m"
    )
    axes[3].legend()

    fig.suptitle("Position Prediction Errors")
    fig.tight_layout()

    return fig


def visualize_prediction_3d(
    position: Tensor | np.ndarray | tuple[float, float, float],
    yaw: float,
    figsize: tuple[float, float] = (12.0, 8.0),
    show_court: bool = True,
    player_color: str = "red",
    arrow_length: float = 1.5,
) -> Figure:
    if hasattr(position, "detach") and hasattr(position, "cpu"):
        pos_np = position.detach().cpu().numpy()
    elif hasattr(position, "cpu") and hasattr(position, "numpy"):
        pos_np = position.cpu().numpy()
    elif hasattr(position, "numpy"):
        pos_np = position.numpy()
    else:
        pos_np = np.asarray(position, dtype=float)

    x, y, z = float(pos_np[0]), float(pos_np[1]), float(pos_np[2])

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")

    if show_court:
        CourtRenderer().render_3d(ax, show_net=True)

    ax.scatter([x], [y], [z], c=player_color, s=80, depthshade=True, zorder=5)

    dx = arrow_length * math.sin(yaw)
    dy = arrow_length * math.cos(yaw)
    ax.quiver(x, y, z, dx, dy, 0.0, color="yellow", arrow_length_ratio=0.3)

    ax.set_title("PLCS Prediction (3D)")

    return fig


def visualize_batch_3d(
    positions: Tensor | np.ndarray,
    yaws: Tensor | np.ndarray,
    figsize: tuple[float, float] = (12.0, 8.0),
    max_players: int = 10,
) -> Figure:
    from torch import (
        Tensor as TorchTensor,  # local import to avoid hard dep in type hints
    )

    if isinstance(positions, TorchTensor):
        positions_np = positions.cpu().numpy()
    else:
        positions_np = np.asarray(positions)

    if isinstance(yaws, TorchTensor):
        yaws_np = yaws.cpu().numpy()
    else:
        yaws_np = np.asarray(yaws)

    num_players = min(len(positions_np), max_players)

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")
    CourtRenderer().render_3d(ax, show_net=True)

    colors = plt.cm.tab10(np.linspace(0, 1, num_players))

    for i in range(num_players):
        x, y, z = map(float, positions_np[i])
        yaw_val = float(yaws_np[i])

        ax.scatter([x], [y], [z], c=[colors[i]], s=70, depthshade=True, zorder=5)

        dx = 1.2 * math.sin(yaw_val)
        dy = 1.2 * math.cos(yaw_val)
        ax.quiver(x, y, z, dx, dy, 0.0, color=colors[i], arrow_length_ratio=0.3)

    ax.set_title(f"PLCS Predictions (3D, n={num_players})")

    return fig


def animate_predictions(
    positions: Tensor | np.ndarray,
    yaws: Tensor | np.ndarray | None = None,
    view: str = "2d_topdown",
    fps: float = 10.0,
) -> FuncAnimation:
    from torch import (
        Tensor as TorchTensor,  # local import to avoid hard dep in type hints
    )

    if isinstance(positions, TorchTensor):
        positions_np = positions.cpu().numpy()
    else:
        positions_np = np.asarray(positions)

    if yaws is not None:
        if isinstance(yaws, TorchTensor):
            yaws_np = yaws.cpu().numpy()
        else:
            yaws_np = np.asarray(yaws)
    else:
        yaws_np = None

    num_frames = positions_np.shape[0]
    interval = 1000.0 / fps

    if view == "3d":
        fig = plt.figure(figsize=(12.0, 8.0))
        ax = fig.add_subplot(111, projection="3d")

        def update(frame_idx: int):
            ax.clear()
            CourtRenderer().render_3d(ax, show_net=True)

            x, y, z = map(float, positions_np[frame_idx])
            ax.scatter([x], [y], [z], c="red", s=80, depthshade=True, zorder=5)

            if yaws_np is not None:
                yaw_val = float(yaws_np[frame_idx])
                dx = 1.2 * math.sin(yaw_val)
                dy = 1.2 * math.cos(yaw_val)
                ax.quiver(x, y, z, dx, dy, 0.0, color="yellow", arrow_length_ratio=0.3)

            ax.set_title(f"Frame {frame_idx + 1}/{num_frames}")
            return []

    elif view == "2d_topdown":
        fig, ax = plt.subplots(figsize=(10.0, 12.0))

        def update(frame_idx: int):
            ax.clear()
            _court_renderer.render_2d(ax, show_fence=False)

            xs = positions_np[: frame_idx + 1, 0]
            ys = positions_np[: frame_idx + 1, 1]
            ax.plot(xs, ys, "-o", color="yellow", markersize=4, linewidth=2, zorder=10)

            x = float(positions_np[frame_idx, 0])
            y = float(positions_np[frame_idx, 1])
            ax.plot(x, y, "o", color="red", markersize=10, zorder=11)

            if yaws_np is not None:
                yaw_val = float(yaws_np[frame_idx])
                dx = 1.2 * math.sin(yaw_val)
                dy = 1.2 * math.cos(yaw_val)
                ax.arrow(
                    x,
                    y,
                    dx,
                    dy,
                    head_width=0.25,
                    head_length=0.2,
                    fc="orange",
                    ec="orange",
                    zorder=12,
                )

            ax.set_title(f"Frame {frame_idx + 1}/{num_frames}")
            return []

    else:
        raise ValueError(f"Unknown view type: {view}")

    anim = FuncAnimation(
        fig,
        update,
        frames=num_frames,
        interval=interval,
        blit=False,
    )

    return anim
