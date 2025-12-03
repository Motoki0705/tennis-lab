"""Human skeleton renderer for 2D and 3D visualization.

This module provides rendering of human pose skeletons with support
for multiple skeleton formats (COCO-17, SMPL-H, etc.).

Example:
    >>> import numpy as np
    >>> from src.utils.rendering import SkeletonRenderer
    >>>
    >>> renderer = SkeletonRenderer(skeleton_type="coco17")
    >>> keypoints = np.random.randn(17, 2)
    >>> fig, ax = plt.subplots()
    >>> renderer.render_2d(ax, keypoints)

"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from mpl_toolkits.mplot3d import Axes3D


class SkeletonType(Enum):
    """Supported skeleton types."""

    COCO17 = "coco17"
    SMPLH = "smplh"
    SMPL = "smpl"


# COCO 17-keypoint skeleton connections
COCO17_SKELETON: list[tuple[int, int]] = [
    # Head
    (0, 1),  # nose -> left_eye
    (0, 2),  # nose -> right_eye
    (1, 3),  # left_eye -> left_ear
    (2, 4),  # right_eye -> right_ear
    # Torso
    (5, 6),  # left_shoulder -> right_shoulder
    (5, 11),  # left_shoulder -> left_hip
    (6, 12),  # right_shoulder -> right_hip
    (11, 12),  # left_hip -> right_hip
    # Left arm
    (5, 7),  # left_shoulder -> left_elbow
    (7, 9),  # left_elbow -> left_wrist
    # Right arm
    (6, 8),  # right_shoulder -> right_elbow
    (8, 10),  # right_elbow -> right_wrist
    # Left leg
    (11, 13),  # left_hip -> left_knee
    (13, 15),  # left_knee -> left_ankle
    # Right leg
    (12, 14),  # right_hip -> right_knee
    (14, 16),  # right_knee -> right_ankle
]

COCO17_KEYPOINT_NAMES: list[str] = [
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
]

# SMPL-H 22-joint skeleton connections (body joints only, excluding hands)
SMPLH_SKELETON: list[tuple[int, int]] = [
    # Spine
    (0, 3),  # pelvis -> spine1
    (3, 6),  # spine1 -> spine2
    (6, 9),  # spine2 -> spine3
    (9, 12),  # spine3 -> neck
    (12, 15),  # neck -> head
    # Left side
    (0, 1),  # pelvis -> left_hip
    (1, 4),  # left_hip -> left_knee
    (4, 7),  # left_knee -> left_ankle
    (7, 10),  # left_ankle -> left_foot
    (9, 13),  # spine3 -> left_collar
    (13, 16),  # left_collar -> left_shoulder
    (16, 18),  # left_shoulder -> left_elbow
    (18, 20),  # left_elbow -> left_wrist
    # Right side
    (0, 2),  # pelvis -> right_hip
    (2, 5),  # right_hip -> right_knee
    (5, 8),  # right_knee -> right_ankle
    (8, 11),  # right_ankle -> right_foot
    (9, 14),  # spine3 -> right_collar
    (14, 17),  # right_collar -> right_shoulder
    (17, 19),  # right_shoulder -> right_elbow
    (19, 21),  # right_elbow -> right_wrist
]

# SMPL 24-joint skeleton (standard SMPL body model)
SMPL_SKELETON: list[tuple[int, int]] = [
    # Spine
    (0, 3),  # pelvis -> spine1
    (3, 6),  # spine1 -> spine2
    (6, 9),  # spine2 -> spine3
    (9, 12),  # spine3 -> neck
    (12, 15),  # neck -> head
    # Left side
    (0, 1),  # pelvis -> left_hip
    (1, 4),  # left_hip -> left_knee
    (4, 7),  # left_knee -> left_ankle
    (7, 10),  # left_ankle -> left_foot
    (9, 13),  # spine3 -> left_collar
    (13, 16),  # left_collar -> left_shoulder
    (16, 18),  # left_shoulder -> left_elbow
    (18, 20),  # left_elbow -> left_wrist
    (20, 22),  # left_wrist -> left_hand
    # Right side
    (0, 2),  # pelvis -> right_hip
    (2, 5),  # right_hip -> right_knee
    (5, 8),  # right_knee -> right_ankle
    (8, 11),  # right_ankle -> right_foot
    (9, 14),  # spine3 -> right_collar
    (14, 17),  # right_collar -> right_shoulder
    (17, 19),  # right_shoulder -> right_elbow
    (19, 21),  # right_elbow -> right_wrist
    (21, 23),  # right_wrist -> right_hand
]

SKELETON_DEFINITIONS: dict[str, list[tuple[int, int]]] = {
    "coco17": COCO17_SKELETON,
    "smplh": SMPLH_SKELETON,
    "smpl": SMPL_SKELETON,
}


@dataclass
class SkeletonStyle:
    """Style configuration for skeleton rendering.

    Attributes:
        joint_color: Color for joint markers.
        bone_color: Color for bone lines.
        joint_size: Size of joint markers in points.
        bone_width: Width of bone lines in points.
        joint_alpha: Alpha transparency for joints.
        bone_alpha: Alpha transparency for bones.

    """

    joint_color: str = "#FF4444"
    bone_color: str = "#4444FF"
    joint_size: float = 20.0
    bone_width: float = 2.0
    joint_alpha: float = 1.0
    bone_alpha: float = 0.8


class SkeletonRenderer:
    """Render human skeleton in 2D or 3D.

    Supports multiple skeleton formats and customizable styling.
    Handles visibility masks for partially occluded poses.

    Example:
        >>> renderer = SkeletonRenderer(skeleton_type="coco17")
        >>> keypoints = np.random.randn(17, 2)
        >>> visibility = np.ones(17, dtype=bool)
        >>> fig, ax = plt.subplots()
        >>> renderer.render_2d(ax, keypoints, visibility)

    """

    def __init__(
        self,
        skeleton_type: str | SkeletonType = "coco17",
        style: SkeletonStyle | None = None,
    ) -> None:
        """Initialize skeleton renderer.

        Args:
            skeleton_type: Type of skeleton format.
            style: Style configuration. If None, uses defaults.

        Raises:
            ValueError: If skeleton_type is not supported.

        """
        if isinstance(skeleton_type, SkeletonType):
            skeleton_type = skeleton_type.value

        if skeleton_type not in SKELETON_DEFINITIONS:
            supported = list(SKELETON_DEFINITIONS.keys())
            raise ValueError(
                f"Unknown skeleton type: {skeleton_type}. Supported: {supported}"
            )

        self.skeleton_type = skeleton_type
        self.skeleton = SKELETON_DEFINITIONS[skeleton_type]
        self.style = style or SkeletonStyle()

    def render_2d(
        self,
        ax: Axes,
        keypoints: np.ndarray,
        visibility: np.ndarray | None = None,
        *,
        label: str | None = None,
        style_override: SkeletonStyle | None = None,
    ) -> None:
        """Render skeleton in 2D.

        Args:
            ax: Matplotlib axes to draw on.
            keypoints: Joint positions, shape (N, 2).
            visibility: Visibility mask, shape (N,). If None, all joints visible.
            label: Optional label for legend.
            style_override: Override default style for this render call.

        """
        style = style_override or self.style
        num_joints = keypoints.shape[0]

        if visibility is None:
            visibility = np.ones(num_joints, dtype=bool)
        visibility = np.asarray(visibility, dtype=bool)

        # Draw bones first (behind joints)
        for i, j in self.skeleton:
            if i < num_joints and j < num_joints:
                if visibility[i] and visibility[j]:
                    ax.plot(
                        [keypoints[i, 0], keypoints[j, 0]],
                        [keypoints[i, 1], keypoints[j, 1]],
                        color=style.bone_color,
                        linewidth=style.bone_width,
                        alpha=style.bone_alpha,
                        zorder=2,
                        solid_capstyle="round",
                    )

        # Draw joints
        visible_kp = keypoints[visibility]
        if len(visible_kp) > 0:
            ax.scatter(
                visible_kp[:, 0],
                visible_kp[:, 1],
                c=style.joint_color,
                s=style.joint_size,
                alpha=style.joint_alpha,
                zorder=3,
                label=label,
                edgecolors="white",
                linewidths=0.5,
            )

    def render_3d(
        self,
        ax: Axes3D,
        keypoints: np.ndarray,
        visibility: np.ndarray | None = None,
        *,
        label: str | None = None,
        style_override: SkeletonStyle | None = None,
    ) -> None:
        """Render skeleton in 3D.

        Args:
            ax: Matplotlib 3D axes to draw on.
            keypoints: Joint positions, shape (N, 3).
            visibility: Visibility mask, shape (N,). If None, all joints visible.
            label: Optional label for legend.
            style_override: Override default style for this render call.

        """
        style = style_override or self.style
        num_joints = keypoints.shape[0]

        if visibility is None:
            visibility = np.ones(num_joints, dtype=bool)
        visibility = np.asarray(visibility, dtype=bool)

        # Draw bones
        for i, j in self.skeleton:
            if i < num_joints and j < num_joints:
                if visibility[i] and visibility[j]:
                    ax.plot(
                        [keypoints[i, 0], keypoints[j, 0]],
                        [keypoints[i, 1], keypoints[j, 1]],
                        [keypoints[i, 2], keypoints[j, 2]],
                        color=style.bone_color,
                        linewidth=style.bone_width,
                        alpha=style.bone_alpha,
                        zorder=2,
                    )

        # Draw joints
        visible_kp = keypoints[visibility]
        if len(visible_kp) > 0:
            ax.scatter(
                visible_kp[:, 0],
                visible_kp[:, 1],
                visible_kp[:, 2],
                c=style.joint_color,
                s=style.joint_size,
                alpha=style.joint_alpha,
                zorder=3,
                label=label,
            )

    def render_sequence_2d(
        self,
        ax: Axes,
        keypoints_seq: np.ndarray,
        visibility_seq: np.ndarray | None = None,
        *,
        num_frames: int = 5,
        alpha_decay: float = 0.7,
    ) -> None:
        """Render skeleton sequence with fading trail effect.

        Args:
            ax: Matplotlib axes to draw on.
            keypoints_seq: Joint position sequence, shape (T, N, 2).
            visibility_seq: Visibility sequence, shape (T, N).
            num_frames: Number of frames to display.
            alpha_decay: Alpha multiplier per frame (0-1).

        """
        T = keypoints_seq.shape[0]
        indices = np.linspace(0, T - 1, num_frames, dtype=int)

        for i, idx in enumerate(indices):
            alpha_factor = alpha_decay ** (num_frames - i - 1)
            kp = keypoints_seq[idx]
            vis = visibility_seq[idx] if visibility_seq is not None else None

            # Create style with adjusted alpha
            style = SkeletonStyle(
                joint_color=self.style.joint_color,
                bone_color=self.style.bone_color,
                joint_size=self.style.joint_size * alpha_factor,
                bone_width=self.style.bone_width * alpha_factor,
                joint_alpha=self.style.joint_alpha * alpha_factor,
                bone_alpha=self.style.bone_alpha * alpha_factor,
            )

            self.render_2d(ax, kp, vis, style_override=style)

    @staticmethod
    def get_skeleton_connections(skeleton_type: str) -> list[tuple[int, int]]:
        """Get skeleton connection definitions for a given type.

        Args:
            skeleton_type: Type of skeleton.

        Returns:
            List of (joint_i, joint_j) tuples defining bones.

        Raises:
            ValueError: If skeleton_type is not supported.

        """
        if skeleton_type not in SKELETON_DEFINITIONS:
            supported = list(SKELETON_DEFINITIONS.keys())
            raise ValueError(
                f"Unknown skeleton type: {skeleton_type}. Supported: {supported}"
            )
        return SKELETON_DEFINITIONS[skeleton_type]
