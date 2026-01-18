"""Data structures for tennis scene reconstruction results.

Example:
    >>> result = SceneResult(...)
    >>> result.save("output.npz")
    >>> loaded = SceneResult.load("output.npz")
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray


@dataclass
class SceneResult:
    """Result of tennis scene 3D reconstruction.

    Attributes:
        num_frames: Number of frames in the video.
        fps: Video frame rate.
        width: Video width in pixels.
        height: Video height in pixels.
        court_kp: Court keypoints (20, 2), normalized [0, 1].
        court_vis: Court keypoint visibility (20,).
        player_position: Player 3D position in court coords (T, 3), meters.
        player_yaw: Player yaw angle (T,), radians.
        smpl_body_pose: SMPL body pose parameters (T, 63).
        smpl_global_orient: SMPL global orientation (T, 3).
        smpl_betas: SMPL shape parameters (10,).
        smpl_vertices_local: Local SMPL vertices (T, V, 3).
        smpl_vertices_global: Global SMPL vertices (T, V, 3), after PLCS transform.
        ball_uv: Ball 2D position (T, 2), normalized [0, 1].
        ball_visibility: Ball visibility mask (T,).
        ball_3d: Ball 3D position in court coords (T, 3), meters.
        human_kp_2d: Human 2D keypoints (T, 17, 2), normalized [0, 1].
        human_kp_vis: Human keypoint visibility (T, 17).
        metadata: Additional metadata dict.

    """

    num_frames: int
    fps: float
    width: int
    height: int

    court_kp: NDArray[np.float32]
    court_vis: NDArray[np.float32]

    player_position: NDArray[np.float32]
    player_yaw: NDArray[np.float32]

    smpl_body_pose: NDArray[np.float32]
    smpl_global_orient: NDArray[np.float32]
    smpl_betas: NDArray[np.float32]
    smpl_vertices_local: NDArray[np.float32] | None = None
    smpl_vertices_global: NDArray[np.float32] | None = None

    ball_uv: NDArray[np.float32] | None = None
    ball_visibility: NDArray[np.bool_] | None = None
    ball_3d: NDArray[np.float32] | None = None

    human_kp_2d: NDArray[np.float32] | None = None
    human_kp_vis: NDArray[np.float32] | None = None

    metadata: dict[str, Any] = field(default_factory=dict)

    def save(self, path: str | Path) -> None:
        """Save scene result to .npz file.

        Args:
            path: Output file path.

        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        data: dict[str, Any] = {
            "num_frames": self.num_frames,
            "fps": self.fps,
            "width": self.width,
            "height": self.height,
            "court_kp": self.court_kp,
            "court_vis": self.court_vis,
            "player_position": self.player_position,
            "player_yaw": self.player_yaw,
            "smpl_body_pose": self.smpl_body_pose,
            "smpl_global_orient": self.smpl_global_orient,
            "smpl_betas": self.smpl_betas,
        }

        if self.smpl_vertices_local is not None:
            data["smpl_vertices_local"] = self.smpl_vertices_local
        if self.smpl_vertices_global is not None:
            data["smpl_vertices_global"] = self.smpl_vertices_global
        if self.ball_uv is not None:
            data["ball_uv"] = self.ball_uv
        if self.ball_visibility is not None:
            data["ball_visibility"] = self.ball_visibility
        if self.ball_3d is not None:
            data["ball_3d"] = self.ball_3d
        if self.human_kp_2d is not None:
            data["human_kp_2d"] = self.human_kp_2d
        if self.human_kp_vis is not None:
            data["human_kp_vis"] = self.human_kp_vis

        if self.metadata:
            data["metadata"] = np.array([self.metadata], dtype=object)

        np.savez_compressed(path, **data)

    @classmethod
    def load(cls, path: str | Path) -> SceneResult:
        """Load scene result from .npz file.

        Args:
            path: Input file path.

        Returns:
            Loaded SceneResult instance.

        """
        data = np.load(path, allow_pickle=True)

        metadata = {}
        if "metadata" in data:
            metadata = data["metadata"].item()

        return cls(
            num_frames=int(data["num_frames"]),
            fps=float(data["fps"]),
            width=int(data["width"]),
            height=int(data["height"]),
            court_kp=data["court_kp"],
            court_vis=data["court_vis"],
            player_position=data["player_position"],
            player_yaw=data["player_yaw"],
            smpl_body_pose=data["smpl_body_pose"],
            smpl_global_orient=data["smpl_global_orient"],
            smpl_betas=data["smpl_betas"],
            smpl_vertices_local=data.get("smpl_vertices_local"),
            smpl_vertices_global=data.get("smpl_vertices_global"),
            ball_uv=data.get("ball_uv"),
            ball_visibility=data.get("ball_visibility"),
            ball_3d=data.get("ball_3d"),
            human_kp_2d=data.get("human_kp_2d"),
            human_kp_vis=data.get("human_kp_vis"),
            metadata=metadata,
        )


if __name__ == "__main__":
    # Smoke test
    import tempfile

    result = SceneResult(
        num_frames=10,
        fps=30.0,
        width=1920,
        height=1080,
        court_kp=np.random.rand(20, 2).astype(np.float32),
        court_vis=np.ones(20, dtype=np.float32),
        player_position=np.random.rand(10, 3).astype(np.float32),
        player_yaw=np.random.rand(10).astype(np.float32),
        smpl_body_pose=np.random.rand(10, 63).astype(np.float32),
        smpl_global_orient=np.random.rand(10, 3).astype(np.float32),
        smpl_betas=np.random.rand(10).astype(np.float32),
        ball_uv=np.random.rand(10, 2).astype(np.float32),
        ball_visibility=np.ones(10, dtype=bool),
        ball_3d=np.random.rand(10, 3).astype(np.float32),
    )

    with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
        result.save(f.name)
        loaded = SceneResult.load(f.name)

    assert loaded.num_frames == result.num_frames
    assert loaded.fps == result.fps
    assert np.allclose(loaded.court_kp, result.court_kp)
    assert np.allclose(loaded.player_position, result.player_position)
    print("SceneResult smoke test passed!")
