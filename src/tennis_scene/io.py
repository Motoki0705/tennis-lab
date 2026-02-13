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
import warnings

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
        player_track_ids: Player track IDs (P,).
        players_position: Multi-player position (P, T, 3).
        players_yaw: Multi-player yaw (P, T).
        players_smpl_body_pose: Multi-player SMPL body pose (P, T, 63).
        players_smpl_global_orient: Multi-player SMPL global orient (P, T, 3).
        players_smpl_betas: Multi-player SMPL betas (P, 10).
        players_smpl_vertices_local: Multi-player local SMPL vertices (P, T, V, 3).
        players_smpl_vertices_global: Multi-player global SMPL vertices (P, T, V, 3).
        players_human_kp_2d: Multi-player 2D keypoints (P, T, 17, 2).
        players_human_kp_vis: Multi-player keypoint visibility (P, T, 17).
        players_kp_3d: Optional multi-player 3D skeleton keypoints (P, T, J, 3).
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

    player_track_ids: NDArray[np.int32] | None = None
    players_position: NDArray[np.float32] | None = None
    players_yaw: NDArray[np.float32] | None = None
    players_smpl_body_pose: NDArray[np.float32] | None = None
    players_smpl_global_orient: NDArray[np.float32] | None = None
    players_smpl_betas: NDArray[np.float32] | None = None
    players_smpl_vertices_local: NDArray[np.float32] | None = None
    players_smpl_vertices_global: NDArray[np.float32] | None = None
    players_human_kp_2d: NDArray[np.float32] | None = None
    players_human_kp_vis: NDArray[np.float32] | None = None
    players_kp_3d: NDArray[np.float32] | None = None

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
        if self.player_track_ids is not None:
            data["player_track_ids"] = self.player_track_ids
        if self.players_position is not None:
            data["players_position"] = self.players_position
        if self.players_yaw is not None:
            data["players_yaw"] = self.players_yaw
        if self.players_smpl_body_pose is not None:
            data["players_smpl_body_pose"] = self.players_smpl_body_pose
        if self.players_smpl_global_orient is not None:
            data["players_smpl_global_orient"] = self.players_smpl_global_orient
        if self.players_smpl_betas is not None:
            data["players_smpl_betas"] = self.players_smpl_betas
        if self.players_smpl_vertices_local is not None:
            data["players_smpl_vertices_local"] = self.players_smpl_vertices_local
        if self.players_smpl_vertices_global is not None:
            data["players_smpl_vertices_global"] = self.players_smpl_vertices_global
        if self.players_human_kp_2d is not None:
            data["players_human_kp_2d"] = self.players_human_kp_2d
        if self.players_human_kp_vis is not None:
            data["players_human_kp_vis"] = self.players_human_kp_vis
        if self.players_kp_3d is not None:
            data["players_kp_3d"] = self.players_kp_3d

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
        if "metadata" in data.files:
            try:
                metadata = data["metadata"].item()
            except Exception as exc:
                warnings.warn(
                    f"Failed to load metadata from {path}: {exc}. "
                    "Proceeding without metadata.",
                    RuntimeWarning,
                )

        players_position = data.get("players_position")
        players_yaw = data.get("players_yaw")
        players_smpl_body_pose = data.get("players_smpl_body_pose")
        players_smpl_global_orient = data.get("players_smpl_global_orient")
        players_smpl_betas = data.get("players_smpl_betas")
        players_smpl_vertices_local = data.get("players_smpl_vertices_local")
        players_smpl_vertices_global = data.get("players_smpl_vertices_global")
        players_human_kp_2d = data.get("players_human_kp_2d")
        players_human_kp_vis = data.get("players_human_kp_vis")
        players_kp_3d = data.get("players_kp_3d")
        player_track_ids = data.get("player_track_ids")

        player_position = data.get("player_position")
        player_yaw = data.get("player_yaw")
        smpl_body_pose = data.get("smpl_body_pose")
        smpl_global_orient = data.get("smpl_global_orient")
        smpl_betas = data.get("smpl_betas")
        smpl_vertices_local = data.get("smpl_vertices_local")
        smpl_vertices_global = data.get("smpl_vertices_global")
        human_kp_2d = data.get("human_kp_2d")
        human_kp_vis = data.get("human_kp_vis")

        # Backward compatibility: old single-player files -> populate multi-player fields.
        if players_position is None and player_position is not None:
            players_position = player_position[None, ...]
            players_yaw = player_yaw[None, ...] if player_yaw is not None else None
            players_smpl_body_pose = (
                smpl_body_pose[None, ...] if smpl_body_pose is not None else None
            )
            players_smpl_global_orient = (
                smpl_global_orient[None, ...] if smpl_global_orient is not None else None
            )
            players_smpl_betas = smpl_betas[None, ...] if smpl_betas is not None else None
            players_smpl_vertices_local = (
                smpl_vertices_local[None, ...] if smpl_vertices_local is not None else None
            )
            players_smpl_vertices_global = (
                smpl_vertices_global[None, ...] if smpl_vertices_global is not None else None
            )
            players_human_kp_2d = (
                human_kp_2d[None, ...] if human_kp_2d is not None else None
            )
            players_human_kp_vis = (
                human_kp_vis[None, ...] if human_kp_vis is not None else None
            )
            player_track_ids = np.array([0], dtype=np.int32)

        # Forward compatibility: new multi-player files without single fields.
        if player_position is None and players_position is not None:
            player_position = players_position[0]
        if player_yaw is None and players_yaw is not None:
            player_yaw = players_yaw[0]
        if smpl_body_pose is None and players_smpl_body_pose is not None:
            smpl_body_pose = players_smpl_body_pose[0]
        if smpl_global_orient is None and players_smpl_global_orient is not None:
            smpl_global_orient = players_smpl_global_orient[0]
        if smpl_betas is None and players_smpl_betas is not None:
            smpl_betas = players_smpl_betas[0]
        if smpl_vertices_local is None and players_smpl_vertices_local is not None:
            smpl_vertices_local = players_smpl_vertices_local[0]
        if smpl_vertices_global is None and players_smpl_vertices_global is not None:
            smpl_vertices_global = players_smpl_vertices_global[0]
        if human_kp_2d is None and players_human_kp_2d is not None:
            human_kp_2d = players_human_kp_2d[0]
        if human_kp_vis is None and players_human_kp_vis is not None:
            human_kp_vis = players_human_kp_vis[0]

        return cls(
            num_frames=int(data["num_frames"]),
            fps=float(data["fps"]),
            width=int(data["width"]),
            height=int(data["height"]),
            court_kp=data["court_kp"],
            court_vis=data["court_vis"],
            player_position=player_position,
            player_yaw=player_yaw,
            smpl_body_pose=smpl_body_pose,
            smpl_global_orient=smpl_global_orient,
            smpl_betas=smpl_betas,
            smpl_vertices_local=smpl_vertices_local,
            smpl_vertices_global=smpl_vertices_global,
            ball_uv=data.get("ball_uv"),
            ball_visibility=data.get("ball_visibility"),
            ball_3d=data.get("ball_3d"),
            human_kp_2d=human_kp_2d,
            human_kp_vis=human_kp_vis,
            player_track_ids=player_track_ids,
            players_position=players_position,
            players_yaw=players_yaw,
            players_smpl_body_pose=players_smpl_body_pose,
            players_smpl_global_orient=players_smpl_global_orient,
            players_smpl_betas=players_smpl_betas,
            players_smpl_vertices_local=players_smpl_vertices_local,
            players_smpl_vertices_global=players_smpl_vertices_global,
            players_human_kp_2d=players_human_kp_2d,
            players_human_kp_vis=players_human_kp_vis,
            players_kp_3d=players_kp_3d,
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
