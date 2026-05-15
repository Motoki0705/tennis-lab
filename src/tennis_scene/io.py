"""Data structures for tennis scene reconstruction results."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any
import warnings

import numpy as np
from numpy.typing import NDArray


@dataclass
class SceneResult:
    """Result of tennis scene 3D reconstruction.

    Player-related arrays use (P, T, ...) as the canonical shape.
    """

    num_frames: int
    fps: float
    width: int
    height: int

    court_kp: NDArray[np.float32]
    court_vis: NDArray[np.float32]

    player_position: NDArray[np.float32]  # (P, T, 3)
    player_yaw: NDArray[np.float32]  # (P, T)

    smpl_body_pose: NDArray[np.float32]  # (P, T, 63)
    smpl_global_orient: NDArray[np.float32]  # (P, T, 3)
    smpl_betas: NDArray[np.float32]  # (P, 10)
    smpl_vertices_local: NDArray[np.float32] | None = None  # (P, T, V, 3)

    ball_uv: NDArray[np.float32] | None = None
    ball_visibility: NDArray[np.bool_] | None = None
    ball_3d: NDArray[np.float32] | None = None

    human_kp_2d: NDArray[np.float32] | None = None  # (P, T, 17, 2)
    human_kp_vis: NDArray[np.float32] | None = None  # (P, T, 17)

    player_track_ids: NDArray[np.int32] | None = None
    player_kp_3d: NDArray[np.float32] | None = None  # (P, T, J, 3)

    metadata: dict[str, Any] = field(default_factory=dict)

    @staticmethod
    def _metadata_sidecar_path(path: Path) -> Path:
        """Return sidecar metadata JSON path for a scene archive."""
        return path.with_suffix(".metadata.json")

    def save(self, path: str | Path) -> None:
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
        if self.player_kp_3d is not None:
            data["player_kp_3d"] = self.player_kp_3d

        np.savez_compressed(path, **data)
        if self.metadata:
            sidecar_path = self._metadata_sidecar_path(path)
            with sidecar_path.open("w", encoding="utf-8") as f:
                json.dump(self.metadata, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> "SceneResult":
        path = Path(path)
        try:
            data = np.load(path, allow_pickle=False)
        except ValueError:
            warnings.warn(
                (
                    f"{path} contains pickle-based arrays (legacy format). "
                    "Falling back to allow_pickle=True."
                ),
                RuntimeWarning,
            )
            data = np.load(path, allow_pickle=True)

        metadata = {}
        sidecar_path = cls._metadata_sidecar_path(path)
        if sidecar_path.exists():
            try:
                with sidecar_path.open("r", encoding="utf-8") as f:
                    metadata = json.load(f)
            except Exception as exc:
                warnings.warn(
                    (
                        f"Failed to load sidecar metadata from {sidecar_path}: {exc}. "
                        "Proceeding without metadata."
                    ),
                    RuntimeWarning,
                )
        elif "metadata" in data.files:
            # Legacy npz fallback
            try:
                metadata = data["metadata"].item()
            except Exception as exc:
                warnings.warn(
                    f"Failed to load metadata from {path}: {exc}. Proceeding without metadata.",
                    RuntimeWarning,
                )

        # Primary fields (new canonical names).
        player_position = np.asarray(data["player_position"], dtype=np.float32)
        player_yaw = np.asarray(data["player_yaw"], dtype=np.float32)
        smpl_body_pose = np.asarray(data["smpl_body_pose"], dtype=np.float32)
        smpl_global_orient = np.asarray(data["smpl_global_orient"], dtype=np.float32)
        smpl_betas = np.asarray(data["smpl_betas"], dtype=np.float32)

        smpl_vertices_local = data.get("smpl_vertices_local")
        if smpl_vertices_local is not None:
            smpl_vertices_local = np.asarray(smpl_vertices_local, dtype=np.float32)

        human_kp_2d = data.get("human_kp_2d")
        if human_kp_2d is not None:
            human_kp_2d = np.asarray(human_kp_2d, dtype=np.float32)

        human_kp_vis = data.get("human_kp_vis")
        if human_kp_vis is not None:
            human_kp_vis = np.asarray(human_kp_vis, dtype=np.float32)

        player_track_ids = data.get("player_track_ids")
        if player_track_ids is not None:
            player_track_ids = np.asarray(player_track_ids, dtype=np.int32)

        player_kp_3d = data.get("player_kp_3d")
        if player_kp_3d is not None:
            player_kp_3d = np.asarray(player_kp_3d, dtype=np.float32)

        return cls(
            num_frames=int(data["num_frames"]),
            fps=float(data["fps"]),
            width=int(data["width"]),
            height=int(data["height"]),
            court_kp=np.asarray(data["court_kp"], dtype=np.float32),
            court_vis=np.asarray(data["court_vis"], dtype=np.float32),
            player_position=player_position,
            player_yaw=player_yaw,
            smpl_body_pose=smpl_body_pose,
            smpl_global_orient=smpl_global_orient,
            smpl_betas=smpl_betas,
            smpl_vertices_local=smpl_vertices_local,
            ball_uv=data.get("ball_uv"),
            ball_visibility=data.get("ball_visibility"),
            ball_3d=data.get("ball_3d"),
            human_kp_2d=human_kp_2d,
            human_kp_vis=human_kp_vis,
            player_track_ids=player_track_ids,
            player_kp_3d=player_kp_3d,
            metadata=metadata,
        )


if __name__ == "__main__":
    import tempfile

    T = 10
    P = 2
    result = SceneResult(
        num_frames=T,
        fps=30.0,
        width=1920,
        height=1080,
        court_kp=np.random.rand(20, 2).astype(np.float32),
        court_vis=np.ones(20, dtype=np.float32),
        player_position=np.random.rand(P, T, 3).astype(np.float32),
        player_yaw=np.random.rand(P, T).astype(np.float32),
        smpl_body_pose=np.random.rand(P, T, 63).astype(np.float32),
        smpl_global_orient=np.random.rand(P, T, 3).astype(np.float32),
        smpl_betas=np.random.rand(P, 10).astype(np.float32),
        ball_uv=np.random.rand(T, 2).astype(np.float32),
        ball_visibility=np.ones(T, dtype=bool),
        ball_3d=np.random.rand(T, 3).astype(np.float32),
        player_track_ids=np.array([0, 1], dtype=np.int32),
    )

    with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
        result.save(f.name)
        loaded = SceneResult.load(f.name)

    assert loaded.player_position.shape == (P, T, 3)
    assert loaded.smpl_body_pose.shape == (P, T, 63)
    print("SceneResult smoke test passed!")
