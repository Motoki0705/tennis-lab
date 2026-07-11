"""Synthetic issue #634 dataset builder.

Generates a small, fully contract-conformant dataset (media videos,
tennis_scene pseudo-annotations, DINOv3 token annotations, dataset index)
with random but physically plausible content. Used by unit/integration/smoke
tests and by dry runs before real data exists; it doubles as executable
documentation of the writer-side contract.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from src.tasks.slcs.data.contract import DatasetIndex
from src.tasks.slcs.data.contract_writer import (
    append_dataset_index,
    write_clip_manifest,
    write_tennis_scene_annotation,
)
from src.tasks.slcs.data.dino_tokens import (
    DinoTokenSpec,
    sample_frame_indices,
    write_dino_tokens,
)
from src.tennis_scene.io import SceneResult

DEFAULT_TEST_DINO_SPEC = DinoTokenSpec(
    backbone="dinov3_vitb16",
    patch_size=16,
    image_height=48,
    image_width=64,
    embed_dim=8,
    frame_stride=10,
)


@dataclass(frozen=True)
class SyntheticDatasetConfig:
    """Shape parameters of the generated dataset."""

    recordings: tuple[str, ...] = ("rec-a", "rec-b", "rec-c")
    clips_per_recording: int = 1
    num_frames: int = 37
    num_players: int = 2
    num_cameras: int = 1
    num_court_kp: int = 14
    width: int = 64
    height: int = 48
    fps: float = 30.0
    seed: int = 7
    ball_visibility: float = 0.85
    dino_spec: DinoTokenSpec = field(default_factory=lambda: DEFAULT_TEST_DINO_SPEC)


def make_synthetic_scene(
    config: SyntheticDatasetConfig, rng: np.random.Generator
) -> SceneResult:
    """Random SceneResult obeying the SLCS contract (near/far player split)."""
    num_frames = config.num_frames
    num_players = config.num_players
    num_cameras = config.num_cameras
    yaw = rng.uniform(-np.pi, np.pi, size=(num_players, num_frames)).astype(np.float32)
    position = rng.uniform(-0.5, 0.5, size=(num_players, num_frames, 3)).astype(np.float32)
    for player in range(num_players):
        # Alternate near (-y) / far (+y) so canonical ordering is unambiguous.
        offset = -5.0 if player % 2 == 0 else 5.0
        position[player, :, 1] += offset
    position[..., 2] = 0.0
    ball = rng.uniform(-1.0, 1.0, size=(num_frames, 3)).astype(np.float32)
    ball[:, 2] = np.abs(ball[:, 2]) + 0.2
    return SceneResult(
        num_frames=num_frames,
        fps=config.fps,
        width=config.width,
        height=config.height,
        court_kp=rng.uniform(
            0, 1, size=(num_cameras, num_frames, config.num_court_kp, 2)
        ).astype(np.float32),
        court_vis=np.ones(
            (num_cameras, num_frames, config.num_court_kp), dtype=np.float32
        ),
        player_position=position,
        player_yaw=yaw,
        smpl_body_pose=np.zeros((num_players, num_frames, 63), dtype=np.float32),
        smpl_global_orient=np.zeros((num_players, num_frames, 3), dtype=np.float32),
        smpl_betas=np.zeros((num_players, 10), dtype=np.float32),
        ball_uv=rng.uniform(0, 1, size=(num_cameras, num_frames, 2)).astype(np.float32),
        ball_vis=rng.random((num_cameras, num_frames)) < config.ball_visibility,
        ball_3d=ball,
        human_kp_2d=rng.uniform(
            0, 1, size=(num_players, num_cameras, num_frames, 17, 2)
        ).astype(np.float32),
        human_kp_vis=rng.uniform(
            0.4, 1.0, size=(num_players, num_cameras, num_frames, 17)
        ).astype(np.float32),
    )


def build_synthetic_dataset(
    dataset_root: str | Path,
    config: SyntheticDatasetConfig | None = None,
) -> DatasetIndex:
    """Materialize a synthetic dataset at ``dataset_root`` and return its index."""
    cfg = config or SyntheticDatasetConfig()
    rng = np.random.default_rng(cfg.seed)
    camera_ids = [f"cam{i}" for i in range(cfg.num_cameras)]
    for recording_id in cfg.recordings:
        for clip_index in range(cfg.clips_per_recording):
            media = {
                camera_id: rng.integers(
                    0, 255, size=(cfg.num_frames, cfg.height, cfg.width, 3), dtype=np.uint8
                )
                for camera_id in camera_ids
            }
            manifest = write_clip_manifest(
                dataset_root,
                recording_id=recording_id,
                clip_name=f"clip_{clip_index:03d}",
                fps=cfg.fps,
                num_frames=cfg.num_frames,
                width=cfg.width,
                height=cfg.height,
                media_videos=media,
                source={"origin": "synthetic"},
            )
            scene = make_synthetic_scene(cfg, rng)
            write_tennis_scene_annotation(
                manifest, scene, generator={"generator": "synthetic"}
            )
            frame_idx = sample_frame_indices(cfg.num_frames, cfg.dino_spec.frame_stride)
            tokens_by_camera = {
                camera_id: (
                    rng.normal(
                        size=(
                            len(frame_idx),
                            cfg.dino_spec.num_tokens,
                            cfg.dino_spec.embed_dim,
                        )
                    ).astype(np.float16),
                    frame_idx,
                )
                for camera_id in camera_ids
            }
            write_dino_tokens(manifest, tokens_by_camera, cfg.dino_spec)
            append_dataset_index(dataset_root, manifest)
    return DatasetIndex.load(dataset_root)


__all__ = [
    "DEFAULT_TEST_DINO_SPEC",
    "SyntheticDatasetConfig",
    "build_synthetic_dataset",
    "make_synthetic_scene",
]
