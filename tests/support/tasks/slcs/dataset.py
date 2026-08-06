"""Deterministic SLCS dataset fixtures composed from canonical production writers."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from src.tasks.slcs.data.annotation import SLCSDataIndex
from src.tasks.slcs.data.dino_tokens import (
    DinoTokenSpec,
    sample_frame_indices,
    write_dino_tokens,
)
from src.tennis_scene.clip_studio.export import (
    CameraExportPlan,
    ClipExportPlan,
    ExportSettings,
    export_clip,
)
from src.tennis_scene.generate_dataset.manifest import ClipManifest
from src.tennis_scene.generate_dataset.pseudo_annotation import (
    generate_pseudo_annotations,
)
from src.tennis_scene.schema import SceneResult
from src.utils.video import probe_video_info, save_video_rgb

DEFAULT_FIXTURE_DINO_SPEC = DinoTokenSpec(
    backbone="dinov3_vitb16",
    patch_size=16,
    image_height=48,
    image_width=64,
    embed_dim=8,
    frame_stride=10,
)


@dataclass(frozen=True, slots=True)
class SLCSFixtureDatasetConfig:
    """Shape and seed parameters for the test-only dataset composer."""

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
    dino_spec: DinoTokenSpec = field(
        default_factory=lambda: DEFAULT_FIXTURE_DINO_SPEC
    )


def make_fixture_scene(
    config: SLCSFixtureDatasetConfig,
    rng: np.random.Generator,
) -> SceneResult:
    """Create one deterministic random scene obeying the SLCS data contract."""
    num_frames = config.num_frames
    num_players = config.num_players
    num_cameras = config.num_cameras
    yaw = rng.uniform(-np.pi, np.pi, size=(num_players, num_frames)).astype(
        np.float32
    )
    position = rng.uniform(-0.5, 0.5, size=(num_players, num_frames, 3)).astype(
        np.float32
    )
    for player in range(num_players):
        position[player, :, 1] += -5.0 if player % 2 == 0 else 5.0
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
        smpl_global_orient=np.zeros(
            (num_players, num_frames, 3), dtype=np.float32
        ),
        smpl_betas=np.zeros((num_players, 10), dtype=np.float32),
        ball_uv=rng.uniform(0, 1, size=(num_cameras, num_frames, 2)).astype(
            np.float32
        ),
        ball_vis=rng.random((num_cameras, num_frames)) < config.ball_visibility,
        ball_3d=ball,
        human_kp_2d=rng.uniform(
            0, 1, size=(num_players, num_cameras, num_frames, 17, 2)
        ).astype(np.float32),
        human_kp_vis=rng.uniform(
            0.4, 1.0, size=(num_players, num_cameras, num_frames, 17)
        ).astype(np.float32),
    )


def _export_clip_fixture(
    dataset_root: Path,
    source_root: Path,
    *,
    recording_id: str,
    clip_name: str,
    config: SLCSFixtureDatasetConfig,
    rng: np.random.Generator,
) -> ClipManifest:
    cameras: list[CameraExportPlan] = []
    for camera_index in range(config.num_cameras):
        camera_id = f"cam{camera_index}"
        source_path = source_root / recording_id / clip_name / f"{camera_id}.mp4"
        frames = rng.integers(
            0,
            255,
            size=(config.num_frames, config.height, config.width, 3),
            dtype=np.uint8,
        )
        save_video_rgb(frames, source_path, fps=config.fps, crf=10)
        source_info = probe_video_info(source_path)
        cameras.append(
            CameraExportPlan(
                camera_id=camera_id,
                source_path=source_path,
                offset_sec=0.0,
                source_info=source_info,
                frame_indices=tuple(range(config.num_frames)),
            )
        )
    plan = ClipExportPlan(
        recording_id=recording_id,
        clip_name=clip_name,
        global_start_sec=0.0,
        global_end_sec=config.num_frames / config.fps,
        fps=config.fps,
        width=config.width,
        height=config.height,
        num_frames=config.num_frames,
        cameras=tuple(cameras),
    )
    result = export_clip(
        plan,
        ExportSettings(
            output_dir=dataset_root,
            fps=config.fps,
            width=config.width,
            height=config.height,
            crf=10,
            overwrite=False,
        ),
    )
    return ClipManifest.load(result.clip_dir)


def build_slcs_dataset_fixture(
    dataset_root: str | Path,
    config: SLCSFixtureDatasetConfig | None = None,
) -> SLCSDataIndex:
    """Compose a complete fixture through canonical manifest/annotation writers."""
    cfg = config or SLCSFixtureDatasetConfig()
    root = Path(dataset_root).resolve()
    rng = np.random.default_rng(cfg.seed)
    source_root = root / ".fixture_sources"
    scenes: dict[str, SceneResult] = {}
    manifests: list[ClipManifest] = []
    for recording_id in cfg.recordings:
        for clip_index in range(cfg.clips_per_recording):
            clip_name = f"clip_{clip_index:03d}"
            manifest = _export_clip_fixture(
                root,
                source_root,
                recording_id=recording_id,
                clip_name=clip_name,
                config=cfg,
                rng=rng,
            )
            manifests.append(manifest)
            scenes[manifest.clip_id] = make_fixture_scene(cfg, rng)

    def scene_runner(
        video_paths: Sequence[Path], camera_ids: Sequence[str]
    ) -> SceneResult:
        del camera_ids
        clip_dir = video_paths[0].parent.parent
        clip_id = f"{clip_dir.parent.name}/{clip_dir.name}"
        return scenes[clip_id]

    outcomes = generate_pseudo_annotations(
        root,
        scene_runner,
        pipeline_config_yaml="# deterministic SLCS test fixture\n",
        clip_ids=[manifest.clip_id for manifest in manifests],
        continue_on_error=False,
    )
    failed = [outcome for outcome in outcomes if outcome.status != "generated"]
    if failed:
        raise RuntimeError(f"Canonical pseudo-annotation writer did not run: {failed}")

    for manifest in manifests:
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
            for camera_id in manifest.camera_ids
        }
        write_dino_tokens(
            manifest,
            tokens_by_camera,
            cfg.dino_spec,
            generator={"fixture": "tests.support.tasks.slcs.dataset"},
        )
    return SLCSDataIndex.load(root)


__all__ = [
    "DEFAULT_FIXTURE_DINO_SPEC",
    "SLCSFixtureDatasetConfig",
    "build_slcs_dataset_fixture",
    "make_fixture_scene",
]
