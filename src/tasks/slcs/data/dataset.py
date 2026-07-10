"""SLCS window dataset over the issue #634 structured clip dataset.

One dataset item is a single-camera temporal window of one clip
(:class:`src.tasks.slcs.data.types.SLCSSample`). Construction is eager and
strict: every clip in the requested split is loaded and validated at init
time, so contract violations surface immediately rather than mid-training.

Canonical player ordering: players are reordered so that slot 0 is the near
side (smaller mean court-Y over label-valid frames) and slot 1 the far side.
The ordering is derived from the pseudo-label positions and is deterministic;
an ambiguous ordering (equal means) is an error.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import torch
from numpy.typing import NDArray
from torch.utils.data import Dataset

from src.tasks.slcs.data.contract import (
    ClipManifest,
    DatasetContractError,
    DatasetIndex,
    IncompleteAnnotationError,
    load_tennis_scene_annotation,
)
from src.tasks.slcs.data.dino_tokens import DinoTokenSpec, load_dino_tokens
from src.tasks.slcs.data.quality import (
    QualityConfig,
    build_label_masks,
    window_label_ratio,
)
from src.tasks.slcs.data.splits import load_split_assignments
from src.tasks.slcs.data.types import SLCSSample, SLCSWindowMeta
from src.tasks.slcs.data.windows import WindowPlan, plan_windows, select_window_tokens
from src.utils.schema.court import COURT_COORD_SCALE_XYZ
from src.utils.schema.player import NUM_HUMAN_KP

if TYPE_CHECKING:
    from omegaconf import DictConfig

# Normalized-UV sanity range for visible observations. Slightly outside [0, 1]
# is legitimate (detections at the frame border); far outside means the data
# is in pixel units and violates the contract.
_UV_TOLERANCE = 0.25


@dataclass(frozen=True)
class SLCSDataConfig:
    """Static configuration of the SLCS data pipeline."""

    window_size: int = 120
    train_stride: int = 60
    eval_stride: int = 120
    num_players: int = 2
    num_court_kp: int = 14
    require_dino: bool = True
    cache_dino_tokens: bool = True
    on_incomplete: Literal["error", "skip"] = "error"
    dino_spec: DinoTokenSpec | None = None
    quality: QualityConfig = field(default_factory=QualityConfig)

    def __post_init__(self) -> None:
        if self.window_size <= 0:
            raise ValueError(f"window_size must be positive, got {self.window_size}.")
        if self.train_stride <= 0 or self.eval_stride <= 0:
            raise ValueError("train_stride and eval_stride must be positive.")
        if self.num_players <= 0:
            raise ValueError(f"num_players must be positive, got {self.num_players}.")
        if self.num_court_kp <= 0:
            raise ValueError(f"num_court_kp must be positive, got {self.num_court_kp}.")
        if self.dino_spec is None:
            raise ValueError(
                "dino_spec is required (even with require_dino=False) so token tensor "
                "shapes are fixed by configuration rather than inferred from data."
            )
        if self.on_incomplete not in ("error", "skip"):
            raise ValueError(
                f"on_incomplete must be 'error' or 'skip', got {self.on_incomplete!r}."
            )

    @classmethod
    def from_config(cls, data_cfg: DictConfig | dict[str, Any]) -> SLCSDataConfig:
        """Build from the ``data`` section of a Hydra config."""
        get = data_cfg.get
        dino_cfg = get("dino", None)
        if dino_cfg is None:
            raise ValueError(
                "data config must declare a 'dino' section (backbone, patch_size, "
                "image_height, image_width, embed_dim, frame_stride)."
            )
        dino_spec = DinoTokenSpec(
            backbone=str(dino_cfg["backbone"]),
            patch_size=int(dino_cfg["patch_size"]),
            image_height=int(dino_cfg["image_height"]),
            image_width=int(dino_cfg["image_width"]),
            embed_dim=int(dino_cfg["embed_dim"]),
            frame_stride=int(dino_cfg["frame_stride"]),
        )
        quality_cfg = get("quality", None)
        quality = (
            QualityConfig.from_dict(dict(quality_cfg)) if quality_cfg is not None else QualityConfig()
        )
        return cls(
            window_size=int(get("window_size", 120)),
            train_stride=int(get("train_stride", 60)),
            eval_stride=int(get("eval_stride", 120)),
            num_players=int(get("num_players", 2)),
            num_court_kp=int(get("num_court_kp", 14)),
            require_dino=bool(get("require_dino", True)),
            cache_dino_tokens=bool(get("cache_dino_tokens", True)),
            on_incomplete=str(get("on_incomplete", "error")),  # type: ignore[arg-type]
            dino_spec=dino_spec,
            quality=quality,
        )


@dataclass
class _ClipData:
    """Validated per-clip arrays shared by all windows of the clip."""

    manifest: ClipManifest
    fps: float
    num_frames: int
    court_kp: NDArray[np.float32]  # (N, T, K, 2)
    court_vis: NDArray[np.float32]  # (N, T, K)
    human_kp_2d: NDArray[np.float32]  # (P, N, T, J, 2)
    human_kp_vis: NDArray[np.float32]  # (P, N, T, J)
    ball_uv: NDArray[np.float32]  # (N, T, 2)
    ball_vis: NDArray[np.bool_]  # (N, T)
    player_position_norm: NDArray[np.float32]  # (P, T, 3)
    player_rotation: NDArray[np.float32]  # (P, T, 2)
    ball_position_norm: NDArray[np.float32]  # (T, 3)
    player_label_valid: NDArray[np.bool_]  # (P, T)
    player_label_weight: NDArray[np.float32]  # (P, T)
    ball_label_valid: NDArray[np.bool_]  # (T,)
    ball_label_weight: NDArray[np.float32]  # (T,)


@dataclass(frozen=True)
class _WindowEntry:
    clip_id: str
    camera_id: str
    camera_index: int
    plan: WindowPlan


def _canonical_player_order(
    player_position: NDArray[np.float32], player_label_valid: NDArray[np.bool_], *, clip_id: str
) -> NDArray[np.int64]:
    """Near-side-first ordering of the player axis (see module docstring)."""
    num_players = player_position.shape[0]
    means = np.empty(num_players, dtype=np.float64)
    for p in range(num_players):
        valid = player_label_valid[p]
        if not valid.any():
            raise DatasetContractError(
                f"{clip_id}: player {p} has no label-valid frame; cannot derive a "
                "canonical player ordering."
            )
        means[p] = float(player_position[p, valid, 1].mean())
    order = np.argsort(means, kind="stable").astype(np.int64)
    if num_players > 1 and np.any(np.diff(means[order]) == 0.0):
        raise DatasetContractError(
            f"{clip_id}: ambiguous player ordering (equal mean court-Y {means.tolist()})."
        )
    return order


def _check_visible_uv(
    uv: NDArray[np.float32], vis: NDArray[Any], *, context: str
) -> None:
    visible = np.asarray(vis) > 0
    if not visible.any():
        return
    values = uv[visible]
    if not np.isfinite(values).all():
        raise DatasetContractError(f"{context}: visible observations contain non-finite UV.")
    if values.min() < -_UV_TOLERANCE or values.max() > 1.0 + _UV_TOLERANCE:
        raise DatasetContractError(
            f"{context}: visible UV outside [{-_UV_TOLERANCE}, {1 + _UV_TOLERANCE}] "
            f"(min={values.min():.3f}, max={values.max():.3f}); expected normalized "
            "[0, 1] coordinates, not pixels."
        )


class SLCSWindowDataset(Dataset[SLCSSample]):
    """Single-camera temporal windows over a split of the #634 dataset."""

    def __init__(
        self,
        *,
        dataset_root: str | Path,
        split_file: str | Path,
        split: str,
        config: SLCSDataConfig,
        stride: int | None = None,
    ) -> None:
        if split not in ("train", "val", "test"):
            raise ValueError(f"split must be train/val/test, got {split!r}.")
        self.config = config
        self.split = split
        self.stride = int(
            stride
            if stride is not None
            else (config.train_stride if split == "train" else config.eval_stride)
        )

        index = DatasetIndex.load(dataset_root)
        assignments = load_split_assignments(split_file, index)

        self._clips: dict[str, _ClipData] = {}
        self._dino_cache: dict[tuple[str, str], tuple[NDArray[np.float32], NDArray[np.int64]]] = {}
        self._entries: list[_WindowEntry] = []
        self.build_report: dict[str, int] = {
            "clips_in_split": 0,
            "clips_loaded": 0,
            "clips_skipped_incomplete": 0,
            "windows_total": 0,
            "windows_dropped_low_label": 0,
        }

        for ref in index.clips:
            if assignments[ref.recording_id] != split:
                continue
            self.build_report["clips_in_split"] += 1
            manifest = ClipManifest.load(index.clip_dir(ref))
            try:
                clip = self._load_clip(manifest)
            except IncompleteAnnotationError:
                if config.on_incomplete == "error":
                    raise
                self.build_report["clips_skipped_incomplete"] += 1
                continue
            self._clips[manifest.clip_id] = clip
            self.build_report["clips_loaded"] += 1
            self._plan_clip_windows(clip)

        if not self._entries:
            raise DatasetContractError(
                f"split {split!r} of dataset {dataset_root} yields no windows "
                f"(report: {self.build_report})."
            )

        self.metas: list[SLCSWindowMeta] = [
            SLCSWindowMeta(
                clip_id=e.clip_id,
                recording_id=self._clips[e.clip_id].manifest.recording_id,
                camera_id=e.camera_id,
                window_start=e.plan.start,
                window_length=e.plan.length,
            )
            for e in self._entries
        ]
        # Sample identifiers, consumed by BaseLightningModule test-prediction saving.
        self.scenes: list[str] = [
            f"{m.clip_id}@{m.camera_id}@{m.window_start:06d}" for m in self.metas
        ]

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    def _load_clip(self, manifest: ClipManifest) -> _ClipData:
        cfg = self.config
        scene = load_tennis_scene_annotation(manifest)
        clip_id = manifest.clip_id

        human_kp_2d = np.asarray(scene.human_kp_2d, dtype=np.float32)
        human_kp_vis = np.asarray(scene.human_kp_vis, dtype=np.float32)
        ball_uv = np.asarray(scene.ball_uv, dtype=np.float32)
        ball_vis = np.asarray(scene.ball_vis, dtype=np.bool_)
        ball_3d = np.asarray(scene.ball_3d, dtype=np.float32)
        court_kp = np.asarray(scene.court_kp, dtype=np.float32)
        court_vis = np.asarray(scene.court_vis, dtype=np.float32)
        player_position = np.asarray(scene.player_position, dtype=np.float32)
        player_yaw = np.asarray(scene.player_yaw, dtype=np.float32)

        num_players = player_position.shape[0]
        if num_players != cfg.num_players:
            raise DatasetContractError(
                f"{clip_id}: scene has P={num_players} players, config expects "
                f"{cfg.num_players}."
            )
        if court_kp.shape[2] != cfg.num_court_kp:
            raise DatasetContractError(
                f"{clip_id}: scene court_kp has K={court_kp.shape[2]}, config expects "
                f"{cfg.num_court_kp}."
            )
        if human_kp_2d.shape[3] != NUM_HUMAN_KP:
            raise DatasetContractError(
                f"{clip_id}: human_kp_2d has J={human_kp_2d.shape[3]}, expected {NUM_HUMAN_KP}."
            )

        _check_visible_uv(ball_uv, ball_vis, context=f"{clip_id}: ball_uv")
        _check_visible_uv(court_kp, court_vis, context=f"{clip_id}: court_kp")
        _check_visible_uv(human_kp_2d, human_kp_vis, context=f"{clip_id}: human_kp_2d")

        masks = build_label_masks(
            human_kp_vis=human_kp_vis,
            ball_vis=ball_vis,
            player_position=player_position,
            player_yaw=player_yaw,
            ball_3d=ball_3d,
            config=cfg.quality,
        )
        order = _canonical_player_order(
            player_position, masks["player_label_valid"], clip_id=clip_id
        )

        scale = np.asarray(COURT_COORD_SCALE_XYZ, dtype=np.float32)
        player_position_norm = (player_position / scale)[order]
        player_rotation = np.stack(
            [np.cos(player_yaw), np.sin(player_yaw)], axis=-1
        ).astype(np.float32)[order]
        ball_position_norm = (ball_3d / scale).astype(np.float32)

        return _ClipData(
            manifest=manifest,
            fps=float(scene.fps),
            num_frames=int(scene.num_frames),
            court_kp=court_kp,
            court_vis=court_vis,
            human_kp_2d=human_kp_2d[order],
            human_kp_vis=human_kp_vis[order],
            ball_uv=ball_uv,
            ball_vis=ball_vis,
            player_position_norm=player_position_norm.astype(np.float32),
            player_rotation=player_rotation,
            ball_position_norm=ball_position_norm,
            player_label_valid=masks["player_label_valid"][order],
            player_label_weight=masks["player_label_weight"][order],
            ball_label_valid=masks["ball_label_valid"],
            ball_label_weight=masks["ball_label_weight"],
        )

    def _plan_clip_windows(self, clip: _ClipData) -> None:
        cfg = self.config
        plans = plan_windows(
            clip.num_frames, window_size=cfg.window_size, stride=self.stride
        )
        for camera_index, camera_id in enumerate(clip.manifest.camera_ids):
            if cfg.require_dino:
                # Validates spec + arrays; raises on missing/incomplete annotation.
                self._dino_arrays(clip.manifest, camera_id)
            for plan in plans:
                self.build_report["windows_total"] += 1
                ratio = window_label_ratio(
                    clip.player_label_valid,
                    clip.ball_label_valid,
                    start=plan.start,
                    length=plan.length,
                )
                if ratio < cfg.quality.min_window_label_ratio:
                    self.build_report["windows_dropped_low_label"] += 1
                    continue
                self._entries.append(
                    _WindowEntry(
                        clip_id=clip.manifest.clip_id,
                        camera_id=camera_id,
                        camera_index=camera_index,
                        plan=plan,
                    )
                )

    def _dino_arrays(
        self, manifest: ClipManifest, camera_id: str
    ) -> tuple[NDArray[np.float32], NDArray[np.int64]]:
        key = (manifest.clip_id, camera_id)
        cached = self._dino_cache.get(key)
        if cached is not None:
            return cached
        tokens, frame_idx, _spec = load_dino_tokens(
            manifest, camera_id, expected_spec=self.config.dino_spec
        )
        if self.config.cache_dino_tokens:
            self._dino_cache[key] = (tokens, frame_idx)
        return tokens, frame_idx

    # ------------------------------------------------------------------
    # Dataset protocol
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._entries)

    def __getitem__(self, index: int) -> SLCSSample:
        entry = self._entries[index]
        clip = self._clips[entry.clip_id]
        plan = entry.plan
        cam = entry.camera_index
        cfg = self.config
        t0, t1 = plan.start, plan.start + plan.length
        pad = plan.pad

        def pad_time(arr: NDArray[Any], time_axis: int) -> NDArray[Any]:
            if pad == 0:
                return arr
            widths = [(0, 0)] * arr.ndim
            widths[time_axis] = (0, pad)
            return np.pad(arr, widths)

        player_kp = pad_time(clip.human_kp_2d[:, cam, t0:t1], 1)
        player_kp_vis = pad_time(clip.human_kp_vis[:, cam, t0:t1], 1)
        player_valid = player_kp_vis.max(axis=-1) > 0

        ball_uv = pad_time(clip.ball_uv[cam, t0:t1], 0)
        ball_vis = pad_time(clip.ball_vis[cam, t0:t1], 0)
        court_kp = pad_time(clip.court_kp[cam, t0:t1], 0)
        court_vis = pad_time(clip.court_vis[cam, t0:t1], 0)

        frame_mask = plan.frame_mask()
        frame_idx = plan.frame_indices()
        timestamp = (frame_idx.astype(np.float32)) / np.float32(clip.fps)

        spec = cfg.dino_spec
        assert spec is not None  # enforced by SLCSDataConfig.__post_init__
        dino_tokens: NDArray[np.float32]
        dino_frame_idx: NDArray[np.int64]
        dino_valid: NDArray[np.bool_]
        if cfg.require_dino:
            tokens, token_frames = self._dino_arrays(clip.manifest, entry.camera_id)
            sel = select_window_tokens(token_frames, plan)
            dino_tokens = tokens[sel]
            dino_frame_idx = (token_frames[sel] - plan.start).astype(np.int64)
            dino_valid = np.ones(len(sel), dtype=np.bool_)
        else:
            dino_tokens = np.zeros((0, spec.num_tokens, spec.embed_dim), dtype=np.float32)
            dino_frame_idx = np.zeros((0,), dtype=np.int64)
            dino_valid = np.zeros((0,), dtype=np.bool_)

        target_player_position = pad_time(clip.player_position_norm[:, t0:t1], 1)
        target_player_rotation = pad_time(clip.player_rotation[:, t0:t1], 1)
        target_player_valid = pad_time(clip.player_label_valid[:, t0:t1], 1)
        target_player_weight = pad_time(clip.player_label_weight[:, t0:t1], 1)
        target_ball_position = pad_time(clip.ball_position_norm[t0:t1], 0)
        target_ball_valid = pad_time(clip.ball_label_valid[t0:t1], 0)
        target_ball_weight = pad_time(clip.ball_label_weight[t0:t1], 0)

        return SLCSSample(
            player_kp=torch.from_numpy(np.ascontiguousarray(player_kp)),
            player_kp_vis=torch.from_numpy(np.ascontiguousarray(player_kp_vis)),
            player_valid=torch.from_numpy(np.ascontiguousarray(player_valid)),
            ball_uv=torch.from_numpy(np.ascontiguousarray(ball_uv)),
            ball_vis=torch.from_numpy(np.ascontiguousarray(ball_vis)),
            court_kp=torch.from_numpy(np.ascontiguousarray(court_kp)),
            court_vis=torch.from_numpy(np.ascontiguousarray(court_vis)),
            dino_tokens=torch.from_numpy(np.ascontiguousarray(dino_tokens)),
            dino_frame_idx=torch.from_numpy(np.ascontiguousarray(dino_frame_idx)),
            dino_valid=torch.from_numpy(np.ascontiguousarray(dino_valid)),
            frame_idx=torch.from_numpy(np.ascontiguousarray(frame_idx)),
            timestamp=torch.from_numpy(np.ascontiguousarray(timestamp)),
            frame_mask=torch.from_numpy(np.ascontiguousarray(frame_mask)),
            target_player_position=torch.from_numpy(
                np.ascontiguousarray(target_player_position)
            ),
            target_player_rotation=torch.from_numpy(
                np.ascontiguousarray(target_player_rotation)
            ),
            target_player_valid=torch.from_numpy(np.ascontiguousarray(target_player_valid)),
            target_player_weight=torch.from_numpy(
                np.ascontiguousarray(target_player_weight)
            ),
            target_ball_position=torch.from_numpy(np.ascontiguousarray(target_ball_position)),
            target_ball_valid=torch.from_numpy(np.ascontiguousarray(target_ball_valid)),
            target_ball_weight=torch.from_numpy(np.ascontiguousarray(target_ball_weight)),
        )


def collate_slcs(samples: list[SLCSSample]) -> dict[str, torch.Tensor]:
    """Collate samples into an :class:`SLCSBatch`-shaped dict.

    Fixed-shape tensors are stacked; the variable DINOv3 sample axis is
    right-padded to the batch maximum (at least 1 slot so downstream tensor
    ops are well-defined) with ``dino_valid=False`` marking the padding.
    """
    if not samples:
        raise ValueError("collate_slcs received an empty sample list.")
    batch: dict[str, torch.Tensor] = {}
    dino_keys = {"dino_tokens", "dino_frame_idx", "dino_valid"}
    for key in samples[0]:
        if key in dino_keys:
            continue
        batch[key] = torch.stack([s[key] for s in samples], dim=0)  # type: ignore[literal-required]

    max_td = max(1, max(int(s["dino_tokens"].shape[0]) for s in samples))
    ref_tokens = samples[0]["dino_tokens"]
    if ref_tokens.ndim != 3:
        raise ValueError(f"dino_tokens must be (T_d, S, C), got shape {tuple(ref_tokens.shape)}.")
    num_patches = int(ref_tokens.shape[1])
    embed_dim = int(ref_tokens.shape[2])
    for s in samples:
        t = s["dino_tokens"]
        if int(t.shape[1]) != num_patches or int(t.shape[2]) != embed_dim:
            raise ValueError(
                f"inconsistent dino token shapes in batch: {tuple(t.shape)} vs "
                f"(*, {num_patches}, {embed_dim})."
            )

    tokens_out = torch.zeros(len(samples), max_td, num_patches, embed_dim, dtype=torch.float32)
    frames_out = torch.zeros(len(samples), max_td, dtype=torch.int64)
    valid_out = torch.zeros(len(samples), max_td, dtype=torch.bool)
    for i, s in enumerate(samples):
        td = int(s["dino_tokens"].shape[0])
        if td > 0:
            tokens_out[i, :td] = s["dino_tokens"]
            frames_out[i, :td] = s["dino_frame_idx"]
            valid_out[i, :td] = s["dino_valid"]
    batch["dino_tokens"] = tokens_out
    batch["dino_frame_idx"] = frames_out
    batch["dino_valid"] = valid_out
    return batch


__all__ = ["SLCSDataConfig", "SLCSWindowDataset", "collate_slcs"]
