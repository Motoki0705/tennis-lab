"""Measure angular velocity of GT canonical-pose angles on the PLCS train split.

Iterates the GT training data, converts each frame's world pose to the
canonical frame, computes frame-to-frame angular velocities for the three
angle families (joint angles, torsion angles, torso twist), and aggregates
per-angle standard deviation and mean-absolute velocity across all valid
frame transitions.  The resulting dominance weights (normalized to mean 1.0)
can be used to upweight the velocity-loss terms for the most dynamic angles.

Usage:
    python -m src.tasks.plcs.scripts.analysis.analyze_angle_velocity
    python -m src.tasks.plcs.scripts.analysis.analyze_angle_velocity \
        analysis.split=train analysis.max_batches=50 \
        run.output_dir=plcs/analysis/angle_velocity

Notes:
    - Configuration is loaded from
      ``src/tasks/plcs/configs/analyze_angle_velocity.yaml`` via Hydra.
    - The script must be run CPU-only (prefix with ``CUDA_VISIBLE_DEVICES=""``).
    - Only valid frame-to-frame transitions are counted: both adjacent frames
      must not be padded according to ``padding_mask``.
    - Torsion angles and torso twist are periodic; their velocity differences are
      wrapped into [-pi, pi] before aggregation.
    - The JSON output contains per-angle std, mean-abs velocity, and normalized
      dominance weights (mean=1.0) for each angle family.
"""

from __future__ import annotations

import math
from typing import Any, cast

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from torch import Tensor

from src.tasks.plcs.configuration import PLCSAnalysisRuntimeConfig
from src.tasks.plcs.data.dataset import SceneDataset, collate_plcs_batch
from src.tasks.plcs.training.losses import (
    compute_joint_angles,
    compute_torsion_angles,
    compute_torso_twist,
)
from src.utils.geometry.court_pose import world_pose_to_canonical_pose
from src.utils.hydra import hydra_main
from src.utils.io import save_json
from src.utils.schema.player import COCO17_JOINT_ANGLE_TRIPLETS as JOINT_ANGLE_TRIPLETS
from src.utils.schema.player import COCO17_TORSION_QUADRUPLETS as TORSION_QUADRUPLETS

# ---------------------------------------------------------------------------
# Welford online statistics (per-channel)
# ---------------------------------------------------------------------------


class ChannelStats:
    """Online (Welford) running statistics for a fixed number of channels."""

    def __init__(self, n_channels: int) -> None:
        self.n_channels = n_channels
        self._count: np.ndarray = np.zeros(n_channels, dtype=np.float64)
        self._mean: np.ndarray = np.zeros(n_channels, dtype=np.float64)
        self._m2: np.ndarray = np.zeros(n_channels, dtype=np.float64)
        self._sum_abs: np.ndarray = np.zeros(n_channels, dtype=np.float64)

    def update(self, values: np.ndarray) -> None:
        """Update running stats with a 2-D array of shape (N, C).

        Args:
            values: Array with shape ``(N, C)`` where ``C == self.n_channels``.
                Each row is one observation.

        """
        if values.ndim != 2 or values.shape[1] != self.n_channels:
            raise ValueError(
                f"Expected shape (N, {self.n_channels}), got {values.shape}"
            )
        if values.shape[0] == 0:
            return
        n_new = values.shape[0]
        batch_mean = values.mean(axis=0)
        batch_var = values.var(axis=0, ddof=0)
        batch_m2 = batch_var * n_new
        batch_sum_abs = np.abs(values).sum(axis=0)

        delta = batch_mean - self._mean
        n_total = self._count + n_new
        # Guard against division by zero on the very first update.
        safe_n_total = np.where(n_total > 0, n_total, 1.0)
        self._mean = self._mean + delta * n_new / safe_n_total
        self._m2 = (
            self._m2 + batch_m2 + (delta * delta) * self._count * n_new / safe_n_total
        )
        self._count = n_total
        self._sum_abs += batch_sum_abs

    def std(self) -> np.ndarray:
        """Return per-channel standard deviation."""
        var = np.asarray(
            np.where(self._count > 0, self._m2 / np.maximum(self._count, 1.0), 0.0),
            dtype=np.float64,
        )
        result: np.ndarray = np.sqrt(np.maximum(var, 0.0))
        return result

    def mean_abs(self) -> np.ndarray:
        """Return per-channel mean absolute value."""
        result: np.ndarray = np.asarray(
            np.where(
                self._count > 0,
                self._sum_abs / np.maximum(self._count, 1.0),
                0.0,
            ),
            dtype=np.float64,
        )
        return result

    def count(self) -> np.ndarray:
        """Return per-channel sample count."""
        return self._count.copy()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _wrap_diff(diff: np.ndarray) -> np.ndarray:
    """Wrap angular differences into [-pi, pi]."""
    result: np.ndarray = np.asarray(
        np.arctan2(np.sin(diff), np.cos(diff)), dtype=np.float64
    )
    return result


def _normalize_weights(stds: np.ndarray) -> list[float]:
    """Normalize std vector to mean=1.0 (dominance weights)."""
    mean_std = float(stds.mean())
    if mean_std < 1e-12:
        return [1.0] * len(stds)
    return [float(v / mean_std) for v in stds]


def _angles_to_numpy(t: Tensor) -> np.ndarray:
    result: np.ndarray = np.asarray(t.detach().cpu().numpy(), dtype=np.float64)
    return result


# ---------------------------------------------------------------------------
# Main analysis loop
# ---------------------------------------------------------------------------


def analyze_angle_velocity(
    dataset: SceneDataset,
    split: str,
    batch_size: int,
    num_workers: int,
    max_batches: int | None,
) -> dict[str, Any]:
    """Compute per-angle GT angular-velocity statistics on the given split.

    Args:
        dataset: Strictly configured PLCS scene dataset.
        batch_size: Analysis loader batch size.
        num_workers: Analysis loader worker count.
        max_batches: Optional cap on the number of batches processed.

    Returns:
        Dictionary with per-family statistics and normalized dominance weights.

    """
    # Build the dataset directly without the collate adapter so that we always
    # have human_kp_3d, position, rotation, and padding_mask in the batch.
    loader: torch.utils.data.DataLoader[dict[str, Tensor]] = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
        drop_last=False,
        collate_fn=collate_plcs_batch,
    )

    n_joint = len(JOINT_ANGLE_TRIPLETS)  # 12
    n_torsion = len(TORSION_QUADRUPLETS)  # 4
    n_twist = 1

    joint_stats = ChannelStats(n_joint)
    torsion_stats = ChannelStats(n_torsion)
    twist_stats = ChannelStats(n_twist)

    batches_processed = 0

    for batch in loader:
        if max_batches is not None and batches_processed >= max_batches:
            break

        human_kp_3d: Tensor | None = cast("Tensor | None", batch.get("human_kp_3d"))
        position: Tensor | None = cast("Tensor | None", batch.get("position"))
        rotation: Tensor | None = cast("Tensor | None", batch.get("rotation"))
        padding_mask: Tensor | None = cast(
            "Tensor | None", batch.get("padding_mask")
        )

        if human_kp_3d is None or position is None or rotation is None:
            batches_processed += 1
            continue

        # human_kp_3d: (B, T, 17, 3)
        # position   : (B, T, 3)
        # rotation   : (B, T, 2)
        # padding_mask: (B,N,T), True=padding.
        if human_kp_3d.ndim != 4:
            batches_processed += 1
            continue

        b, t, _j, _ = human_kp_3d.shape

        if t < 2:
            batches_processed += 1
            continue

        # Resolve validity mask: (B, T) boolean
        if padding_mask is not None:
            if padding_mask.ndim == 3:
                # (B,N,T) -> (B,T): valid if any camera is not padded.
                mask_bt: Tensor = (~padding_mask).any(dim=1)
            elif padding_mask.ndim == 2:
                mask_bt = ~padding_mask
            else:
                mask_bt = torch.ones(b, t, dtype=torch.bool)
        else:
            mask_bt = torch.ones(b, t, dtype=torch.bool)

        with torch.no_grad():
            canon = world_pose_to_canonical_pose(
                human_kp_3d, position, rotation
            )  # (B, T, 17, 3)

            # --- joint angles (B, T, 12), non-periodic ---
            joint_a = compute_joint_angles(canon)  # (B, T, 12)
            joint_vel = joint_a[:, 1:, :] - joint_a[:, :-1, :]  # (B, T-1, 12)

            # --- torsion angles (B, T, 4), periodic ---
            torsion_a = compute_torsion_angles(canon)  # (B, T, 4)
            torsion_diff = torsion_a[:, 1:, :] - torsion_a[:, :-1, :]
            torsion_vel = torch.atan2(
                torch.sin(torsion_diff), torch.cos(torsion_diff)
            )  # (B, T-1, 4)

            # --- torso twist (B, T), periodic ---
            twist_a = compute_torso_twist(canon)  # (B, T)
            twist_diff = twist_a[:, 1:] - twist_a[:, :-1]
            twist_vel = torch.atan2(
                torch.sin(twist_diff), torch.cos(twist_diff)
            ).unsqueeze(-1)  # (B, T-1, 1)

        # Validity mask for transitions: both frame t and t+1 must be valid
        trans_mask = mask_bt[:, 1:] & mask_bt[:, :-1]  # (B, T-1) bool

        # Gather valid transitions into numpy arrays (N_valid, C)
        joint_np = _angles_to_numpy(joint_vel)  # (B, T-1, 12)
        torsion_np = _angles_to_numpy(torsion_vel)  # (B, T-1, 4)
        twist_np = _angles_to_numpy(twist_vel)  # (B, T-1, 1)
        trans_np = trans_mask.cpu().numpy()  # (B, T-1) bool

        # Flatten batch x time and filter by mask
        flat_mask = trans_np.reshape(-1)  # (B*(T-1),)
        joint_flat = joint_np.reshape(-1, n_joint)[flat_mask]
        torsion_flat = torsion_np.reshape(-1, n_torsion)[flat_mask]
        twist_flat = twist_np.reshape(-1, n_twist)[flat_mask]

        if joint_flat.shape[0] > 0:
            joint_stats.update(joint_flat)
            torsion_stats.update(torsion_flat)
            twist_stats.update(twist_flat)

        batches_processed += 1

    # --- collect results ---
    joint_std = joint_stats.std()
    torsion_std = torsion_stats.std()
    twist_std = twist_stats.std()

    joint_mean_abs = joint_stats.mean_abs()
    torsion_mean_abs = torsion_stats.mean_abs()
    twist_mean_abs = twist_stats.mean_abs()

    joint_weights = _normalize_weights(joint_std)
    torsion_weights = _normalize_weights(torsion_std)

    joint_labels = [
        "L_elbow",
        "R_elbow",
        "L_knee",
        "R_knee",
        "L_shoulder",
        "R_shoulder",
        "L_hip",
        "R_hip",
        "L_shoulder_torso",
        "R_shoulder_torso",
        "R_hip_torso",
        "L_hip_torso",
    ]
    torsion_labels = ["L_arm", "R_arm", "L_leg", "R_leg"]

    return {
        "split": split,
        "batches_processed": batches_processed,
        "joint_angles": {
            "labels": joint_labels,
            "std": joint_std.tolist(),
            "mean_abs": joint_mean_abs.tolist(),
            "dominance_weights": joint_weights,
            "sample_count": int(joint_stats.count()[0]),
        },
        "torsion_angles": {
            "labels": torsion_labels,
            "std": torsion_std.tolist(),
            "mean_abs": torsion_mean_abs.tolist(),
            "dominance_weights": torsion_weights,
            "sample_count": int(torsion_stats.count()[0]),
        },
        "torso_twist": {
            "std": float(twist_std[0]),
            "mean_abs": float(twist_mean_abs[0]),
            "sample_count": int(twist_stats.count()[0]),
        },
    }


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def _print_report(result: dict[str, Any]) -> None:
    """Pretty-print the dominance analysis results."""
    print("=" * 72)
    print("PLCS angular-velocity dominance analysis")
    print("=" * 72)
    print(f"split          : {result['split']}")
    print(f"batches        : {result['batches_processed']}")

    print()
    print("Joint angles (12 channels)  [non-periodic, rad/frame]")
    print(f"{'label':<22}{'std':>10}{'mean_abs':>12}{'weight':>10}")
    print("-" * 54)
    ja = result["joint_angles"]
    for label, std, mabs, w in zip(
        ja["labels"], ja["std"], ja["mean_abs"], ja["dominance_weights"], strict=True
    ):
        print(f"{label:<22}{std:>10.5f}{mabs:>12.5f}{w:>10.4f}")
    print(f"  sample_count : {ja['sample_count']}")

    print()
    print("Torsion angles (4 channels)  [periodic, rad/frame]")
    print(f"{'label':<22}{'std':>10}{'mean_abs':>12}{'weight':>10}")
    print("-" * 54)
    ta = result["torsion_angles"]
    for label, std, mabs, w in zip(
        ta["labels"], ta["std"], ta["mean_abs"], ta["dominance_weights"], strict=True
    ):
        print(f"{label:<22}{std:>10.5f}{mabs:>12.5f}{w:>10.4f}")
    print(f"  sample_count : {ta['sample_count']}")

    print()
    print("Torso twist (1 channel)  [periodic, rad/frame]")
    tt = result["torso_twist"]
    print(f"  std         : {tt['std']:.5f}")
    print(f"  mean_abs    : {tt['mean_abs']:.5f}")
    print(f"  sample_count: {tt['sample_count']}")
    print("=" * 72)

    # Dominant channel summary
    all_stds = list(zip(ja["labels"], ja["std"], ["joint"] * 12, strict=True))
    all_stds += list(zip(ta["labels"], ta["std"], ["torsion"] * 4, strict=True))
    all_stds += [("torso_twist", tt["std"], "twist")]
    dominant = max(all_stds, key=lambda x: x[1])
    print(
        f"Most dynamic channel: {dominant[0]} ({dominant[2]}) "
        f"std={dominant[1]:.5f} rad/frame"
    )
    print("=" * 72)


# ---------------------------------------------------------------------------
# Hydra entry point
# ---------------------------------------------------------------------------


@hydra_main(
    config_path="../../configs",
    config_name="analyze_angle_velocity",
    version_base="1.3",
    validation_boundary="plcs.analyze_angle_velocity",
)
def main(cfg: DictConfig) -> int:  # pragma: no cover - CLI entry
    seed = int(cfg.run.seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    runtime = PLCSAnalysisRuntimeConfig.angle_velocity(cfg)
    out_dir = runtime.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, out_dir / "config.yaml")

    max_batches = cfg.analysis.max_batches
    if max_batches is not None:
        max_batches = int(max_batches)

    split = str(cfg.analysis.split)

    if runtime.scene_dir is None or runtime.result_path is None:
        raise AssertionError("PLCS angle-velocity paths were not resolved.")
    dataset = SceneDataset(
        scene_dir=runtime.scene_dir,
        split_file=f"{split}.txt",
        config=cfg,
        seed=seed,
        augment=False,
    )

    result = analyze_angle_velocity(
        dataset=dataset,
        split=split,
        batch_size=int(cfg.data.batch_size),
        num_workers=int(cfg.data.num_workers),
        max_batches=max_batches,
    )

    _print_report(result)

    # Serialize to JSON (convert any remaining numpy types)
    def _to_python(obj: Any) -> Any:
        if isinstance(obj, (np.floating, float)):
            v = float(obj)
            return None if math.isnan(v) or math.isinf(v) else v
        if isinstance(obj, (np.integer, int)):
            return int(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return {k: _to_python(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_to_python(v) for v in obj]
        return obj

    out_path = runtime.result_path
    save_json(_to_python(result), out_path)
    print(f"Saved results to {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
