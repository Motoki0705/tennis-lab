"""Build an auditable train-only hard-scene resampling overlay and residual init."""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf, open_dict

from src.tasks.plcs.geometry.footpoint import footpoint_prior
from src.tasks.plcs.training.lightning_module import PLCSLightningModule
from src.utils.schema.court import COURT_COORD_SCALE_XYZ


def score_scene(path: str) -> dict[str, object]:
    torch.set_num_threads(1)
    p = Path(path)
    meta = json.loads((p / "meta.json").read_text())
    gt = np.load(p / "position.npy", mmap_mode="r")
    indices = np.unique(np.linspace(0, len(gt) - 1, 64).astype(int))
    values: dict[str, list[np.ndarray]] = {
        k: [] for k in ["human_kp", "court_kp", "human_vis", "court_vis"]
    }
    views = meta["court_keypoint_views"]
    for v in range(4):
        order = np.argsort(views[v]["semantic_to_physical"])
        for key, suffix in [
            ("human_kp", "human_kp_uv"),
            ("court_kp", "court_kp_uv"),
            ("human_vis", "human_kp_vis"),
            ("court_vis", "court_kp_vis"),
        ]:
            a = np.load(p / f"cam_{v}_{suffix}.npy", mmap_mode="r")[indices].copy()
            if key.startswith("court"):
                a = a[:, order[:14]]
            values[key].append(a)
    tensors = [
        torch.from_numpy(np.stack(values[k])).float().unsqueeze(0)
        for k in ["human_kp", "court_kp", "human_vis", "court_vis"]
    ]
    _, valid, anchor = footpoint_prior(
        tensors[0],
        tensors[1],
        tensors[2],
        tensors[3],
        torch.zeros(1, 4, len(indices), dtype=torch.bool),
    )
    good = valid.any(1)[0].numpy()
    delta = (anchor[0, :, :2].numpy() - gt[indices, :2]) * np.array(
        COURT_COORD_SCALE_XYZ[:2]
    )
    error = np.linalg.norm(delta, axis=-1)
    if good.sum() < len(indices) // 2:
        raise ValueError(f"Insufficient ankle geometry: {p}")
    return {
        "scene": p.name,
        "mean_xy_m": float(error[good].mean()),
        "errors_xy_m": error[good].tolist(),
        "valid_fraction": float(good.mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    source = args.source.resolve()
    train = (source / "train.txt").read_text().splitlines()
    stats_path = out / "train_geometry.json"
    if not stats_path.exists():
        rows = []
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            for i, row in enumerate(
                executor.map(
                    score_scene,
                    [str(source / "scenes" / s) for s in train],
                    chunksize=8,
                )
            ):
                rows.append(row)
                if (i + 1) % 500 == 0:
                    print(f"scored {i + 1}/{len(train)}", flush=True)
        stats_path.write_text(json.dumps(rows))
    rows = json.loads(stats_path.read_text())
    if [r["scene"] for r in rows] != train:
        raise ValueError("Cached statistics do not match train split")
    scores = np.array([r["mean_xy_m"] for r in rows])
    hard = scores >= np.quantile(scores, 0.8)
    weights = 0.5 * np.ones(len(train)) / len(train) + 0.5 * hard / hard.sum()
    rng = np.random.default_rng(42)
    sampled = rng.choice(len(train), size=len(train), replace=True, p=weights)
    overlay = out / "sampled_scenes"
    overlay.mkdir(exist_ok=True)
    for name in ["scenes", "meta.json", "scenes_meta.json", "val.txt", "test.txt"]:
        dest = overlay / name
        if not dest.exists():
            dest.symlink_to(source / name)
    (overlay / "train.txt").write_text("\n".join(train) + "\n")
    (overlay / "sampling_weights.json").write_text(
        json.dumps(dict(zip(train, weights.tolist(), strict=True)))
    )
    bins = [0, 0.25, 0.5, 1.0, 2.0, float("inf")]
    hist = np.array(
        [np.histogram(r["errors_xy_m"], bins=bins)[0] for r in rows], dtype=float
    )
    hist = hist / hist.sum(1, keepdims=True)
    summary = {
        "seed": 42,
        "sampling": "0.5 natural + 0.5 top-20%-mean-horizontal-error scenes",
        "source": str(source),
        "train_examples": len(train),
        "hard_threshold_mean_xy_m": float(scores[hard].min()),
        "natural_hard_scene_fraction": float(hard.mean()),
        "sampled_hard_scene_fraction": float(hard[sampled].mean()),
        "frame_xy_bins_m": ["0-.25", ".25-.5", ".5-1", "1-2", "2+"],
        "natural_frame_fractions": hist.mean(0).tolist(),
        "resampled_frame_fractions": hist[sampled].mean(0).tolist(),
        "natural_mean_scene_xy_m": float(scores.mean()),
        "resampled_mean_scene_xy_m": float(scores[sampled].mean()),
        "validation_test_unchanged": True,
        "statistics_use_clean_observations": True,
    }
    (out / "sampling_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)
    torch.set_num_threads(2)
    checkpoint = torch.load(args.baseline, map_location="cpu", weights_only=False)
    config = OmegaConf.create(checkpoint["hyper_parameters"]["config"])
    config.model.name = "plcs_multiview_axial_foot_residual"
    config.data.scene_dir = "sampled_scenes"
    with open_dict(config.data):
        config.data.sampling_weights = "sampling_weights.json"
    config.training.compile.enabled = False
    config.training.trainer.max_epochs = 20
    config.training.trainer.check_val_every_n_epoch = 1
    config.training.trainer.enable_progress_bar = False
    config.training.learning_rate = 0.00005
    config.training.warmup_steps = 250
    config.training.checkpoint.save_top_k = 1
    config.training.early_stopping.enabled = True
    config.training.early_stopping.patience = 5
    config.training.qualitative_logging.enabled = False
    config.run.output_dir = "training"
    config.run.init_weights = None
    config.paths.project_root = str(Path.cwd())
    config.paths.data_root = str(out)
    config.paths.external_asset_root = str(source.parents[1])
    config.paths.output_root = str(out)
    config.paths.checkpoint_root = str(out)
    torch.manual_seed(42)
    model = PLCSLightningModule(config)
    state = checkpoint["state_dict"]
    # This explicit initialization conversion preserves all old trunk/rotation
    # weights but resets final absolute-position layers to zero residual.
    expected = model.state_dict()
    missing = set(expected) - set(state)
    unexpected = set(state) - set(expected)
    if unexpected or any(not k.startswith("model.geometry_embed.") for k in missing):
        raise ValueError(
            f"Unexpected initialization differences: {missing=}, {unexpected=}"
        )
    for key in expected:
        if key in state:
            expected[key] = state[key]
    for head in ["position_head", "aux_position_head"]:
        for suffix in ["weight", "bias"]:
            key = f"model.{head}.mlp.8.{suffix}"
            expected[key] = torch.zeros_like(expected[key])
    for suffix in ["weight", "bias"]:
        key = f"model.geometry_embed.2.{suffix}"
        expected[key] = torch.zeros_like(expected[key])
    model.load_state_dict(expected, strict=True)
    init = {
        k: checkpoint[k] for k in ["court_coordinate_normalization", "court_keypoints"]
    }
    init.update(
        state_dict=model.state_dict(),
        hyper_parameters={"config": OmegaConf.to_container(config, resolve=True)},
    )
    init_path = out / "residual_init.ckpt"
    torch.save(init, init_path)
    config.run.init_weights = "residual_init.ckpt"
    OmegaConf.save(config, out / "train.yaml")


if __name__ == "__main__":
    main()
