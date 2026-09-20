"""One-run train-only velocity scale analysis; no DINO or model execution."""

import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf

from src.tasks.slcs.configuration import SLCSTrainingRuntimeConfig
from src.tasks.slcs.data.annotation import SLCSDataIndex
from src.tasks.slcs.data.dataset import SLCSWindowDataset
from src.tasks.slcs.data.splits import load_split_assignments
from src.utils.schema.court import COURT_COORD_SCALE_XYZ


def main() -> None:
    config_path = Path(sys.argv[1])
    runtime = SLCSTrainingRuntimeConfig.from_config(OmegaConf.load(config_path))
    data = runtime.data
    if data.overfit or data.pipeline.on_incomplete != "error":
        raise ValueError("Calibration requires held-out splits and complete labels")
    index = SLCSDataIndex.load(data.dataset_root)
    assignments = load_split_assignments(data.split_file, index)
    fps = {(r.video_id, r.clip_id): float(r.fps) for r in index.clips}
    dataset = SLCSWindowDataset(
        dataset_root=data.dataset_root,
        split_file=data.split_file,
        split="train",
        config=replace(data.pipeline, require_dino=False),
        augment=False,
        stride=data.pipeline.train_stride,
    )
    scale = np.asarray(COURT_COORD_SCALE_XYZ, dtype=np.float64)
    pairs: dict[tuple[str, str, str, int], tuple[np.ndarray, float]] = {}
    for i, meta in enumerate(dataset.metas):
        assert assignments[meta.video_id] == "train"
        sample = dataset[i]
        position = sample["target_ball_position"].numpy().astype(np.float64)
        frames = sample["frame_idx"].numpy()
        weight = sample["target_ball_weight"].numpy().astype(np.float64)
        valid = sample["target_ball_valid"].numpy() & ~sample["padding_mask"].numpy()
        pair_weight = np.minimum(weight[1:], weight[:-1])
        pair_valid = valid[1:] & valid[:-1] & (np.diff(frames) == 1) & (pair_weight > 0)
        frame_rate = fps[(meta.video_id, meta.clip_id)]
        assert np.isfinite(frame_rate) and frame_rate > 0
        for t in np.flatnonzero(pair_valid):
            key = (meta.video_id, meta.clip_id, meta.camera_id, int(frames[t]))
            velocity = (position[t + 1] - position[t]) * scale * frame_rate
            confidence = float(pair_weight[t])
            assert np.isfinite(velocity).all()
            if key in pairs:
                previous_velocity, previous_confidence = pairs[key]
                assert (
                    np.array_equal(velocity, previous_velocity)
                    and confidence == previous_confidence
                )
            else:
                pairs[key] = (velocity, confidence)
    values = list(pairs.values())
    speeds = np.linalg.norm(np.asarray([v for v, _ in values]), axis=-1)
    weights = np.asarray([w for _, w in values])
    distributions = []
    for threshold in (0.0, 0.25, 0.5, 0.75, 1.0):
        selected = speeds[weights >= threshold]
        distributions.append(
            {
                "min_pair_confidence": threshold,
                "pairs": int(selected.size),
                "speed_quantiles_mps": {
                    name: float(np.quantile(selected, quantile))
                    for name, quantile in zip(
                        ("min", "p25", "median", "p75", "p95", "p99", "max"),
                        (0, 0.25, 0.5, 0.75, 0.95, 0.99, 1),
                        strict=True,
                    )
                }
                if selected.size
                else None,
            }
        )
    high_confidence = speeds[weights >= 0.5]
    selected_scale = float(np.median(high_confidence))
    assert np.isfinite(selected_scale) and selected_scale > 0
    print(
        json.dumps(
            {
                "source_config": str(config_path),
                "split": "train",
                "train_videos": sorted(
                    video for video, split in assignments.items() if split == "train"
                ),
                "windows": len(dataset),
                "unique_positive_weight_camera_frame_pairs": len(pairs),
                "selected_scale_mps": selected_scale,
                "selection": "Unweighted median teacher speed among unique train camera-frame pairs with min endpoint confidence >= 0.5; no val/test data or DINO read.",
                "distributions": distributions,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
