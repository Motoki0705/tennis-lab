"""Audit raw/refined pseudo-label consistency without repeating GPU inference."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from src.tennis_scene.archive import load_scene_result, save_scene_result
from src.tennis_scene.dataset_pipeline.quality import evaluate_reconstruction
from src.tennis_scene.dataset_pipeline.refinement import (
    RefinementSettings,
    check_label_coverage,
    refine_scene,
)
from src.tennis_scene.reference_pipeline.observations import sha256
from src.utils.io import save_json_atomic


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--court", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config-name", default="build_slcs_dataset")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Choose a new evaluation run directory")
    with initialize_config_dir(
        config_dir=str(Path("src/tennis_scene/configs").resolve()), version_base="1.3"
    ):
        cfg = compose(config_name=args.config_name)
    scene = load_scene_result(args.scene)
    with np.load(args.court) as data:
        matrices = data["homographies"]
    turns = list(cfg.view_half_turns) if cfg.coordinate_mode == "reference" else [False]
    before, _ = evaluate_reconstruction(scene, matrices, turns)
    settings = RefinementSettings.from_config(cfg.refinement)
    evidence = refine_scene(scene, matrices, settings)
    after, arrays = evaluate_reconstruction(scene, matrices, turns)
    args.output.mkdir(parents=True)
    save_json_atomic(
        {
            "before": before,
            "after": after,
            "evidence": evidence,
            "inputs": {
                "scene": {
                    "path": str(args.scene.resolve()),
                    "sha256": sha256(args.scene),
                },
                "court": {
                    "path": str(args.court.resolve()),
                    "sha256": sha256(args.court),
                },
            },
            "config": OmegaConf.to_container(cfg.refinement, resolve=True),
        },
        args.output / "metrics.json",
    )
    save_scene_result(scene, args.output / "scene.npz")
    np.savez_compressed(args.output / "quality_arrays.npz", allow_pickle=False, **arrays)
    check_label_coverage(evidence, settings)
    print(evidence)


if __name__ == "__main__":
    main()
