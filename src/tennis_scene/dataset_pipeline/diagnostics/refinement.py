"""Audit raw/refined pseudo-label consistency without repeating GPU inference."""

from __future__ import annotations

import numpy as np

from src.tennis_scene.archive import load_scene_result, save_scene_result
from src.tennis_scene.dataset_pipeline.quality import evaluate_reconstruction
from src.tennis_scene.dataset_pipeline.refinement import (
    check_label_coverage,
    refine_scene,
)
from src.tennis_scene.reference_pipeline.observations import sha256
from src.utils.io import save_json_atomic

from .configuration import RefinementEvaluationConfig


def evaluate(args: RefinementEvaluationConfig) -> None:
    if args.output.exists():
        raise FileExistsError("Choose a new evaluation run directory")
    scene = load_scene_result(args.scene)
    with np.load(args.court) as data:
        matrices = data["homographies"]
    turns = list(args.view_half_turns)
    before, _ = evaluate_reconstruction(scene, matrices, turns)
    settings = args.settings
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
            "config": args.refinement_config,
        },
        args.output / "metrics.json",
    )
    save_scene_result(scene, args.output / "scene.npz")
    np.savez_compressed(
        args.output / "quality_arrays.npz", allow_pickle=False, **arrays
    )
    check_label_coverage(evidence, settings)
    print(evidence)
