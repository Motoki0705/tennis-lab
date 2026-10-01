"""Synthetic robustness benchmark and threshold selection of the ball side decision.

Example (from repository root; CPU only):
    .venv/bin/python -m src.tasks.court_side.scripts.benchmark_synthetic \
        --data-root /absolute/data --dataset blcs/single_object_camera_view_v2 \
        --output-root /absolute/outputs --experiment synthetic_blcs_v2 --run-id <run-id>

Scenes are ``--scenes`` of the dataset's sorted test split from ``--scene-offset``.
The run directory ``<output-root>/court_side/evaluate/<experiment>/<run-id>/``
must not exist. It receives ``conditions.json``, ``evidence.jsonl``
(threshold-free scores of every trial), ``thresholds.json`` (the grid) and
``report.json``: the selected thresholds and per-condition wrong/stop rates,
compared with the previous method (thresholds of one Meiji clip, no static
deduplication) on the same perturbed observations. ``--fixed-thresholds``
judges a held-out run with the thresholds selected by an earlier report
instead of selecting.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import numpy as np

from src.tasks.court_side.benchmark import (
    Perturbation,
    ThresholdGrid,
    judge,
    load_blcs_scene,
    make_trials,
    outcome_row,
    select_thresholds,
)
from src.tasks.court_side.hypothesis import CourtSideConfig
from src.utils.configuration import (
    BoundaryPathField,
    NonHydraPathBoundary,
    PathDirection,
    PathKind,
    PathResolver,
    PathRole,
    RuntimePathRoots,
)
from src.utils.configuration.output_layout import new_run_id, task_output_path

PATH_BOUNDARY = NonHydraPathBoundary(
    name="court_side.benchmark_synthetic",
    fields=(
        BoundaryPathField("dataset", PathRole.DATA, PathDirection.INPUT, PathKind.DIRECTORY, must_exist=True),
        BoundaryPathField("output", PathRole.OUTPUT, PathDirection.OUTPUT, PathKind.DIRECTORY),
    ),
)
# The method before this benchmark: thresholds of one Meiji clip, every repeated observation counted.
PREVIOUS = CourtSideConfig(reprojection_px=20., min_motion_px=0., min_frames=8, max_cost=.5, min_support=.5, min_margin=.1)
# Scoring of the selected method; the grid below replaces its decision thresholds.
SCORING = replace(PREVIOUS, min_motion_px=5.)
GRID = ThresholdGrid(max_cost=(.3, .4, .5, .6, .7, .8, .9), min_support=(.1, .2, .3, .4, .5, .6),
                     min_margin=(.02, .05, .1, .15, .2, .3), min_frames=(4, 8, 15, 30, 60))
NOMINAL = Perturbation("nominal")
CONDITIONS: tuple[Perturbation, ...] = (
    NOMINAL,
    *(replace(NOMINAL, name=f"missing_{r:.2f}", missing_rate=r) for r in (0., .3, .5, .7, .85)),
    *(replace(NOMINAL, name=f"false_{r:.2f}", false_rate=r) for r in (.05, .1, .2, .3, .5)),
    *(replace(NOMINAL, name=f"false_shared_{r:.2f}", false_rate=r, false_shared=True) for r in (.1, .3)),
    *(replace(NOMINAL, name=f"sync_{k}f", sync_offset_frames=k) for k in (1, 2, 4, 8)),
    *(replace(NOMINAL, name=f"pixel_sigma_{s:g}px", pixel_sigma_px=s) for s in (0., 5., 10.)),
    *(replace(NOMINAL, name=f"calibration_x{s:g}", calibration_scale=s) for s in (0., 2., 4.)),
    *(replace(NOMINAL, name=f"window_{w}f", window_frames=w) for w in (30, 60, 150)),
    replace(NOMINAL, name="cameras_4", cameras=4),
    replace(NOMINAL, name="combined", missing_rate=.3, false_rate=.1, sync_offset_frames=1, pixel_sigma_px=3.),
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--dataset", default="blcs/single_object_camera_view_v2", help="DATA-relative BLCS dataset")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--experiment", default="synthetic_blcs_v2")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--scenes", type=int, default=300)
    parser.add_argument("--scene-offset", type=int, default=0)
    parser.add_argument("--fixed-thresholds", type=Path, default=None, help="report.json whose selected thresholds are judged")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if not args.data_root.is_absolute() or not args.output_root.is_absolute():
        parser.error("--data-root and --output-root must be absolute")
    roots = RuntimePathRoots(project_root=args.output_root, data_root=args.data_root, checkpoint_root=args.data_root,
                             artifact_root=args.output_root, output_root=args.output_root, cache_root=args.output_root,
                             external_asset_root=args.data_root)
    resolver = PathResolver(roots)
    run = task_output_path("court_side", "evaluate", args.experiment, args.run_id or new_run_id())
    paths = PATH_BOUNDARY.validate({"dataset": resolver.resolve(PathRole.DATA, args.dataset),
                                    "output": resolver.resolve(PathRole.OUTPUT, run)}, resolver=resolver)
    dataset, output = paths.declared("dataset").path, paths.declared("output").path
    output.mkdir(parents=True, exist_ok=False)
    names = sorted((dataset / "test.txt").read_text().split())[args.scene_offset:args.scene_offset + args.scenes]
    if len(names) != args.scenes:
        raise ValueError(f"The test split has only {len(names)} scenes from offset {args.scene_offset}")
    fixed = None
    if args.fixed_thresholds is not None:
        selected_document = json.loads(args.fixed_thresholds.read_text())["selected"]
        fixed = CourtSideConfig(**{**selected_document, "height_range_m": tuple(selected_document["height_range_m"])})
    scenes = [load_blcs_scene(dataset / "scenes" / name) for name in names]
    (output / "conditions.json").write_text(json.dumps({"dataset": str(dataset), "scenes": names, "seed": args.seed,
        "scene_offset": args.scene_offset, "conditions": [asdict(c) for c in CONDITIONS], "grid": asdict(GRID),
        "previous": asdict(PREVIOUS), "scoring": asdict(SCORING), "fixed_thresholds": None if fixed is None else asdict(fixed)}, indent=1))
    if fixed is not None and (fixed.reprojection_px, fixed.min_motion_px) != (SCORING.reprojection_px, SCORING.min_motion_px):
        raise ValueError("Fixed thresholds were selected under a different scoring")
    rng = np.random.default_rng(args.seed)
    pairs = [make_trials(scene, condition, (SCORING, PREVIOUS), rng) for condition in CONDITIONS for scene in scenes]
    trials, previous_trials = [pair[0] for pair in pairs], [pair[1] for pair in pairs]
    with (output / "evidence.jsonl").open("w") as stream:
        for trial in trials:
            evidence = trial.evidence
            stream.write(json.dumps({"scene": trial.scene_id, "condition": trial.condition, "expected": list(trial.expected),
                "frames": evidence.frames, "pair_frames": evidence.pair_record(),
                "hypotheses": [asdict(h) for h in evidence.hypotheses]}) + "\n")
    if fixed is None:
        selected, table = select_thresholds(trials, GRID, SCORING)
        (output / "thresholds.json").write_text(json.dumps([{**asdict(row["config"]), "wrong": row["wrong"],
            "mean_stop_rate": row["mean_stop_rate"], "safe": row["safe"]} for row in table], indent=1))
    else:
        selected = fixed
    report: dict[str, Any] = {"selected": asdict(selected), "previous": asdict(PREVIOUS),
                              "selection": "fixed" if fixed is not None else "grid", "conditions": {}}
    for condition in CONDITIONS:
        items = [t for t in trials if t.condition == condition.name]
        previous_items = [t for t in previous_trials if t.condition == condition.name]
        report["conditions"][condition.name] = {"selected": outcome_row(judge(items, selected)),
                                                "previous": outcome_row(judge(previous_items, PREVIOUS))}
    (output / "report.json").write_text(json.dumps(report, indent=1))
    print(f"{'condition':22s} {'wrong':>7s} {'stop':>7s} | previous {'wrong':>7s} {'stop':>7s}")
    for name, row in report["conditions"].items():
        new, old = row["selected"], row["previous"]
        print(f"{name:22s} {new['wrong_rate']:7.3f} {new['stop_rate']:7.3f} | {'':8s} {old['wrong_rate']:7.3f} {old['stop_rate']:7.3f}")
    print(json.dumps({"selected": report["selected"], "output": str(output)}))


if __name__ == "__main__":
    main()
