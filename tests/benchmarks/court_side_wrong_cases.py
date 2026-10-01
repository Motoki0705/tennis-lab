"""Reconstruct the three run-29 wrong decisions down to cameras and point rows."""
from __future__ import annotations

import argparse
import copy
import json
from dataclasses import asdict, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from omegaconf import OmegaConf

from src.tasks.ball_refiner.refiner_2d.confidence import PointConfidenceRule
from src.tasks.court_side.benchmark import load_blcs_scene
from src.tasks.court_side.hypothesis import (
    CourtSideConfig,
    collect_side_evidence,
    distinct_observation_frames,
)
from src.tasks.court_side.scripts.benchmark_synthetic import CONDITIONS
from src.utils.geometry.multiview_consistency import score_multiview_points

if TYPE_CHECKING:
    from tests.benchmarks.court_side_confidence import replay_mask
    from tests.benchmarks.court_side_correlated import capture, record
else:
    # Standalone benchmark scripts use sibling imports: the environment also
    # contains an unrelated installed ``tests`` package.
    from court_side_confidence import replay_mask
    from court_side_correlated import capture, record


def analyse(dataset: Path, confidence: Path, original: Path, previous: Path, rule_path: Path, output: Path) -> None:
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    selected = json.loads(original.read_text())["selected"]
    cfg = CourtSideConfig(**{**selected, "height_range_m": tuple(selected["height_range_m"])})
    rule = PointConfidenceRule(**dict(OmegaConf.load(rule_path)))
    targets = {(r["condition"], r["scene"]): r for r in json.loads(previous.read_text())}
    blocks = []
    for clip in range(1, 12):
        cameras = []
        for camera in range(3):
            with np.load(confidence / f"clip_{clip:03d}-cam{camera}.npz", allow_pickle=False) as data:
                cameras.append((data["presence"][::2], data["area"][::2]))
        blocks.append((np.stack([c[0] for c in cameras]), np.stack([c[1] for c in cameras])))
    names = sorted((dataset / "test.txt").read_text().split())[600:1000]
    scenes = [load_blcs_scene(dataset / "scenes" / name) for name in names]
    rng, confidence_rng = np.random.default_rng(1), np.random.default_rng(29001)
    findings = []
    for condition in CONDITIONS:
        for scene in scenes:
            frames = min(condition.window_frames, len(scene.ball_xyz) - condition.sync_offset_frames)
            mask = replay_mask(blocks, frames=frames, views=condition.cameras, size=scene.image_size, rule=rule, rng=confidence_rng)
            before = copy.deepcopy(rng)
            trial, obs = capture(scene, condition, cfg, rng, score=False)
            key = (condition.name, scene.scene_id)
            if key not in targets:
                continue
            _, clean = capture(scene, replace(condition, false_rate=0., pixel_sigma_px=0.), cfg, copy.deepcopy(before), score=False)
            _, noisy = capture(scene, replace(condition, false_rate=0.), cfg, before, score=False)
            if condition.sync_offset_frames:
                raise ValueError("This three-case diagnostic requires the recorded zero-sync-offset cases")
            physical = next(c for c in scene.cameras if c.camera_id == obs["cameras"][0].camera_id)
            projected = physical.project(scene.ball_xyz)[0].astype(np.float32)
            starts = [i for i in range(len(projected) - frames + 1)
                      if np.array_equal(projected[i:i + frames], clean["uv"][0])]
            if len(starts) != 1:
                raise ValueError("Source-frame window could not be uniquely reconstructed")
            source_frames = np.arange(starts[0], starts[0] + frames)
            false = np.any(obs["uv"] != noisy["uv"], axis=-1)
            error = np.linalg.norm(obs["uv"] - clean["uv"], axis=-1)
            summary: dict[str, Any] = {"condition": key[0], "scene": key[1], "expected": trial.expected,
                "camera_ids": [c.camera_id for c in obs["cameras"]], "paths": {}}
            arrays = {"uv": obs["uv"], "truth_uv": clean["uv"], "visible": obs["visible"], "filter_keep": mask,
                      "synthetic_false": false, "error_px": error, "source_frames": source_frames}
            for label, visible in (("original", obs["visible"]), ("filtered", obs["visible"] & mask)):
                evidence = collect_side_evidence(obs["cameras"], obs["reference"], obs["uv"], visible, obs["config"])
                current = record(replace(trial, evidence=evidence), label)
                # JSON-equivalence checks the original wrong-case record, including
                # every hypothesis float and pair count (new camera field excluded).
                current.pop("cameras")
                if json.loads(json.dumps(current)) != targets[key][label]:
                    raise ValueError("Run-29 wrong-case replay differs")
                distinct = distinct_observation_frames(obs["uv"], visible, obs["config"].min_motion_px)
                at = np.flatnonzero(distinct & (visible.sum(0) >= 2))
                camera_rows = []
                for i, camera in enumerate(obs["cameras"]):
                    used = np.zeros(frames, bool)
                    used[at] = visible[i, at]
                    camera_rows.append({"camera": camera.camera_id, "visible": int(visible[i].sum()),
                        "distinct_multiview_observations": int(used.sum()),
                        "false_points": int((used & false[i]).sum()),
                        "good_points_le20px": int((used & (error[i] <= 20)).sum()),
                        "point_frames": np.flatnonzero(used).tolist(), "source_frames": source_frames[used].tolist()})
                hypothesis_rows = []
                for hypothesis in evidence.hypotheses:
                    turned_cameras = tuple(c.half_turned(t) for c, t in zip(obs["cameras"], hypothesis.view_half_turns, strict=True))
                    points = score_multiview_points(obs["uv"][:, at], visible[:, at], turned_cameras,
                        threshold_px=obs["config"].reprojection_px, bounds=cfg.bounds)
                    hypothesis_rows.append({**asdict(hypothesis), "supported_frame_ids": at[points.support].tolist(),
                        "supported_with_false": int((points.support & (false[:, at] & visible[:, at]).any(0)).sum()),
                        "supported_without_false": int((points.support & ~(false[:, at] & visible[:, at]).any(0)).sum())})
                arrays[f"{label}_distinct"] = distinct
                summary["paths"][label] = {"frames": evidence.frames, "pair_frames": evidence.pair_record(),
                    "cameras": camera_rows, "hypotheses": hypothesis_rows,
                    "margin": evidence.hypotheses[1].cost - evidence.hypotheses[0].cost}
            filename = f"{key[0]}-{key[1]}.npz"
            np.savez_compressed(output / filename, **arrays)
            summary["points_file"] = filename
            findings.append(summary)
    if len(findings) != 3:
        raise ValueError("Must reproduce all three previous wrong cases")
    (output / "analysis.json").write_text(json.dumps(findings, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("dataset", "confidence", "original", "previous", "rule", "output"):
        parser.add_argument(f"--{key}", type=Path, required=True)
    args = parser.parse_args()
    analyse(args.dataset, args.confidence, args.original, args.previous, args.rule, args.output)
