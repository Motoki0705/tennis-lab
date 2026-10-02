"""Reconcile every before metric, immutable field, CV fold and published digest."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    before = json.loads((args.source / "metrics.json").read_text())
    after = json.loads((args.evaluation / "metrics.json").read_text())
    checked_metrics = 0
    for key, value in after.items():
        if key.startswith("before/"):
            old = before[key.replace("before/", "variant/", 1)]
            assert all(old[k] == v for k, v in value.items()), key
            checked_metrics += 1
    original = json.loads((args.source / "manifest.json").read_text())
    entries = {(a["clip_id"], a["condition"]): a for a in original["artifacts"]}
    manifest = json.loads((args.evaluation / "manifest.json").read_text())
    fit = json.loads((args.evaluation / "fit.json").read_text())
    held_rows = defaultdict(list)
    checked_frames = 0
    for entry in manifest["artifacts"]:
        identity = entry["clip_id"], entry["condition"]
        path = args.evaluation / entry["path"]
        original_path = args.source / entries[identity]["path"]
        assert sha(path) == entry["sha256"] and sha(original_path) == entries[identity]["sha256"]
        with np.load(path, allow_pickle=False) as current, np.load(original_path, allow_pickle=False) as previous:
            for key in ("means", "mixture_logits", "presence_logits", "target_uv", "target_reason",
                        "gap_mask", "presence", "presence_valid", "frame_index", "pts", "error_px", "presence_nll"):
                np.testing.assert_array_equal(current[key], previous[key])
            a, b = (z["scale_tril"].astype(np.float64) for z in (current, previous))
            np.testing.assert_allclose(a @ a.swapaxes(-1, -2),
                                       entry["covariance_multiplier"] * (b @ b.swapaxes(-1, -2)), rtol=3e-7, atol=1e-12)
            if entry["half"] == "calibration":
                group = entry["clip_id"].rsplit("/", 1)[0]
                fold = fit["folds"][group]
                assert group not in fold["groups"] and len(fold["groups"]) == 5
                assert entry["covariance_multiplier"] == fold["covariance_multiplier"]
                mask = previous["target_reason"] == 0
                if entry["condition"] == "evidence_gap":
                    mask &= previous["gap_mask"]
                held_rows[group, entry["condition"]].append(
                    np.stack([previous["nll_px"][mask], current["nll_px"][mask]], axis=1))
            else:
                assert entry["covariance_multiplier"] == fit["full"]["covariance_multiplier"]
            checked_frames += len(current["frame_index"])
    macro = {"/".join(key): np.concatenate(parts).mean(0).tolist() for key, parts in held_rows.items()}
    # Each temporal clip and its two conditions receives equal weight.
    balanced = np.asarray(list(macro.values())).mean(0)
    artifact = Path(manifest["calibration_path"])
    assert sha(artifact) == manifest["calibration_sha256"]
    source_code = ["src/tasks/ball_refiner/evaluation/calibration_fit.py",
                   "src/tasks/ball_refiner/evaluation/covariance_calibration.py",
                   "src/tasks/ball_refiner/refiner_2d/calibration.py"]
    project = Path(__file__).resolve().parents[3]
    result = {"status": "verified", "unchanged_before_strata": checked_metrics,
              "npz_files": len(manifest["artifacts"]), "frame_condition_rows": checked_frames,
              "held_out_clip_groups": len(fit["folds"]), "clip_condition_nll_px_before_after": macro,
              "clip_condition_balanced_oof_nll_px_before_after": balanced.tolist(),
              "calibration_sha256": sha(artifact), "evaluation_manifest_sha256": sha(args.evaluation / "manifest.json"),
              "source_code_sha256": {p: sha(project / p) for p in source_code},
              "evaluation_output_bytes": sum(p.stat().st_size for p in args.evaluation.rglob("*") if p.is_file())}
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
