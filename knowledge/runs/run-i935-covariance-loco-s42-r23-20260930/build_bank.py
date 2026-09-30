"""Use the frozen #936 consumer in this checkout; never access its worktree."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import types
from pathlib import Path

import numpy as np


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    bundle = Path(__file__).resolve().parent
    contract = json.loads((bundle / "consumer-contract.json").read_text())
    project = bundle.parents[2]
    source = subprocess.check_output(["git", "-C", str(project), "show", f"{contract['commit']}:{contract['path']}"])
    if hashlib.sha256(source).hexdigest() != contract["sha256"]:
        raise ValueError("Frozen consumer code differs")
    consumer = types.ModuleType("i936_frozen_calibration")
    sys.modules[consumer.__name__] = consumer
    exec(compile(source, contract["path"], "exec"), consumer.__dict__)
    evaluation = json.loads((args.evaluation / "manifest.json").read_text())
    if json.loads((args.evaluation / "run_state.json").read_text())["status"] != "complete":
        raise ValueError("C must be complete before D")
    scale_path = Path(evaluation["calibration_path"])
    if sha(scale_path) != evaluation["calibration_sha256"]:
        raise ValueError("Calibration identity differs")
    report = consumer.build_calibration(
        args.evaluation / "bank_input", args.output,
        status="anchored_e9_s42_covariance_scaled_calibration_fit_frames",
    )
    bank_path = args.output / "bank.npz"
    bank = consumer.load_calibration(bank_path, report["bank_sha256"])
    arrays = bank.arrays
    cut_count = 0
    checked_rows = 0
    for i, entry in enumerate(report["source_predictions"]):
        source_path = args.evaluation / "bank_input" / entry["path"]
        if sha(source_path) != entry["sha256"]:
            raise ValueError("Calibrated source prediction changed")
        rows = np.flatnonzero(arrays["source_artifact"] == i)
        with np.load(source_path, allow_pickle=False) as z:
            offsets = np.searchsorted(z["frame_index"], arrays["source_frame"][rows])
            np.testing.assert_array_equal(z["frame_index"][offsets], arrays["source_frame"][rows])
            # Preserve annotation segment boundaries in addition to frame gaps.
            for left, right in zip(range(len(rows) - 1), range(1, len(rows)), strict=True):
                if arrays["continues"][rows[left]] and z["segment_break"][offsets[right]]:
                    arrays["continues"][rows[left]] = False
                    cut_count += 1
            for name, actual in (
                ("error_uv", z["means"][offsets] - z["target_uv"][offsets, None]),
                ("scale_tril_uv", z["scale_tril"][offsets]),
                ("mixture_logits", z["mixture_logits"][offsets]),
                ("presence_logits", z["presence_logits"][offsets]),
            ):
                np.testing.assert_array_equal(arrays[name][rows], actual)
            assert z["position_valid"][offsets].all()
            if entry["condition"] == "evidence_gap":
                assert z["gap_mask"][offsets].all()
            checked_rows += len(rows)
    consumer.CalibrationBank(arrays)
    np.savez_compressed(bank_path, **arrays)
    report["bank_sha256"] = sha(bank_path)
    report["consumer_contract"] = contract
    report["covariance_calibration"] = {
        "path": str(scale_path), "sha256": sha(scale_path),
        "multiplier": json.loads(scale_path.read_text())["covariance_multiplier"],
        "fit_frame_reuse": "Bank uses the full-six-clip fitted scale on those same fit frames; NOT OOF performance.",
        "oof_metrics": str(args.evaluation / "metrics.json"),
        "oof_metrics_sha256": sha(args.evaluation / "metrics.json"),
    }
    report["continuity"] = {
        "rule": "consecutive source frames within one camera/condition/artifact, without a new annotation segment",
        "additional_segment_cuts": cut_count,
    }
    report["limits"] = [
        "covariance-only calibration; means, component logits and presence unchanged",
        "scale fitted on the six calibration clips that also supply this empirical bank; performance is reported separately by LOCO",
        "e9 detector and anchored_12k seed42; seed43/44 confirmation pending; pipeline default unchanged",
        "positive-only presence, no negative calibration; out-of-frame logit remains an explicit consumer hypothesis",
        "camera errors resampled independently; up to 16-frame blocks; longer gaps extrapolate",
        "synthetic translation/head clipping may change errors; no new synthetic generation or real 3D evaluation",
    ]
    report_path = args.output / "calibration.json"
    report_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    restored = consumer.load_calibration(bank_path, report["bank_sha256"])
    settings = consumer.with_calibration_report({"calibration": {}}, report_path)
    assert restored.components == 4 and settings["components_per_camera"] == 4
    assert settings["max_components"] == 125 and checked_rows == report["rows"]
    draws = []
    for camera in range(3):
        for condition in (False, True):
            drawn = restored.draw_rows(camera, np.full(128, condition, dtype=bool),
                                       np.random.default_rng(93523), block_frames=16)
            assert (arrays["camera_index"][drawn] == camera).all()
            assert (arrays["condition_index"][drawn] == condition).all()
            draws.append({"camera": camera, "condition": condition, "frames": len(drawn)})
    verification = {"consumer_contract": contract, "rows_equal_to_sources": checked_rows,
                    "source_predictions": len(report["source_predictions"]), "components": restored.components,
                    "segment_cuts": cut_count, "draw_contract_checks": draws,
                    "bank_path": str(bank_path), "bank_sha256": sha(bank_path),
                    "report_path": str(report_path), "report_sha256": sha(report_path),
                    "calibration_path": str(scale_path), "calibration_sha256": sha(scale_path),
                    "output_bytes": sum(p.stat().st_size for p in args.output.rglob("*") if p.is_file())}
    (args.output / "verification.json").write_text(json.dumps(verification, indent=2) + "\n")
    print(json.dumps(verification), flush=True)


if __name__ == "__main__":
    main()
