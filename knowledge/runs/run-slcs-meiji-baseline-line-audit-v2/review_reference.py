"""Record the parent's explicit RGB white-band selections after image review."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

SELECTED_PAIRS = {0: 0, 453: 0, 907: 1}


def line_y(endpoints: list[list[float]], x: float) -> float:
    (x0, y0), (x1, y1) = endpoints
    if x1 <= x0:
        raise ValueError("Expected endpoints with increasing x")
    return y0 + (x - x0) * (y1 - y0) / (x1 - x0)


def review(report: dict[str, Any]) -> dict[str, Any]:
    selected = []
    for frame in report["frames"]:
        index = frame["frame_index"]
        pair = next(
            p for p in frame["unadopted_edge_pairs"] if p["id"] == SELECTED_PAIRS[index]
        )
        selected.append({"frame_index": index, "pair": pair})
    low = max(item["pair"]["center_endpoints_px"][0][0] for item in selected)
    high = min(item["pair"]["center_endpoints_px"][1][0] for item in selected)
    if not low < high:
        raise ValueError("Selected white-band segments have no common x interval")
    # The executed probe computes the H slope from float32 court points.
    # Keep that operation's dtype when checking its saved comparison values.
    h = np.asarray(report["H_baseline_endpoints_px"], dtype=np.float32)
    slope = float((h[1, 1] - h[0, 1]) / (h[1, 0] - h[0, 0]))

    def h_y(x: float) -> float:
        return float(h[0, 1]) + (x - float(h[0, 0])) * slope

    rows = []
    for item in selected:
        pair = item["pair"]
        # Check arithmetic against the executed probe's three saved samples.
        for sample in pair["provisional_center_comparison"]:
            delta = h_y(sample["x"]) - line_y(
                pair["center_endpoints_px"], sample["x"]
            )
            if not math.isclose(
                delta, sample["H_minus_candidate_vertical_px"], abs_tol=1e-10
            ):
                raise ValueError("Stored comparison differs from recomputation")
        for x in (low, (low + high) / 2, high):
            reference_y = line_y(pair["center_endpoints_px"], x)
            delta = h_y(x) - reference_y
            rows.append(
                {
                    "frame_index": item["frame_index"],
                    "pair_id": pair["id"],
                    "edge_segment_ids": pair["edge_segment_ids"],
                    "x_px": x,
                    "white_band_center_y_px": reference_y,
                    "H_minus_white_band_vertical_px": delta,
                    "distance_to_H_line_px": abs(delta) / math.sqrt(1 + slope**2),
                }
            )
    values = [row["H_minus_white_band_vertical_px"] for row in rows]
    return {
        "status": "parent_reviewed_local_pixel_reference",
        "reviewer": "parent_codex",
        "date": "2026-09-19",
        "clip": report["clip"],
        "camera": report["camera"],
        "selected_pair_ids": SELECTED_PAIRS,
        "selection_reason": "In all three edge-pair sheets the chosen pair brackets the visible bright near-baseline stripe on the right side, without player occlusion. Original frame crops and full frames were also inspected. Dark shadow pairs were not selected.",
        "not_blind_to_H": True,
        "selection_did_not_minimize_H_residual": True,
        "H_slope_dtype": "float32, matching the executed probe",
        "selected_center_endpoints_px": [
            {
                "frame_index": item["frame_index"],
                "endpoints": item["pair"]["center_endpoints_px"],
            }
            for item in selected
        ],
        "common_x_interval_px": [low, high],
        "samples": rows,
        "vertical_difference_min_px": min(values),
        "vertical_difference_max_px": max(values),
        "vertical_difference_mean_px": sum(values) / len(values),
        "interpretation": "Positive difference means the saved projected baseline lies below the visible white-band center. This is a local RGB pixel diagnostic, not human ground truth, full-court error, or measured 3D accuracy.",
        "limitations": "One camera, three frames, and the common right-side segment only. Edge extraction and stripe thickness introduce uncertainty; no calibrated uncertainty interval is claimed. No calibration or dataset was changed. Candidate generation did not use H; the parent review saw its overlay.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = Path(__file__).with_name("results.json")
    result = review(json.loads(source.read_text()))
    result["results_sha256"] = hashlib.sha256(source.read_bytes()).hexdigest()
    result["review_script_sha256"] = hashlib.sha256(
        Path(__file__).read_bytes()
    ).hexdigest()
    with args.output.open("x") as handle:
        json.dump(result, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")
    print(
        json.dumps(
            {
                k: result[k]
                for k in (
                    "common_x_interval_px",
                    "vertical_difference_min_px",
                    "vertical_difference_max_px",
                    "vertical_difference_mean_px",
                )
            }
        )
    )


if __name__ == "__main__":
    main()
