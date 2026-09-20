"""Recompute confidence-aware H from immutable saved KP predictions on CPU."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np
from common import REPO, ROOT, sha256, sources, write_json

sys.path.insert(0, str(REPO))
from src.tasks.court_detection.geometry.confidence_homography import (  # noqa: E402
    PROSAC_OPTIONS,
    estimate_confidence_homography,
)

BUNDLE = ROOT / "evidence/homography"
IMPLEMENTATION = REPO / "src/tasks/court_detection/geometry/confidence_homography.py"
THRESHOLD_DIAGONAL_RATIO = 0.005
MIN_SCORE = 0.05


def fit_prediction(
    points: np.ndarray, scores: np.ndarray, width: int, height: int
) -> dict:
    template = np.asarray(
        json.loads((BUNDLE / "template.json").read_text())["points_xy"]
    )
    threshold = THRESHOLD_DIAGONAL_RATIO * float(np.hypot(width, height))
    result = estimate_confidence_homography(
        template,
        points,
        scores,
        reprojection_threshold_px=threshold,
        min_score=MIN_SCORE,
    )
    top = result.ranked_indices[:4]
    return {
        "status": result.status,
        "matrix": None if result.matrix is None else result.matrix.tolist(),
        "aligned_kp": None if result.matrix is None else result.projected.tolist(),
        "ranked_indices": result.ranked_indices.tolist(),
        "top4_indices": top.tolist(),
        "top4_scores": scores[top].tolist(),
        "top4_template_collinear": bool(
            len(top) == 4
            and np.linalg.matrix_rank(template[top] - template[top].mean(axis=0)) < 2
        ),
        "fit_inliers": result.fit_inliers.tolist(),
        "inliers": result.inliers.tolist(),
        "residuals_px": [
            float(v) if np.isfinite(v) else None for v in result.residuals_px
        ],
        "threshold_px": threshold,
        "inlier_count": int(result.inliers.sum()),
        "inlier_rms_px": None
        if not result.inliers.any()
        else float(np.sqrt(np.mean(result.residuals_px[result.inliers] ** 2))),
    }


def collect() -> None:
    metadata = json.loads((ROOT / "evidence/inference_both.json").read_text())
    images = {}
    for record in sources():
        path = ROOT / f"evidence/predictions/{record['id']}_ours.npz"
        digest = sha256(path)
        if digest != metadata["prediction_sha256"][path.name]:
            raise ValueError("Original inference prediction hash changed")
        with np.load(path) as data:
            result = fit_prediction(
                data["raw_kp"], data["kp_scores"], record["width"], record["height"]
            )
        result["source_sha256"] = digest
        images[record["id"]] = result
    write_json(
        BUNDLE / "results.json",
        {
            "schema": "court_confidence_homography_v1",
            "opencv_version": cv2.__version__,
            "implementation_sha256": sha256(IMPLEMENTATION),
            "driver_sha256": sha256(Path(__file__)),
            "template_sha256": sha256(BUNDLE / "template.json"),
            "input_manifest_sha256": sha256(ROOT / "evidence/inputs.json"),
            "threshold_diagonal_ratio": THRESHOLD_DIAGONAL_RATIO,
            "method": "Stable descending KP score; USAC PROSAC + MSAC; inner LO; explicit inlier-only least-squares/LM refit; reclassify final inliers.",
            "parameters": {"min_score": MIN_SCORE, **PROSAC_OPTIONS},
            "scope": "Postprocessing only; original model predictions and all checkpoint hashes unchanged. Residuals measure self-consistency, not accuracy against GT. No photograph-specific threshold tuning.",
            "images": images,
        },
    )


def read_results() -> dict:
    result = json.loads((BUNDLE / "results.json").read_text())
    expected = {
        "schema": "court_confidence_homography_v1",
        "opencv_version": cv2.__version__,
        "implementation_sha256": sha256(IMPLEMENTATION),
        "driver_sha256": sha256(Path(__file__)),
        "template_sha256": sha256(BUNDLE / "template.json"),
        "input_manifest_sha256": sha256(ROOT / "evidence/inputs.json"),
        "threshold_diagonal_ratio": THRESHOLD_DIAGONAL_RATIO,
        "parameters": {"min_score": MIN_SCORE, **PROSAC_OPTIONS},
    }
    for key, value in expected.items():
        if result[key] != value:
            raise ValueError(f"Stale confidence homography evidence: {key}")
    if list(result["images"]) != [r["id"] for r in sources()]:
        raise ValueError("Confidence homography image set differs")
    for record in sources():
        item = result["images"][record["id"]]
        path = ROOT / f"evidence/predictions/{record['id']}_ours.npz"
        if sha256(path) != item["source_sha256"]:
            raise ValueError("Confidence homography source prediction changed")
        with np.load(path) as data:
            recalculated = fit_prediction(
                data["raw_kp"], data["kp_scores"], record["width"], record["height"]
            )
        for key, value in recalculated.items():
            if item[key] != value:
                raise ValueError(
                    f"Confidence homography result differs: {record['id']}/{key}"
                )
    return result


def table_text(results: dict) -> str:
    rows = list(results["images"].values())
    status = ["可" if r["status"] == "ok" else "不可" for r in rows]
    counts = [str(r["inlier_count"]) for r in rows]
    rms = [
        "--" if r["inlier_rms_px"] is None else f"{r['inlier_rms_px']:.2f}"
        for r in rows
    ]
    baseline_counts, baseline_h, argmax_h = [], [], []
    for record in sources():
        with np.load(
            ROOT / f"evidence/predictions/{record['id']}_baseline.npz"
        ) as data:
            baseline_counts.append(str(np.isfinite(data["raw_kp"]).all(1).sum()))
            baseline_h.append("可" if data["homography_found"] else "不可")
            argmax_h.append("可" if data["argmax_homography_found"] else "不可")
    entries = [
        ("TCD公式：検出点数 / 14", baseline_counts),
        ("TCD公式：H生成", baseline_h),
        ("TCD argmax：H生成", argmax_h),
        ("本モデル：PROSAC H生成", status),
        ("本モデル：インライア / 14", counts),
        ("本モデル：残差RMS [px]", rms),
    ]
    lines = [r"\begin{tabular}{@{}lrrrr@{}}\toprule", r" & A & B & C & D\\\midrule"]
    lines += [label + " & " + " & ".join(values) + r"\\" for label, values in entries]
    lines += [r"\bottomrule\end{tabular}"]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    collect()
    results = read_results()
    (ROOT / "tables/homography.tex").write_text(table_text(results))
    for key, item in results["images"].items():
        print(key, item["status"], item["inlier_count"], item["inlier_rms_px"])
