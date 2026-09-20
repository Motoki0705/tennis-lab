"""Recompute and verify hard-trimmed KP/LINE registration from saved inference."""

from __future__ import annotations

import json
import sys
from dataclasses import asdict
from pathlib import Path

import cv2
import numpy as np
import scipy
from common import REPO, ROOT, sha256, sources, write_json

sys.path.insert(0, str(REPO))
from src.tasks.court_detection.geometry.confidence_homography import (  # noqa: E402
    PROSAC_OPTIONS,
)
from src.tasks.court_detection.geometry.hybrid_homography import (  # noqa: E402
    DEFAULT_HYBRID_CONFIG,
    estimate_hybrid_homography,
)
from src.tasks.court_detection.geometry.line_evidence import (  # noqa: E402
    LineEvidence,
    line_residuals,
)

BUNDLE = ROOT / "evidence/homography"
IMPLEMENTATION = REPO / "src/tasks/court_detection/geometry/confidence_homography.py"
CONFIG = DEFAULT_HYBRID_CONFIG
THRESHOLD_DIAGONAL_RATIO = CONFIG.threshold_diagonal_ratio
MIN_SCORE = CONFIG.min_score
EDGES = np.asarray(
    [(0, 1), (2, 3), (0, 2), (1, 3), (4, 5), (6, 7), (8, 9), (10, 11), (12, 13)]
)


def contract() -> dict:
    return {
        "schema": "court_hybrid_homography_v2",
        "opencv_version": cv2.__version__,
        "numpy_version": np.__version__,
        "scipy_version": scipy.__version__,
        "implementation_sha256": {
            name: sha256(IMPLEMENTATION.parent / name)
            for name in (
                "confidence_homography.py",
                "hybrid_homography.py",
                "line_evidence.py",
            )
        },
        "driver_sha256": sha256(Path(__file__)),
        "template_sha256": sha256(BUNDLE / "template.json"),
        "input_manifest_sha256": sha256(ROOT / "evidence/inputs.json"),
        "parameters": {"hybrid": asdict(CONFIG), "prosac": PROSAC_OPTIONS},
        "edges": EDGES.tolist(),
    }


def fit_prediction(
    points: np.ndarray,
    scores: np.ndarray,
    width: int,
    height: int,
    line_probability: np.ndarray,
) -> dict:
    template = np.asarray(
        json.loads((BUNDLE / "template.json").read_text())["points_xy"], dtype=float
    )
    threshold = THRESHOLD_DIAGONAL_RATIO * np.hypot(width, height)
    result = estimate_hybrid_homography(
        template,
        points,
        scores,
        line_probability,
        edges=EDGES,
        image_size_hw=(height, width),
        config=CONFIG,
    )
    baseline = result.kp_only
    top4 = baseline.ranked_indices[:4]
    top4_collinear = (
        len(top4) == 4
        and np.linalg.matrix_rank(template[top4] - template[top4].mean(axis=0)) < 2
    )
    stages = {}
    evidence = None
    if result.line_selection is not None:
        evidence = LineEvidence.from_probability(
            line_probability,
            (height, width),
            threshold=CONFIG.line_probability_threshold,
            max_observations=CONFIG.max_line_observations,
        )
    if baseline.matrix is not None:
        line = (
            None
            if evidence is None
            else line_residuals(
                baseline.projected,
                EDGES,
                evidence,
                threshold,
                samples_per_line=CONFIG.samples_per_line,
            )[1]
        )
        stages["kp_only"] = {
            "matrix": baseline.matrix.tolist(),
            "aligned_kp": baseline.projected.tolist(),
            "selected": baseline.inliers.tolist(),
            "line": line,
        }
    for name, hypothesis in (
        ("line_selection", result.line_selection),
        ("hybrid", result.best),
    ):
        if hypothesis is not None:
            stages[name] = {
                "matrix": hypothesis.matrix.tolist(),
                "aligned_kp": hypothesis.projected.tolist(),
                "selected": hypothesis.selected.tolist(),
                "line": hypothesis.line,
                "objective": hypothesis.score,
                "selection_history": [
                    list(indices) for indices in hypothesis.selection_history
                ],
            }
    line_distances = (
        np.full(len(points), np.nan)
        if evidence is None
        else evidence.distances(np.nan_to_num(points, nan=-1e5))
    )
    reasons = []
    for i in range(len(points)):
        if result.selected[i]:
            reason = "selected"
        elif not np.isfinite(points[i]).all():
            reason = "invalid_coordinate"
        elif scores[i] < MIN_SCORE:
            reason = "low_score"
        elif result.matrix is None:
            reason = "no_accepted_homography"
        elif result.residuals_px[i] > threshold:
            reason = "reprojection_outlier"
        elif line_distances[i] > threshold:
            reason = "unsupported_by_line"
        else:
            reason = "selection_cap"
        reasons.append(reason)
    return {
        "status": result.status,
        "matrix": None if result.matrix is None else result.matrix.tolist(),
        "aligned_kp": None if result.matrix is None else result.projected.tolist(),
        "ranked_indices": baseline.ranked_indices.tolist(),
        "top4_indices": top4.tolist(),
        "top4_scores": scores[top4].tolist(),
        "top4_template_collinear": bool(top4_collinear),
        "fit_inliers": result.selected.tolist(),
        "inliers": result.selected.tolist(),
        "residuals_px": [
            None if not np.isfinite(x) else float(x) for x in result.residuals_px
        ],
        "kp_line_distance_px": [
            None if not np.isfinite(x) else float(x) for x in line_distances
        ],
        "kp_rejection_reasons": reasons,
        "threshold_px": float(threshold),
        "inlier_count": int(result.selected.sum()),
        "inlier_rms_px": float(
            np.sqrt(np.mean(result.residuals_px[result.selected] ** 2))
        )
        if result.selected.any()
        else None,
        "candidate_count": result.candidate_count,
        "refined_count": result.refined_count,
        "alternative_score_gap": result.alternative_score_gap,
        "stages": stages,
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
                data["raw_kp"],
                data["kp_scores"],
                record["width"],
                record["height"],
                data["line_probability"],
            )
        result["source_sha256"] = digest
        images[record["id"]] = result
    write_json(
        BUNDLE / "results.json",
        {
            **contract(),
            "method": "PROSAC plus ranked nondegenerate 4-point candidates; bidirectional LINE selection; alternating hard KP selection and coarse/fine joint optimization.",
            "scope": "CPU postprocessing of immutable original KP/scores/LINE; no new neural forward or training. LINE support and selected-KP residuals are internal agreement, not GT accuracy. One shared configuration for all photos.",
            "images": images,
        },
    )


def read_results() -> dict:
    result = json.loads((BUNDLE / "results.json").read_text())
    expected = contract()
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
                data["raw_kp"],
                data["kp_scores"],
                record["width"],
                record["height"],
                data["line_probability"],
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
        ("本モデル：KP+LINE H生成", status),
        ("本モデル：KP損失の採用点 / 14", counts),
        ("本モデル：採用KP残差RMS [px]", rms),
    ]
    for name, label in (
        ("kp_only", "KPのみ"),
        ("line_selection", "LINEで候補選択"),
        ("hybrid", "KP+LINE共同最適化"),
    ):
        values = []
        for row in rows:
            line = row["stages"].get(name, {}).get("line")
            values.append(
                "--"
                if line is None
                else f"{100 * line['forward_support']:.1f}/{100 * line['reverse_support']:.1f}"
            )
        entries.append((label + "：線支持率 [\\%]", values))
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
