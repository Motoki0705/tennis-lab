"""Frame-weighted location metrics and validation-only absence calibration."""

from __future__ import annotations

from fractions import Fraction
from typing import Any

import numpy as np

from src.tasks.ball_detection.data.windows.all_frames import EDGE_BINS, LABEL_KINDS

CALIBRATION_METHOD = "minimize_balanced_fpr_fnr; tie_lower_fpr_then_higher_threshold; tied_scores_indivisible"


def present(score: float, calibration: dict[str, Any]) -> bool:
    mode = calibration["mode"]
    if mode == "all_present":
        return True
    if mode == "all_absent":
        return False
    if mode != "threshold" or not 0 <= calibration["threshold"] <= 1:
        raise ValueError("Invalid heatmap absence threshold/mode")
    return bool(score >= calibration["threshold"])


def calibrate(records: list[dict[str, Any]], *, split: str) -> dict[str, Any]:
    if split != "val":
        raise ValueError("Absence threshold calibration is only allowed on validation")
    known = sorted((r for r in records if r["kind"] != "masked"), key=lambda r: r["score"])
    positives = sum(r["kind"] == "observed" for r in known)
    negatives = len(known) - positives
    if not positives or not negatives:
        raise ValueError("Validation needs both observed and confirmed-negative frames for calibration")
    fp, fn = negatives, 0
    best_key: tuple[Fraction, Fraction, float] | None = None
    best: dict[str, Any] = {}

    def consider(mode: str, threshold: float | None, order: float) -> None:
        nonlocal best, best_key
        fpr, fnr = Fraction(fp, negatives), Fraction(fn, positives)
        key = ((fpr + fnr) / 2, fpr, -order)
        if best_key is None or key < best_key:
            best_key = key
            best = dict(mode=mode, threshold=threshold, method=CALIBRATION_METHOD, split="val",
                        balanced_error=float(key[0]), observed=positives, negatives=negatives,
                        false_positive=fp, false_negative=fn, true_positive=positives - fn, true_negative=negatives - fp)

    consider("all_present", None, -1.)
    index = 0
    while index < len(known):
        score = known[index]["score"]
        while index < len(known) and known[index]["score"] == score:
            positive = known[index]["kind"] == "observed"
            fn += int(positive)
            fp -= int(not positive)
            index += 1
        if index < len(known):
            consider("threshold", known[index]["score"], known[index]["score"])
        else:
            consider("all_absent", None, 2.)
    return best


def frame_metrics(records: list[dict[str, Any]], calibration: dict[str, Any], *, distance_px: float) -> dict[str, Any]:
    errors = [r["error_px"] for r in records if r["kind"] == "observed"]
    counts = {kind: sum(r["kind"] == kind for r in records) for kind in LABEL_KINDS}
    fp = {kind: sum(present(r["score"], calibration) for r in records if r["kind"] == kind)
          for kind in ("reviewed_empty", "out_of_frame")}
    fn = sum(not present(r["score"], calibration) for r in records if r["kind"] == "observed")
    negatives = counts["reviewed_empty"] + counts["out_of_frame"]
    return dict(
        support=dict(frames=len(records), scored_frames=len(records) - counts["masked"], **counts),
        mean_error_px=float(np.mean(errors)) if errors else None,
        median_error_px=float(np.median(errors)) if errors else None,
        p95_error_px=float(np.quantile(errors, .95, method="linear")) if errors else None,
        within_distance_rate=sum(e <= distance_px for e in errors) / len(errors) if errors else None,
        distance_threshold_source_px=distance_px,
        false_negative=fn, observed_fnr=fn / counts["observed"] if counts["observed"] else None,
        false_positive=sum(fp.values()), negative_fpr=sum(fp.values()) / negatives if negatives else None,
        by_negative_kind={kind: dict(support=counts[kind], false_positive=fp[kind],
                                   fpr=fp[kind] / counts[kind] if counts[kind] else None,
                                   observed_fnr=None, mean_error_px=None, median_error_px=None, p95_error_px=None)
                          for kind in fp},
    )


def metrics_report(records: list[dict[str, Any]], calibration: dict[str, Any], *, distance_px: float) -> dict[str, Any]:
    report = frame_metrics(records, calibration, distance_px=distance_px)
    for field, group in (("source", "by_source"), ("native_fps", "by_fps"), ("source_group", "by_source_group"),
                         ("natural_occlusion", "by_natural_occlusion")):
        report[group] = {str(value): frame_metrics([r for r in records if r[field] == value], calibration, distance_px=distance_px)
                         for value in sorted({r[field] for r in records}, key=str)}
    report["by_edge_distance"] = {str(name): frame_metrics([r for r in records if r["edge_bin"] == i], calibration, distance_px=distance_px)
                                  for i, name in enumerate(EDGE_BINS)}
    return report
