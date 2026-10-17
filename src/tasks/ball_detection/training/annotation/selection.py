"""Apply an explicitly agreed policy to already validated fixed-validation data."""

from __future__ import annotations

import json
from typing import Any

from src.tasks.ball_detection.evaluation.coordinates import selection_score
from src.tasks.base.annotation.artifacts import verify_reference
from src.tasks.base.annotation.contracts import ArtifactRef
from src.tasks.base.annotation.paths import AnnotationLayout

from .contracts import BalancedSelectionPolicy, BallSelectionPolicy


def read_policy(layout: AnnotationLayout, reference: ArtifactRef) -> BallSelectionPolicy | BalancedSelectionPolicy:
    payload = verify_reference(layout, reference).read_bytes()
    raw = json.loads(payload)
    if raw.get("schema_version") == "ball.annotation-balanced-selection.v1":
        balanced: BalancedSelectionPolicy = BalancedSelectionPolicy.model_validate_json(payload)
        return balanced
    legacy: BallSelectionPolicy = BallSelectionPolicy.model_validate_json(payload)
    return legacy


def rejection_reasons(
    comparison: dict[str, Any], policy: BallSelectionPolicy | BalancedSelectionPolicy
) -> tuple[str, ...]:
    baseline = comparison["baseline"]["evaluation"]
    candidate = comparison["candidate"]["evaluation"]
    reasons = []
    if isinstance(policy, BalancedSelectionPolicy):
        if comparison["split"] != "test" or comparison["test_usage"] != "final_comparison":
            raise ValueError("Balanced adoption requires the final test comparison")
        for name in ("existing", "new"):
            if any(g["observed_frames"] < policy.minimum_frames_per_fps
                   for g in candidate["scopes"][name]["by_frame_step"].values()):
                reasons.append(f"{name} test support is below the agreed minimum")
        if selection_score(candidate, "balanced_existing_new") >= selection_score(baseline, "balanced_existing_new"):
            reasons.append("Equally weighted existing/new test error did not improve")
        if selection_score(candidate, "existing") > selection_score(baseline, "existing"):
            reasons.append("Existing test error worsened")
        return tuple(reasons)
    common = candidate["scopes"]["common"]
    if any(
        group["observed_frames"] < policy.minimum_common_frames_per_fps
        for group in common["by_frame_step"].values()
    ):
        reasons.append("Common validation support is below the agreed minimum")
    candidate_common = selection_score(candidate, "common")
    if candidate_common > policy.maximum_common_error_px:
        reasons.append("Common coordinate error exceeds the agreed maximum")
    if (
        selection_score(baseline, "common") - candidate_common
        < policy.minimum_improvement_px
    ):
        reasons.append("Common improvement is below the agreed minimum")
    if (
        selection_score(candidate, "full") - selection_score(baseline, "full")
        > policy.maximum_full_regression_px
    ):
        reasons.append("Full coordinate regression exceeds the agreed maximum")
    for limit in policy.source_regression_limits:
        before = baseline["scopes"]["full"]["by_source"].get(limit.source)
        after = candidate["scopes"]["full"]["by_source"].get(limit.source)
        if (
            before is None
            or after is None
            or any(
                group["observed_frames"] < limit.minimum_frames_per_fps
                for group in after["by_frame_step"].values()
            )
        ):
            reasons.append(f"{limit.source} support is below the agreed minimum")
        elif (
            after["macro_mean_error_px"] - before["macro_mean_error_px"]
            > limit.maximum_regression_px
        ):
            reasons.append(f"{limit.source} regression exceeds the agreed maximum")
    return tuple(reasons)
