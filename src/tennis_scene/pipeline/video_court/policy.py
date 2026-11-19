"""Explicit execution policy for immutable drift measurements."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from src.tasks.court_detection.geometry.video_drift import summarize_clip
from src.tennis_scene.pipeline.errors import ReconstructionUnavailable


def validate_unobservable_policy(value: str) -> str:
    if value not in {'warn', 'fail'}:
        raise ValueError('video_shared.unobservable_policy must be warn or fail')
    return value


def apply_drift_policy(report: dict[str, Any], policy: str, *, common_status: str) -> dict[str, Any]:
    """Reclassify unobservable drift without claiming that motion was measured.

    The original status, numeric values, reasons and source frames survive.
    Reliable tolerance failures and an unavailable common court remain failed.
    """
    validate_unobservable_policy(policy)
    if common_status not in {'accepted', 'failed'}:
        raise ValueError('Unknown common court status')
    result = deepcopy(report)
    for frame in result['frames']:
        measured = frame.get('measurement_status', frame['status'])
        frame['measurement_status'] = measured
        frame['status'] = measured
        if (common_status == 'accepted' and policy == 'warn' and measured == 'failed'
                and frame.get('reason') in {'drift_unobservable', 'non_similarity_or_conflicting_line'}):
            frame['status'] = 'warning'
    result.update(summarize_clip(result['frames']))
    result['measurement_status'] = report.get('measurement_status', report['status'])
    result['unobservable_policy'] = policy
    return result


def require_accepted_drift(camera: dict[str, Any], clip_id: str, measured: dict[str, Any], policy: str, *,
                           artifact_digest: str, drift_digest: str | None) -> dict[str, Any]:
    """Apply the same admission rule to a clip calibration and to its use as evidence."""
    report = apply_drift_policy(measured, policy, common_status=camera['status'])
    if camera['status'] != 'accepted' or report['status'] == 'failed':
        reasons = camera['diagnostics'].get('reasons', [])
        reason = report['reason'] or ', '.join(reasons) or 'common court unavailable'
        raise ReconstructionUnavailable('video_shared_court_failed', f"Shared court unavailable for {clip_id}/{camera['camera_id']}: {reason}",
                                        diagnostics={'clip_id': clip_id, 'camera_id': camera['camera_id'], 'camera_status': camera['status'],
                                                     'artifact_digest': artifact_digest, 'drift_digest': drift_digest,
                                                     'camera_reasons': reasons, 'clip': report})
    return report
