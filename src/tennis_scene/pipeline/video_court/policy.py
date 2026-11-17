"""Explicit execution policy for immutable drift measurements."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from src.tasks.court_detection.geometry.video_drift import summarize_clip


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
