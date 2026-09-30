"""Three fixed candidates, leave-one-video-out validation, then one final fit."""
from __future__ import annotations

from collections import Counter
from dataclasses import asdict
from typing import Any

import numpy as np

from src.tasks.player_association.association.associate import (
    AssociationConfig,
    AssociationUndecided,
    associate,
)
from src.tasks.player_association.calibration.fitting import (
    CalibrationRejected,
    Scales,
    fit_scales,
    fitted_config,
)
from src.tasks.player_association.calibration.samples import (
    CalibrationClip,
    PairWindow,
    hierarchical_weights,
    pair_windows,
)

CANDIDATES = (('A', 1., .5), ('B', 2., .4), ('C', 4., .3))


def score_pairs(clips: list[CalibrationClip], pairs: list[PairWindow],
                config: AssociationConfig) -> dict[str, Any]:
    """Score only shared observations; undecided/excluded is retained as zero."""
    predictions: dict[str, dict[str, np.ndarray] | None] = {}
    stops = {}
    for clip in clips:
        try:
            result = associate(clip.cameras, clip.fps, config)
        except AssociationUndecided as error:
            predictions[clip.key] = None
            stops[clip.key] = {'reason': error.reason, 'message': str(error), 'diagnostics': error.diagnostics}
        else:
            if result.camera_ids != tuple(c.camera.camera_id for c in clip.cameras):
                raise ValueError('Association reordered calibration cameras')
            predictions[clip.key] = dict(zip(result.camera_ids, result.player_ids, strict=True))
    rates, rejected = [], []
    for pair in pairs:
        ids = predictions[pair.clip]
        if ids is None:
            rates.append(0.)
            rejected.append(1.)
            continue
        frames = np.array(pair.frames)
        a = ids[pair.cameras[0]][pair.rows[0], frames]
        b = ids[pair.cameras[1]][pair.rows[1], frames]
        rates.append(float(((a >= 0) & (a == b)).mean()))
        rejected.append(float(((a < 0) | (b < 0)).mean()))
    return {'same_id_rates': rates, 'rejected_rates': rejected, 'stops': stops}


def rates(pairs: list[PairWindow], scores: dict[str, Any]) -> dict[str, float]:
    weights = hierarchical_weights(pairs)
    positive = np.array([p.positive for p in pairs], bool)
    same, rejected = np.array(scores['same_id_rates']), np.array(scores['rejected_rates'])
    if same.shape != weights.shape or rejected.shape != weights.shape \
            or not np.isfinite([same, rejected]).all() \
            or ((same < 0) | (same > 1) | (rejected < 0) | (rejected > 1)).any():
        raise ValueError('Invalid pair-window decision rates')
    if not positive.any() or positive.all():
        raise CalibrationRejected('Validation needs both pseudo classes')
    return {'positive_recall': float(np.average(same[positive], weights=weights[positive])),
            'negative_false_join': float(np.average(same[~positive], weights=weights[~positive])),
            'positive_rejection': float(np.average(rejected[positive], weights=weights[positive])),
            'negative_rejection': float(np.average(rejected[~positive], weights=weights[~positive]))}


def select_candidate(results: dict[str, dict[str, Any]]) -> str:
    if set(results) != {row[0] for row in CANDIDATES}:
        raise ValueError('Exactly the three precommitted candidates are required')
    admissible: list[tuple[str, float]] = []
    for name, _, _ in CANDIDATES:
        result = results[name]
        recall = result['overall']['positive_recall']
        negatives = [fold['metrics']['negative_false_join'] for fold in result['folds']]
        if len(negatives) != 3 or not np.isfinite([recall, *negatives]).all():
            raise ValueError('Incomplete or nonfinite validation folds')
        if max(negatives) <= .01 and recall >= .8:
            admissible.append((name, recall))
    if not admissible:
        raise CalibrationRejected('No fixed candidate meets <=1% per-video false joins and >=80% recall')
    best = max(recall for _, recall in admissible)
    return next(name for name, recall in admissible if best - recall <= 1e-9)


def check_stability(scales: list[Scales]) -> None:
    if len(scales) != 3:
        raise ValueError('Exactly three fitted folds are required')
    sigma, slope = [s.sigma_m for s in scales], [s.slope for s in scales]
    if not np.isfinite([sigma, slope]).all() or min(sigma + slope) <= 0 \
            or max(sigma) / min(sigma) > 2 or max(slope) / min(slope) > 3:
        raise CalibrationRejected('Unstable fold scales (sigma ratio >2 or slope ratio >3)')


def calibrate(clips: list[CalibrationClip], base: AssociationConfig) -> tuple[AssociationConfig | None, dict[str, Any]]:
    """Return no configuration on rejection; preserve all completed evidence."""
    keys = [c.key for c in clips]
    videos = sorted({key.split('/')[0] for key in keys})
    if len(keys) != 6 or len(set(keys)) != 6 or len(videos) != 3 \
            or any(sum(k.startswith(v + '/') for k in keys) != 2 for v in videos):
        raise ValueError('Calibration requires the complete six-clip/three-video set')
    pairs, audits = [], []
    for clip in clips:
        sample, audit = pair_windows(clip, base)
        pairs.extend(sample)
        audits.extend(audit)
    support_counts = Counter(
        (row['clip'], camera, depth, row['reason'])
        for row in audits if row['reason'] in ('positive', 'negative')
        for camera, depth in zip(row['cameras'], row['near_far'], strict=True))
    evidence: dict[str, Any] = {'status': 'rejected', 'pairs': [asdict(p) for p in pairs], 'audit': audits,
                               'folds': [], 'candidates': {}, 'final_fit_count': 0,
                               'camera_near_far_pair_support': [
                                   {'clip': key[0], 'camera': key[1], 'depth': key[2], 'class': key[3], 'pairs': count}
                                   for key, count in sorted(support_counts.items())]}
    try:
        if not pairs:
            raise CalibrationRejected('No eligible geometry pair-windows')
        evidence['weights'] = hierarchical_weights(pairs).tolist()
        folds: list[tuple[str, Scales]] = []
        for video in videos:
            train = [p for p in pairs if p.video != video]
            held = [p for p in pairs if p.video == video]
            support = [sum(p.positive == label for p in held) for label in (True, False)]
            if min(support) < 10:
                raise CalibrationRejected(f'{video}: validation positive/negative support {support}, need >=10 each')
            fitted = fit_scales(train)
            folds.append((video, fitted))
            evidence['folds'].append({'video': video, 'fit_videos': [v for v in videos if v != video],
                                      'scales': asdict(fitted), 'validation_support': support})
        for name, margin, runner_up in CANDIDATES:
            scores_by_video, fold_reports = {}, []
            for video, scales in folds:
                held = [p for p in pairs if p.video == video]
                config = fitted_config(base, scales, margin, runner_up)
                score = score_pairs([c for c in clips if c.key.split('/')[0] == video], held, config)
                scores_by_video[video] = score
                fold_reports.append({'video': video, 'metrics': rates(held, score), **score})
            ordered = [p for video in videos for p in pairs if p.video == video]
            all_scores = {field: [r for video in videos for r in scores_by_video[video][field]]
                          for field in ('same_id_rates', 'rejected_rates')}
            evidence['candidates'][name] = {'folds': fold_reports, 'overall': rates(ordered, all_scores)}
        selected = select_candidate(evidence['candidates'])
        evidence['selected_candidate'] = selected
        check_stability([s for _, s in folds])
        # No final all-data fit if support, validation or stability already failed.
        final = fit_scales(pairs)
        evidence.update(final_fit_count=1, final_scales=asdict(final))
        _, margin, runner_up = next(c for c in CANDIDATES if c[0] == selected)
        config = fitted_config(base, final, margin, runner_up)
    except CalibrationRejected as error:
        evidence['reason'] = str(error)
        return None, evidence
    evidence['status'] = 'accepted'
    return config, evidence
