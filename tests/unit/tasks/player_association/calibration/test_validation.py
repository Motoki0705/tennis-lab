from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, replace
from typing import Any

import numpy as np
import pytest

from src.tasks.player_association.association.associate import AssociationUndecided
from src.tasks.player_association.association.config import load_association_config
from src.tasks.player_association.calibration import validation
from src.tasks.player_association.calibration.fitting import CalibrationRejected, Scales
from src.tasks.player_association.calibration.samples import (
    CalibrationClip,
    PairWindow,
    pair_windows,
)
from src.tasks.player_association.calibration.validation import (
    calibrate,
    check_stability,
    rates,
    score_pairs,
    select_candidate,
)


def test_full_synthetic_fit_holds_out_each_video_and_fits_all_data_once(
        scene: Callable[..., CalibrationClip], monkeypatch: pytest.MonkeyPatch) -> None:
    clips = [scene(f'video_{v:03}/clip_{c:03}') for v in range(3) for c in (10, 11)]
    base = load_association_config(players_per_side=1)
    before = asdict(base)
    fitted_sets = []
    original = validation.fit_scales

    def spy(pairs: list[PairWindow]) -> Scales:
        fitted_sets.append({p.video for p in pairs})
        return original(pairs)

    monkeypatch.setattr(validation, 'fit_scales', spy)
    config, evidence = calibrate(clips, base)
    assert evidence['status'] == 'accepted', evidence.get('reason')
    assert evidence['selected_candidate'] == 'A' and evidence['final_fit_count'] == 1
    videos = {f'video_{v:03}' for v in range(3)}
    assert fitted_sets == [videos - {v} for v in sorted(videos)] + [videos]
    assert config is not None and config.geometry.sigma_m != base.geometry.sigma_m
    assert asdict(base) == before  # no production default mutation
    for row in evidence['candidates'].values():
        assert row['overall']['positive_recall'] == 1. and row['overall']['negative_false_join'] == 0.


def test_undecided_keeps_both_denominators_and_reports_rejection(
        scene: Callable[..., CalibrationClip], monkeypatch: pytest.MonkeyPatch) -> None:
    clip = scene()
    config = load_association_config(players_per_side=1)
    pairs, _ = pair_windows(clip, config)

    def stop(*_: Any) -> None:
        raise AssociationUndecided('synthetic_stop', 'test', {})

    monkeypatch.setattr(validation, 'associate', stop)
    scores = score_pairs([clip], pairs, config)
    metrics = rates(pairs, scores)
    assert metrics == {'positive_recall': 0., 'negative_false_join': 0.,
                       'positive_rejection': 1., 'negative_rejection': 1.}
    assert len(scores['same_id_rates']) == len(pairs) and clip.key in scores['stops']


def test_scoring_counts_shared_observations_only(scene: Callable[..., CalibrationClip],
                                                monkeypatch: pytest.MonkeyPatch) -> None:
    clip = scene()
    config = load_association_config(players_per_side=1)
    pairs, _ = pair_windows(clip, config)
    pair = next(p for p in pairs if p.positive)
    pairs = [replace(pair, frames=(0, 2, 4, 6))]
    from types import SimpleNamespace
    ids: list[np.ndarray] = [np.zeros((2, 24), np.int64) for _ in range(3)]
    ids[1][pair.rows[1], 4] = -1
    ids[1][pair.rows[1], 6] = 1
    monkeypatch.setattr(validation, 'associate', lambda *a: SimpleNamespace(
        camera_ids=('cam0', 'cam1', 'cam2'), player_ids=tuple(ids)))
    result = score_pairs([clip], pairs, config)
    assert result['same_id_rates'] == [.5] and result['rejected_rates'] == [.25]


def test_selection_checks_each_video_and_breaks_ties_in_fixed_order() -> None:
    def candidate(recall: float, negatives: list[float]) -> dict[str, Any]:
        return {'overall': {'positive_recall': recall},
                'folds': [{'metrics': {'negative_false_join': n}} for n in negatives]}
    values = {'A': candidate(.9, [0, 0, .011]), 'B': candidate(.85, [0, .01, 0]),
              'C': candidate(.85 + 5e-10, [0, 0, 0])}
    assert select_candidate(values) == 'B'
    values['C'] = candidate(.86, [0, 0, 0])
    assert select_candidate(values) == 'C'
    values = {'A': candidate(.85, [0, 0, 0]), 'B': candidate(.85 + 8e-10, [0, 0, 0]),
              'C': candidate(.85 + 1.5e-9, [0, 0, 0])}
    assert select_candidate(values) == 'B'  # Tie is measured against the best recall.
    values['A'] = candidate(.9, [0, 0, .011])
    values['B'] = values['C'] = candidate(.79, [0, 0, 0])
    with pytest.raises(CalibrationRejected, match='No fixed candidate'):
        select_candidate(values)


def test_insufficient_support_and_unstable_folds_do_not_publish_config(scene: Callable[..., CalibrationClip]) -> None:
    clips = [scene(f'video_{v:03}/clip_{c:03}', frames=3) for v in range(3) for c in (10, 11)]
    config, evidence = calibrate(clips, load_association_config(players_per_side=1))
    assert config is None and evidence['status'] == 'rejected' and evidence['final_fit_count'] == 0
    with pytest.raises(ValueError, match='complete six'):
        calibrate(clips[:-1], load_association_config(players_per_side=1))
    scales = Scales(1., 10., .5, .1, 5, 10, 10)
    check_stability([scales, scales, replace(scales, sigma_m=2., slope=30.)])
    for changed in (replace(scales, sigma_m=2.01), replace(scales, slope=30.01)):
        with pytest.raises(CalibrationRejected, match='Unstable'):
            check_stability([scales, scales, changed])
