from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from src.tasks.player_association.association.config import load_association_config
from src.tasks.player_association.calibration import fitting
from src.tasks.player_association.calibration.fitting import (
    CalibrationRejected,
    fit_scales,
)
from src.tasks.player_association.calibration.samples import (
    CalibrationClip,
    hierarchical_weights,
    pair_windows,
)


def test_weighted_rayleigh_and_regularized_balanced_logistic(scene: Callable[..., CalibrationClip]) -> None:
    pairs, _ = pair_windows(scene(), load_association_config(players_per_side=1))
    # Missing appearance must still contribute to the geometry scale.
    pairs[0] = replace(pairs[0], distance_m=3., cosine=None)
    result = fit_scales(pairs)
    weights = hierarchical_weights(pairs)
    positive = np.array([p.positive for p in pairs])
    expected = np.sqrt(np.average([p.distance_m ** 2 for p in pairs if p.positive], weights=weights[positive]) / 2)
    assert result.sigma_m == pytest.approx(expected)
    assert result.center == pytest.approx(.5, abs=.02) and result.slope > 0
    assert result.logistic_loss > 0 and result.iterations < 1000
    assert result.appearance_positive == 17 and result.appearance_negative == 18


def test_missing_support_and_reversed_appearance_fail_closed(scene: Callable[..., CalibrationClip]) -> None:
    pairs, _ = pair_windows(scene(), load_association_config(players_per_side=1))
    with pytest.raises(CalibrationRejected, match='>=10'):
        fit_scales([replace(p, cosine=None) if p.positive else p for p in pairs])
    with pytest.raises(CalibrationRejected, match='slope'):
        fit_scales([replace(p, cosine=0. if p.positive else 1.) for p in pairs])


def test_optimizer_nonconvergence_never_restores_old_scales(scene: Callable[..., CalibrationClip],
                                                           monkeypatch: pytest.MonkeyPatch) -> None:
    pairs, _ = pair_windows(scene(), load_association_config(players_per_side=1))
    monkeypatch.setattr(fitting, 'minimize', lambda *a, **k: SimpleNamespace(
        success=False, x=np.array([62.7, -62.7 * .847]), fun=.1, message='synthetic failure'))
    with pytest.raises(CalibrationRejected, match='synthetic failure'):
        fit_scales(pairs)
