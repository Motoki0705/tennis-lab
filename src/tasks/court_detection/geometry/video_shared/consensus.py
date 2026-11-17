"""Combine measurements of the same physical stripes, not H coefficients."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import replace

import numpy as np

from ..line_evidence import FloatArray
from .fit import PaintGroup, SharedFit, SharedPaintConfig, fit_common_court


def merge_measurements(initial: FloatArray, hypotheses: tuple[tuple[PaintGroup, ...], ...],
                       size_wh: tuple[int, int], config: SharedPaintConfig) -> SharedFit:
    """Small H differences can be the same stripe sampled at different positions.

    Compare actual source-frame measurements bidirectionally. Incompatible
    stripe assignments remain a failure. Duplicate initializers cannot turn one
    frame into extra temporal support: there is exactly one merged group per
    clip/frame/physical line, and fit_common_court balances those groups.
    """
    by_key: dict[tuple[str, int, int], list[PaintGroup]] = defaultdict(list)
    for hypothesis in hypotheses:
        keys = [(g.clip_id, g.frame_index, g.line_index) for g in hypothesis]
        if len(set(keys)) != len(keys):
            raise ValueError('Duplicate observation within one hypothesis')
        for key, group in zip(keys, hypothesis, strict=True):
            by_key[key].append(group)
    merged = []
    incompatible = []
    for key, groups in sorted(by_key.items()):
        equations = []
        for group in groups:
            centre = group.points.mean(axis=0)
            _, _, vh = np.linalg.svd(group.points - centre, full_matrices=False)
            equations.append(np.r_[vh[-1], -vh[-1] @ centre])
        for i, group in enumerate(groups):
            for j, equation in enumerate(equations):
                if i == j:
                    continue
                distance = float(np.percentile(np.abs(np.c_[group.points, np.ones(len(group.points))] @ equation), 95))
                if distance > config.paint.ambiguity_separation_px:
                    incompatible.append({'group': list(key), 'candidate_pair': [i, j], 'p95_px': distance})
        # Subpixel locations are observations, not extra votes for an initializer.
        points = np.unique(np.concatenate([g.points for g in groups]).round(6), axis=0)
        endings = []
        for side in (0, 1):
            states = {g.endings[side] for g in groups}
            endings.append('continues' if 'continues' in states else 'ended' if 'ended' in states
                           else 'ambiguous' if 'ambiguous' in states else 'unobservable')
        merged.append(replace(groups[0], points=points, endings=(endings[0], endings[1])))
    if incompatible:
        return SharedFit(None, None, ('ambiguous_observed_stripes',), {'incompatible_observations': incompatible,
                                                                      'hypotheses': len(hypotheses)})
    fitted = fit_common_court(initial, tuple(merged), size_wh, config)
    return replace(fitted, diagnostic={**fitted.diagnostic, 'merged_hypotheses': len(hypotheses),
                                       'merged_groups': len(merged)})
