"""Independent geometric examples for the diagnostic partition, not new gates."""
import numpy as np
from classify_candidates import error_pattern
from common import truth


def candidate(points, selected):
    mask = np.zeros(14, bool); mask[selected] = True
    return {'points': points, 'selected': mask}


def test_uniform_twenty_pixel_offset_is_not_an_extrapolation_collapse():
    gt, _ = truth('cam1')
    result = error_pattern(candidate(gt + [20, 0], [0, 1, 2, 3]), 'cam1')
    assert result['pattern'] == 'coherent_offset'
    np.testing.assert_allclose(result['oracle_translation_xy'], [20, 0], atol=1e-12)


def test_single_unsupported_corner_is_local_extrapolation():
    gt, _ = truth('cam1'); predicted = gt.copy(); predicted[2] += [60, 0]
    result = error_pattern(candidate(predicted, [8, 9, 10, 11]), 'cam1')
    assert result['pattern'] == 'localized_extrapolation'
    assert result['localized_indices'] == [2]


def test_a_disjoint_court_is_reported_before_oracle_translation():
    gt, _ = truth('cam1')
    result = error_pattern(candidate(gt + [2000, 0], [0, 1, 2, 3]), 'cam1')
    assert result['pattern'] == 'different_region_or_orientation'
    assert result['court_polygon_iou'] == 0
