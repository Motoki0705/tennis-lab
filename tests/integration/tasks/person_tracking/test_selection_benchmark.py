import importlib.util
from types import SimpleNamespace

import numpy as np
from ultralytics.engine.results import Boxes
from ultralytics.trackers.bot_sort import BOTSORT

from src.utils.paths import PROJECT_ROOT


def test_real_botsort_keeps_low_score_people_beyond_six_and_returns_source_rows() -> None:
    spec = importlib.util.spec_from_file_location('selection_benchmark', PROJECT_ROOT / 'tests/benchmarks/person_selection_cpu.py')
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    tracker = BOTSORT(SimpleNamespace(**module.TRACKER_CONFIG))
    box_data = np.asarray([[10 + i * 60, 10, 30 + i * 60, 70, .01, 0] for i in range(8)], np.float32)
    frame: np.ndarray = np.zeros((100, 500, 3), np.uint8)
    output = tracker.update(Boxes(box_data, orig_shape=frame.shape[:2]), frame)
    assert output.shape == (8, 8)
    assert sorted(output[:, 7].tolist()) == list(range(8))
    assert len(set(output[:, 4])) == 8
    np.testing.assert_allclose(output[:, 5], .01)
