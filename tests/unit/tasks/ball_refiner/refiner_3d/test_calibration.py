"""Empirical degradation keeps all modes and prevents source/split ambiguity."""

from dataclasses import replace

import numpy as np
import pytest
import yaml

from src.tasks.ball_refiner.refiner_3d.synthetic.calibration import load_calibration
from src.utils.paths import PROJECT_ROOT


def bank():
    plan = yaml.safe_load((PROJECT_ROOT / "src/tasks/ball_refiner/refiner_3d/dataset_plan.yaml").read_text())
    settings = plan["degradation"]["calibration"]
    return load_calibration(PROJECT_ROOT / settings["bank"], settings["bank_sha256"])


def test_bootstrap_uses_all_four_modes_and_requested_camera_gap_stratum():
    source = bank()
    gap = np.r_[np.zeros(40, dtype=bool), np.ones(64, dtype=bool), np.zeros(40, dtype=bool)]
    for camera in range(3):
        rows = source.draw_rows(camera, gap, np.random.default_rng(936), block_frames=16)
        repeat = source.draw_rows(camera, gap, np.random.default_rng(936), block_frames=16)
        np.testing.assert_array_equal(rows, repeat)
        assert source.components == 4
        np.testing.assert_array_equal(source.arrays["condition_index"][rows], gap.astype(int))
        assert (source.arrays["camera_index"][rows] == camera).all()
        assert source.arrays["error_uv"][rows].shape == (144, 4, 2)
        # Sequential runs follow the saved frame axis, never an unreviewed hole.
        for a, b in zip(rows[:-1], rows[1:], strict=True):
            if b == a + 1 and source.arrays["continues"][a]:
                assert source.arrays["source_frame"][b] == source.arrays["source_frame"][a] + 1
                assert source.arrays["source_artifact"][b] == source.arrays["source_artifact"][a]


def test_bootstrap_rejects_hash_mismatch_and_cross_clip_continuation():
    path = PROJECT_ROOT / "knowledge/runs/run-i936-provisional-degradation-r5-s936/bank.npz"
    with pytest.raises(ValueError, match="SHA mismatch"):
        load_calibration(path, "0" * 64)
    source = bank()
    arrays = {k: v.copy() for k, v in source.arrays.items()}
    boundary = np.flatnonzero(np.diff(arrays["source_artifact"]) != 0)[0]
    arrays["continues"][boundary] = True
    with pytest.raises(ValueError, match="source boundary"):
        replace(source, arrays=arrays)
