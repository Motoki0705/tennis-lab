"""Empirical degradation keeps all modes and prevents source/split ambiguity."""

from dataclasses import replace

import numpy as np
import pytest
import yaml
from numpy.typing import NDArray

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


@pytest.mark.parametrize('mutation', ['flag', 'cap', 'weights', 'camera', 'presence'])
def test_reader_rejects_false_convergence_and_modified_calibration(mutation):
    from src.tasks.ball_refiner.refiner_3d.synthetic.dataset import _validate_v2
    source = bank()
    plan = yaml.safe_load((PROJECT_ROOT / 'src/tasks/ball_refiner/refiner_3d/dataset_plan.yaml').read_text())
    rows = np.array([[np.flatnonzero((source.arrays['camera_index'] == camera) & (source.arrays['condition_index'] == condition))[0] for condition in range(2)] for camera in range(3)])
    components = 125
    changes = np.zeros((2, components, 3))
    changes[1, -1, 0] = .06
    flags: NDArray[np.bool_] = np.ones((2, components), dtype=bool)
    flags[1, -1] = False
    arrays = {
        'integration_component_changes': changes, 'integration_component_converged': flags,
        'integration_converged': np.array([True, False]), 'integration_rounds': np.array([3, 5]),
        'integration_nll_delta_nat': np.array([.01, .06]), 'calibration_rows': rows,
        'occlusion_mask': np.tile([False, True], (3, 1)), 'out_of_frame_mask': np.zeros((3, 2), dtype=bool),
        'gmm2d_scale_tril_uv': source.arrays['scale_tril_uv'][rows].copy(),
        'gmm2d_mixture_logits': source.arrays['mixture_logits'][rows].copy(),
        'gmm2d_presence_logits': source.arrays['presence_logits'][rows].copy(),
    }
    record = {'frames': 2, 'integration': {'rule': plan['degradation']['boundary_convergence'], 'converged_frames': 1, 'nonconverged_frames': 1}}
    _validate_v2(arrays, record, plan, components)  # capped frame remains readable
    if mutation == 'flag':
        arrays['integration_converged'][1] = True
    elif mutation == 'cap':
        arrays['integration_rounds'][1] = 2
    elif mutation == 'weights':
        arrays['gmm2d_mixture_logits'][0, 0, -1] += 1
    elif mutation == 'camera':
        arrays['calibration_rows'][0, 0] = rows[1, 0]
    else:
        arrays['gmm2d_presence_logits'][0, 1] = -4
    with pytest.raises(ValueError):
        _validate_v2(arrays, record, plan, components)


def test_relative_calibration_paths_resolve_inside_explicit_project_root(tmp_path):
    import hashlib

    from src.tasks.ball_refiner.refiner_3d.synthetic.configuration import load_plan
    from src.utils.configuration import PathResolver, RuntimePathRoots

    values = yaml.safe_load((PROJECT_ROOT / 'src/tasks/ball_refiner/refiner_3d/dataset_plan.yaml').read_text())
    for i, source in enumerate(values['geometry']['sources']):
        # Loading a plan verifies source identity; camera decoding has its own tests.
        camera = tmp_path / f'camera-{i}.json'
        camera.write_text('{}')
        source['path'] = str(camera)
        source['sha256'] = hashlib.sha256(camera.read_bytes()).hexdigest()
    path = tmp_path / 'plan.yaml'
    path.write_text(yaml.safe_dump(values))
    resolver = PathResolver(RuntimePathRoots(project_root=PROJECT_ROOT, data_root=tmp_path,
        artifact_root=tmp_path, output_root=tmp_path, checkpoint_root=tmp_path,
        cache_root=tmp_path, external_asset_root=tmp_path))
    plan = load_plan(path, resolver)
    assert plan.calibration.components == 4
    assert len(plan.camera_paths) == 3
    assert PROJECT_ROOT / values['degradation']['calibration']['bank'] in plan.input_paths
    plan.verify_inputs()
    values['geometry']['sources'][0]['camera_keys'].reverse()
    path.write_text(yaml.safe_dump(values))
    with pytest.raises(ValueError, match='Camera order'):
        load_plan(path, resolver)
