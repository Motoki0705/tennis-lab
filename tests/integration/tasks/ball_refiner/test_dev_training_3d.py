"""CPU integration of both arms; poison test data and compare initialization/order."""
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
import yaml

from src.tasks.ball_refiner.refiner_3d.baseline_comparison import training_comparison
from src.tasks.ball_refiner.refiner_3d.diffusion import dev_training
from src.tasks.ball_refiner.refiner_3d.diffusion.dev_config import load_config
from src.tasks.ball_refiner.refiner_3d.diffusion.dev_evaluation import (
    RallyInput,
    evaluate_dev,
)
from src.tasks.ball_refiner.refiner_3d.diffusion.model import (
    DenoiserOutput,
    MixtureCondition,
    TrajectoryDenoiser,
)
from src.utils.paths import PROJECT_ROOT
from src.utils.schema.court_normalization import denormalize_court_position


def setup_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, frames: int = 8) -> tuple[Path, Path, list[str]]:
    times = np.arange(frames) * 1001 / 60000
    positions = np.column_stack((times, times * 0, times * 0 + 2)).astype(np.float32)
    rotation = np.tile(np.eye(3), (3, 1, 1))
    intrinsic = np.tile(np.array([[100., 0, 100], [0, 100, 100], [0, 0, 1.]]), (3, 1, 1))
    translation = np.tile([0., 0., 20.], (3, 1))
    arrays: dict[str, Any] = {
        'positions_3d_m': positions, 'timestamps_seconds': times,
        'gmm3d_means_m': np.broadcast_to(positions[:, None], (frames, 125, 3)).copy(),
        'gmm3d_covariance_m2': np.tile(np.eye(3, dtype=np.float32), (frames, 125, 1, 1)),
        'gmm3d_weights': np.full((frames, 125), 1 / 125, dtype=np.float32),
        'gmm3d_camera_subsets': np.ones((frames, 125, 3), dtype=bool),
        'prior_only_probability': np.zeros(frames, dtype=np.float32),
        'gmm2d_means_uv': np.full((3, frames, 4, 2), .5, dtype=np.float32),
        'gmm2d_scale_tril_uv': np.tile(np.eye(2, dtype=np.float32) * .02, (3, frames, 4, 1, 1)),
        'gmm2d_mixture_logits': np.zeros((3, frames, 4), dtype=np.float32),
        'gmm2d_presence_logits': np.full((3, frames), 4., dtype=np.float32),
        'source_size_wh': np.full((3, 2), 201),
        'integration_converged': np.zeros(frames, dtype=bool),
        'integration_convergence_assessed': np.zeros(frames, dtype=bool),
        'event_labels': np.zeros((frames, 2), dtype=bool),
        'event_region_mask': np.arange(frames) >= 4,
        'free_flight_mask': np.arange(frames) < 4,
        'occlusion_mask': np.tile(np.arange(frames) >= 4, (3, 1)),
        'out_of_frame_mask': np.zeros((3, frames), dtype=bool),
        'camera_true_K': intrinsic, 'camera_true_R': rotation, 'camera_true_t': translation,
        'camera_estimated_K': intrinsic, 'camera_estimated_R': rotation, 'camera_estimated_t': translation,
    }
    loaded: list[str] = []
    records = [{'rally_id': f'{split}-{index:05d}', 'split': split, 'frames': frames, 'seed': 936 + index,
                'npz_sha256': 'fixture', 'physics': {'gravity': 9.81}}
               for split, count in (('train', 2), ('val', 1), ('test', 1)) for index in range(count)]

    class Source:
        def __init__(self, path: Path) -> None:
            self.records = records
            self.manifest = {'counts': {'train': 2, 'val': 1, 'test': 1},
                             'plan': {'degradation': {'boundary_convergence': {'method': 'fixed_hybrid'}}}}

        def load(self, record: dict[str, Any]) -> dict[str, Any]:
            assert record['split'] != 'test', 'Test split must never be opened'
            loaded.append(record['rally_id'])
            return arrays

    monkeypatch.setattr(dev_training, 'SyntheticDataset', Source)
    dataset = tmp_path / 'data'
    dataset.mkdir()
    (dataset / 'manifest.json').write_text('{}')
    config = yaml.safe_load((PROJECT_ROOT / 'src/tasks/ball_refiner/refiner_3d/training_dev.yaml').read_text())
    config.update(updates=2, evaluate_updates=[0, 1, 2], frames=4, stride=4, batch_size=2, samples=2, steps=2,
                  expected_counts={'train': 2, 'val': 1, 'test': 1},
                  model={'width': 16, 'layers': 1, 'heads': 2, 'feedforward_multiplier': 2, 'time_frequencies': 2, 'dropout': 0.})
    config_path = tmp_path / 'config.yaml'
    config_path.write_text(yaml.safe_dump(config))
    return dataset, config_path, loaded


@pytest.mark.parametrize('validation_frames', [None, 4])
def test_two_arms_train_from_identical_initialization_without_test_reads(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, validation_frames: int | None) -> None:
    dataset, config, loaded = setup_run(tmp_path, monkeypatch)
    raw = yaml.safe_load(config.read_text())
    raw['validation_frames'] = validation_frames
    config.write_text(yaml.safe_dump(raw))
    output = tmp_path / 'result'
    result = dev_training.run_dev_training(dataset, config, output, device='cpu')
    assert result['status'] == 'complete'
    assert loaded == ['train-00000', 'train-00001', 'val-00000']
    assert result['input_diagnostics'] == {'frames': 24, 'unassessed_frames': 24, 'nonconverged_frames': 0}
    assert len(result['windows']) == 4
    initial = [torch.load(output / arm / 'initial-state.pt', weights_only=True) for arm in ('flow', 'regression')]
    for key in initial[0]:
        torch.testing.assert_close(initial[0][key], initial[1][key])
    orders = []
    for arm in ('flow', 'regression'):
        checkpoint = torch.load(output / arm / 'dev-only.pt', weights_only=True)
        assert not torch.equal(checkpoint['state_dict']['position_head.weight'], initial[0]['position_head.weight'])
        rows = [json.loads(line) for line in (output / arm / 'updates.jsonl').read_text().splitlines()]
        orders.append([r['window_indices'] for r in rows])
        assert len(rows) == 2
        val = result['arms'][arm]['validation'][-1]
        assert val['metrics']['mean']['rmse_m_overall']['count'] == 8
        assert val['metrics']['samples']['rmse_m_overall']['count'] == (16 if arm == 'flow' else 8)
        assert val['validation_frames'] == validation_frames
        if validation_frames is not None:
            assert [w['owned_start'] for w in val['windows']['val-00000']] == [0, 4]
        assert [v['update'] for v in result['arms'][arm]['validation']] == [0, 1, 2]
        for update in (0, 1, 2):
            assert (output / arm / 'predictions' / f'update-{update:05d}' / 'val-00000.npz').is_file()
        assert val['by_visible_cameras']['mean']['0']['rmse_m_overall']['count'] == 4
        assert val['by_visible_cameras']['mean']['3']['rmse_m_overall']['count'] == 4
        assert (output / arm / 'curves.png').stat().st_size > 1000
    assert orders[0] == orders[1]
    assert result['baselines']['methods']['mixture_mean']['metrics']['rmse_m_overall']['value'] < 1e-6
    assert result['baselines']['frames'] == 8
    comparison = json.loads((output / 'comparison.json').read_text())
    assert comparison['primary_update'] == 2
    assert set(comparison['methods']) == {
        'truth', 'mixture_mean', 'top_component', 'mixture_mean_rts',
        *(f'{arm}_{update:05d}_{kind}' for arm in ('flow', 'regression') for update in (0, 1, 2) for kind in ('mean', 'samples')),
    }
    assert comparison['methods']['truth'] == result['baselines']['methods']['truth']
    assert 'accel p95 all/free' in (output / 'comparison.md').read_text()
    from copy import deepcopy
    mismatched = deepcopy(result)
    mismatched['arms']['flow']['validation'][0]['metrics']['truth']['rmse_m_overall']['count'] += 1
    with pytest.raises(ValueError, match='Historical support differs'):
        training_comparison(mismatched)
    with pytest.raises(FileExistsError):
        dev_training.run_dev_training(dataset, config, output, device='cpu')


def test_nonfinite_loss_records_failure_without_regression_retry(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    dataset, config, _ = setup_run(tmp_path, monkeypatch)

    def invalid(*args: Any, **kwargs: Any) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        return torch.tensor(float('nan')), {}

    monkeypatch.setattr(dev_training, 'training_objective', invalid)
    output = tmp_path / 'failed'
    with pytest.raises(FloatingPointError, match='Nonfinite'):
        dev_training.run_dev_training(dataset, config, output, device='cpu')
    assert json.loads((output / 'manifest.json').read_text())['status'] == 'failed'
    assert not (output / 'regression').exists()


@pytest.mark.parametrize('key,value', [('maximum_seconds', 5101), ('maximum_device_bytes', 10_000_000_001), ('allocator_limit_gib', 7.), ('updates', 20001)])
def test_excess_budget_is_rejected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, key: str, value: float) -> None:
    _, path, _ = setup_run(tmp_path, monkeypatch)
    raw = yaml.safe_load(path.read_text())
    raw[key] = value
    path.write_text(yaml.safe_dump(raw))
    with pytest.raises(ValueError):
        load_config(path)


@pytest.mark.parametrize('schedule', [[1, 2], [0, 1], [0, 2, 1, 2], [0, True, 2], [0, 1, 1, 2]])
def test_invalid_evaluation_schedule_is_rejected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, schedule: list[int]) -> None:
    _, path, _ = setup_run(tmp_path, monkeypatch)
    raw = yaml.safe_load(path.read_text())
    raw['evaluate_updates'] = schedule
    path.write_text(yaml.safe_dump(raw))
    with pytest.raises(ValueError):
        load_config(path)


def test_long_config_changes_only_updates_evaluation_schedule_and_wall_budget() -> None:
    short = load_config(PROJECT_ROOT / 'src/tasks/ball_refiner/refiner_3d/training_dev.yaml')
    long = load_config(PROJECT_ROOT / 'src/tasks/ball_refiner/refiner_3d/training_dev_long.yaml')
    from dataclasses import asdict
    differences = {key for key, value in asdict(short).items() if asdict(long)[key] != value}
    assert differences == {'updates', 'evaluate_updates', 'maximum_seconds'}
    assert long.updates == 20000
    assert long.evaluate_updates == (0, 2000, 5000, 10000, 15000, 20000)


def test_anchored_config_changes_only_validation_context_and_declared_schedule() -> None:
    from dataclasses import asdict
    root = PROJECT_ROOT / 'src/tasks/ball_refiner/refiner_3d'
    old = load_config(root / 'training_dev_long.yaml')
    new = load_config(root / 'training_dev_anchored_t128.yaml')
    assert {key for key, value in asdict(old).items() if asdict(new)[key] != value} == {'validation_frames', 'evaluate_updates'}
    assert new.validation_frames == new.frames == new.stride == 128
    assert new.evaluate_updates == (0, 2000, 5000, 10000, 20000)


@pytest.mark.parametrize('frames', [True, 3, 8, 4.0])
def test_invalid_validation_context_is_rejected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, frames: Any) -> None:
    _, path, _ = setup_run(tmp_path, monkeypatch)
    raw = yaml.safe_load(path.read_text())
    raw['validation_frames'] = frames
    path.write_text(yaml.safe_dump(raw))
    with pytest.raises(ValueError, match='match training'):
        load_config(path)


def test_validation_preserves_context_tail_and_derivative_seams(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    dataset, path, _ = setup_run(tmp_path, monkeypatch, frames=9)
    config = replace(load_config(path), validation_frames=4)
    source = dev_training.SyntheticDataset(dataset)
    record = next(r for r in source.records if r['split'] == 'val')
    arrays = source.load(record)
    arrays['free_flight_mask'][:] = True
    arrays['event_region_mask'][:] = False
    lengths = []

    class ContextMean(TrajectoryDenoiser):
        def forward(self, state: torch.Tensor, time: torch.Tensor, condition: MixtureCondition) -> DenoiserOutput:
            lengths.append(state.shape[1])
            valid = ~condition.padding_mask
            stamps = condition.timestamps_seconds
            center = (stamps * valid).sum(1) / valid.sum(1)
            positions = (stamps + center[:, None])[..., None].expand(-1, -1, 3)
            return DenoiserOutput(positions, torch.zeros_like(state[..., :2]))

    model = ContextMean(config.model).train()
    output = tmp_path / 'windowed'
    result = evaluate_dev(model, [RallyInput(record, arrays)], config, device='cpu',
                          objective='regression', check_budget=lambda: None, predictions=output)
    assert model.training
    assert lengths == [4, 4, 4]
    assert result['windows']['val-00000'][-1] == {
        'window_start': 6, 'real_stop': 9, 'owned_start': 8, 'owned_stop': 9, 'padded_frames': 1,
    }
    stamps = torch.from_numpy(arrays['timestamps_seconds']).float()
    centers = torch.cat((stamps[:4].mean().expand(4), stamps[4:8].mean().expand(4), stamps[6:].mean().expand(1)))
    expected = denormalize_court_position((stamps + centers)[:, None].expand(-1, 3)).numpy()
    with np.load(output / 'val-00000.npz') as saved:
        np.testing.assert_allclose(saved['mean_m'], expected, rtol=0, atol=1e-6)
        assert len(saved['mean_m']) == 9
    metric = result['metrics']['mean']
    # All seven second differences include seams, instead of 2 per complete window.
    assert metric['acceleration_free_flight']['count'] == 7
    assert metric['jerk_free_flight']['count'] == 6
    assert metric['acceleration_free_flight']['p95'] > 1
