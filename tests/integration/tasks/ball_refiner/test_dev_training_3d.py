"""CPU integration of both arms; poison test data and compare initialization/order."""
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
import yaml

from src.tasks.ball_refiner.refiner_3d.diffusion import dev_training
from src.tasks.ball_refiner.refiner_3d.diffusion.dev_config import load_config
from src.utils.paths import PROJECT_ROOT


def setup_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path, list[str]]:
    frames = 8
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
        'event_region_mask': np.array([False] * 4 + [True] * 4),
        'free_flight_mask': np.array([True] * 4 + [False] * 4),
        'occlusion_mask': np.tile([False] * 4 + [True] * 4, (3, 1)),
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
    config.update(updates=2, evaluate_every=2, frames=4, stride=4, batch_size=2, samples=2, steps=2,
                  expected_counts={'train': 2, 'val': 1, 'test': 1},
                  model={'width': 16, 'layers': 1, 'heads': 2, 'feedforward_multiplier': 2, 'time_frequencies': 2, 'dropout': 0.})
    config_path = tmp_path / 'config.yaml'
    config_path.write_text(yaml.safe_dump(config))
    return dataset, config_path, loaded


def test_two_arms_train_from_identical_initialization_without_test_reads(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    dataset, config, loaded = setup_run(tmp_path, monkeypatch)
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
        assert (output / arm / 'predictions/val-00000.npz').is_file()
        assert (output / arm / 'curves.png').stat().st_size > 1000
    assert orders[0] == orders[1]
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


@pytest.mark.parametrize('key,value', [('maximum_seconds', 3301), ('maximum_device_bytes', 10_000_000_001), ('allocator_limit_gib', 7.)])
def test_excess_budget_is_rejected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, key: str, value: float) -> None:
    _, path, _ = setup_run(tmp_path, monkeypatch)
    raw = yaml.safe_load(path.read_text())
    raw[key] = value
    path.write_text(yaml.safe_dump(raw))
    with pytest.raises(ValueError):
        load_config(path)
