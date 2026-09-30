"""The paired runner uses val only and rejects changed historical input identity."""
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from src.tasks.ball_refiner.refiner_3d.diffusion import context_probe as probe
from src.tasks.ball_refiner.refiner_3d.diffusion.memory_fixture import (
    analytic_memory_batch,
)
from src.tasks.ball_refiner.refiner_3d.diffusion.model import (
    ModelConfig,
    TrajectoryDenoiser,
)
from src.tasks.ball_refiner.refiner_3d.synthetic.configuration import sha256
from src.utils.paths import PROJECT_ROOT


def test_context_comparison_never_loads_train_or_test(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config = ModelConfig(16, 1, 4, 2, 4, 0.)
    counts = {'train': 1, 'val': 1, 'test': 1}
    dataset, training, output = (tmp_path / name for name in ('data', 'training', 'probe'))
    dataset.mkdir()
    training.mkdir()
    (dataset / 'manifest.json').write_text(json.dumps({'counts': counts}))
    records = []
    for split in ('train', 'val', 'test'):
        path = dataset / (split + '-00000.npz')
        path.write_bytes(b'identity fixture; intercepted reader')
        records.append({'split': split, 'rally_id': path.stem, 'seed': 936, 'npz_sha256': sha256(path), 'frames': 17})
    checkpoint_config = {'model': asdict(config), 'evaluate_updates': (0, 20000), 'expected_counts': counts,
                         'updates': 20000, 'frames': 128, 'stride': 128, 'samples': 2, 'steps': 1,
                         'loss': {'x0': 1., 'reprojection': .01, 'physics': .0001, 'event': .1}}
    manifest: dict[str, Any] = {'status': 'complete', 'source_manifest_sha256': sha256(dataset / 'manifest.json'),
                'config': checkpoint_config,
                'read_rallies': [{'rally_id': r['rally_id'], 'npz_sha256': r['npz_sha256']} for r in records[:2]], 'arms': {}}
    for arm in ('flow', 'regression'):
        (training / arm).mkdir()
        checkpoint = training / arm / 'dev-only.pt'
        torch.save({'diagnostic_only': True, 'objective': arm, 'updates': 20000, 'config': checkpoint_config,
                    'state_dict': TrajectoryDenoiser(config).state_dict()}, checkpoint)
        manifest['arms'][arm] = {'status': 'complete', 'checkpoint_sha256': sha256(checkpoint)}
    (training / 'manifest.json').write_text(json.dumps(manifest))
    loaded = []
    batch = analytic_memory_batch(PROJECT_ROOT / 'src/tasks/ball_refiner/refiner_3d/fixtures/meiji_video_002_clip_010.json', batch_size=1, frames=17, seed=936)

    class Source:
        def __init__(self, directory: Path) -> None:
            self.records = records
            self.manifest = {'counts': counts}

        def load(self, record: dict[str, Any]) -> dict[str, Any]:
            assert record['split'] == 'val'
            loaded.append(record['rally_id'])
            return {'positions_3d_m': batch.target_positions_m[0].numpy(),
                    'timestamps_seconds': np.arange(17) * 1001/60000,
                    'occlusion_mask': np.zeros((3, 17), bool), 'out_of_frame_mask': np.zeros((3, 17), bool),
                    'event_region_mask': np.zeros(17, bool), 'free_flight_mask': np.ones(17, bool),
                    'camera_true_K': np.tile(np.eye(3), (3, 1, 1)),
                    'camera_true_R': np.tile(np.eye(3), (3, 1, 1)), 'camera_true_t': np.tile([0., 0., 100.], (3, 1))}

    monkeypatch.setattr(probe, 'SyntheticDataset', Source)
    monkeypatch.setattr(probe, 'rally_window', lambda *args, **kwargs: batch)
    result = probe.run_context_probe(dataset, training, output)
    assert result['status'] == 'complete'
    assert loaded == ['val-00000']
    assert result['frames'] == 17
    assert result['results']['flow']['t128']['samples']['metrics']['rmse_m_overall']['count'] == 34
    assert result['resources']['gpu_jobs'] == 0
    with pytest.raises(FileExistsError):
        probe.run_context_probe(dataset, training, output)
    (dataset / 'manifest.json').write_text('{"counts": {}}')
    with pytest.raises(ValueError, match='exact dataset'):
        probe.run_context_probe(dataset, training, tmp_path / 'bad')
