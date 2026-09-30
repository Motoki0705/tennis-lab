"""End-to-end CPU probe boundary: exact source identity and train/val isolation."""
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from src.tasks.ball_refiner.refiner_3d.diffusion import conditioning_probe as probe
from src.tasks.ball_refiner.refiner_3d.diffusion.memory_fixture import (
    analytic_memory_batch,
)
from src.tasks.ball_refiner.refiner_3d.diffusion.model import (
    ModelConfig,
    TrajectoryDenoiser,
)
from src.tasks.ball_refiner.refiner_3d.synthetic.configuration import sha256
from src.utils.paths import PROJECT_ROOT


def test_probe_fits_train_only_and_never_opens_test_rally(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    torch.set_num_threads(1)
    config = ModelConfig(16, 1, 4, 2, 4, 0.)
    counts = {'train': 1, 'val': 1, 'test': 1}
    dataset, training, output = (tmp_path / name for name in ('data', 'training', 'probe'))
    dataset.mkdir()
    (training / 'flow').mkdir(parents=True)
    (dataset / 'manifest.json').write_text(json.dumps({'counts': counts}))
    records = []
    for split in ('train', 'val', 'test'):
        path = dataset / (split + '-00000.npz')
        path.write_bytes(b'identity fixture; intercepted reader')
        records.append({'split': split, 'rally_id': path.stem, 'npz_sha256': sha256(path), 'frames': 16})
    checkpoint_config = {'model': asdict(config), 'evaluate_updates': (0, 20000), 'expected_counts': counts}
    checkpoint = training / 'flow/dev-only.pt'
    torch.save({'diagnostic_only': True, 'objective': 'flow', 'updates': 20000, 'config': checkpoint_config,
                'state_dict': TrajectoryDenoiser(config).state_dict()}, checkpoint)
    manifest = {'status': 'complete', 'source_manifest_sha256': sha256(dataset / 'manifest.json'),
                'config': checkpoint_config,
                'read_rallies': [{'rally_id': r['rally_id'], 'npz_sha256': r['npz_sha256']} for r in records[:2]],
                'arms': {'flow': {'status': 'complete', 'updates': 20000, 'checkpoint_sha256': sha256(checkpoint)}}}
    (training / 'manifest.json').write_text(json.dumps(manifest))
    loaded = []

    class Source:
        def __init__(self, directory: Path) -> None:
            assert directory == dataset
            self.records = records
            self.manifest = {'counts': counts}

        def load(self, record: dict[str, Any]) -> dict[str, Any]:
            assert record['split'] != 'test'
            loaded.append(record['rally_id'])
            return {'occlusion_mask': np.zeros((3, 16), bool), 'out_of_frame_mask': np.zeros((3, 16), bool)}

    batch = analytic_memory_batch(PROJECT_ROOT / 'src/tasks/ball_refiner/refiner_3d/fixtures/meiji_video_002_clip_010.json',
                                  batch_size=1, frames=16, seed=936)
    monkeypatch.setattr(probe, 'SyntheticDataset', Source)
    monkeypatch.setattr(probe, 'rally_window', lambda *args, **kwargs: batch)
    fit = probe.fit_readout
    fitted_frames = []

    def fit_train(tokens: torch.Tensor, targets: torch.Tensor) -> probe.LinearReadout:
        fitted_frames.append(len(targets))
        return fit(tokens, targets)

    monkeypatch.setattr(probe, 'fit_readout', fit_train)
    result = probe.run_conditioning_probe(dataset, training, output)
    assert result['status'] == 'complete'
    assert loaded == ['train-00000', 'val-00000']
    assert fitted_frames == [16]
    assert result['results']['val']['overall']['frames'] == 16
    assert result['results']['val']['by_visible_cameras']['0']['frames'] == 0
    assert result['resources']['gpu_jobs'] == 0
    assert result['solver']['rank'] <= 16
    with pytest.raises(FileExistsError):
        probe.run_conditioning_probe(dataset, training, output)
    # A mismatched historical dataset fails before creating any new output.
    (dataset / 'manifest.json').write_text('{"counts": {}}')
    with pytest.raises(ValueError, match='exact dataset'):
        probe.run_conditioning_probe(dataset, training, tmp_path / 'bad-probe')
    assert not (tmp_path / 'bad-probe').exists()
