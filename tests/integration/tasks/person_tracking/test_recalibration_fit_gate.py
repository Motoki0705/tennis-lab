"""The real-data entrypoint must reject missing cameras before any fit."""
import importlib
from pathlib import Path

import pytest

from src.utils.paths import PROJECT_ROOT


def test_real_data_fit_gate_requires_exact_18_completed_cameras(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.syspath_prepend(str(PROJECT_ROOT / 'tests/benchmarks'))
    module = importlib.import_module('association_recalibration_fit')
    clips = [f'video_{v:03}/clip_{c:03}' for v in range(3) for c in (10, 11)]
    keys: dict[str, dict[str, object]] = {f'{c}/cam{i}': {} for c in clips for i in range(3)}
    plan = {'selected_clips': clips}
    complete = {'status': 'ok', 'records': keys, 'detections': keys}
    module.require_complete(plan, complete)
    for incomplete in ({**complete, 'status': 'running'},
                       {**complete, 'records': {k: v for k, v in keys.items() if not k.endswith('cam2')}},
                       {**complete, 'detections': {}},
                       {**complete, 'records': {**keys, 'video_999/clip_000/cam0': {}}}):
        with pytest.raises(ValueError, match='all 18'):
            module.require_complete(plan, incomplete)


def test_absent_final_feature_manifest_does_not_create_a_fit(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.syspath_prepend(str(PROJECT_ROOT / 'tests/benchmarks'))
    module = importlib.import_module('association_recalibration_fit')
    monkeypatch.setattr(module, 'calibrate', lambda *_: pytest.fail('fit must stay closed'))
    report = tmp_path / 'fit'
    with pytest.raises(FileNotFoundError):
        module.run(tmp_path, tmp_path / 'features', tmp_path / 'unused.json', report)
    assert not report.exists()
