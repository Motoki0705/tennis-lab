import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import pytest

from src.utils.paths import PROJECT_ROOT


@pytest.fixture
def guard(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location('feature_guard', PROJECT_ROOT / 'tests/benchmarks/association_feature_guard.py')
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module.signal, 'signal', lambda *args: None)
    (tmp_path / 'plan.json').write_text(json.dumps({'budget': {
        'vram_stop_bytes': 9_500_000_000, 'disk_limit_bytes': 5_000_000_000, 'wall_seconds': 7200}}))
    return module


def test_guard_does_not_start_child_if_vram_is_already_over_limit(
    guard: ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    monkeypatch.setattr(guard, 'gpu_bytes', lambda: 9_500_000_000)
    with pytest.raises(RuntimeError, match='VRAM'):
        guard.supervise(tmp_path, ['this-must-never-run'])
    receipt = json.loads((tmp_path / 'resource_guard.json').read_text())
    assert receipt['status'] == 'failed' and receipt['peak_gpu_bytes'] == 9_500_000_000


def test_guard_reaps_only_its_child_when_monitor_fails(
    guard: ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    calls: list[bool] = []
    children = []
    real_popen = guard.subprocess.Popen

    def popen(*args: object, **kwargs: object) -> object:
        child = real_popen(*args, **kwargs)
        children.append(child)
        return child

    def memory() -> int:
        if calls:
            raise RuntimeError('NVML unavailable')
        calls.append(True)
        return 100

    monkeypatch.setattr(guard, 'gpu_bytes', memory)
    monkeypatch.setattr(guard.subprocess, 'Popen', popen)
    monkeypatch.setattr(guard.time, 'sleep', lambda _: None)
    with pytest.raises(RuntimeError, match='NVML unavailable'):
        guard.supervise(tmp_path, [sys.executable, '-c', 'import time; time.sleep(60)'])
    assert len(children) == 1 and children[0].poll() is not None
    assert json.loads((tmp_path / 'resource_guard.json').read_text())['status'] == 'failed'
