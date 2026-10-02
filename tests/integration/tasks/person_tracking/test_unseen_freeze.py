"""The reserved clips may only open after the exact person freeze was pushed."""
import importlib
import json
import subprocess
from pathlib import Path
from typing import Any

import pytest

from src.tennis_scene.pipeline.definition import file_identity
from src.utils.paths import PROJECT_ROOT


@pytest.fixture
def freeze_module(monkeypatch: pytest.MonkeyPatch) -> Any:
    monkeypatch.syspath_prepend(str(PROJECT_ROOT / 'tests/benchmarks'))
    return importlib.import_module('person_unseen_freeze')


def test_freeze_requires_push_and_unchanged_manifest(freeze_module: Any, tmp_path: Path) -> None:
    def git(*args: str) -> str:
        return subprocess.check_output(['git', '-C', str(tmp_path), *args], text=True, stderr=subprocess.DEVNULL).strip()

    git('init', '-b', 'fixture')
    path = tmp_path / 'freeze.json'
    path.write_text(json.dumps({'schema': 'i964_person_freeze_v1', 'unseen_media_opened': False,
                               'unseen_labels_opened': False, 'scoring_batches': 0}))
    git('add', '.')
    git('-c', 'core.hooksPath=/dev/null', '-c', 'user.name=Fixture', '-c', 'user.email=fixture@example.invalid',
        'commit', '-m', 'freeze')
    commit = git('rev-parse', 'HEAD')
    remote = 'origin/frozen'
    with pytest.raises(subprocess.CalledProcessError):
        freeze_module.require_pushed(path, commit, root=tmp_path, remote=remote)
    git('update-ref', f'refs/remotes/{remote}', commit)
    assert freeze_module.require_pushed(path, commit, root=tmp_path, remote=remote)['scoring_batches'] == 0
    path.write_text(path.read_text() + '\n')
    with pytest.raises(ValueError, match='changed'):
        freeze_module.require_pushed(path, commit, root=tmp_path, remote=remote)


def test_changed_source_or_checkpoint_stops_before_media(
    freeze_module: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    (tmp_path / 'src').mkdir()
    code = tmp_path / 'src/config.yaml'
    code.write_text('fixed: true\n')
    weights = tmp_path / 'model.pth'
    weights.write_bytes(b'frozen checkpoint')
    external = tmp_path / 'dino'
    external.mkdir()
    manifest = {
        'source': freeze_module.source_hashes(tmp_path), 'assets': {'model': file_identity(weights)},
        'environment_files': [], 'person': {'association_config': file_identity(code)},
        'reservation': file_identity(code), 'generator': file_identity(code),
        'dino_source': {'root': str(external), 'files': {}}, 'packages': {},
    }
    monkeypatch.setattr(freeze_module, 'CODE', tmp_path)
    freeze_module.verify(manifest)
    weights.write_bytes(b'changed checkpoint')
    with pytest.raises(ValueError, match='Frozen input'):
        freeze_module.verify(manifest)
    code.write_text('fixed: false\n')
    with pytest.raises(ValueError, match='source/config'):
        freeze_module.verify(manifest)
