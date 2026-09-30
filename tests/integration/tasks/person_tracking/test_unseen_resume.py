"""Only a pushed execution addendum can resume the unopened, stopped preparation."""
import importlib
import json
import subprocess
from pathlib import Path
from typing import Any

import pytest

from src.tennis_scene.pipeline.definition import file_identity
from src.utils.paths import PROJECT_ROOT


@pytest.fixture
def resume(monkeypatch: pytest.MonkeyPatch) -> Any:
    monkeypatch.syspath_prepend(str(PROJECT_ROOT / 'tests/benchmarks'))
    return importlib.import_module('person_unseen_resume')


@pytest.mark.parametrize('violation', ['none', 'unpublished', 'modified', 'receipt_changed', 'grant_changed'])
def test_resume_requires_published_addendum_original_receipts_and_exact_budget(
    resume: Any, tmp_path: Path, violation: str,
) -> None:
    def git(*args: str) -> str:
        return subprocess.check_output(['git', '-C', str(tmp_path), *args], text=True, stderr=subprocess.DEVNULL).strip()

    def write(path: Path, value: dict[str, Any]) -> None:
        path.write_text(json.dumps(value))

    git('init', '-b', 'fixture')
    report = tmp_path / 'report'
    report.mkdir()
    freeze, side = tmp_path / 'freeze.json', tmp_path / 'side.json'
    frozen = {'report': str(report), 'budget': {'resource': 'all', 'wall_seconds': 7170,
              'outer_timeout_seconds': 7190, 'disk_limit_bytes': 4_800_000_000,
              'allocator_bytes': 7 * 1024**3, 'vram_stop_bytes': 9_000_000_000}}
    write(freeze, frozen)
    write(side, {'historical': True})
    identity = {'freeze_commit': 'original-person-freeze', 'inference_attempts': 0, 'scoring_batches': 0}
    opening, stop = report / 'opening.json', tmp_path / 'preflight-stop.json'
    write(opening, {**identity, 'freeze': file_identity(freeze)})
    write(stop, {**identity, 'status': 'not_enqueued', 'reserved_media_decoded': False,
                 'reserved_person_labels_opened': False, 'side_reference': file_identity(side)})
    addendum = tmp_path / 'addendum.json'
    value = {'schema': 'i964_unseen_resume_v1', 'freeze': file_identity(freeze),
             'freeze_commit': identity['freeze_commit'], 'report': str(report),
             'budget': resume.resumed_budget(frozen), 'allowed_missing_sides': ['video_001/clip_003'],
             'previous_opening': file_identity(opening), 'previous_stop': file_identity(stop)}
    if violation == 'grant_changed':
        value['budget']['allocator_bytes'] += 1
    write(addendum, value)
    git('add', '.')
    git('-c', 'core.hooksPath=/dev/null', '-c', 'user.name=Fixture', '-c', 'user.email=fixture@example.invalid',
        'commit', '-m', 'explicit resumption')
    commit, remote = git('rev-parse', 'HEAD'), 'origin/fixture'
    if violation != 'unpublished':
        git('update-ref', f'refs/remotes/{remote}', commit)
    if violation == 'modified':
        addendum.write_text(addendum.read_text() + '\n')
    if violation == 'receipt_changed':
        stop.write_text(stop.read_text() + '\n')
    originals = {p: p.read_bytes() for p in (freeze, opening, stop)}
    if violation == 'none':
        assert resume.load_addendum(addendum, commit, freeze, identity['freeze_commit'], frozen,
                                    root=tmp_path, remote=remote) == value
    else:
        expected = subprocess.CalledProcessError if violation == 'unpublished' else ValueError
        with pytest.raises(expected):
            resume.load_addendum(addendum, commit, freeze, identity['freeze_commit'], frozen,
                                 root=tmp_path, remote=remote)
    assert all(p.read_bytes() == content for p, content in originals.items())


@pytest.mark.parametrize('extra', ['resumed-opening-r17.json', 'plan.json', 'attempt.json', 'scoring'])
def test_resume_cannot_retry_preparation_or_inference(resume: Any, tmp_path: Path, extra: str) -> None:
    (tmp_path / 'opening.json').write_text('original receipt')
    resume.require_unstarted_directory(tmp_path)
    (tmp_path / extra).touch()
    with pytest.raises(FileExistsError, match='original opening'):
        resume.require_unstarted_directory(tmp_path)
