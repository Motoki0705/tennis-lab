"""ACCAD review resolves body models independently of input motion data."""

from pathlib import Path
from unittest.mock import Mock

import pytest

from src.tasks.plcs.scripts import review_accad_motion as cli


@pytest.mark.parametrize('custom_root', [False, True])
def test_review_uses_checkpoint_root_for_body_models(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, custom_root: bool,
) -> None:
    data_root = tmp_path / 'motions'
    accad = data_root / 'ACCAD'
    checkpoint_root = tmp_path / ('published' if custom_root else 'ckpt')
    models = checkpoint_root / 'body_models/smplh'
    accad.mkdir(parents=True)
    models.mkdir(parents=True)
    argv = ['review', '--data-root', str(data_root)]
    if custom_root:
        argv += ['--checkpoint-root', str(checkpoint_root)]
    monkeypatch.setattr('sys.argv', argv)
    monkeypatch.setattr(cli, 'PROJECT_ROOT', tmp_path)
    service, run = Mock(), Mock()
    monkeypatch.setattr(cli, 'ReviewService', service)
    monkeypatch.setattr(cli, 'create_app', lambda _: 'app')
    monkeypatch.setattr(cli.uvicorn, 'run', run)
    cli.main()
    service.assert_called_once_with(accad, models)
    run.assert_called_once_with('app', host='127.0.0.1', port=8769)
