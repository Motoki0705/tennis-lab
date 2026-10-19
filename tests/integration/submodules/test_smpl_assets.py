"""User-installed SMPL auxiliary files retain the original released bytes."""

import hashlib

import pytest

from src.submodules.smpl_assets import SMPL_ASSET_SOURCES
from src.utils.paths import PROJECT_ROOT


@pytest.mark.integration
@pytest.mark.local_data
@pytest.mark.parametrize("name", SMPL_ASSET_SOURCES)
def test_installed_asset_matches_released_bytes(name: str) -> None:
    source = SMPL_ASSET_SOURCES[name]
    path = PROJECT_ROOT / "ckpt" / source.checkpoint_path
    if not path.is_file():
        pytest.skip(f"missing local SMPL asset: {path}")
    assert hashlib.sha256(path.read_bytes()).hexdigest() == source.sha256
