"""Capacity fixture for tests that run WSL launch paths on ordinary Linux."""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.support.fake_d_capacity import sitecustomize

PROBE_PATH = str(Path(sitecustomize.__file__).parent)
CONFIG_ENV = "TENNIS_D_CAPACITY_CONFIG"
GIB = 1 << 30


def set_available(path: Path, available_bytes: int) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(str(available_bytes), encoding="ascii")
    temporary.replace(path)


@pytest.fixture
def fake_d_drive(
    tmp_path_factory: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch
) -> Path:
    probe_dir: Path = tmp_path_factory.mktemp("d-capacity-probe")
    path = probe_dir / "available-d-bytes"
    set_available(path, 55 * GIB)
    monkeypatch.setenv(sitecustomize.BYTES_FILE, str(path))
    config = probe_dir / "d-capacity.toml"
    config.write_text("threshold_gib = 20\n", encoding="ascii")
    monkeypatch.setenv(CONFIG_ENV, str(config))
    sitecustomize.install(monkeypatch.setattr)
    return path
