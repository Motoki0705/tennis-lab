from __future__ import annotations

from pathlib import Path

import pytest

from src.tasks.base.annotation.workers import isolation


def test_private_codex_home_shares_credentials_without_other_sessions_or_config(
    tmp_path: Path,
) -> None:
    source = tmp_path / "shared"
    source.mkdir()
    (source / "auth.json").write_text('{"synthetic_credentials":true}')
    (source / "config.toml").write_text('model="unwanted"')
    (source / "sessions").mkdir()
    (source / "sessions/other-camera.jsonl").write_text("other-camera input")
    (tmp_path / "attempt/work").mkdir(parents=True)
    home = isolation.private_home(tmp_path / "attempt", source)
    assert (home / "auth.json").read_bytes() == (source / "auth.json").read_bytes()
    assert sorted(p.name for p in home.iterdir()) == ["auth.json"]
    assert (home / "auth.json").stat().st_mode & 0o777 == 0o600
    assert isolation.private_home(tmp_path / "attempt", source) == home


def test_unavailable_kernel_backend_never_runs_without_a_read_boundary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(isolation.platform, "system", lambda: "unsupported")
    with pytest.raises(RuntimeError, match="Landlock"):
        isolation.restrict_attempt(tmp_path, Path("/usr/bin/true"))


def test_original_media_cannot_be_placed_inside_readable_system_runtime(
    tmp_path: Path,
) -> None:
    isolation.require_private_paths((tmp_path / "repository", tmp_path / "cam2.mp4"))
    for path in ("/usr/lib/cam2.mp4", "/etc/ssl/certs/cam2.mp4"):
        with pytest.raises(ValueError, match="readable system runtime"):
            isolation.require_private_paths((tmp_path / "cam0.mp4", Path(path)))
