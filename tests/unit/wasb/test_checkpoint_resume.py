from __future__ import annotations

import time
from pathlib import Path

import pytest
from omegaconf import OmegaConf

from src.wasb.utils.checkpoint import resolve_resume_ckpt_path


@pytest.mark.unit
def test_resolve_resume_ckpt_path_prefers_latest_last_ckpt(tmp_path: Path) -> None:
    output_dir = tmp_path

    ckpt_a = output_dir / "logs" / "version_0" / "checkpoints" / "last.ckpt"
    ckpt_a.parent.mkdir(parents=True, exist_ok=True)
    ckpt_a.write_text("a")

    time.sleep(0.01)

    ckpt_b = output_dir / "logs" / "version_1" / "checkpoints" / "last.ckpt"
    ckpt_b.parent.mkdir(parents=True, exist_ok=True)
    ckpt_b.write_text("b")

    cfg = OmegaConf.create(
        {
            "training": {
                "resume": {
                    "enabled": True,
                    "ckpt_path": None,
                    "auto_last": True,
                }
            }
        }
    )

    resolved = resolve_resume_ckpt_path(
        args_resume=None,
        config=cfg,
        output_dir=output_dir,
    )
    assert resolved == str(ckpt_b)


@pytest.mark.unit
def test_resolve_resume_ckpt_path_raises_when_enabled_but_missing(tmp_path: Path) -> None:
    cfg = OmegaConf.create(
        {
            "training": {
                "resume": {
                    "enabled": True,
                    "ckpt_path": None,
                    "auto_last": True,
                }
            }
        }
    )

    with pytest.raises(FileNotFoundError):
        resolve_resume_ckpt_path(
            args_resume=None,
            config=cfg,
            output_dir=tmp_path,
        )


@pytest.mark.unit
def test_resolve_resume_ckpt_path_cli_takes_priority(tmp_path: Path) -> None:
    ckpt = tmp_path / "some.ckpt"
    ckpt.write_text("x")

    cfg = OmegaConf.create(
        {
            "training": {
                "resume": {
                    "enabled": True,
                    "ckpt_path": None,
                    "auto_last": True,
                }
            }
        }
    )

    resolved = resolve_resume_ckpt_path(
        args_resume=str(ckpt),
        config=cfg,
        output_dir=tmp_path,
    )
    assert resolved == str(ckpt)


@pytest.mark.unit
def test_resolve_resume_ckpt_path_disabled_returns_none(tmp_path: Path) -> None:
    cfg = OmegaConf.create(
        {
            "training": {
                "resume": {
                    "enabled": False,
                    "ckpt_path": None,
                    "auto_last": True,
                }
            }
        }
    )

    resolved = resolve_resume_ckpt_path(
        args_resume=None,
        config=cfg,
        output_dir=tmp_path,
    )
    assert resolved is None
