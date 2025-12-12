from __future__ import annotations

from pathlib import Path
from typing import Any


def _cfg_get(cfg: Any, key: str, default: Any = None) -> Any:
    if cfg is None:
        return default
    if hasattr(cfg, "get"):
        try:
            return cfg.get(key, default)
        except TypeError:
            return default
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return default


def resolve_resume_ckpt_path(
    *,
    args_resume: str | None,
    config: Any,
    output_dir: Path,
) -> str | None:
    def _validate_file(p: Path, *, context: str) -> str:
        if not p.is_file():
            raise FileNotFoundError(f"Resume checkpoint not found ({context}): {p}")
        return str(p)

    if args_resume is not None:
        return _validate_file(Path(args_resume), context="--resume")

    training_cfg = _cfg_get(config, "training", {})
    resume_cfg = _cfg_get(training_cfg, "resume", None)

    if resume_cfg is None:
        return None

    enabled = bool(_cfg_get(resume_cfg, "enabled", False))
    if not enabled:
        return None

    ckpt_path = _cfg_get(resume_cfg, "ckpt_path", None)
    if ckpt_path:
        return _validate_file(Path(ckpt_path), context="training.resume.ckpt_path")

    auto_last = bool(_cfg_get(resume_cfg, "auto_last", False))
    if auto_last:
        candidates = list(output_dir.glob("logs/version_*/checkpoints/last.ckpt"))
        if not candidates:
            raise FileNotFoundError(
                f"Resume enabled but no last.ckpt found under: {output_dir / 'logs'}"
            )
        latest = max(candidates, key=lambda p: p.stat().st_mtime)
        return _validate_file(latest, context="training.resume.auto_last")

    raise ValueError(
        "Resume enabled but neither training.resume.ckpt_path nor training.resume.auto_last is set."
    )
