"""Package ball detection code and dataset into 7z archives.

Example commands:
    `uv run python -m src.tasks.ball_detection.scripts.package`
    `uv run python -m src.tasks.ball_detection.scripts.package package_target=code`
    `uv run python -m src.tasks.ball_detection.scripts.package package_target=data overwrite=true`

Config entry point: `src/tasks/ball_detection/configs/package.yaml`
"""

from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import hydra
from hydra.core.hydra_config import HydraConfig
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf

if TYPE_CHECKING:
    from collections.abc import Iterable


VALID_PACKAGE_TARGETS = {"both", "code", "data"}
EXCLUDE_PATTERNS = (
    "-xr!__pycache__",
    "-xr!.pytest_cache",
    "-xr!.mypy_cache",
    "-x!*.pyc",
    "-x!*.pyo",
)


@dataclass(slots=True)
class ArchiveSummary:
    """Metadata for one generated archive."""

    target: str
    archive_name: str
    archive_path: str
    included_paths: list[str]
    size_bytes: int


def _repo_root() -> Path:
    return Path(to_absolute_path(".")).resolve()


def _resolve_output_dir(cfg: DictConfig) -> Path:
    output_dir = Path(to_absolute_path(str(cfg.output_dir))).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def _resolve_archive_paths(paths: Iterable[object]) -> list[Path]:
    repo_root = _repo_root()
    resolved_paths: list[Path] = []
    for raw_path in paths:
        path_str = str(raw_path).strip()
        if not path_str:
            raise ValueError("Archive path entries must not be empty.")
        candidate = (repo_root / path_str).resolve()
        if not candidate.exists():
            raise FileNotFoundError(f"Archive input path does not exist: {path_str}")
        resolved_paths.append(candidate)
    return resolved_paths


def _relative_archive_paths(paths: Iterable[Path]) -> list[str]:
    repo_root = _repo_root()
    relative_paths: list[str] = []
    for path in paths:
        try:
            relative_paths.append(path.relative_to(repo_root).as_posix())
        except ValueError as exc:
            raise ValueError(
                f"Archive input path must stay inside the repository: {path}"
            ) from exc
    return relative_paths


def _validate_target(cfg: DictConfig) -> str:
    target = str(cfg.package_target).strip().lower()
    if target not in VALID_PACKAGE_TARGETS:
        raise ValueError(
            "`package_target` must be one of: both, code, data. "
            f"Received: {cfg.package_target!r}"
        )
    return target


def _target_names(target: str) -> list[str]:
    if target == "both":
        return ["code", "data"]
    return [target]


def _resolve_7z_binary() -> str:
    binary = shutil.which("7z")
    if binary is None:
        raise RuntimeError("7z command not found. Install p7zip/7-Zip before packaging.")
    return binary


def _build_archive_command(
    *,
    archive_path: Path,
    relative_inputs: list[str],
    compression_level: int,
) -> list[str]:
    return [
        _resolve_7z_binary(),
        "a",
        f"-mx={compression_level}",
        *EXCLUDE_PATTERNS,
        str(archive_path),
        *relative_inputs,
    ]


def _package_target(
    *,
    cfg: DictConfig,
    target_name: str,
    output_dir: Path,
) -> ArchiveSummary:
    target_cfg = cfg.get(target_name)
    if target_cfg is None:
        raise ValueError(f"Missing `{target_name}` section in package config.")

    archive_name = str(target_cfg.archive_name).strip()
    if not archive_name:
        raise ValueError(f"`{target_name}.archive_name` must not be empty.")

    archive_path = output_dir / archive_name
    if archive_path.exists():
        if not bool(cfg.overwrite):
            raise FileExistsError(
                f"Refusing to overwrite existing archive: {archive_path}"
            )
        archive_path.unlink()

    raw_paths = OmegaConf.to_container(target_cfg.paths, resolve=True)
    if not isinstance(raw_paths, list) or not raw_paths:
        raise ValueError(f"`{target_name}.paths` must be a non-empty list.")

    resolved_paths = _resolve_archive_paths(raw_paths)
    relative_inputs = _relative_archive_paths(resolved_paths)
    command = _build_archive_command(
        archive_path=archive_path,
        relative_inputs=relative_inputs,
        compression_level=int(cfg.compression.level),
    )
    subprocess.run(
        command,
        check=True,
        cwd=_repo_root(),
    )

    return ArchiveSummary(
        target=target_name,
        archive_name=archive_name,
        archive_path=str(archive_path),
        included_paths=relative_inputs,
        size_bytes=archive_path.stat().st_size,
    )


def _write_summary(output_dir: Path, summaries: list[ArchiveSummary]) -> None:
    hydra_output_dir = Path(HydraConfig.get().runtime.output_dir)
    hydra_output_dir.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "output_dir": str(output_dir),
        "archives": [asdict(summary) for summary in summaries],
    }
    summary_text = json.dumps(payload, indent=2, ensure_ascii=False)
    (hydra_output_dir / "package_summary.json").write_text(
        summary_text,
        encoding="utf-8",
    )


@hydra.main(
    version_base=None,
    config_path="../configs",
    config_name="package",
)
def main(cfg: DictConfig) -> None:
    """Create one or both archives under outputs/ball_detection/packages."""
    target = _validate_target(cfg)
    output_dir = _resolve_output_dir(cfg)
    summaries = [
        _package_target(cfg=cfg, target_name=target_name, output_dir=output_dir)
        for target_name in _target_names(target)
    ]
    _write_summary(output_dir, summaries)


if __name__ == "__main__":
    main()
