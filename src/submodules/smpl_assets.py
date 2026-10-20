"""Provenance and local installation of user-acquired SMPL auxiliary assets."""

from __future__ import annotations

import hashlib
import shutil
from dataclasses import dataclass
from pathlib import Path

GVHMR_ASSET_COMMIT = "ee960bb6e2ea2d381aa97f08e9b71ef320b624b1"
GVHMR_ASSET_ROOT = f"https://github.com/zju3dv/GVHMR/blob/{GVHMR_ASSET_COMMIT}"
SMPL_ASSET_README = "src/submodules/README.md#smpl補助資産の取得"


@dataclass(frozen=True, slots=True)
class SmplAssetSource:
    """One byte-pinned file published in the official GVHMR repository."""

    upstream_path: str
    sha256: str

    @property
    def source_url(self) -> str:
        return f"{GVHMR_ASSET_ROOT}/{self.upstream_path}"

    @property
    def checkpoint_path(self) -> Path:
        return Path("body_models/gvhmr") / Path(self.upstream_path).name


SMPL_ASSET_SOURCES = {
    "hmr2_mean_params": SmplAssetSource(
        "hmr4d/network/hmr2/configs/smpl_mean_params.npz",
        "6fd6dd687800da946d0a0492383f973b92ec20f166a0b829775882868c35fcdd",
    ),
    "smplx_to_smpl": SmplAssetSource(
        "hmr4d/utils/body_model/smplx2smpl_sparse.pt",
        "0fc821a9e79ec3e76d6a9796b96d5bef8cd67055e18497bbe370d8aed9e07e06",
    ),
    "smpl_coco17_regressor": SmplAssetSource(
        "hmr4d/utils/body_model/smpl_coco17_J_regressor.pt",
        "bacdaf756629493994cc869f4c27d179f5e4a5d06b8797ee3dcb94571522079f",
    ),
    "smplx_verts437": SmplAssetSource(
        "hmr4d/utils/body_model/smplx_verts437.pt",
        "ef0ea64c470a1fea80adb5e4b866c78a792a04f5f98744c9367b15e57bfb4a4d",
    ),
    "smpl_neutral_joint_regressor": SmplAssetSource(
        "hmr4d/utils/body_model/smpl_neutral_J_regressor.pt",
        "70e3213bd30fe8d8ce37b54675282745e406f915a51511a003aeff99b6da04cf",
    ),
}


def require_smpl_asset(path: Path, name: str) -> None:
    """Stop with the exact upstream file and placement instructions if absent."""
    source = SMPL_ASSET_SOURCES[name]
    if not path.is_file():
        raise FileNotFoundError(
            f"Required SMPL asset '{Path(source.upstream_path).name}' is missing: {path}. "
            f"Obtain it from the official GVHMR repository: {source.source_url}. "
            f"Follow {SMPL_ASSET_README} to accept the upstream terms and install "
            "the user-acquired file at the required path above."
        )


def install_smpl_assets(*, gvhmr_root: Path, checkpoint_root: Path) -> tuple[Path, ...]:
    """Verify all five upstream files before copying any into checkpoint storage.

    This function performs no downloads and does not regenerate model data.
    Existing files must already match; a conflicting file is never overwritten.
    """
    checkpoint_root = checkpoint_root.resolve()
    pairs: list[tuple[Path, Path]] = []
    for name, source in SMPL_ASSET_SOURCES.items():
        original = gvhmr_root / source.upstream_path
        require_smpl_asset(original, name)
        if hashlib.sha256(original.read_bytes()).hexdigest() != source.sha256:
            raise ValueError(
                f"SHA-256 mismatch for {original}; expected {source.sha256}. "
                f"Use the official GVHMR asset at {source.source_url}."
            )
        target = checkpoint_root / source.checkpoint_path
        if not target.resolve().is_relative_to(checkpoint_root):
            raise ValueError(f"SMPL asset destination escapes checkpoint root: {target}")
        if target.exists() and (
            not target.is_file()
            or hashlib.sha256(target.read_bytes()).hexdigest() != source.sha256
        ):
            raise FileExistsError(f"Conflicting SMPL asset already exists: {target}")
        pairs.append((original, target))
    for original, target in pairs:
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(original, target)
    return tuple(target for _, target in pairs)
