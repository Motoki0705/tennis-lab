"""The verified inputs to the published CameraHMR-based FastHMR model."""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from src.utils.models.loading.source_assets import python_source_assets

UPSTREAM_COMMIT = "dc5c8daf0beea56a419c631f494ef7bd7ce35d3c"
CHECKPOINT_REVISION = "898332e1059fc0579f986e03cab11471d4e27b65"
CHECKPOINT_SHA256 = {
    "backbone": "8d814cbf95b0efac89a939db055b6f67feacab0c85a43d3d307f49de313c9d4e",
    "diffusion": "ba9dcf7dfa3bed46c01fb814c25b88cece68919328ea7b757bbf30a20b1fcccc",
    "vae": "218055d749b0a00118cfaf26efc4dac8b1c7c6840884e083b108d3da33c95b07",
    "segmentor": "4e53a5f5fd3ee2ae3361c62169c6bb3ed4ae251dd0e57606e230955aa52d919c",
    "hmr2_backbone": "07a3bc7fca47010b3f4e0dbe75a4a1341b832955e42ca97ddf228a2e1ff5be30",
    "hmr2_diffusion": "603e56b947cb51867c03b285a6b9a55095b241530ff3541b492f6e9b09aca2f8",
}

IMAGE_SIZE = 256
WINDOW_LENGTH = 243
MERGE_RATIO = 40
MODEL_ARCHITECTURE = "fasthmr_camerahmr_eclm_mask_tome_diffusion"

FastHMRBackend = Literal["pt32", "pt16"]


@dataclass(frozen=True)
class FastHMRConfig:
    repository: Path
    assets: Path
    coco17_regressor: Path
    device: str
    backend: FastHMRBackend
    feature_batch_size: int
    seed: int

    @property
    def geometry_weights(self) -> dict[str, Path]:
        return {
            "smpl": self.assets / "body_models/SMPL_NEUTRAL.pkl",
            "h36m": self.assets / "body_models/J_regressor_h36m.npy",
            "coco17": self.coco17_regressor,
        }

    @property
    def inference_weights(self) -> dict[str, Path]:
        return {
            "backbone": self.assets / "feature_extractors/camerahmr.pth.tr",
            "fov": self.assets / "feature_extractors/cam_model_cleaned.ckpt",
            "segmentor": self.assets / "feature_extractors/yolo11x-seg.pt",
            "diffusion": self.assets / "fasthmr_ckpts/camerahmr_diffdecoder.pth.tr",
            "vae": self.assets / "fasthmr_ckpts/vae.pth.tr",
            "mean_params": self.assets / "body_models/smpl_mean_params.npz",
        }

    @property
    def demo_config(self) -> Path:
        return self.repository / "configs/demo.yaml"

    def implementation_assets(self) -> dict[str, Path]:
        return dict(python_source_assets(self.repository, prefix="fasthmr_source"))

    def verify_repository(self) -> dict[str, Path]:
        """Use the exact release and reject local code or configuration changes."""
        if not self.repository.is_dir():
            raise FileNotFoundError(f"FastHMR release checkout is missing: {self.repository}")
        try:
            root = self._git("rev-parse", "--show-toplevel")
            head = self._git("rev-parse", "HEAD")
            changes = self._git("status", "--porcelain", "--untracked-files=no")
            tracked_sources = set(self._git("ls-files", "--", "*.py").splitlines())
        except subprocess.CalledProcessError as error:
            raise ValueError(f"FastHMR requires a clean Git checkout at {self.repository}") from error
        if Path(root).resolve() != self.repository.resolve() or head != UPSTREAM_COMMIT or changes:
            raise ValueError(f"FastHMR requires its own clean checkout at {UPSTREAM_COMMIT}: {self.repository}")
        sources = self.implementation_assets()
        if {path.relative_to(self.repository).as_posix() for path in sources.values()} != tracked_sources:
            raise ValueError(f"FastHMR checkout has unexpected Python sources: {self.repository}")
        return sources

    def _git(self, *args: str) -> str:
        return subprocess.run(
            ["git", "-C", str(self.repository), *args],
            check=True, capture_output=True, text=True, timeout=15,
        ).stdout.strip()
