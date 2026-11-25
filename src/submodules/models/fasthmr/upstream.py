"""Bind the published FastHMR modules to an explicit checkout and asset paths."""

from __future__ import annotations

import sys
from collections.abc import Iterator
from contextlib import contextmanager
from importlib import import_module
from pathlib import Path
from threading import RLock
from typing import Any

import numpy as np
import torch
from torch import Tensor
from yaml import safe_load

from src.submodules.models.fasthmr.assets import (
    MERGE_RATIO,
    FastHMRConfig,
    require_checkpoint_sha256,
)
from src.utils.checksum import dual_sha256
from src.utils.models.loading.source_assets import validate_source_modules

_PACKAGE_ROOTS = ("model", "configs", "data", "tome")
_IMPORT_LOCK = RLock()


def _require_pinned_asset(name: str, path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Required FastHMR asset is missing: {path}")
    require_checkpoint_sha256(name, path, dual_sha256(path))


@contextmanager
def _official_imports(config: FastHMRConfig) -> Iterator[None]:
    # Upstream uses absolute imports and inserts its checkout on sys.path.
    # Retain its imported classes, while limiting the search path adjustment
    # and removing any partial upstream modules after initialization errors.
    with _IMPORT_LOCK:
        sources = config.verify_repository()
        validate_source_modules(config.repository, sources, _PACKAGE_ROOTS)
        previous_path, previous_modules = sys.path[:], set(sys.modules)
        sys.path.insert(0, str(config.repository))
        try:
            yield
            validate_source_modules(config.repository, sources, _PACKAGE_ROOTS)
        except BaseException:
            for name in sys.modules.keys() - previous_modules:
                if name.split(".", 1)[0] in _PACKAGE_ROOTS:
                    sys.modules.pop(name)
            raise
        finally:
            sys.path[:] = previous_path


def load_official_smpl(config: FastHMRConfig) -> Any:
    weights = config.geometry_weights
    for name in ("h36m", "smpl"):
        _require_pinned_asset(name, weights[name])
    with _official_imports(config):
        order = import_module("configs.constants").KEYPOINTS.H36M_TO_J14
        layer = import_module("model.smpl.SMPL").SMPL_layer
        regressor = np.load(weights["h36m"], allow_pickle=False)
        if regressor.shape != (17, 6890) or not np.isfinite(regressor).all():
            raise ValueError("FastHMR Human3.6M regressor must be finite (17, 6890)")
        return layer(model_path=str(weights["smpl"]),
                     h36m_jregressor=torch.from_numpy(regressor[order]).float()).eval().to(config.device)


def load_official_camera_models(config: FastHMRConfig) -> tuple[Any, Any, Any, Any, Any]:
    """Load the released ECLM backbone, mask patch and diffusion/VAE decoder."""
    weights = config.inference_weights
    for name, path in weights.items():
        _require_pinned_asset(name, path)
    with _official_imports(config):
        camera = import_module("data.preprocess.camerahmr.camerahmr_model")
        head = import_module("data.preprocess.camerahmr.heads.smpl_head_cliff")
        fov_type = import_module("data.preprocess.camerahmr.cam_model.fl_net").FLNet
        patch = import_module("model.hmr2_tools.patch_vit")
        diffusion_type = import_module("model.fasthmr").FastHMR
        crop = import_module("data.preprocess.camerahmr.datasets.utils").generate_image_patch_cv2

        saved = torch.load(weights["backbone"], map_location="cpu", weights_only=True)
        previous_mean_params = head.__dict__["SMPL_MEAN_PARAMS_FILE"]
        try:
            head.__dict__["SMPL_MEAN_PARAMS_FILE"] = str(weights["mean_params"])
            backbone = camera.CameraHMR(num_layers=saved["n_blocks"])
        finally:
            head.__dict__["SMPL_MEAN_PARAMS_FILE"] = previous_mean_params
        backbone.load_state_dict(saved["weights"], strict=True)
        del saved
        patch.apply_patch(backbone.backbone)
        if not all(isinstance(block, patch.ToMeBlock) for block in backbone.backbone.blocks):
            raise ValueError("The official Mask-ToMe patch did not attach to every CameraHMR ViT block")
        backbone.backbone.r = MERGE_RATIO
        fov = fov_type()
        # The licensed Lightning checkpoint includes the release's metadata.
        fov.load_state_dict(torch.load(weights["fov"], map_location="cpu", weights_only=False)["state_dict"], strict=True)

        options = safe_load(config.demo_config.read_bytes())
        del options["model_type"]
        options["vae_ckpt_path"] = str(weights["vae"])

        class FastHMRWithCPUWeights(diffusion_type):  # type: ignore[valid-type,misc]
            def load_vae_weights(self, checkpoint_path: str) -> None:
                # The release stores CUDA-tagged tensors. Restore them without
                # allocating on the device recorded by the checkpoint.
                checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
                self.vae.load_state_dict(checkpoint["model"], strict=True)

        with torch.device("cpu"):
            diffusion = FastHMRWithCPUWeights(**options)
        diffusion.load_state_dict(torch.load(weights["diffusion"], map_location="cpu", weights_only=True)["model"], strict=True)

    from ultralytics import YOLO

    return (backbone.eval().to(config.device), fov.eval().to(config.device),
            diffusion.eval().to(config.device), YOLO(str(weights["segmentor"])), crop)


class OfficialSMPL:
    """Use the same licensed SMPL layer for HybrIK and downstream geometry."""

    def __init__(self, config: FastHMRConfig) -> None:
        self.config = config
        self.layer: Any = None
        self.coco17: Tensor | None = None

    def load(self) -> None:
        self.layer = load_official_smpl(self.config)
        regressor = torch.load(self.config.coco17_regressor, map_location="cpu", weights_only=True)
        if not isinstance(regressor, Tensor) or regressor.shape != (17, 6890) or not bool(regressor.isfinite().all()):
            raise ValueError("FastHMR COCO17 requires the SMPL 6890-vertex regressor")
        self.coco17 = regressor.to(device=self.config.device, dtype=torch.float32)

    def unload(self) -> None:
        self.layer, self.coco17 = None, None

    def rotations(self, pose: Tensor, shape: Tensor) -> Tensor:
        if self.layer is None:
            raise RuntimeError("FastHMR SMPL geometry has not been loaded")
        result = self.layer.hybrik(
            pose_skeleton=pose[:, :87].reshape(-1, 29, 3),
            phis=pose[:, 87:].reshape(-1, 23, 2),
            betas=shape, global_orient=None,
        )
        return result.rot_mats
