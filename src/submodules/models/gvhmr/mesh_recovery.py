"""GVHMR SMPL-X mesh recovery (typed wrapper over the vendored pipeline)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

from src.submodules.configuration import (
    BundledModelAssetPaths,
    require_absolute_path,
)
from src.submodules.models._base import BaseInferenceModel
from src.submodules.vendor.gvhmr.body_model import make_smplx
from src.submodules.vendor.gvhmr.pipeline import GvhmrDemoModel, build_gvhmr_demo_model
from src.submodules.vendor.gvhmr.utils.geo_transform import compute_cam_angvel
from src.submodules.vendor.gvhmr.utils.hmr_cam import estimate_K


@dataclass(frozen=True)
class GvhmrRequest:
    """Request for SMPL-X recovery of one tracked person.

    Attributes:
        kp2d: COCO-17 keypoints ``(F, 17, 3)`` (x, y, confidence) in pixels.
        bbx_xys: Square person boxes ``(F, 3)`` (center_x, center_y, size).
        f_imgseq: HMR2 features ``(F, 1024)``.
        width: Full-image width in pixels.
        height: Full-image height in pixels.
        K_fullimg: Optional pinhole intrinsics ``(3, 3)``; estimated from the
            image size when omitted.
        R_w2c: Optional world-to-camera rotations ``(F, 3, 3)``; identity
            (static camera) when omitted.
        static_cam: Use the static-camera post-processing prior.
    """

    kp2d: torch.Tensor
    bbx_xys: torch.Tensor
    f_imgseq: torch.Tensor
    width: int
    height: int
    static_cam: bool
    K_fullimg: torch.Tensor | None = None
    R_w2c: torch.Tensor | None = None


@dataclass(frozen=True)
class GvhmrResult:
    """SMPL-X parameters predicted by GVHMR for one person.

    Both dicts contain ``body_pose (F, 63)``, ``betas (F, 10)``,
    ``global_orient (F, 3)`` and ``transl (F, 3)`` (float32, CPU).

    Attributes:
        smpl_params_incam: Parameters in the camera coordinate frame.
        smpl_params_global: Parameters in the gravity-aligned world frame.
        K_fullimg: Intrinsics used for the prediction ``(F, 3, 3)``.
    """

    smpl_params_incam: dict[str, torch.Tensor]
    smpl_params_global: dict[str, torch.Tensor]
    K_fullimg: torch.Tensor


class GvhmrMeshRecovery(BaseInferenceModel[GvhmrRequest, GvhmrResult]):
    """GVHMR (SIGA24 release) video-to-SMPL-X regressor."""

    def __init__(
        self,
        checkpoint: str | Path,
        body_models_dir: str | Path,
        *,
        device: str | torch.device,
        allow_device_fallback: bool,
        bundled_assets: BundledModelAssetPaths,
    ) -> None:
        super().__init__(device, allow_device_fallback=allow_device_fallback)
        self.checkpoint = require_absolute_path(checkpoint, name="GVHMR checkpoint")
        self.body_models_dir = require_absolute_path(
            body_models_dir, name="GVHMR body-model directory"
        )
        if not isinstance(bundled_assets, BundledModelAssetPaths):
            raise TypeError("bundled_assets must be BundledModelAssetPaths.")
        bundled_assets.require_files()
        self.bundled_assets = bundled_assets
        self._model: GvhmrDemoModel | None = None

    def _load_impl(self) -> None:
        if not self.checkpoint.exists():
            raise FileNotFoundError(f"GVHMR checkpoint not found: {self.checkpoint}")
        model = build_gvhmr_demo_model(
            checkpoint_path=self.checkpoint,
            body_models_dir=self.body_models_dir,
            bundled_assets=self.bundled_assets,
        )
        self._model = model.eval().to(self._device)

    def _unload_impl(self) -> None:
        self._model = None

    def _predict_impl(self, request: GvhmrRequest) -> GvhmrResult:
        assert self._model is not None
        num_frames = request.kp2d.shape[0]

        K = request.K_fullimg
        if K is None:
            K = estimate_K(request.width, request.height)
        K_fullimg = K.expand(num_frames, -1, -1).float()

        R_w2c = request.R_w2c
        if R_w2c is None:
            R_w2c = torch.eye(3).repeat(num_frames, 1, 1)
        cam_angvel = compute_cam_angvel(R_w2c)

        data = {
            "length": torch.tensor(num_frames),
            "kp2d": request.kp2d.float(),
            "bbx_xys": request.bbx_xys.float(),
            "K_fullimg": K_fullimg,
            "cam_angvel": cam_angvel,
            "f_imgseq": request.f_imgseq.float(),
        }
        pred = self._model.predict(data, static_cam=request.static_cam)

        def to_cpu(params: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
            return {k: v.detach().float().cpu() for k, v in params.items()}

        return GvhmrResult(
            smpl_params_incam=to_cpu(pred["smpl_params_incam"]),
            smpl_params_global=to_cpu(pred["smpl_params_global"]),
            K_fullimg=K_fullimg,
        )


class SmplVertexReconstructor:
    """Reconstruct SMPL vertices ``(F, 6890, 3)`` from GVHMR SMPL-X params.

    Runs the SMPL-X body model and converts vertices to the SMPL topology via
    the bundled sparse conversion matrix. Requires the licensed
    ``SMPLX_NEUTRAL.npz`` (see vendor README).
    """

    def __init__(
        self,
        body_models_dir: str | Path,
        *,
        device: str | torch.device,
        allow_device_fallback: bool,
        bundled_assets: BundledModelAssetPaths,
    ) -> None:
        from src.utils.device import resolve_device

        if type(allow_device_fallback) is not bool:
            raise TypeError("allow_device_fallback must be a bool.")
        self._device = resolve_device(
            device,
            allow_fallback=allow_device_fallback,
        )
        self._body_models_dir = require_absolute_path(
            body_models_dir, name="SMPL-X body-model directory"
        )
        if not isinstance(bundled_assets, BundledModelAssetPaths):
            raise TypeError("bundled_assets must be BundledModelAssetPaths.")
        bundled_assets.require_files()
        self._bundled_assets = bundled_assets
        self._smplx: torch.nn.Module | None = None
        self._smplx2smpl: torch.Tensor | None = None

    def _ensure_loaded(self) -> None:
        if self._smplx is not None:
            return
        smplx = make_smplx(
            "supermotion",
            model_path=self._body_models_dir,
            bundled_assets=self._bundled_assets,
        )
        self._smplx = smplx.to(self._device).eval()
        smplx2smpl = torch.load(
            self._bundled_assets.smplx_to_smpl,
            map_location="cpu",
            weights_only=False,
        )
        if not isinstance(smplx2smpl, torch.Tensor):
            smplx2smpl = torch.as_tensor(smplx2smpl)
        self._smplx2smpl = smplx2smpl.to(device=self._device, dtype=torch.float32)

    def reconstruct(self, smpl_params: dict[str, torch.Tensor]) -> torch.Tensor:
        """SMPL-X params (each ``(F, C)``) -> SMPL vertices ``(F, 6890, 3)``, CPU."""
        with torch.no_grad():
            self._ensure_loaded()
            assert self._smplx is not None and self._smplx2smpl is not None
            params = {k: v.to(self._device) for k, v in smpl_params.items()}
            smplx_out = self._smplx(**params)
            vertices = torch.stack(
                [torch.matmul(self._smplx2smpl, verts) for verts in smplx_out.vertices],
                dim=0,
            )
            return vertices.float().cpu()
