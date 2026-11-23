"""GVHMR SMPL-X mesh recovery (typed wrapper over the vendored pipeline)."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

import torch

from src.submodules.configuration import (
    SmplAssetPaths,
    require_absolute_path,
)
from src.submodules.models._base.inference_model import BaseInferenceModel
from src.submodules.vendor.gvhmr.body_model import make_smplx
from src.submodules.vendor.gvhmr.pipeline import GvhmrDemoModel, build_gvhmr_demo_model
from src.submodules.vendor.gvhmr.utils.geo_transform import compute_cam_angvel
from src.submodules.vendor.gvhmr.utils.hmr_cam import estimate_K
from src.utils.inference_recording import model_action, span
from src.utils.models.tensorrt_artifacts import PLAN_FILE, EngineBundle
from src.utils.models.tensorrt_runtime import TensorRTRunner


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

    def __post_init__(self) -> None:
        _validate_gvhmr_request(self)


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


class _TensorRTDenoiser(torch.nn.Module):
    def __init__(
        self, runner: Callable[[Mapping[str, torch.Tensor]], dict[str, torch.Tensor]]
    ) -> None:
        super().__init__()
        self.runner = runner

    def forward(self, *, length: torch.Tensor, **conditions: torch.Tensor) -> dict[str, torch.Tensor]:
        return self.runner({"length": length, **conditions})


def _adapt_tensorrt_denoiser_output(
    length: torch.Tensor, outputs: dict[str, torch.Tensor]
) -> dict[str, torch.Tensor]:
    with span("cuda_sync", operation="validate_sequence", model="GVHMR"):
        if not torch.equal(outputs.pop("length_out"), length) or not bool(outputs.pop("mask_out").all()):
            raise ValueError("GVHMR TensorRT changed the complete sequence length or mask")
    return {name: value.float() for name, value in outputs.items()}


class GvhmrMeshRecovery(BaseInferenceModel[GvhmrRequest, GvhmrResult]):
    """GVHMR (SIGA24 release) video-to-SMPL-X regressor."""

    def __init__(
        self,
        checkpoint: str | Path,
        body_models_dir: str | Path,
        *,
        device: str | torch.device,
        smpl_assets: SmplAssetPaths,
        engine: EngineBundle | None = None,
    ) -> None:
        super().__init__(device)
        self.checkpoint = require_absolute_path(checkpoint, name="GVHMR checkpoint")
        self.body_models_dir = require_absolute_path(
            body_models_dir, name="GVHMR body-model directory"
        )
        if not isinstance(smpl_assets, SmplAssetPaths):
            raise TypeError("smpl_assets must be SmplAssetPaths.")
        smpl_assets.require_files()
        self.smpl_assets = smpl_assets
        self.engine = engine
        self._model: GvhmrDemoModel | None = None

    def recording_assets(self) -> dict[str, Path]:
        inference = {"checkpoint": self.checkpoint} if self.engine is None else {"engine": self.engine.root / PLAN_FILE}
        return {**inference, "smplx": self.body_models_dir / "smplx" / "SMPLX_NEUTRAL.npz", **{field.name: getattr(self.smpl_assets, field.name) for field in fields(self.smpl_assets)}}

    def _load_impl(self) -> None:
        if not self.checkpoint.exists():
            raise FileNotFoundError(f"GVHMR checkpoint not found: {self.checkpoint}")
        kwargs: dict[str, Any] = {}
        if self.engine is not None:
            kwargs["denoiser"] = _TensorRTDenoiser(TensorRTRunner(self.engine, self.device))
            kwargs["adapt_denoiser_output"] = _adapt_tensorrt_denoiser_output
        model = build_gvhmr_demo_model(
            checkpoint_path=self.checkpoint,
            body_models_dir=self.body_models_dir,
            smpl_assets=self.smpl_assets,
            **kwargs,
        )
        self._model = model.eval().to(self._device)

    def _unload_impl(self) -> None:
        self._model = None

    def _predict_impl(self, request: GvhmrRequest) -> GvhmrResult:
        if self._model is None:
            raise RuntimeError("GVHMR model did not load before prediction.")
        with span("preprocess", model="GVHMR", operation="prepare_sequence"):
            data = self._prepare(request)
        with span("postprocess", model="GVHMR", operation="recover_body"):
            pred = self._model.predict(data, static_cam=request.static_cam)
        return self._result(pred, data["K_fullimg"])

    def predict_batch(self, requests: Sequence[GvhmrRequest]) -> list[GvhmrResult]:
        if not requests:
            return []
        if len(requests) == 1:
            return [self.predict(requests[0])]
        frames, static = len(requests[0].kp2d), requests[0].static_cam
        if any(len(request.kp2d) != frames or request.static_cam != static for request in requests):
            raise ValueError("A GVHMR batch requires equal lengths and a shared camera mode")
        with span("preprocess", model="GVHMR", operation="prepare_sequence"):
            data = [self._prepare(request) for request in requests]
        self.load()
        if self._model is None:
            raise RuntimeError("GVHMR model did not load before prediction.")
        with span("postprocess", model="GVHMR", operation="recover_body"):
            predictions = self._model.predict_batch(data, static_cam=static)
        return [self._result(pred, row["K_fullimg"]) for pred, row in zip(predictions, data, strict=True)]

    @staticmethod
    def _prepare(request: GvhmrRequest) -> dict[str, torch.Tensor]:
        # Revalidate immediately before the vendored model boundary.  Frozen
        # dataclasses prevent field replacement through the public API, but
        # tensors remain mutable and therefore cannot be trusted solely from
        # construction-time validation.
        _validate_gvhmr_request(request)
        num_frames = request.kp2d.shape[0]

        K = request.K_fullimg
        if K is None:
            K = estimate_K(request.width, request.height)
        K_fullimg = K.expand(num_frames, -1, -1).float()

        R_w2c = request.R_w2c
        if R_w2c is None:
            R_w2c = torch.eye(3).repeat(num_frames, 1, 1)
        cam_angvel = compute_cam_angvel(R_w2c)

        return {
            "length": torch.tensor(num_frames),
            "kp2d": request.kp2d.float(),
            "bbx_xys": request.bbx_xys.float(),
            "K_fullimg": K_fullimg,
            "cam_angvel": cam_angvel,
            "f_imgseq": request.f_imgseq.float(),
        }

    def _result(self, pred: dict[str, Any], K_fullimg: torch.Tensor) -> GvhmrResult:
        def to_cpu(params: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
            return {k: v.detach().float().cpu() for k, v in params.items()}

        with span("transfer", model="GVHMR", operation="output_to_host", device=str(self.device)):
            return GvhmrResult(
                smpl_params_incam=to_cpu(pred["smpl_params_incam"]),
                smpl_params_global=to_cpu(pred["smpl_params_global"]),
                K_fullimg=K_fullimg,
            )


def _validate_gvhmr_request(request: GvhmrRequest) -> None:
    """Validate a mesh request before any vendored GVHMR code is entered."""
    tensors = {
        "kp2d": request.kp2d,
        "bbx_xys": request.bbx_xys,
        "f_imgseq": request.f_imgseq,
    }
    for name, tensor in tensors.items():
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"GvhmrRequest.{name} must be a torch.Tensor.")
        if tensor.dtype != torch.float32:
            raise TypeError(
                f"GvhmrRequest.{name} must have dtype torch.float32, "
                f"got {tensor.dtype}."
            )

    num_frames = request.kp2d.shape[0] if request.kp2d.ndim > 0 else 0
    expected_shapes = {
        "kp2d": (num_frames, 17, 3),
        "bbx_xys": (num_frames, 3),
        "f_imgseq": (num_frames, 1024),
    }
    for name, expected_shape in expected_shapes.items():
        tensor = tensors[name]
        if tuple(tensor.shape) != expected_shape:
            raise ValueError(
                f"GvhmrRequest.{name} must have shape {expected_shape}, "
                f"got {tuple(tensor.shape)}."
            )
    if num_frames <= 0:
        raise ValueError("GvhmrRequest must contain at least one frame.")

    reference = request.kp2d
    for name in ("bbx_xys", "f_imgseq"):
        tensor = tensors[name]
        if tensor.device != reference.device:
            raise ValueError(
                "GvhmrRequest tensors must share one device; "
                f"kp2d is on {reference.device} but {name} is on {tensor.device}."
            )
    for name, tensor in tensors.items():
        if not bool(torch.isfinite(tensor).all()):
            raise ValueError(f"GvhmrRequest.{name} must contain only finite values.")
    if not bool((request.kp2d[..., 2] >= 0.0).all()):
        raise ValueError("GvhmrRequest.kp2d confidence values must be non-negative.")
    if not bool((request.bbx_xys[:, 2] > 0.0).all()):
        raise ValueError("GvhmrRequest.bbx_xys sizes must be positive.")

    for name, value in {"width": request.width, "height": request.height}.items():
        if type(value) is not int:
            raise TypeError(f"GvhmrRequest.{name} must be an integer.")
        if value <= 0:
            raise ValueError(f"GvhmrRequest.{name} must be positive, got {value}.")
    if type(request.static_cam) is not bool:
        raise TypeError("GvhmrRequest.static_cam must be a bool.")

    if request.K_fullimg is not None:
        _validate_optional_camera_tensor(
            request.K_fullimg,
            name="K_fullimg",
            expected_shape=(3, 3),
            reference=reference,
        )
        if not bool((request.K_fullimg.diagonal()[:2] > 0.0).all()):
            raise ValueError("GvhmrRequest.K_fullimg focal lengths must be positive.")
    if request.R_w2c is not None:
        _validate_optional_camera_tensor(
            request.R_w2c,
            name="R_w2c",
            expected_shape=(num_frames, 3, 3),
            reference=reference,
        )


def _validate_optional_camera_tensor(
    tensor: torch.Tensor,
    *,
    name: str,
    expected_shape: tuple[int, ...],
    reference: torch.Tensor,
) -> None:
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"GvhmrRequest.{name} must be a torch.Tensor.")
    if tuple(tensor.shape) != expected_shape:
        raise ValueError(
            f"GvhmrRequest.{name} must have shape {expected_shape}, "
            f"got {tuple(tensor.shape)}."
        )
    if tensor.dtype != reference.dtype:
        raise TypeError(
            f"GvhmrRequest.{name} must have dtype {reference.dtype}, "
            f"got {tensor.dtype}."
        )
    if tensor.device != reference.device:
        raise ValueError(
            f"GvhmrRequest.{name} must be on {reference.device}, got {tensor.device}."
        )
    if not bool(torch.isfinite(tensor).all()):
        raise ValueError(f"GvhmrRequest.{name} must contain only finite values.")


class SmplVertexReconstructor:
    """Reconstruct SMPL vertices ``(F, 6890, 3)`` from GVHMR SMPL-X params.

    Runs the SMPL-X body model and converts vertices to the SMPL topology via
    the user-acquired sparse conversion matrix. Requires the licensed
    ``SMPLX_NEUTRAL.npz`` (see vendor README).
    """

    def __init__(
        self,
        body_models_dir: str | Path,
        *,
        device: str | torch.device,
        smpl_assets: SmplAssetPaths,
    ) -> None:
        from src.utils.device import resolve_device

        self._device = resolve_device(device)
        self._body_models_dir = require_absolute_path(
            body_models_dir, name="SMPL-X body-model directory"
        )
        if not isinstance(smpl_assets, SmplAssetPaths):
            raise TypeError("smpl_assets must be SmplAssetPaths.")
        smpl_assets.require_files()
        self._smpl_assets = smpl_assets
        self._smplx: torch.nn.Module | None = None
        self._smplx2smpl: torch.Tensor | None = None

    def load(self) -> None:
        self._ensure_loaded()

    @model_action("load", "SmplVertexReconstructor", weights=lambda owner: {"smplx": owner._body_models_dir / "smplx" / "SMPLX_NEUTRAL.npz", **{field.name: getattr(owner._smpl_assets, field.name) for field in fields(owner._smpl_assets)}},
                  device=lambda owner: str(owner._device), loaded=lambda owner: owner._smplx is not None)
    def _ensure_loaded(self) -> None:
        if self._smplx is not None:
            return
        smplx = make_smplx(
            "supermotion",
            model_path=self._body_models_dir,
            smpl_assets=self._smpl_assets,
        )
        self._smplx = smplx.to(self._device).eval()
        smplx2smpl = torch.load(
            self._smpl_assets.smplx_to_smpl,
            map_location="cpu",
            weights_only=False,
        )
        if not isinstance(smplx2smpl, torch.Tensor):
            raise TypeError(
                "SMPL-X to SMPL conversion asset must contain a tensor."
            )
        self._smplx2smpl = smplx2smpl.to(device=self._device, dtype=torch.float32)

    @model_action("unload", "SmplVertexReconstructor", weights=lambda owner: {"smplx": owner._body_models_dir / "smplx" / "SMPLX_NEUTRAL.npz", **{field.name: getattr(owner._smpl_assets, field.name) for field in fields(owner._smpl_assets)}},
                  device=lambda owner: str(owner._device), loaded=lambda owner: owner._smplx is not None)
    def unload(self) -> None:
        """Release the lazily loaded body model and topology conversion."""
        self._smplx = None
        self._smplx2smpl = None

    def reconstruct(self, smpl_params: dict[str, torch.Tensor]) -> torch.Tensor:
        """SMPL-X params (each ``(F, C)``) -> SMPL vertices ``(F, 6890, 3)``, CPU."""
        with torch.no_grad():
            self._ensure_loaded()
            if self._smplx is None or self._smplx2smpl is None:
                raise RuntimeError(
                    "SMPL vertex assets did not load before reconstruction."
                )
            with span("transfer", model="SMPL-vertices"):
                params = {k: v.to(self._device) for k, v in smpl_params.items()}
            with span("inference", model="SMPL-vertices", batch_shape=list(params["body_pose"].shape)):
                smplx_out = self._smplx(**params)
            vertices = torch.stack(
                [torch.matmul(self._smplx2smpl, verts) for verts in smplx_out.vertices],
                dim=0,
            )
            return vertices.float().cpu()


class SmplCoco17Reconstructor:
    """Reconstruct only COCO-17 joints from GVHMR SMPL-X parameters.

    Unlike :class:`SmplVertexReconstructor`, this path evaluates only the 132
    SMPL-X vertices touched by the user-acquired COCO-17 regressor. It is therefore
    the canonical boundary for motion datasets that do not need a renderable
    mesh.
    """

    _WIDTHS = {
        "body_pose": 63,
        "betas": 10,
        "global_orient": 3,
        "transl": 3,
    }

    def __init__(
        self,
        body_models_dir: str | Path,
        *,
        device: str | torch.device,
        smpl_assets: SmplAssetPaths,
    ) -> None:
        from src.utils.device import resolve_device

        self._device = resolve_device(device)
        self._body_models_dir = require_absolute_path(
            body_models_dir, name="SMPL-X body-model directory"
        )
        if not isinstance(smpl_assets, SmplAssetPaths):
            raise TypeError("smpl_assets must be SmplAssetPaths.")
        smpl_assets.require_files()
        self._smpl_assets = smpl_assets
        self._model: torch.nn.Module | None = None

    def load(self) -> None:
        self._ensure_loaded()

    @model_action("load", "SmplCoco17Reconstructor", weights=lambda owner: {"smplx": owner._body_models_dir / "smplx" / "SMPLX_NEUTRAL.npz", **{field.name: getattr(owner._smpl_assets, field.name) for field in fields(owner._smpl_assets)}},
                  device=lambda owner: str(owner._device), loaded=lambda owner: owner._model is not None)
    def _ensure_loaded(self) -> None:
        if self._model is None:
            self._model = (
                make_smplx(
                    "supermotion_coco17",
                    model_path=self._body_models_dir,
                    smpl_assets=self._smpl_assets,
                )
                .to(self._device)
                .eval()
            )

    @model_action("unload", "SmplCoco17Reconstructor", weights=lambda owner: {"smplx": owner._body_models_dir / "smplx" / "SMPLX_NEUTRAL.npz", **{field.name: getattr(owner._smpl_assets, field.name) for field in fields(owner._smpl_assets)}},
                  device=lambda owner: str(owner._device), loaded=lambda owner: owner._model is not None)
    def unload(self) -> None:
        """Release the lazily loaded body model."""
        self._model = None

    def reconstruct(self, smpl_params: dict[str, torch.Tensor]) -> torch.Tensor:
        """SMPL-X params (each ``(F,C)``) -> COCO-17 joints, float32 CPU."""
        if not isinstance(smpl_params, dict) or set(smpl_params) != set(self._WIDTHS):
            raise ValueError(
                f"SMPL COCO-17 reconstruction requires exactly {sorted(self._WIDTHS)}."
            )
        frame_count: int | None = None
        validated: dict[str, torch.Tensor] = {}
        for name, width in self._WIDTHS.items():
            tensor = smpl_params[name]
            if not isinstance(tensor, torch.Tensor):
                raise TypeError(f"SMPL parameter {name} must be a torch.Tensor.")
            if (
                tensor.dtype != torch.float32
                or tensor.ndim != 2
                or tensor.shape[1] != width
            ):
                raise ValueError(
                    f"SMPL parameter {name} must have float32 shape (F,{width}), "
                    f"got {tensor.dtype} {tuple(tensor.shape)}."
                )
            if not bool(torch.isfinite(tensor).all()):
                raise ValueError(f"SMPL parameter {name} contains NaN or infinity.")
            current_frames = int(tensor.shape[0])
            if current_frames <= 0:
                raise ValueError("SMPL parameters must contain at least one frame.")
            if frame_count is None:
                frame_count = current_frames
            elif current_frames != frame_count:
                raise ValueError(
                    "All SMPL parameters must contain the same frame count."
                )
            with span("transfer", model="SMPL-COCO17"):
                validated[name] = tensor.to(self._device)

        with torch.no_grad():
            self._ensure_loaded()
            if self._model is None:
                raise RuntimeError("SMPL COCO-17 model did not load.")
            with span("inference", model="SMPL-COCO17", batch_shape=list(validated["body_pose"].shape)):
                joints = self._model(**validated)
        if not isinstance(joints, torch.Tensor) or joints.shape != (frame_count, 17, 3):
            raise RuntimeError(
                "SMPL COCO-17 model returned an invalid shape: "
                f"{getattr(joints, 'shape', None)}."
            )
        return joints.detach().float().cpu()
