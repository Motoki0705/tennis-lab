#!/usr/bin/env python3
"""Fit only movable-asset features to one frozen target NHT appearance model."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from gsplat.nht.deferred_shader import DeferredShaderModule
from gsplat.rendering import rasterization
from PIL import Image
from plyfile import PlyData

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.synthetic_data_generation.blcs.calibration import (  # noqa: E402
    load_ball_calibration_bundle,
)

CONVERSION_REPORT_SCHEMA = "tennis_ball_asset_conversion_report_v1"
CONVERSION_DETAIL_SCHEMA = "tennis_ball_nht_feature_fit_detail_v1"
CONVERSION_METHOD = "frozen_target_nht_feature_optimization_v1"
INDEPENDENT_NHT_SOURCE = "independent_nht_tensor_pack_v1"
VANILLA_3DGS_SOURCE = "vanilla_3dgs_ply_v1"
SOURCE_FORMATS = (INDEPENDENT_NHT_SOURCE, VANILLA_3DGS_SOURCE)
PREPARED_TENSOR_KEYS = {
    "means",
    "quats",
    "scales",
    "opacities",
    "features",
    "instance_ids",
}
MIN_VALIDATION_PSNR_DB = 20.0
DEFAULT_FEATURE_LR = 0.015
DEFAULT_FINAL_LR_FRACTION = 0.1
DEFAULT_OPTIMIZATION_STEPS = 600


@dataclass(frozen=True)
class RenderCalibrationBundle:
    """CUDA-ready calibration tensors loaded from the shared BLCS contract."""

    root: Path
    manifest: dict[str, object]
    camera_to_asset: torch.Tensor
    intrinsics: torch.Tensor
    rgb: torch.Tensor
    mask: torch.Tensor
    split: torch.Tensor

    @property
    def width(self) -> int:
        return int(self.rgb.shape[2])

    @property
    def height(self) -> int:
        return int(self.rgb.shape[1])

    @property
    def train_indices(self) -> tuple[int, ...]:
        return tuple(
            int(index)
            for index in torch.nonzero(self.split == 0, as_tuple=False).flatten()
        )

    @property
    def validation_indices(self) -> tuple[int, ...]:
        return tuple(
            int(index)
            for index in torch.nonzero(self.split == 1, as_tuple=False).flatten()
        )


@dataclass(frozen=True)
class SourceGeometry:
    """Frozen Gaussian geometry and source metadata."""

    means: torch.Tensor
    quats: torch.Tensor
    log_scales: torch.Tensor
    opacity_logits: torch.Tensor
    source_feature_dim: int | None

    @property
    def gaussian_count(self) -> int:
        return int(self.means.shape[0])


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Optimize per-Gaussian NHT features while freezing geometry, opacity, "
            "and the target deferred shader."
        )
    )
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--source-format", choices=SOURCE_FORMATS, required=True)
    parser.add_argument("--calibration-bundle", type=Path, required=True)
    parser.add_argument("--target-appearance", type=Path, required=True)
    parser.add_argument(
        "--target-appearance-space-sha256",
        required=True,
        help="Expected target appearance-space identity; never inferred silently.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--optimization-steps", type=int, default=DEFAULT_OPTIMIZATION_STEPS
    )
    parser.add_argument("--feature-lr", type=float, default=DEFAULT_FEATURE_LR)
    parser.add_argument(
        "--final-lr-fraction",
        type=float,
        default=DEFAULT_FINAL_LR_FRACTION,
    )
    parser.add_argument(
        "--min-validation-psnr-db",
        type=float,
        default=MIN_VALIDATION_PSNR_DB,
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def _file_ref(root: Path, path: Path) -> dict[str, object]:
    return {
        "relative_path": path.relative_to(root).as_posix(),
        "sha256": _sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def _absolute_file_ref(path: Path) -> dict[str, object]:
    return {
        "uri": path.resolve().as_uri(),
        "sha256": _sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def _strict_mapping(
    value: object,
    *,
    name: str,
    keys: set[str],
) -> dict[str, object]:
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise TypeError(f"{name} must be a JSON object with string keys.")
    if set(value) != keys:
        raise ValueError(
            f"{name} keys differ: missing={sorted(keys - set(value))}, "
            f"extra={sorted(set(value) - keys)}."
        )
    return value


def _positive_int(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer.")
    return value


def _sha256(value: object, *, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest.")
    return value


def _git_revision(path: Path) -> str:
    revision = subprocess.check_output(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        text=True,
    ).strip()
    dirty = subprocess.check_output(
        ["git", "-C", str(path), "status", "--porcelain"],
        text=True,
    ).strip()
    if dirty:
        raise RuntimeError(f"Refusing modified renderer checkout: {path}")
    return revision


def _runtime_revisions() -> dict[str, str]:
    root = Path(__file__).resolve().parent
    expected: dict[str, str] = {}
    for line in (root / "pins.env").read_text(encoding="utf-8").splitlines():
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            expected[key] = value
    nht_revision = _git_revision(root / "upstream")
    gsplat_revision = _git_revision(root / "upstream" / "gsplat")
    if nht_revision != expected.get("NHT_COMMIT"):
        raise RuntimeError(f"NHT revision differs from pins.env: {nht_revision}.")
    if gsplat_revision != expected.get("GSPLAT_COMMIT"):
        raise RuntimeError(f"gsplat revision differs from pins.env: {gsplat_revision}.")
    return {
        "nht_commit": nht_revision,
        "gsplat_commit": gsplat_revision,
        "worker_sha256": _sha256_file(Path(__file__).resolve()),
        "torch_version": torch.__version__,
        "torch_cuda_version": str(torch.version.cuda),
        "numpy_version": np.__version__,
        "plyfile_version": version("plyfile"),
    }


def _float_tensor(value: object, *, name: str, shape: tuple[int, ...]) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a tensor.")
    if value.dtype != torch.float32 or tuple(value.shape) != shape:
        raise ValueError(f"{name} must have float32 shape {shape}.")
    if not bool(torch.isfinite(value).all()):
        raise ValueError(f"{name} contains non-finite values.")
    return value.detach().cpu().contiguous()


def _load_independent_nht(path: Path) -> SourceGeometry:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or set(payload) != PREPARED_TENSOR_KEYS:
        actual = (
            sorted(payload) if isinstance(payload, dict) else type(payload).__name__
        )
        raise ValueError(f"Independent NHT tensor keys differ: {actual}.")
    means_value = payload["means"]
    if not isinstance(means_value, torch.Tensor) or means_value.ndim != 2:
        raise ValueError("Independent NHT means must have shape [N,3].")
    count = int(means_value.shape[0])
    if count <= 0:
        raise ValueError("Independent NHT source must contain Gaussians.")
    means = _float_tensor(means_value, name="means", shape=(count, 3))
    quats = _float_tensor(payload["quats"], name="quats", shape=(count, 4))
    scales = _float_tensor(payload["scales"], name="scales", shape=(count, 3))
    opacities = _float_tensor(
        payload["opacities"],
        name="opacities",
        shape=(count,),
    )
    features_value = payload["features"]
    if (
        not isinstance(features_value, torch.Tensor)
        or features_value.dtype != torch.float32
        or features_value.ndim != 2
        or features_value.shape[0] != count
        or not bool(torch.isfinite(features_value).all())
    ):
        raise ValueError("Independent NHT features must be finite float32 [N,F].")
    instance_ids = payload["instance_ids"]
    if (
        not isinstance(instance_ids, torch.Tensor)
        or instance_ids.dtype != torch.int64
        or tuple(instance_ids.shape) != (count,)
    ):
        raise ValueError("Independent NHT instance_ids must be int64 [N].")
    if not bool((instance_ids == 0).all()):
        raise ValueError("An independent movable asset must use instance_id zero.")
    quat_norm = torch.linalg.vector_norm(quats, dim=-1)
    if not torch.allclose(quat_norm, torch.ones_like(quat_norm), atol=1.0e-3):
        raise ValueError("Independent NHT quaternions are not normalized.")
    return SourceGeometry(
        means=means,
        quats=quats,
        log_scales=scales,
        opacity_logits=opacities,
        source_feature_dim=int(features_value.shape[1]),
    )


def _numeric_property_names(names: list[str], prefix: str) -> list[str]:
    selected = [name for name in names if name.startswith(prefix)]
    try:
        return sorted(selected, key=lambda name: int(name.removeprefix(prefix)))
    except ValueError as error:
        raise ValueError(
            f"PLY {prefix} properties must have numeric suffixes."
        ) from error


def _load_vanilla_3dgs(path: Path) -> SourceGeometry:
    ply = PlyData.read(path)
    if "vertex" not in ply:
        raise ValueError("Vanilla 3DGS PLY has no vertex element.")
    vertex = ply["vertex"]
    names = list(vertex.data.dtype.names or ())
    required = {
        "x",
        "y",
        "z",
        "opacity",
        "f_dc_0",
        "f_dc_1",
        "f_dc_2",
        "scale_0",
        "scale_1",
        "scale_2",
        "rot_0",
        "rot_1",
        "rot_2",
        "rot_3",
    }
    if not required.issubset(names):
        raise ValueError(
            f"Vanilla 3DGS PLY is missing: {sorted(required - set(names))}."
        )
    rest_names = _numeric_property_names(names, "f_rest_")
    if len(rest_names) % 3 != 0:
        raise ValueError("Vanilla 3DGS f_rest property count must be divisible by 3.")
    count = len(vertex)
    if count <= 0:
        raise ValueError("Vanilla 3DGS PLY must contain Gaussians.")

    def stacked(properties: list[str]) -> torch.Tensor:
        array = np.stack(
            [np.asarray(vertex[name], dtype=np.float32) for name in properties],
            axis=-1,
        )
        return torch.from_numpy(array.copy())

    means = stacked(["x", "y", "z"])
    quats = stacked(["rot_0", "rot_1", "rot_2", "rot_3"])
    log_scales = stacked(["scale_0", "scale_1", "scale_2"])
    opacity_logits = torch.from_numpy(
        np.asarray(vertex["opacity"], dtype=np.float32).copy()
    )
    for name, value in {
        "means": means,
        "quats": quats,
        "scales": log_scales,
        "opacities": opacity_logits,
    }.items():
        if not bool(torch.isfinite(value).all()):
            raise ValueError(f"Vanilla 3DGS {name} contains non-finite values.")
    quat_norm = torch.linalg.vector_norm(quats, dim=-1)
    if bool((quat_norm <= 1.0e-8).any()):
        raise ValueError("Vanilla 3DGS PLY contains a zero quaternion.")
    return SourceGeometry(
        means=means.contiguous(),
        quats=F.normalize(quats, dim=-1).contiguous(),
        log_scales=log_scales.contiguous(),
        opacity_logits=opacity_logits.contiguous(),
        source_feature_dim=None,
    )


def load_source_geometry(path: Path, source_format: str) -> SourceGeometry:
    """Load strict independent-NHT or standard INRIA 3DGS geometry."""
    source = path.resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Ball Gaussian source is missing: {source}")
    if source_format == INDEPENDENT_NHT_SOURCE:
        return _load_independent_nht(source)
    if source_format == VANILLA_3DGS_SOURCE:
        return _load_vanilla_3dgs(source)
    raise ValueError(f"Unsupported source format: {source_format!r}.")


def _load_shader(
    appearance_path: Path,
    *,
    device: torch.device,
) -> tuple[DeferredShaderModule, int, dict[str, object]]:
    payload = torch.load(appearance_path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or set(payload) != {"config", "state_dict"}:
        raise ValueError("Target appearance must contain only config and state_dict.")
    config = payload["config"]
    state_dict = payload["state_dict"]
    if not isinstance(config, dict) or not isinstance(state_dict, dict):
        raise TypeError("Target appearance config/state_dict must be mappings.")
    feature_dim = _positive_int(config.get("feature_dim"), name="feature_dim")
    shader = DeferredShaderModule(**config).to(device)
    shader.load_state_dict(state_dict, strict=True)
    shader.eval()
    for parameter in shader.parameters():
        parameter.requires_grad_(False)
    return shader, feature_dim, {str(key): value for key, value in config.items()}


def _render(
    geometry: SourceGeometry,
    features: torch.Tensor,
    shader: DeferredShaderModule,
    camera_to_asset: torch.Tensor,
    intrinsics: torch.Tensor,
    *,
    width: int,
    height: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    rendered_features, alpha, _ = rasterization(
        means=geometry.means,
        quats=geometry.quats,
        scales=torch.exp(geometry.log_scales),
        opacities=torch.sigmoid(geometry.opacity_logits),
        colors=features,
        viewmats=torch.linalg.inv(camera_to_asset.unsqueeze(0)),
        Ks=intrinsics.unsqueeze(0),
        width=width,
        height=height,
        sh_degree=None,
        near_plane=0.01,
        far_plane=1.0e10,
        render_mode="RGB",
        packed=False,
        tile_size=16,
        with_ut=True,
        with_eval3d=True,
        nht=True,
        center_ray_mode=shader.center_ray_encoding,
        ray_dir_scale=shader.ray_dir_scale,
    )
    rgb, extras = shader(rendered_features)
    if extras is not None:
        raise RuntimeError("RGB-only NHT rendering unexpectedly returned extras.")
    composed = (rgb[..., :3] + (1.0 - alpha)).clamp(0.0, 1.0)
    return composed[0], alpha[0, ..., 0]


def _masked_mse(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    expanded = mask.unsqueeze(-1).expand_as(prediction)
    if not bool(expanded.any()):
        raise RuntimeError("Foreground mask is empty.")
    return torch.mean(torch.square(prediction[expanded] - target[expanded]))


def _evaluate(
    geometry: SourceGeometry,
    features: torch.Tensor,
    shader: DeferredShaderModule,
    bundle: RenderCalibrationBundle,
    indices: tuple[int, ...],
) -> tuple[list[float], list[tuple[torch.Tensor, torch.Tensor]]]:
    psnr_values: list[float] = []
    renders: list[tuple[torch.Tensor, torch.Tensor]] = []
    with torch.no_grad():
        for index in indices:
            prediction, alpha = _render(
                geometry,
                features,
                shader,
                bundle.camera_to_asset[index],
                bundle.intrinsics[index],
                width=bundle.width,
                height=bundle.height,
            )
            mse = _masked_mse(prediction, bundle.rgb[index], bundle.mask[index])
            psnr_values.append(float(-10.0 * torch.log10(mse.clamp_min(1.0e-12))))
            renders.append((prediction.detach().cpu(), alpha.detach().cpu()))
    return psnr_values, renders


def _save_diagnostic(
    path: Path,
    *,
    target: torch.Tensor,
    prediction: torch.Tensor,
    mask: torch.Tensor,
) -> None:
    target_image = (target.clamp(0.0, 1.0).numpy() * 255.0).round().astype(np.uint8)
    prediction_image = (
        (prediction.clamp(0.0, 1.0).numpy() * 255.0).round().astype(np.uint8)
    )
    difference = np.abs(
        prediction.numpy().astype(np.float32) - target.numpy().astype(np.float32)
    )
    difference_image = (np.clip(difference * 8.0, 0.0, 1.0) * 255.0).astype(np.uint8)
    mask_image = np.repeat(mask.numpy()[..., None], 3, axis=-1).astype(np.uint8) * 255
    panel = np.concatenate(
        [target_image, prediction_image, difference_image, mask_image],
        axis=1,
    )
    Image.fromarray(panel).save(path)


def _validate_hyperparameters(args: argparse.Namespace) -> None:
    if args.optimization_steps <= 0:
        raise SystemExit("optimization-steps must be positive.")
    if not math.isfinite(args.feature_lr) or args.feature_lr <= 0.0:
        raise SystemExit("feature-lr must be finite and positive.")
    if (
        not math.isfinite(args.final_lr_fraction)
        or not 0.0 < args.final_lr_fraction <= 1.0
    ):
        raise SystemExit("final-lr-fraction must lie in (0, 1].")
    if (
        not math.isfinite(args.min_validation_psnr_db)
        or args.min_validation_psnr_db < MIN_VALIDATION_PSNR_DB
    ):
        raise SystemExit(
            f"min-validation-psnr-db must be at least {MIN_VALIDATION_PSNR_DB}."
        )
    if args.seed < 0:
        raise SystemExit("seed must be non-negative.")


def main() -> None:
    args = _parse_args()
    _validate_hyperparameters(args)
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise SystemExit(f"Refusing to overwrite output directory: {output_dir}")
    source_path = args.source.resolve()
    appearance_path = args.target_appearance.resolve()
    if not appearance_path.is_file():
        raise SystemExit(f"Target appearance is missing: {appearance_path}")
    appearance_space = _sha256(
        args.target_appearance_space_sha256,
        name="target appearance-space SHA-256",
    )
    runtime = _runtime_revisions()
    loaded_bundle = load_ball_calibration_bundle(args.calibration_bundle)
    source_geometry_cpu = load_source_geometry(source_path, args.source_format)
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable for NHT feature optimization.")
    device = torch.device(args.device)
    if device.type != "cuda":
        raise SystemExit("NHT feature optimization requires a CUDA device.")

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.use_deterministic_algorithms(True)
    shader, feature_dim, shader_config = _load_shader(
        appearance_path,
        device=device,
    )
    geometry = SourceGeometry(
        means=source_geometry_cpu.means.to(device),
        quats=source_geometry_cpu.quats.to(device),
        log_scales=source_geometry_cpu.log_scales.to(device),
        opacity_logits=source_geometry_cpu.opacity_logits.to(device),
        source_feature_dim=source_geometry_cpu.source_feature_dim,
    )
    bundle = RenderCalibrationBundle(
        root=loaded_bundle.root,
        manifest=loaded_bundle.manifest,
        camera_to_asset=torch.from_numpy(loaded_bundle.camera_to_asset).to(device),
        intrinsics=torch.from_numpy(loaded_bundle.intrinsics).to(device),
        rgb=torch.from_numpy(loaded_bundle.rgb).float().div_(255.0).to(device),
        mask=torch.from_numpy(loaded_bundle.mask).to(device),
        split=torch.from_numpy(loaded_bundle.split).to(device),
    )
    features = torch.nn.Parameter(
        torch.zeros(
            (geometry.gaussian_count, feature_dim),
            dtype=torch.float32,
            device=device,
        )
    )
    optimizer = torch.optim.Adam((features,), lr=args.feature_lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.optimization_steps,
        eta_min=args.feature_lr * args.final_lr_fraction,
    )
    history: list[dict[str, float | int]] = []
    train_indices = bundle.train_indices
    for step in range(args.optimization_steps):
        index = train_indices[step % len(train_indices)]
        prediction, _ = _render(
            geometry,
            features,
            shader,
            bundle.camera_to_asset[index],
            bundle.intrinsics[index],
            width=bundle.width,
            height=bundle.height,
        )
        loss = _masked_mse(prediction, bundle.rgb[index], bundle.mask[index])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if features.grad is None or not bool(torch.isfinite(features.grad).all()):
            raise RuntimeError("Feature optimization produced invalid gradients.")
        optimizer.step()
        scheduler.step()
        if step == 0 or (step + 1) % 100 == 0 or step + 1 == args.optimization_steps:
            history.append(
                {
                    "step": step + 1,
                    "train_view_index": index,
                    "masked_mse": float(loss.detach()),
                    "feature_lr": float(scheduler.get_last_lr()[0]),
                    "max_abs_feature": float(features.detach().abs().max()),
                }
            )
    validation_psnr, validation_renders = _evaluate(
        geometry,
        features,
        shader,
        bundle,
        bundle.validation_indices,
    )
    validation_psnr_db = float(np.mean(validation_psnr))
    if (
        not math.isfinite(validation_psnr_db)
        or validation_psnr_db < args.min_validation_psnr_db
    ):
        raise RuntimeError(
            f"Validation PSNR {validation_psnr_db:.6f} dB is below "
            f"{args.min_validation_psnr_db:.6f} dB; no output was published."
        )
    torch.cuda.synchronize(device)

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(
            prefix=f".{output_dir.name}.",
            suffix=".tmp",
            dir=output_dir.parent,
        )
    )
    try:
        prepared_path = temporary / "prepared-nht-tensors.pt"
        torch.save(
            {
                "means": geometry.means.detach().cpu().contiguous(),
                "quats": geometry.quats.detach().cpu().contiguous(),
                "scales": geometry.log_scales.detach().cpu().contiguous(),
                "opacities": geometry.opacity_logits.detach().cpu().contiguous(),
                "features": features.detach().cpu().contiguous(),
                "instance_ids": torch.zeros(
                    (geometry.gaussian_count,),
                    dtype=torch.int64,
                ),
            },
            prepared_path,
        )
        prepared_sha256 = _sha256_file(prepared_path)
        appearance_payload_sha256 = _sha256_file(appearance_path)
        report = {
            "schema": CONVERSION_REPORT_SCHEMA,
            "status": "passed",
            "method": CONVERSION_METHOD,
            "source_format": args.source_format,
            "target_appearance_space_sha256": appearance_space,
            "target_appearance_payload_sha256": appearance_payload_sha256,
            "prepared_tensors_sha256": prepared_sha256,
            "gaussian_count": geometry.gaussian_count,
            "feature_dim": feature_dim,
            "optimization_steps": args.optimization_steps,
            "validation_views": len(bundle.validation_indices),
            "validation_psnr_db": validation_psnr_db,
        }
        _write_json(temporary / "conversion-report.json", report)
        _write_json(temporary / "optimization-history.json", history)
        diagnostics = temporary / "validation"
        diagnostics.mkdir()
        for diagnostic_index, (view_index, rendered) in enumerate(
            zip(bundle.validation_indices, validation_renders, strict=True)
        ):
            prediction, alpha = rendered
            np.save(
                diagnostics / f"view-{view_index:03d}-alpha.npy",
                alpha.numpy().astype(np.float32),
            )
            _save_diagnostic(
                diagnostics / f"view-{view_index:03d}-panel.png",
                target=bundle.rgb[view_index].detach().cpu(),
                prediction=prediction,
                mask=bundle.mask[view_index].detach().cpu(),
            )
            _write_json(
                diagnostics / f"view-{view_index:03d}-metrics.json",
                {
                    "view_index": view_index,
                    "masked_psnr_db": validation_psnr[diagnostic_index],
                    "mask_pixel_count": int(bundle.mask[view_index].sum()),
                },
            )
        detail_unsigned: dict[str, object] = {
            "schema": CONVERSION_DETAIL_SCHEMA,
            "status": "passed",
            "source_format": args.source_format,
            "source": _absolute_file_ref(source_path),
            "source_feature_dim": source_geometry_cpu.source_feature_dim,
            "calibration_bundle": {
                "uri": bundle.root.resolve().as_uri(),
                "content_fingerprint": bundle.manifest["content_fingerprint"],
            },
            "target_appearance": _absolute_file_ref(appearance_path),
            "target_appearance_space_sha256": appearance_space,
            "renderer": runtime,
            "shader_config": shader_config,
            "frozen_parameters": [
                "means",
                "quats",
                "scales",
                "opacities",
                "target_deferred_shader",
            ],
            "geometry_preprocessing": (
                "strict-float32-copy-v1"
                if args.source_format == INDEPENDENT_NHT_SOURCE
                else "standard-inria-ply-float32-unit-quaternion-v1"
            ),
            "optimized_parameters": ["features"],
            "initialization": "all-zero-target-feature-space-v1",
            "optimizer": {
                "name": "Adam",
                "initial_feature_lr": args.feature_lr,
                "schedule": "cosine",
                "final_lr_fraction": args.final_lr_fraction,
                "steps": args.optimization_steps,
                "seed": args.seed,
                "train_view_schedule": "round-robin-v1",
            },
            "validation": {
                "minimum_psnr_db": args.min_validation_psnr_db,
                "per_view_psnr_db": validation_psnr,
                "mean_psnr_db": validation_psnr_db,
            },
            "files": {
                "prepared_tensors": _file_ref(temporary, prepared_path),
                "conversion_report": _file_ref(
                    temporary,
                    temporary / "conversion-report.json",
                ),
                "optimization_history": _file_ref(
                    temporary,
                    temporary / "optimization-history.json",
                ),
                "validation_diagnostics": [
                    _file_ref(temporary, path)
                    for path in sorted(diagnostics.iterdir())
                    if path.is_file()
                ],
            },
        }
        detail = {
            **detail_unsigned,
            "content_fingerprint": _canonical_sha256(detail_unsigned),
        }
        _write_json(temporary / "manifest.json", detail)
        if _sha256_file(prepared_path) != prepared_sha256:
            raise RuntimeError("Prepared tensor file changed during publication.")
        temporary.rename(output_dir)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
