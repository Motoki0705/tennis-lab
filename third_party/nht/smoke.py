#!/usr/bin/env python3
"""Verify a finite CUDA forward/backward pass in the pinned NHT stack."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from gsplat import csrc
from gsplat.nht.deferred_shader import DeferredShaderModule
from gsplat.rendering import rasterization


def _git_head(path: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        text=True,
    ).strip()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tensor_stats(tensor: torch.Tensor) -> dict[str, Any]:
    detached = tensor.detach().float()
    return {
        "shape": list(detached.shape),
        "min": float(detached.min().cpu()),
        "max": float(detached.max().cpu()),
        "mean": float(detached.mean().cpu()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    nht_root = Path(__file__).resolve().parent
    upstream = nht_root / "upstream"
    csrc_path = Path(csrc.__file__).resolve()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable")

    torch.manual_seed(42)
    device = torch.device("cuda:0")
    count, feature_dim = 128, 16
    width, height = 32, 24

    means = torch.nn.Parameter(torch.randn(count, 3, device=device) * 0.3)
    quats = F.normalize(torch.randn(count, 4, device=device), dim=-1)
    log_scales = torch.nn.Parameter(torch.full((count, 3), -1.5, device=device))
    opacity_logits = torch.nn.Parameter(torch.zeros(count, device=device))
    features = torch.nn.Parameter(torch.randn(count, feature_dim, device=device))
    camera_to_world = torch.eye(4, device=device).unsqueeze(0)
    camera_to_world[:, 2, 3] = -3.0
    viewmats = torch.linalg.inv(camera_to_world)
    intrinsics = torch.tensor(
        [
            [
                [100.0, 0.0, width / 2],
                [0.0, 100.0, height / 2],
                [0.0, 0.0, 1.0],
            ]
        ],
        device=device,
    )

    rendered_features, rendered_alpha, _ = rasterization(
        means=means,
        quats=quats,
        scales=torch.exp(log_scales),
        opacities=torch.sigmoid(opacity_logits),
        colors=features,
        viewmats=viewmats,
        Ks=intrinsics,
        width=width,
        height=height,
        nht=True,
        with_eval3d=True,
        with_ut=True,
        packed=False,
        sh_degree=None,
    )
    shader = DeferredShaderModule(
        feature_dim=feature_dim,
        enable_view_encoding=True,
    ).to(device)
    rendered_rgb, extras = shader(rendered_features)
    loss = rendered_rgb.mean() + 0.01 * rendered_alpha.mean()
    loss.backward()
    torch.cuda.synchronize()

    gradients = {
        "means": means.grad,
        "log_scales": log_scales.grad,
        "opacity_logits": opacity_logits.grad,
        "features": features.grad,
    }
    for name, gradient in gradients.items():
        if gradient is None or not torch.isfinite(gradient).all():
            raise RuntimeError(f"Missing or non-finite gradient: {name}")
    if not torch.isfinite(rendered_features).all():
        raise RuntimeError("Non-finite NHT rasterized features")
    if not torch.isfinite(rendered_rgb).all():
        raise RuntimeError("Non-finite deferred RGB")
    if extras is not None:
        raise RuntimeError("Unexpected auxiliary output in RGB-only smoke test")

    shader_gradients = [
        parameter.grad
        for parameter in shader.parameters()
        if parameter.grad is not None
    ]
    if not shader_gradients or not all(
        torch.isfinite(gradient).all() for gradient in shader_gradients
    ):
        raise RuntimeError("Missing or non-finite deferred shader gradient")

    report = {
        "schema": "tennis_lab_nht_smoke_v1",
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        "device_capability": list(torch.cuda.get_device_capability(0)),
        "nht_commit": _git_head(upstream),
        "gsplat_commit": _git_head(upstream / "gsplat"),
        "csrc": {
            "path": str(csrc_path),
            "bytes": csrc_path.stat().st_size,
            "sha256": _sha256(csrc_path),
        },
        "rendered_features": _tensor_stats(rendered_features),
        "rendered_alpha": _tensor_stats(rendered_alpha),
        "rendered_rgb": _tensor_stats(rendered_rgb),
        "loss": float(loss.detach().cpu()),
        "gradient_l1": {
            name: float(gradient.detach().abs().sum().cpu())
            for name, gradient in gradients.items()
            if gradient is not None
        },
        "deferred_shader_gradient_tensors": len(shader_gradients),
        "all_finite": True,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2) + "\n")
    temporary.replace(args.output)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
