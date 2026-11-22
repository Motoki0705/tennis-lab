"""Explicitly export a scene model to ONNX or build a TensorRT FP16 engine."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch
from hydra import compose, initialize_config_dir

from src.submodules.models.dino.onnx_export import export_dino_onnx
from src.submodules.models.gvhmr.onnx_export import ROTARY_MAX_FRAMES, export_gvhmr_onnx
from src.submodules.smpl_assets import require_smpl_asset
from src.submodules.vendor.gvhmr.hmr2 import load_hmr2
from src.submodules.vendor.gvhmr.vitpose import build_vitpose_huge
from src.tasks.player_association.appearance.encoders import (
    CLIP_REID_PROFILE,
    ClipReIDEncoder,
    encoder_weights,
)
from src.tennis_scene.configuration import PipelineRuntimeConfig
from src.tennis_scene.pipeline.model_resources import (
    clip_tensorrt,
    dino_tensorrt,
    gvhmr_tensorrt,
    hmr2_tensorrt,
    vitpose_tensorrt,
)
from src.tennis_scene.pipeline.tensorrt_assets import TensorRTModel
from src.utils.configuration import (
    BoundaryPathField,
    NonHydraPathBoundary,
    PathDirection,
    PathKind,
    PathResolver,
    PathRole,
    RuntimePathRoots,
)
from src.utils.models.tensorrt_artifacts import (
    EXPORT_MANIFEST,
    ExportManifest,
    InputProfile,
)
from src.utils.models.tensorrt_build import build_bundle, export_bundle
from src.utils.paths import PROJECT_ROOT

PATH_BOUNDARY = NonHydraPathBoundary(
    name="tennis_scene.tensorrt",
    fields=(
        BoundaryPathField(
            "directory", PathRole.OUTPUT, PathDirection.OUTPUT, PathKind.DIRECTORY,
            allow_role_root=True,
        ),
        BoundaryPathField(
            "export_manifest", PathRole.ARTIFACT, PathDirection.INPUT, PathKind.FILE,
            must_exist=True, required=False,
        ),
    ),
)


def export_image_model(path: Path, model: torch.nn.Module, spec: TensorRTModel) -> None:
    """Export an image model with one dynamic batch dimension."""
    pixels = spec.inputs["pixels"]
    batch = spec.batch_multiplier
    with torch.no_grad():
        torch.onnx.export(
            model,
            (
                torch.zeros(
                    batch, *pixels.shape[1:], dtype=getattr(torch, pixels.dtype)
                ),
            ),
            str(path),
            input_names=list(spec.inputs),
            output_names=list(spec.outputs),
            dynamic_axes={name: {0: "batch"} for name in (*spec.inputs, *spec.outputs)},
            opset_version=19,
            dynamo=False,
        )


class _HMR2Bindings(torch.nn.Module):
    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.hmr = model

    def forward(self, pixels: torch.Tensor) -> torch.Tensor:
        result: torch.Tensor = self.hmr({"img": pixels})
        return result


class _ClipBindings(torch.nn.Module):
    def __init__(self, visual: torch.nn.Module) -> None:
        super().__init__()
        self.visual = visual

    def forward(self, pixels: torch.Tensor) -> torch.Tensor:
        return torch.cat(self.visual(pixels), dim=-1)


def _align_hmr2_token(
    _module: torch.nn.Module, args: tuple[Any, ...], kwargs: dict[str, Any]
) -> tuple[tuple[Any, ...], dict[str, Any]]:
    # HMR2 creates its singleton token in FP32. Follow the export context dtype.
    return (args[0].to(kwargs["context"].dtype), *args[1:]), kwargs


def _export_model(args: argparse.Namespace, cfg: PipelineRuntimeConfig) -> None:
    config = cfg.people
    if args.model == "dino":
        if args.image_size is None:
            raise ValueError(
                "DINO export requires --image-size HEIGHT WIDTH after resize"
            )
        height, width = args.image_size
        spec = dino_tensorrt(config, height, width)
    else:
        if args.image_size is not None:
            raise ValueError("--image-size is used only for DINO")
        if args.model == "clip":
            checkpoint = encoder_weights(
                CLIP_REID_PROFILE,
                checkpoint_root=cfg.roots.checkpoint_root,
                external_root=cfg.roots.external_asset_root,
            )
            spec = clip_tensorrt(checkpoint)
        else:
            spec = {
                "vitpose": vitpose_tensorrt,
                "hmr2": hmr2_tensorrt,
                "gvhmr": gvhmr_tensorrt,
            }[args.model](config)

    def export(path: Path) -> None:
        if args.model == "dino":
            export_dino_onnx(
                config.dino_checkpoint, config.dino_repository, path, height, width
            )
            return
        if args.model == "gvhmr":
            export_gvhmr_onnx(config.gvhmr_checkpoint, path)
            return
        model: torch.nn.Module
        if args.model == "vitpose":
            model = build_vitpose_huge(
                str(config.vitpose_checkpoint), head_config=config.runtime.vitpose.head
            )
        elif args.model == "hmr2":
            mean = config.smpl_assets.hmr2_mean_params
            require_smpl_asset(mean, "hmr2_mean_params")
            hmr = load_hmr2(str(config.hmr2_checkpoint), mean_params_path=mean)
            hmr.smpl_head.transformer.register_forward_pre_hook(
                _align_hmr2_token, with_kwargs=True
            )
            model = _HMR2Bindings(hmr)
        else:
            model = _ClipBindings(
                ClipReIDEncoder(CLIP_REID_PROFILE, spec.assets["checkpoint"]).model
            )
        export_image_model(path, model.half().eval(), spec)

    export_bundle(
        args.directory,
        model=spec.model,
        settings=spec.settings,
        sources=spec.assets,
        export=export,
    )


def _profile(
    args: argparse.Namespace, export: ExportManifest
) -> dict[str, InputProfile]:
    batch_multiplier = 2 if export.settings.get("flip_test") is True else 1
    if not 1 <= args.opt_batch <= args.max_batch:
        raise ValueError("Expected 1 <= opt-batch <= max-batch")
    batches = (1, args.opt_batch, args.max_batch)
    is_gvhmr = export.model == "gvhmr_siga24_denoiser"
    if is_gvhmr and (args.max_batch != 1 or args.frames is None):
        raise ValueError("GVHMR TensorRT requires batch 1 and --frames MIN OPT MAX")
    if not is_gvhmr and args.frames is not None:
        raise ValueError("--frames is used only for GVHMR")
    if args.frames is not None:
        minimum, optimum, maximum = args.frames
        if not 1 <= minimum <= optimum <= maximum <= ROTARY_MAX_FRAMES:
            raise ValueError(
                f"Expected 1 <= min frames <= opt frames <= max frames <= {ROTARY_MAX_FRAMES}"
            )
    profiles = {}
    for name, spec in export.inputs.items():
        if is_gvhmr and name != "length":
            dims = tuple((1, frames, *spec.shape[2:]) for frames in args.frames)
        else:
            dims = tuple(
                (batch_multiplier * batch, *spec.shape[1:]) for batch in batches
            )
        profiles[name] = InputProfile(minimum=dims[0], optimum=dims[1], maximum=dims[2])
    return profiles


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_subparsers(dest="action", required=True)
    export = actions.add_parser(
        "export", help="Export the configured model to typed ONNX"
    )
    export.add_argument("model", choices=("vitpose", "hmr2", "clip", "dino", "gvhmr"))
    export.add_argument("directory", type=Path)
    export.add_argument("--image-size", type=int, nargs=2, metavar=("HEIGHT", "WIDTH"))
    export.add_argument(
        "--override", action="append", default=[], help="Pipeline Hydra override"
    )
    build = actions.add_parser(
        "build", help="Build a plan from a completed ONNX export"
    )
    build.add_argument("directory", type=Path)
    build.add_argument("--device", default="cuda:0")
    build.add_argument("--opt-batch", type=int, default=1)
    build.add_argument("--max-batch", type=int, default=1)
    build.add_argument("--frames", type=int, nargs=3, metavar=("MIN", "OPT", "MAX"))
    build.add_argument("--workspace-mib", type=int, default=1536)
    args = parser.parse_args()
    directory = (PROJECT_ROOT / args.directory.expanduser()).resolve()
    roots = RuntimePathRoots(
        project_root=PROJECT_ROOT,
        data_root=PROJECT_ROOT,
        checkpoint_root=PROJECT_ROOT,
        artifact_root=directory,
        output_root=directory,
        cache_root=PROJECT_ROOT,
        external_asset_root=PROJECT_ROOT,
    )
    path_values = {"directory": directory}
    if args.action == "build":
        if args.workspace_mib < 1:
            parser.error("--workspace-mib must be positive")
        path_values["export_manifest"] = directory / EXPORT_MANIFEST
    paths = PATH_BOUNDARY.validate(path_values, resolver=PathResolver(roots))
    args.directory = paths.declared("directory").path
    if args.action == "export":
        with initialize_config_dir(
            config_dir=str(PROJECT_ROOT / "src/tennis_scene/configs"),
            version_base="1.3",
        ):
            raw = compose(config_name="pipeline", overrides=args.override)
        cfg = PipelineRuntimeConfig.from_config(raw, bind_inputs=False)
        torch.backends.mha.set_fastpath_enabled(False)
        _export_model(args, cfg)
    else:
        manifest = ExportManifest.model_validate_json(
            paths.declared("export_manifest").path.read_bytes()
        )
        result = build_bundle(
            args.directory,
            _profile(args, manifest),
            device=torch.device(args.device),
            workspace_bytes=args.workspace_mib * 1024**2,
        )
        print(result.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
