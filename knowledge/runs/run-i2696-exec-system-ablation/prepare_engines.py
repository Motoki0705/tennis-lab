"""Check #2363 plan provenance against current model contracts or rebuild."""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import torch
from hydra import compose, initialize_config_dir
from run import BUNDLE, CONDITION, REPO, command, pipeline_overrides, read_json

from src.tasks.player_association.appearance.encoders import (
    CLIP_REID_PROFILE,
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
from src.utils.checksum import dual_sha256
from src.utils.io import write_json_atomic
from src.utils.models.tensorrt_artifacts import ENGINE_MANIFEST, EngineBundle
from src.utils.models.tensorrt_runtime import engine_environment, tensorrt_module

TRT_COMMAND = [sys.executable, "-B", "-m", "src.tennis_scene.scripts.tensorrt"]
EXPORTED_NETWORK_SOURCES = (
    "src/submodules/models/dino",
    "src/submodules/models/gvhmr",
    "src/submodules/vendor/gvhmr",
    "src/tasks/player_association/appearance/encoders.py",
    "src/utils/models/tensorrt_build.py",
    "third_party/DINO",
)


def require_reference_plan(bundle: EngineBundle, reference: dict[str, Any]) -> None:
    if bundle.engine.model_dump(mode="json") != reference:
        raise ValueError(
            f"Engine manifest differs from the #2363 qualification: {bundle.root}"
        )


def rebuild(
    run: Path, root: Path, model: str, batch: int, overrides: list[str]
) -> list[dict[str, Any]]:
    build = ["build", str(root), "--opt-batch", str(batch), "--max-batch", str(batch)]
    export = ["export", model, str(root)]
    if model == "dino":
        export += ["--image-size", *map(str, CONDITION["dino_image_size"])]
    if model == "gvhmr":
        export.append("--keep-mask-reference")
        build += ["--frames", *map(str, CONDITION["gvhmr_engine_profile_frames"])]
    export += [arg for override in overrides for arg in ("--override", override)]
    receipts = []
    for action in (export, build):
        log = run / "engines" / f"{model}-{action[0]}.log"
        result = command([*TRT_COMMAND, *action], log)
        if result["exit_code"] != 0:
            raise RuntimeError(f"{model} {action[0]} failed. See {log}")
        receipts.append(result)
    if model == "gvhmr":
        log = run / "engines/gvhmr-qualification.log"
        result = command(
            [
                sys.executable,
                "-B",
                "-m",
                "src.tennis_scene.scripts.qualify_gvhmr_tensorrt",
                str(root),
                str(run / "engines/gvhmr-qualification.json"),
            ],
            log,
        )
        if result["exit_code"] != 0:
            raise RuntimeError(f"Rebuilt GVHMR qualification failed. See {log}")
        receipts.append(result)
    return receipts


def prepare(prior: Path, output_root: Path, run: Path) -> None:
    reference = read_json(
        BUNDLE.parent / "run-i2363-trt-player-video/qualification-v1.json"
    )
    published = read_json(BUNDLE.parent / "run-i2363-trt-player-video/full-video.json")
    changes = subprocess.check_output(
        [
            "git",
            "diff",
            "--name-only",
            reference["commit"],
            "HEAD",
            "--",
            *EXPORTED_NETWORK_SOURCES,
        ],
        text=True,
    ).splitlines()
    overrides = pipeline_overrides(output_root, run, None)
    with initialize_config_dir(
        config_dir=str(REPO / "src/tennis_scene/configs"), version_base="1.3"
    ):
        raw = compose(config_name="pipeline", overrides=overrides)
    cfg = PipelineRuntimeConfig.from_config(raw, bind_inputs=False)
    clip = encoder_weights(
        CLIP_REID_PROFILE,
        checkpoint_root=cfg.roots.checkpoint_root,
        external_root=cfg.roots.external_asset_root,
    )
    specs = {
        "dino": dino_tensorrt(cfg.people, *CONDITION["dino_image_size"]),
        "vitpose": vitpose_tensorrt(cfg.people),
        "clip": clip_tensorrt(clip),
        "hmr2": hmr2_tensorrt(cfg.people),
        "gvhmr": gvhmr_tensorrt(cfg.people),
    }
    trt = tensorrt_module()
    environment = engine_environment(torch.device(cfg.device), trt)
    if environment.device_name != "NVIDIA GeForce RTX 5060 Ti":
        raise ValueError(
            f"Wrong accelerator for the fixed experiment: {environment.device_name}"
        )
    result: dict[str, Any] = {
        "qualification_commit": reference["commit"],
        "environment": environment.model_dump(mode="json"),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "changed_network_sources_since_qualification": changes,
        "models": {},
    }
    (run / "engines").mkdir()
    for name, spec in specs.items():
        print(f"Verifying {name}", flush=True)
        batch = CONDITION["batches"][name]
        relative = reference["engines"]["directories"][name]
        root = run / "engines" / Path(relative).parent.relative_to("tensorrt")
        rebuild_reason = None
        operations = []
        try:
            if changes:
                raise ValueError(
                    "An exported-network implementation changed since qualification"
                )
            source = spec.bind(prior / relative, device=cfg.device, batch_size=batch)
            require_reference_plan(source, reference["engines"]["manifests"][name])
            if (source.engine.export_sha256, source.engine.plan_sha256) != (
                published["engines"][name]["export_sha256"],
                published["engines"][name]["plan_sha256"],
            ):
                raise ValueError("SHA-256 differs from the #2363 inference receipt")
        except (OSError, ValueError) as error:
            rebuild_reason = str(error)
            print(f"Rebuilding {name}: {error}", flush=True)
            profile_batch = (
                CONDITION["dino_engine_profile_batch"] if name == "dino" else batch
            )
            operations = rebuild(run, root, name, profile_batch, overrides)
        else:
            root.mkdir(parents=True)
            for path in source.files:
                shutil.copy2(path, root / path.name)
            if dual_sha256(root / ENGINE_MANIFEST) != dual_sha256(
                source.root / ENGINE_MANIFEST
            ):
                raise ValueError(f"Copied engine manifest differs for {name}")
        installed = spec.bind(root, device=cfg.device, batch_size=batch)
        if rebuild_reason is None:
            require_reference_plan(installed, reference["engines"]["manifests"][name])
        result["models"][name] = {
            "directory": str(root.relative_to(output_root)),
            "reused": rebuild_reason is None,
            "rebuild_reason": rebuild_reason,
            "operations": operations,
            "engine": installed.engine.model_dump(mode="json"),
            "engine_manifest_sha256": dual_sha256(root / ENGINE_MANIFEST),
            "export": installed.export.model_dump(mode="json"),
        }
        write_json_atomic(run / "engines.json", result)


if __name__ == "__main__":
    prepare(*map(Path, sys.argv[1:]))
