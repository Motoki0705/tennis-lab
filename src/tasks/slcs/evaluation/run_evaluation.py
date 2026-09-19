"""Select retained checkpoints using validation only, then evaluate paired inputs.

Independent test data is never used for checkpoint selection. Test evaluation
requires an explicit split request. Training's terminal test may use a different
checkpoint and is not a selection signal here.
"""

from __future__ import annotations

import math
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

from src.tasks.slcs.configuration import SLCSEvaluationConfig
from src.tasks.slcs.data.annotation import SLCSDataIndex
from src.tasks.slcs.evaluation.comparison import (
    CONDITIONS,
    compare_conditions,
    save_comparison,
)
from src.tasks.slcs.evaluation.evaluate import (
    evaluate_split,
    evaluation_context,
    save_evaluation,
)
from src.tasks.slcs.evaluation.motion import summarize_motion
from src.tasks.slcs.inference.predictor import SLCSPredictor
from src.tennis_scene.generate_dataset.manifest import ClipManifest
from src.utils.configuration.paths import PathResolver, PathRole, RuntimePathRoots
from src.utils.io import save_json

MONITOR = "val/scene_position_error_m_epoch"


def select_checkpoint(training_run: Path) -> tuple[Path, dict[str, Any]]:
    """Use one last checkpoint's retained validation scores, never file times."""
    root = training_run.resolve(strict=True)
    last_paths = list(root.rglob("last.ckpt"))
    if len(last_paths) != 1:
        raise ValueError(
            f"Expected exactly one last.ckpt in {root}, found {len(last_paths)}"
        )
    last_path = last_paths[0].resolve(strict=True)
    if not last_path.is_relative_to(root):
        raise ValueError("last checkpoint must belong to the training run")
    last = torch.load(last_path, map_location="cpu", weights_only=False)
    callbacks = [
        value
        for value in last.get("callbacks", {}).values()
        if isinstance(value, dict) and value.get("monitor") == MONITOR
    ]
    if len(callbacks) != 1:
        raise ValueError(f"Expected exactly one callback monitoring {MONITOR}")
    retained = callbacks[0].get("best_k_models")
    if not isinstance(retained, dict) or not retained:
        raise ValueError("Validation callback has no retained best_k_models")
    candidates: list[dict[str, Any]] = []
    for name, score in retained.items():
        path = Path(name)
        if not path.is_absolute():
            raise ValueError(
                "Retained checkpoint paths must be absolute; no relocation guessing"
            )
        path = path.resolve(strict=True)
        if not path.is_relative_to(root) or not path.is_file():
            raise ValueError(
                "Retained checkpoint must be a file in the same training run"
            )
        value = float(score)
        if not math.isfinite(value):
            raise ValueError("Retained validation scores must be finite")
        candidates.append({"path": str(path), "validation_score": value})
    candidates.sort(key=lambda row: (row["validation_score"], row["path"]))
    selected = candidates[0]
    checkpoint = Path(selected["path"])
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    return checkpoint, {
        "selection": "minimum retained validation monitor; ties resolved by path; no test selection",
        "monitor": MONITOR,
        "mode": "min",
        "candidates": candidates,
        "selected": {**selected, "epoch_zero_based": int(state["epoch"])},
        "last_checkpoint": str(last_path),
        "last_epoch_zero_based": int(last["epoch"]),
    }


def evaluation_config(
    training: DictConfig,
    *,
    training_run: Path,
    checkpoint: Path,
    output_root: Path,
    output: str,
    split: str,
    device: str,
    batch_size: int,
) -> DictConfig:
    """Preserve the saved training data contract; disable only augmentation."""
    if OmegaConf.select(training, "training.checkpoint.monitor") != MONITOR:
        raise ValueError(f"Training configuration must monitor {MONITOR}")
    mode = OmegaConf.select(training, "training.checkpoint.mode")
    if mode is not None and mode != "min":
        raise ValueError("Validation error checkpoint mode must be min")
    paths = cast(dict[str, Any], OmegaConf.to_container(training.paths, resolve=True))
    paths["checkpoint_root"] = str(training_run.resolve())
    paths["output_root"] = str(output_root.resolve())
    data = cast(dict[str, Any], OmegaConf.to_container(training.data, resolve=True))
    data["augmentation"]["enabled"] = False
    result = OmegaConf.create(
        {
            "paths": paths,
            "data": data,
            "evaluate": {
                "checkpoint": str(checkpoint.relative_to(training_run.resolve())),
                "split": split,
                "device": device,
                "batch_size": batch_size,
                "input_mode": "full",
                "checkpoint_strict": True,
                "checkpoint_weights_only": False,
                "output_dir": output,
            },
        }
    )
    return cast(DictConfig, result)


def clip_fps(
    dataset_root: Path, arrays: Mapping[str, np.ndarray]
) -> dict[tuple[str, str], float]:
    """Read each evaluated clip's physical FPS from the canonical manifest."""
    keys = set(
        zip(arrays["video_ids"].tolist(), arrays["clip_ids"].tolist(), strict=True)
    )
    index = SLCSDataIndex.load(dataset_root)
    result = {}
    for record in index.clips:
        key = (record.video_id, record.clip_id)
        if key not in keys:
            continue
        manifest = ClipManifest.load(index.clip_dir(record))
        if (manifest.video_id, manifest.clip_id) != key or manifest.fps != record.fps:
            raise ValueError(f"Dataset/clip manifest mismatch for {key}")
        if not math.isfinite(manifest.fps) or manifest.fps <= 0:
            raise ValueError(f"FPS must be finite and positive for {key}")
        result[key] = manifest.fps
    if set(result) != keys:
        raise ValueError("Missing evaluated clip in dataset manifest")
    return result


def _domains(
    videos: np.ndarray, prefixes: Sequence[tuple[str, str]], default: str
) -> dict[str, str]:
    if not default.strip() or any(
        not prefix or not domain.strip() for prefix, domain in prefixes
    ):
        raise ValueError("Domain rules must have nonempty prefixes and domains")
    result = {}
    for video in videos.tolist():
        matches = [domain for prefix, domain in prefixes if video.startswith(prefix)]
        if len(matches) > 1:
            raise ValueError(f"Overlapping domain rules for {video}")
        result[video] = matches[0] if matches else default
    return result


def evaluate_training_run(
    *,
    training_run: Path,
    output_root: Path,
    output: str,
    splits: Sequence[str] = ("val",),
    device: str = "cpu",
    batch_size: int = 4,
    domain_prefixes: Sequence[tuple[str, str]],
    default_domain: str,
) -> Path:
    """Create a fresh evaluation bundle with configs, selection, metrics and FPS."""
    if (
        not splits
        or len(set(splits)) != len(splits)
        or set(splits) - {"train", "val", "test"}
    ):
        raise ValueError(
            "splits must be distinct train, val or explicitly requested test"
        )
    parsed_device = torch.device(device)
    if parsed_device.type not in {"cpu", "cuda"}:
        raise ValueError("device must be cpu or cuda[:index]")
    if parsed_device.type == "cuda" and not all(
        os.environ.get(name) for name in ("TENNIS_RUN_ID", "TENNIS_REPRO_DIR")
    ):
        raise ValueError(
            "CUDA evaluation must run through training queue (TENNIS_RUN_ID/TENNIS_REPRO_DIR)"
        )
    _domains(np.array([], dtype=str), domain_prefixes, default_domain)
    root = output_root.resolve()
    project = Path.cwd().resolve()
    resolver = PathResolver(
        RuntimePathRoots(
            project_root=project,
            data_root=project / "data",
            checkpoint_root=project / "ckpt",
            output_root=root,
            artifact_root=root,
            cache_root=project / ".cache",
            external_asset_root=project / "third_party",
        )
    )
    fragment = Path(output)
    if len(fragment.parts) != 4 or fragment.parts[:2] != ("slcs", "evaluate"):
        raise ValueError("output must be slcs/evaluate/<experiment>/<run-id>")
    destination: Path = resolver.resolve(PathRole.OUTPUT, output)
    if destination.exists():
        raise FileExistsError(f"Refusing existing evaluation directory: {destination}")
    training_run = resolver.resolve(PathRole.OUTPUT, training_run)
    training = OmegaConf.load(training_run / "config.yaml")
    if not isinstance(training, DictConfig):
        raise ValueError("Training config must be a mapping")
    checkpoint, receipt = select_checkpoint(training_run)
    config = evaluation_config(
        training,
        training_run=training_run,
        checkpoint=checkpoint,
        output_root=root,
        output=output,
        split=splits[0],
        device=device,
        batch_size=batch_size,
    )
    runtime = SLCSEvaluationConfig.from_config(config)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.mkdir()  # Exclusive: a failed/partial previous run is never overwritten.
    base_context = evaluation_context(checkpoint, input_mode="full")
    receipt["checkpoint_sha256"] = base_context["checkpoint_sha256"]
    save_json(receipt, destination / "selection.json")
    OmegaConf.save(config, destination / "evaluation_config.yaml", resolve=True)
    predictor = SLCSPredictor.load_from_checkpoint(
        checkpoint,
        resolver=runtime.resolver,
        device=device,
        strict=True,
        weights_only=False,
    )
    predictor.model.float()
    for split in splits:
        bundles = {}
        for condition in CONDITIONS:
            config.evaluate.split = split
            config.evaluate.input_mode = condition
            directory = destination / split / condition
            config.evaluate.output_dir = str(directory.relative_to(root))
            report, arrays = evaluate_split(
                predictor,
                dataset_root=runtime.data.dataset_root,
                split_file=runtime.data.split_file,
                split=split,
                data_config=runtime.data.pipeline,
                batch_size=batch_size,
                input_mode=condition,
            )
            fps = clip_fps(runtime.data.dataset_root, arrays)
            context = {
                **base_context,
                "input_mode": condition,
                "split": split,
                "training_run": str(training_run),
                "selection": receipt,
                "evaluation_config": OmegaConf.to_container(config, resolve=True),
                "precision": f"{parsed_device.type} float32 (no autocast)",
                "augmentation_applied": False,
                "fps_by_clip": [
                    {"video_id": video, "clip_id": clip, "fps": value}
                    for (video, clip), value in sorted(fps.items())
                ],
            }
            save_evaluation(directory, report, arrays, context=context)
            OmegaConf.save(config, directory / "evaluation_config.yaml", resolve=True)
            save_json(
                summarize_motion(
                    arrays, fps, position_representation="normalized_court"
                ),
                directory / "motion.json",
            )
            bundles[condition] = directory
        comparison = compare_conditions(
            bundles, _domains(arrays["video_ids"], domain_prefixes, default_domain)
        )
        comparison["domain_rules"] = {
            "prefixes": list(domain_prefixes),
            "default": default_domain,
        }
        save_comparison(comparison, destination / split / "comparison")
    return destination
