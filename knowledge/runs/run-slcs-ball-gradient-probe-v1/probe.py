"""CPU-only local checkpoint gradient diagnostic; no training or causal inference.

Run from the worktree with PYTHONPATH=. and the repository .venv/bin/python.
One first eligible full train window per domain, one seed, float32, batch size 1.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
import platform
import random
import subprocess
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

# Set before importing numerical libraries, including when the caller omitted them.
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

from src.tasks.slcs.configuration import SLCSTrainingRuntimeConfig
from src.tasks.slcs.data.annotation import (
    SLCS_ANNOTATION_FILENAME,
    SLCS_SCENE_ARCHIVE_FILENAME,
    slcs_annotation_dir,
)
from src.tasks.slcs.data.dataset import SLCSWindowDataset, collate_slcs
from src.tasks.slcs.data.dino_tokens import DINO_ANNOTATION_FILENAME, dino_dir
from src.tasks.slcs.inference.predictor import SLCSPredictor
from src.tasks.slcs.training.losses import build_slcs_loss_inputs
from src.tennis_scene.generate_dataset.manifest import DATASET_MANIFEST_FILENAME
from src.utils.checksum import dual_sha256
from src.utils.configuration import PathResolver
from src.utils.schema.court import COURT_COORD_SCALE_XYZ

MAIN = Path("/home/kamimura/projects/tennis-lab")
CHECKPOINT = MAIN / "ckpt/slcs/real-rgb-pilot-augmented-e60-v2.ckpt"
CONFIG = MAIN / "outputs/slcs/train/real_rgb_pilot_augmented/s42-002/config.yaml"
EXPECTED_SHA = "b925fc2a8cc2eb3dc80aacb7e2e6f8389d40ab48f8afb3c74408b97cbd17d30c"
SEED = 42


def finite(tensor: torch.Tensor, context: str) -> None:
    if not bool(torch.isfinite(tensor).all()):
        raise ValueError(f"Nonfinite values: {context}")


def tensor_sha(tensor: torch.Tensor) -> str:
    array = tensor.detach().cpu().contiguous().numpy()
    digest = hashlib.sha256()
    digest.update(str((array.dtype.str, array.shape)).encode())
    digest.update(array.tobytes())
    return digest.hexdigest()


def gradient_summary(
    objectives: dict[str, torch.Tensor], parameters: list[torch.Tensor]
) -> dict[str, Any]:
    """Float64 dot/squared-norm sums, equivalent to a common flat concatenation.

    Unused autograd entries are represented explicitly as zero vectors. A zero
    norm makes cosine undefined (JSON null), not an invented cosine of zero.
    """
    if not parameters:
        raise ValueError("Empty differentiation parameter group")
    gradients = {}
    counts = {}
    for name, objective in objectives.items():
        raw = torch.autograd.grad(
            objective, parameters, retain_graph=True, allow_unused=True
        )
        counts[name] = {
            "unused_tensors": sum(g is None for g in raw),
            "unused_elements": sum(
                p.numel() for p, g in zip(parameters, raw, strict=True) if g is None
            ),
        }
        gradients[name] = [
            torch.zeros_like(p, dtype=torch.float64)
            if g is None
            else g.detach().double()
            for p, g in zip(parameters, raw, strict=True)
        ]
        for g in gradients[name]:
            finite(g, f"gradient {name}")
    squares = {
        name: sum(float(g.square().sum()) for g in gs) for name, gs in gradients.items()
    }
    pairs = {}
    for left, right in itertools.combinations(gradients, 2):
        dot = sum(
            float((a * b).sum())
            for a, b in zip(gradients[left], gradients[right], strict=True)
        )
        denom = (squares[left] * squares[right]) ** 0.5
        pairs[f"{left}__{right}"] = {
            "dot": dot,
            "cosine": dot / denom if denom > 0 else None,
        }
    return {
        "tensor_count": len(parameters),
        "element_count": sum(p.numel() for p in parameters),
        "norms": {name: value**0.5 for name, value in squares.items()},
        "unused_as_zero": counts,
        "pairs": pairs,
    }


def identify_domain(dataset: SLCSWindowDataset, index: int) -> str:
    meta = dataset.metas[index]
    manifest = dataset._clips[meta.clip_id].manifest
    sources = [str(camera["source_path"]) for camera in manifest.cameras]
    if all("/meiji_3cam/" in source for source in sources):
        return "meiji"
    if meta.video_id.startswith("broadcast_"):
        return "broadcast"
    raise ValueError(f"Unrecognized domain: {meta.video_id}, {sources}")


def selected_windows(dataset: SLCSWindowDataset) -> dict[str, int]:
    selected = {}
    for index, meta in enumerate(dataset.metas):
        domain = identify_domain(dataset, index)
        if domain in selected or meta.window_length != 120:
            continue
        sample = dataset[index]
        if not bool(sample["padding_mask"].any()) and bool(
            (sample["target_ball_valid"] & (sample["target_ball_weight"] > 0)).any()
        ):
            selected[domain] = index
    if set(selected) != {"meiji", "broadcast"}:
        raise ValueError(f"Both full-window supervised domains required: {selected}")
    return selected


def probe_mode(
    predictor: SLCSPredictor, batch: dict[str, torch.Tensor], mode: str
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    predictor.model.train(mode == "train_dropout")
    targets = predictor.model_adapter.build_training_targets(batch)
    output = predictor.model_io.run(batch)
    inputs = build_slcs_loss_inputs(output, targets)
    if inputs.pred_ball_position.shape != (1, 120, 3):
        raise ValueError(f"Unexpected ball output: {inputs.pred_ball_position.shape}")
    for name, value in vars(inputs).items():
        finite(value, name)
    loss_fn = predictor.lightning_module.loss_fn
    losses = loss_fn(inputs)
    weighted = {
        name: losses[name] * weight for name, _, weight in loss_fn.weighted_terms
    }
    for name, value in losses.items():
        finite(value, name)
    objectives = {
        "ball_supervised": weighted["ball_position"] + weighted["ball_position_nll"],
        "ball_smooth": weighted["ball_position_smoothness"],
        "player": sum(
            value for name, value in weighted.items() if name.startswith("player_")
        ),
        "ground": weighted["ground_penetration"],
    }
    if not torch.allclose(
        sum(objectives.values()), losses["total"], rtol=1e-5, atol=1e-6
    ):
        raise ValueError("Gradient objectives do not partition production total loss")
    prefixes = ("entity_layers.", "time_layers.", "dino_cross_layers.", "final_norm.")
    groups: dict[str, list[tuple[str, torch.Tensor]]] = {
        "shared_axial_trunk": [
            (name, p)
            for name, p in predictor.model.named_parameters()
            if name.startswith(prefixes)
        ],
        "ball_position_head": [
            (name, p)
            for name, p in predictor.model.named_parameters()
            if name.startswith("ball_position_head.")
        ],
        "pred_ball_position": [("pred_ball_position", inputs.pred_ball_position)],
    }
    gradient_results = {}
    for name, pairs in groups.items():
        gradient_results[name] = gradient_summary(objectives, [p for _, p in pairs])
        gradient_results[name]["names_in_order"] = [name for name, _ in pairs]
    mask = inputs.ball_mask
    scale = torch.tensor(COURT_COORD_SCALE_XYZ)
    pred_m = inputs.pred_ball_position[mask] * scale
    target_m = inputs.target_ball_position[mask] * scale
    if pred_m.shape[0] < 2:
        raise ValueError("Need at least two valid ball frames for temporal std")
    penalty = loss_fn.temporal_smoothness
    # Replace invalid targets before differencing. Only consecutive all-valid
    # stencils contribute, so placeholders never influence the reported value.
    safe_target = inputs.target_ball_position.masked_fill(~mask[..., None], 0)
    stencil_mask = mask.unfold(1, penalty.window_width, 1).all(-1)
    supported = int(stencil_mask.sum())
    if supported == 0:
        raise ValueError("No valid consecutive ball stencil for target smoothness")
    result = {
        "mode": mode,
        "seed": SEED,
        "precision": "float32; autocast disabled",
        "raw_losses": {name: float(value.detach()) for name, value in losses.items()},
        "weighted_losses": {
            name: float(value.detach()) for name, value in weighted.items()
        },
        "gradient_objective_values": {
            name: float(value.detach()) for name, value in objectives.items()
        },
        "gradients": gradient_results,
        "ball_valid_count": int(mask.sum()),
        "ball_error_m_mean": float((pred_m - target_m).norm(dim=-1).mean().detach()),
        "pred_ball_std_m_xyz_population": pred_m.detach()
        .std(dim=0, correction=0)
        .tolist(),
        "target_ball_std_m_xyz_population": target_m.std(dim=0, correction=0).tolist(),
        "ball_b_normalized_mean": float(
            inputs.pred_ball_position_log_b[mask].exp().mean().detach()
        ),
        "smoothness": {
            "order": penalty.order,
            "beta": penalty.beta,
            "unit": "normalized court coordinates; per-frame differences; no dt division",
            "valid_stencils": supported,
            "model_all_real_frames": float(
                penalty(inputs.pred_ball_position, ~inputs.padding_mask).detach()
            ),
            "model_valid_supported": float(
                penalty(inputs.pred_ball_position, mask).detach()
            ),
            "target_valid_supported": float(penalty(safe_target, mask)),
        },
    }
    arrays = {name: value.detach().numpy() for name, value in vars(inputs).items()}
    return result, arrays


def publish_results(output_dir: Path, report: dict[str, Any]) -> None:
    """Atomic replacement keeps readers from seeing partially written JSON."""
    text = json.dumps(report, indent=2, allow_nan=False) + "\n"
    temporary = output_dir / "results.json.tmp"
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(output_dir / "results.json")


@contextmanager
def publish_run(output_dir: Path) -> Iterator[dict[str, Any]]:
    """Preserve completed progress and re-raise the original error without retry."""
    report: dict[str, Any] = {
        "status": "running",
        "stage": "initializing",
        "domains": {},
    }
    publish_results(output_dir, report)
    try:
        yield report
    except BaseException as error:
        report["status"] = "failed"
        report["error"] = {"type": type(error).__name__, "message": str(error)}
        try:
            publish_results(output_dir, report)
        except Exception as publication_error:
            error.add_note(f"Could not publish failure results: {publication_error!r}")
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="Fresh output directory; existing paths are rejected",
    )
    args = parser.parse_args()
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    with publish_run(args.output_dir) as report:
        run_probe(args.output_dir, report)


def run_probe(output_dir: Path, report: dict[str, Any]) -> None:
    torch.set_num_threads(2)
    torch.set_num_interop_threads(2)
    torch.use_deterministic_algorithms(True)
    config = OmegaConf.load(CONFIG)
    if not isinstance(config, DictConfig):
        raise TypeError("Training configuration must be a mapping")
    runtime = SLCSTrainingRuntimeConfig.from_config(config)
    if runtime.data.pipeline.window_size != 120:
        raise ValueError("This diagnostic requires the fixed 120-frame pilot config")
    checkpoint_sha = dual_sha256(CHECKPOINT)
    if checkpoint_sha != EXPECTED_SHA:
        raise ValueError(f"Checkpoint SHA mismatch: {checkpoint_sha}")
    # The selected checkpoint was exported outside the original training root.
    # Narrowly relocate checkpoint authority while retaining all other roots.
    resolver = PathResolver(
        replace(runtime.resolver.roots, checkpoint_root=CHECKPOINT.parent.resolve())
    )
    predictor = SLCSPredictor.load_from_checkpoint(
        CHECKPOINT, resolver=resolver, device="cpu", strict=True, weights_only=False
    )
    for section in ("model", "data", "loss"):
        if OmegaConf.to_container(
            predictor.lightning_module.config[section], resolve=True
        ) != OmegaConf.to_container(config[section], resolve=True):
            raise ValueError(f"Saved checkpoint and fixed config disagree: {section}")
    if any(
        p.device.type != "cpu" or p.dtype != torch.float32
        for p in predictor.model.parameters()
    ):
        raise ValueError("Expected CPU float32 model parameters")
    state_before = {
        name: tensor_sha(value) for name, value in predictor.model.state_dict().items()
    }
    dataset = SLCSWindowDataset(
        dataset_root=runtime.data.dataset_root,
        split_file=runtime.data.split_file,
        split="train",
        config=replace(runtime.data.pipeline, augmentation=None),
        stride=runtime.data.pipeline.train_stride,
        augment=False,
    )
    chosen = selected_windows(dataset)
    OmegaConf.save(config, output_dir / "resolved_training_config.yaml", resolve=True)
    report.update(
        {
            "status": "running",
            "scope": "Local checkpoint diagnostic, two batch-size-1 train windows; not proof of training causality or population gradient dominance.",
            "checkpoint": str(CHECKPOINT),
            "checkpoint_sha256": checkpoint_sha,
            "config": str(CONFIG),
            "seed": SEED,
            "device": "cpu",
            "threads": 2,
            "training_precision": str(config.training.trainer.precision),
            "diagnostic_precision": "float32",
            "augmentation": {
                "dataset_augment": False,
                "pipeline_augmentation": None,
                "same_batch_between_modes": True,
            },
            "selection": "First eligible dataset index per domain, full 120 frames, positive-weight ball supervision; no random search",
            "shared_group_definition": "Shared axial entity/time/DINO cross layers and final norm, excludes input embeddings/projections and task heads; see exact names",
            "versions": {
                "python": platform.python_version(),
                "torch": torch.__version__,
                "numpy": np.__version__,
            },
            "command": [sys.executable, *sys.argv],
            "cwd": str(Path.cwd()),
            "git_head": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], text=True
            ).strip(),
            "git_status": subprocess.check_output(
                ["git", "status", "--short"], text=True
            ),
            "court_coordinate_scale_xyz": list(COURT_COORD_SCALE_XYZ),
            "input_sha256": {},
            "domains": {},
        }
    )
    publish_results(output_dir, report)
    source_paths = {
        CONFIG,
        CHECKPOINT,
        Path(__file__).resolve(),
        runtime.data.split_file,
        runtime.data.dataset_root / DATASET_MANIFEST_FILENAME,
    }
    for index in chosen.values():
        meta = dataset.metas[index]
        manifest = dataset._clips[meta.clip_id].manifest
        ann = slcs_annotation_dir(manifest.clip_dir)
        source_paths.update(
            (
                manifest.manifest_path,
                ann / SLCS_ANNOTATION_FILENAME,
                ann / SLCS_SCENE_ARCHIVE_FILENAME,
                (ann / SLCS_SCENE_ARCHIVE_FILENAME).with_suffix(".metadata.json"),
                dino_dir(manifest.clip_dir) / DINO_ANNOTATION_FILENAME,
                dino_dir(manifest.clip_dir) / f"{meta.camera_id}.npz",
            )
        )
    report["input_sha256"] = {
        str(path): dual_sha256(path) for path in sorted(source_paths)
    }
    for domain, index in chosen.items():
        batch = collate_slcs([dataset[index]])
        for key, value in batch.items():
            finite(value, f"input {key}")
        before = {name: tensor_sha(value) for name, value in batch.items()}
        meta = dataset.metas[index]
        manifest = dataset._clips[meta.clip_id].manifest
        camera = manifest.cameras[manifest.camera_index(meta.camera_id)]
        local_frames = batch["frame_idx"][0].tolist()
        np.savez_compressed(
            output_dir / f"{domain}_inputs.npz",
            allow_pickle=False,
            **{key: value.numpy() for key, value in batch.items()},
        )
        record: dict[str, Any] = {
            "dataset_index": index,
            "meta": asdict(meta),
            "prediction_id": dataset.prediction_ids[index],
            "fps": manifest.fps,
            "camera_metadata": camera,
            "frame_idx_clip": local_frames,
            "frame_idx_source": [
                camera["source_frame_start"] + frame for frame in local_frames
            ],
            "input_tensor_sha256": before,
            "modes": {},
        }
        report["domains"][domain] = record
        for mode in ("eval", "train_dropout"):
            report["stage"] = f"{domain}/{mode}"
            publish_results(output_dir, report)
            result, arrays = probe_mode(predictor, batch, mode)
            record["modes"][mode] = result
            np.savez_compressed(
                output_dir / f"{domain}_{mode}.npz", allow_pickle=False, **arrays
            )
            if before != {name: tensor_sha(value) for name, value in batch.items()}:
                raise ValueError("Input batch mutated")
            publish_results(output_dir, report)
    report["stage"] = "verifying_immutability"
    publish_results(output_dir, report)
    if state_before != {
        name: tensor_sha(value) for name, value in predictor.model.state_dict().items()
    }:
        raise ValueError("Model state mutated during diagnostic")
    if any(parameter.grad is not None for parameter in predictor.model.parameters()):
        raise ValueError("Unexpected accumulated parameter gradients")
    for path, digest in report["input_sha256"].items():
        if dual_sha256(path) != digest:
            raise ValueError(f"Input changed during diagnostic: {path}")
    report["status"] = "completed"
    report["state_unchanged"] = True
    report["stage"] = "finished"
    publish_results(output_dir, report)
    print(
        json.dumps(
            {
                "status": "completed",
                "results": str(output_dir / "results.json"),
                "selected": chosen,
            }
        )
    )


if __name__ == "__main__":
    main()
