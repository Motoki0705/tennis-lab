"""One-time migration of the recorded deployment artifact; not an application CLI."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
from omegaconf import open_dict

SOURCE_SHA256 = "67e01836b1016ca1d0fb2acf088bfe44aa49a804c91aa8fb9cc132ec115dd197"


def sha256(path: Path) -> str:
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def predict(checkpoint: Path, output: Path) -> None:
    # The caller deliberately selects the original or current checkout by cwd.
    sys.path.insert(0, str(Path.cwd()))
    from src.tasks.blcs.inference.predictor import BLCSPredictor
    from src.utils.configuration import PathResolver, RuntimePathRoots

    root = Path.cwd()
    resolver = PathResolver(
        RuntimePathRoots(
            project_root=root,
            data_root=root / "data",
            output_root=root / "outputs",
            checkpoint_root=checkpoint.parent,
            artifact_root=output.parent,
            cache_root=root / ".cache",
            external_asset_root=root / "third_party",
        )
    )
    predictor = BLCSPredictor.load_from_checkpoint(
        checkpoint, resolver=resolver, device="cpu"
    )
    generator = torch.Generator().manual_seed(1044)
    batch = {
        "ball_uv": torch.rand(1, 2, 8, 2, generator=generator),
        "ball_vis": torch.ones(1, 2, 8),
        "court_kp": torch.rand(1, 2, 8, 14, 2, generator=generator),
        "court_vis": torch.ones(1, 2, 8, 14),
        "padding_mask": torch.zeros(1, 2, 8, dtype=torch.bool),
    }
    result = predictor.predict_batch(batch, denormalize=True)
    if result.velocity is not None:
        raise ValueError("The deployment source has no velocity head")
    np.savez_compressed(output, position_m=result.position.numpy())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--original-checkout", type=Path)
    parser.add_argument("--evidence-dir", type=Path)
    parser.add_argument("--predict", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(2)
    if args.predict:
        predict(args.source, args.output)
        return
    if args.original_checkout is None or args.evidence_dir is None:
        parser.error("Migration requires --original-checkout and --evidence-dir")
    if args.output.exists():
        raise FileExistsError(args.output)
    if sha256(args.source) != SOURCE_SHA256:
        raise ValueError("This migration is restricted to the recorded deployment SHA")
    raw = torch.load(args.source, map_location="cpu", weights_only=False, mmap=True)
    config = copy.deepcopy(raw["hyper_parameters"]["config"])
    if (
        "physics_head" in config["model"]
        or config["model"]["predict_velocity"] is not False
    ):
        raise ValueError("Unexpected source model contract")
    with open_dict(config["model"]):
        config["model"]["physics_head"] = False
    migrated = dict(raw)
    migrated["hyper_parameters"] = {**raw["hyper_parameters"], "config": config}
    migrated["inference_migration"] = {
        "schema": "blcs.inference_metadata_migration.v1",
        "source_sha256": SOURCE_SHA256,
        "model_changes": {"physics_head": False},
        "purpose": "inference; original training loss/metrics metadata preserved",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(".pending")
    if temporary.exists():
        raise FileExistsError(temporary)
    torch.save(migrated, temporary)
    restored = torch.load(temporary, map_location="cpu", weights_only=False, mmap=True)
    original_state, migrated_state = raw["state_dict"], restored["state_dict"]
    if original_state.keys() != migrated_state.keys() or any(
        not torch.equal(value, migrated_state[key])
        for key, value in original_state.items()
    ):
        raise AssertionError("Migration changed model weights")
    for section in ("loss", "metrics"):
        if (
            restored["hyper_parameters"]["config"][section]
            != raw["hyper_parameters"]["config"][section]
        ):
            raise AssertionError(f"Historical {section} metadata changed")
    args.evidence_dir.mkdir(parents=True, exist_ok=True)
    script = Path(__file__).resolve()
    for name, checkout, checkpoint in (
        ("original", args.original_checkout, args.source),
        ("migrated", Path.cwd(), temporary),
    ):
        subprocess.run(
            [
                sys.executable,
                str(script),
                "--predict",
                "--source",
                str(checkpoint),
                "--output",
                str(args.evidence_dir / f"{name}-prediction.npz"),
            ],
            cwd=checkout,
            check=True,
            env={**os.environ, "CUDA_VISIBLE_DEVICES": "", "PYTHONPATH": str(checkout)},
        )
    with (
        np.load(args.evidence_dir / "original-prediction.npz") as a,
        np.load(args.evidence_dir / "migrated-prediction.npz") as b,
    ):
        np.testing.assert_array_equal(a["position_m"], b["position_m"])
        maximum_error = float(np.max(np.abs(a["position_m"] - b["position_m"])))
    # Publish only after both the original and current public predictors agree.
    temporary.rename(args.output)
    receipt = {
        "source": str(args.source),
        "source_sha256": SOURCE_SHA256,
        "checkpoint": str(args.output),
        "checkpoint_sha256": sha256(args.output),
        "size_bytes": args.output.stat().st_size,
        "model_changes": {"physics_head": False},
        "state_dict_tensors_equal": len(original_state),
        "original_training_metadata_preserved": True,
        "strict_public_predictor_load": True,
        "original_checkout_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=args.original_checkout, text=True
        ).strip(),
        "verification": "original public predictor versus migrated public predictor; CPU FP32, fixed random 2-view/8-frame input",
        "maximum_position_difference_m": maximum_error,
    }
    (args.evidence_dir / "migration.json").write_text(
        json.dumps(receipt, indent=2) + "\n"
    )
    print(json.dumps(receipt), flush=True)


if __name__ == "__main__":
    main()
