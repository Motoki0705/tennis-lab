"""Reproduce the inference checkpoints published in tennis-lab-models.

Run from the repository root with the locked virtualenv (CPU only)::

    CUDA_VISIBLE_DEVICES='' .venv/bin/python scripts/export_detector_weights.py \
        --source-root /absolute/path/to/original/ckpt \
        --output-root /absolute/path/to/new/bundle \
        --external-asset-root /absolute/path/to/third_party

Only the two SHA-pinned, trusted local training checkpoints are deserialized.
Existing output files are never replaced. Lightning's state_dict and complete
hyper_parameters are preserved, including the court backbone and target bundle.
Training progress, callbacks, optimizer, scheduler and loop states are omitted.
File-handle serialization fixes the internal archive name across output paths.
The printed JSON records original/export hashes, exact tensor comparisons, and
CPU float32 inference parity through the production strict checkpoint loaders.
Use --verify-only to check an existing bundle (including a fresh Hub download).
This release reproduction script is intentionally outside the runtime src tree.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.checksum import dual_sha256  # noqa: E402
from src.utils.configuration import (  # noqa: E402
    PathResolver,
    PathRole,
    RuntimePathRoots,
)
from src.utils.model_weights import PUBLISHED_WEIGHTS  # noqa: E402
from src.utils.paths import PROJECT_ROOT  # noqa: E402

SOURCES = {
    "court_detection/multiscale_depth3/b863df1f01f0.ckpt": "b863df1f01f00d2ff00a21d56a461879e3c5af193a9f32cec47a88d2842f8383",
    "ball_detection/i935-mixed-ft-s42-epoch09.ckpt": "37f4c59aead00062829280ad591b3874886978891104a290f704a2c33c3c1b36",
}
RETAINED_KEYS = (
    "pytorch-lightning_version",
    "state_dict",
    "hparams_name",
    "hyper_parameters",
)


def export_checkpoint(
    source: Path, destination: Path, expected_sha256: str
) -> dict[str, Any]:
    """Strip training state only after authenticating the original checkpoint."""
    source_hash = dual_sha256(source)
    if source_hash != expected_sha256:
        raise ValueError(f"Source SHA-256 mismatch: {source}: {source_hash}")
    original = torch.load(source, map_location="cpu", weights_only=False, mmap=True)
    exported = {key: original[key] for key in RETAINED_KEYS}
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("xb") as handle:
        torch.save(exported, handle)
    restored = torch.load(
        destination, map_location="cpu", weights_only=False, mmap=True
    )
    assert tuple(restored) == RETAINED_KEYS
    assert restored["hyper_parameters"] == original["hyper_parameters"]
    assert restored["state_dict"].keys() == original["state_dict"].keys()
    for name, value in original["state_dict"].items():
        torch.testing.assert_close(restored["state_dict"][name], value, rtol=0, atol=0)
    return {
        "source_sha256": source_hash,
        "sha256": dual_sha256(destination),
        "source_bytes": source.stat().st_size,
        "bytes": destination.stat().st_size,
        "retained_keys": list(RETAINED_KEYS),
        "removed_keys": sorted(set(original) - set(RETAINED_KEYS)),
        "tensors": len(restored["state_dict"]),
        "all_tensors_equal": True,
    }


def inference_outputs(
    filename: str,
    *,
    artifact_root: Path,
    checkpoint_root: Path,
    external_asset_root: Path,
) -> dict[str, torch.Tensor]:
    """Load one real model at a time, retaining only its small CPU outputs."""
    resolver = PathResolver(
        RuntimePathRoots(
            project_root=PROJECT_ROOT,
            data_root=PROJECT_ROOT / "data",
            checkpoint_root=checkpoint_root,
            artifact_root=artifact_root,
            output_root=PROJECT_ROOT / "outputs",
            cache_root=PROJECT_ROOT / ".cache",
            external_asset_root=external_asset_root,
        )
    )
    generator = torch.Generator().manual_seed(42)
    outputs: dict[str, torch.Tensor]
    with torch.inference_mode():
        if filename.startswith("court_detection/"):
            from src.tasks.court_detection.inference.predictor import CourtPredictor

            predictor = CourtPredictor.load_from_checkpoint(
                filename,
                device="cpu",
                resolver=resolver,
                checkpoint_role=PathRole.ARTIFACT,
                strict=True,
                weights_only=False,
            )
            images = torch.rand((1, 3, 32, 32), generator=generator)
            mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
            call = predictor.adapter.prepare_images(
                (images - mean) / std,
                content_size_hw=torch.tensor([[32, 32]]),
            )
            prediction = predictor.model(*call.model_args)
            outputs = {
                f"dense/{name}": value.clone()
                for name, value in prediction.dense_logits.items()
            }
            assert prediction.pose is not None
            outputs["pose"] = prediction.pose.values.clone()
        elif filename.startswith("ball_detection/"):
            from src.tasks.ball_detection.inference.predictor import (
                BallDetectionPredictor,
            )

            ball = BallDetectionPredictor.load_from_checkpoint(
                filename,
                device="cpu",
                resolver=resolver,
                checkpoint_role=PathRole.ARTIFACT,
                strict=True,
                weights_only=False,
                subpixel_refine=True,
            )
            frames = torch.rand((1, 8, 3, 32, 32), generator=generator)
            result = ball.predict(frames)
            outputs = {
                "coords": result.coords.clone(),
                "confidence": result.confidence.clone(),
                "heatmaps": result.heatmaps.clone(),
            }
        else:
            raise ValueError(f"Unsupported detector checkpoint: {filename}")
    return outputs


def verify_inference(
    source_root: Path, output_root: Path, external_asset_root: Path
) -> dict[str, Any]:
    """Authenticate the two versions before checking exact CPU predictions."""
    report = {}
    for weight in PUBLISHED_WEIGHTS:
        source_hash = dual_sha256(source_root / weight.filename)
        exported_hash = dual_sha256(output_root / weight.filename)
        if source_hash != SOURCES[weight.filename] or exported_hash != weight.sha256:
            raise ValueError(f"Checkpoint SHA-256 mismatch: {weight.filename}")
        original = inference_outputs(
            weight.filename,
            artifact_root=source_root,
            checkpoint_root=source_root,
            external_asset_root=external_asset_root,
        )
        exported = inference_outputs(
            weight.filename,
            artifact_root=output_root,
            checkpoint_root=source_root,
            external_asset_root=external_asset_root,
        )
        assert original.keys() == exported.keys()
        for name in original:
            torch.testing.assert_close(exported[name], original[name], rtol=0, atol=0)
        report[weight.filename] = {
            "source_sha256": source_hash,
            "sha256": exported_hash,
            "device": "cpu",
            "dtype": "float32",
            "strict": True,
            "input_shape": [1, 3, 32, 32]
            if weight.name == "court"
            else [1, 8, 3, 32, 32],
            "output_shapes": {
                name: list(tensor.shape) for name, tensor in original.items()
            },
            "max_abs_difference": max(
                float((original[name] - exported[name]).abs().max())
                for name in original
            ),
        }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--external-asset-root", type=Path, required=True)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    if not all(
        path.is_absolute()
        for path in (args.source_root, args.output_root, args.external_asset_root)
    ):
        parser.error("All roots must be absolute paths")
    torch.set_num_threads(2)
    results = {}
    if not args.verify_only:
        results["export"] = {
            name: export_checkpoint(
                args.source_root / name, args.output_root / name, digest
            )
            for name, digest in SOURCES.items()
        }
    results["inference_parity"] = verify_inference(
        args.source_root.resolve(),
        args.output_root.resolve(),
        args.external_asset_root.resolve(),
    )
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
