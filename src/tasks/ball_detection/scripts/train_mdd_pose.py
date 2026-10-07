"""Explicit coordinate-model training entrypoint; run only after proposal review.

CUDA execution must be submitted through the repository's shared training queue.
The heatmap trainer remains specific to ConvNeXt; no coordinates-to-heatmap shim.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import torch
from torch import Tensor
from torch.utils.data import DataLoader

from src.tasks.ball_detection.data.pose_windows import (
    PoseWindowDataset,
    collate_pose_windows,
    coordinate_loss,
)
from src.tasks.ball_detection.model_io.mdd_pose import (
    MDDPoseAdapter,
    MDDPoseInput,
    prepare_mdd_pose_inputs,
)
from src.tasks.ball_detection.models.mdd_pose import MDDPoseConfig, MDDPoseDetector
from src.utils.checksum import dual_sha256
from src.utils.configuration import (
    BoundaryPathField,
    NonHydraPathBoundary,
    PathDirection,
    PathKind,
    PathResolver,
    PathRole,
    RuntimePathRoots,
)

PATH_BOUNDARY = NonHydraPathBoundary(
    name="ball_detection.train_mdd_pose",
    fields=(
        BoundaryPathField("manifest", PathRole.ARTIFACT, PathDirection.INPUT, PathKind.FILE, must_exist=True),
        BoundaryPathField("model_config", PathRole.PROJECT, PathDirection.INPUT, PathKind.FILE, must_exist=True),
        BoundaryPathField("output", PathRole.OUTPUT, PathDirection.OUTPUT, PathKind.DIRECTORY),
    ),
)


def predict(model: MDDPoseDetector, batch: dict[str, Any], device: torch.device) -> Tensor:
    inputs = MDDPoseInput(*(batch[key].to(device) for key in ("mdd", "pose", "pose_valid", "timestamps")))
    return MDDPoseAdapter(model.config).decode_output(model(*prepare_mdd_pose_inputs(model.config, inputs)))


@torch.no_grad()
def evaluate(model: MDDPoseDetector, loader: DataLoader[Any], device: torch.device) -> dict[str, float | int]:
    """Count each observed frame once; ownership is centre distance, then start."""
    model.eval()
    frames: dict[tuple[str, int], tuple[tuple[float, int], float]] = {}
    for batch in loader:
        uv = predict(model, batch, device).cpu()
        errors = ((uv - batch["uv"]) * (batch["source_size"][:, None] - 1)).norm(dim=-1)
        for row, clip in enumerate(batch["clip_id"]):
            start = batch["start"][row]
            for frame in range(32):
                if not bool(batch["position_valid"][row, frame]):
                    continue
                key, owner = (clip, start + frame), (abs(frame - 15.5), start)
                if key not in frames or owner < frames[key][0]:
                    frames[key] = owner, float(errors[row, frame])
    if not frames:
        raise ValueError("Validation has no observed coordinates")
    values = torch.tensor([value for _, value in frames.values()], dtype=torch.float64)
    return {"observed_frames": len(frames), "mean_error_px": float(values.mean()),
            "median_error_px": float(values.median()), "p95_error_px": float(values.quantile(.95))}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-config", type=Path, required=True)
    parser.add_argument("--compression", choices=("conv2d", "average", "unshuffle", "haar"))
    parser.add_argument("--pose-pooling", choices=("deepsets", "attention", "hierarchical", "gnn"))
    parser.add_argument("--readout", choices=("query", "pose"))
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--learning-rate", type=float, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--mdd-a", type=float, default=.2)
    parser.add_argument("--mdd-b", type=float, default=.15)
    args = parser.parse_args()
    if not args.manifest.is_absolute() or not args.output.is_absolute():
        parser.error("Manifest and output must be absolute paths")
    if min(args.epochs, args.batch_size) < 1 or not 0 < args.learning_rate < float("inf"):
        parser.error("Training budget and learning rate must be positive")
    if not args.model_config.is_absolute():
        parser.error("Model config must be an absolute path")
    roots = RuntimePathRoots(project_root=args.model_config.parent, data_root=args.manifest.parent,
                             artifact_root=args.manifest.parent, checkpoint_root=args.output.parent,
                             cache_root=args.output.parent, output_root=args.output.parent,
                             external_asset_root=args.output.parent)
    paths = PATH_BOUNDARY.validate({"manifest": args.manifest, "model_config": args.model_config,
                                   "output": args.output}, resolver=PathResolver(roots))
    args.manifest, args.model_config, args.output = (paths.declared(k).path for k in ("manifest", "model_config", "output"))
    config = MDDPoseConfig.load(args.model_config)
    config = replace(config, **{key: getattr(args, key) for key in ("compression", "pose_pooling", "readout")
                               if getattr(args, key) is not None})
    identity = dual_sha256(args.manifest)
    training = PoseWindowDataset(args.manifest, split="train", mdd_a=args.mdd_a, mdd_b=args.mdd_b)
    validation = PoseWindowDataset(args.manifest, split="val", mdd_a=args.mdd_a, mdd_b=args.mdd_b)
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    model = MDDPoseDetector(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=.01)
    train_loader: DataLoader[Any] = DataLoader(training, batch_size=args.batch_size, shuffle=True,
                                               collate_fn=collate_pose_windows, num_workers=0,
                                               generator=torch.Generator().manual_seed(args.seed))
    val_loader: DataLoader[Any] = DataLoader(validation, batch_size=args.batch_size,
                                             collate_fn=collate_pose_windows, num_workers=0)
    args.output.mkdir(parents=True, exist_ok=False)
    settings = dict(model=asdict(config), manifest_sha256=identity,
                    arguments={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                    selection="unique observed validation frames; mean source-pixel error; no test access",
                    input_rgb="stored JPEG at native size; [0,1] luminance; no ImageNet normalization",
                    mdd_first_frame="zero, no RGB predecessor outside the 32-frame window")
    (args.output / "config.json").write_text(json.dumps(settings, indent=2) + "\n")
    (args.output / "data_manifest.json").write_bytes(args.manifest.read_bytes())
    best = float("inf")
    for epoch in range(args.epochs):
        model.train()
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            loss = coordinate_loss(predict(model, batch, device), batch["uv"].to(device), batch["position_valid"].to(device))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            optimizer.step()
        report = evaluate(model, val_loader, device)
        if dual_sha256(args.manifest) != identity:
            raise ValueError("Training manifest changed during the run")
        path = args.output / f"epoch-{epoch:03d}.pt"
        torch.save(dict(schema="mdd_pose_coordinates.v1", model_config=asdict(config), state_dict=model.state_dict(),
                        optimizer=optimizer.state_dict(), epoch=epoch, manifest_sha256=identity,
                        torch_rng=torch.get_rng_state(), validation=report), path)
        if report["mean_error_px"] < best:
            best = float(report["mean_error_px"])
            (args.output / "best.json").write_text(json.dumps(dict(epoch=epoch, checkpoint=path.name,
                checkpoint_sha256=dual_sha256(path), **report), indent=2) + "\n")
        with (args.output / "metrics.jsonl").open("a") as stream:
            stream.write(json.dumps(dict(epoch=epoch, **report)) + "\n")
        print(json.dumps(dict(epoch=epoch, **report)), flush=True)


if __name__ == "__main__":
    main()
