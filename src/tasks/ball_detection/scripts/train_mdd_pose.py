"""Pose-conditioned and pose-free coordinate training with mixed FPS.

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

from src.tasks.ball_detection.data.coordinate_dataset import (
    CoordinateWindowDataset,
    collate_coordinate_windows,
)
from src.tasks.ball_detection.data.pose_windows import coordinate_loss
from src.tasks.ball_detection.models.mdd_pose import (
    MDDPoseConfig,
    MDDPoseDetector,
    MDDQueryDetector,
)
from src.tasks.ball_detection.training.coordinate_evaluation import (
    CoordinateModel,
    evaluate_coordinates,
    predict_coordinates,
    selection_score,
)
from src.tasks.ball_detection.training.coordinate_provenance import (
    coordinate_source_identity,
)
from src.tasks.ball_detection.training.coordinate_sampling import FPSMixSampler
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


def predict(model: CoordinateModel, batch: dict[str, Any], device: torch.device) -> Tensor:
    return predict_coordinates(model, batch, device)


@torch.no_grad()
def evaluate(model: CoordinateModel, loader: DataLoader[Any], device: torch.device) -> dict[str, Any]:
    if not isinstance(loader.dataset, CoordinateWindowDataset):
        raise ValueError("Evaluation requires a declared coordinate window dataset")
    return evaluate_coordinates(model, loader, device, loader.dataset.frame_steps)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-config", type=Path, required=True)
    parser.add_argument("--compression", choices=("conv2d", "average", "unshuffle", "haar"))
    parser.add_argument("--pose-pooling", choices=("deepsets", "attention", "hierarchical", "gnn", "none"))
    parser.add_argument("--readout", choices=("query", "pose", "query_only"))
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--learning-rate", type=float, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--mdd-a", type=float, default=.2)
    parser.add_argument("--mdd-b", type=float, default=.15)
    parser.add_argument("--windows-per-epoch", type=int,
                        help="Equal-FPS mixture size; defaults to the native-FPS training window count")
    parser.add_argument("--selection-scope", choices=("common", "full"), default="common",
                        help="Checkpoint selection uses the equal-FPS mean error in this validation scope")
    args = parser.parse_args()
    if not args.manifest.is_absolute() or not args.output.is_absolute():
        parser.error("Manifest and output must be absolute paths")
    if min(args.epochs, args.batch_size) < 1 or not 0 < args.learning_rate < float("inf"):
        parser.error("Training budget and learning rate must be positive")
    if args.windows_per_epoch is not None and args.windows_per_epoch < 1:
        parser.error("Epoch window budget must be positive")
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
    overrides = {key: getattr(args, key) for key in ("compression", "pose_pooling", "readout")
                 if getattr(args, key) is not None}
    if overrides.get("pose_pooling") == "none":
        overrides["pose_pooling"] = None
    config = replace(config, **overrides)
    identity = dual_sha256(args.manifest)
    training = CoordinateWindowDataset(args.manifest, split="train", requires_pose=config.requires_pose,
                                       mdd_a=args.mdd_a, mdd_b=args.mdd_b)
    validation = CoordinateWindowDataset(args.manifest, split="val", requires_pose=config.requires_pose,
                                         mdd_a=args.mdd_a, mdd_b=args.mdd_b)
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    model = (MDDPoseDetector(config) if config.requires_pose else MDDQueryDetector(config)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=.01)
    epoch_windows = args.windows_per_epoch or sum(window.frame_step == 1 for _, window in training.windows)
    sampler = FPSMixSampler(training, windows_per_epoch=epoch_windows, seed=args.seed)
    train_loader: DataLoader[Any] = DataLoader(training, batch_size=args.batch_size, sampler=sampler,
                                               collate_fn=collate_coordinate_windows, num_workers=0)
    val_loader: DataLoader[Any] = DataLoader(validation, batch_size=args.batch_size,
                                             collate_fn=collate_coordinate_windows, num_workers=0)
    args.output.mkdir(parents=True, exist_ok=False)
    code = coordinate_source_identity()
    settings = dict(model=asdict(config), manifest_sha256=identity, code=code,
                    model_config_sha256=dual_sha256(args.model_config),
                    arguments={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                    selection=dict(scope=args.selection_scope, aggregation="equal mean of per-FPS means",
                                   ownership="unique frames within FPS; centre distance, then start"),
                    fps_sampling=dict(frame_steps=training.frame_steps, windows_per_epoch=epoch_windows,
                                      mixture="equal-FPS shuffled cycles; sampled RGB before MDD"),
                    test_usage="no test samples used for training or checkpoint selection",
                    input_rgb="stored JPEG at native size; [0,1] luminance; no ImageNet normalization",
                    mdd_first_frame="zero, no RGB predecessor outside the 32-frame window")
    (args.output / "config.json").write_text(json.dumps(settings, indent=2) + "\n")
    (args.output / "data_manifest.json").write_bytes(args.manifest.read_bytes())
    best = float("inf")
    for epoch in range(args.epochs):
        sampler.set_epoch(epoch)
        model.train()
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            loss = coordinate_loss(predict(model, batch, device), batch["uv"].to(device), batch["position_valid"].to(device))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            optimizer.step()
        report = evaluate(model, val_loader, device)
        score = selection_score(report, args.selection_scope)
        if dual_sha256(args.manifest) != identity:
            raise ValueError("Training manifest changed during the run")
        path = args.output / f"epoch-{epoch:03d}.pt"
        torch.save(dict(schema="mdd_coordinates.v2", model_config=asdict(config), state_dict=model.state_dict(),
                        optimizer=optimizer.state_dict(), epoch=epoch, manifest_sha256=identity,
                        torch_rng=torch.get_rng_state(), validation=report,
                        selection_scope=args.selection_scope, selection_error_px=score,
                        mdd_a=args.mdd_a, mdd_b=args.mdd_b, code=code), path)
        if score < best:
            best = score
            (args.output / "best.json").write_text(json.dumps(dict(epoch=epoch, checkpoint=path.name,
                checkpoint_sha256=dual_sha256(path), selection_scope=args.selection_scope,
                selection_error_px=score, **report), indent=2) + "\n")
        with (args.output / "metrics.jsonl").open("a") as stream:
            stream.write(json.dumps(dict(epoch=epoch, **report)) + "\n")
        print(json.dumps(dict(epoch=epoch, **report)), flush=True)


if __name__ == "__main__":
    main()
