"""Queue-only BF16 GPU probe of augmentation and compiled freeze/unfreeze."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from src.tasks.ball_detection.data.coordinate_dataset import (
    CoordinateWindowDataset,
    collate_coordinate_windows,
)
from src.tasks.ball_detection.data.pose_windows import coordinate_loss
from src.tasks.ball_detection.models.mdd_pretrain import DeepMDDQueryDetector
from src.tasks.ball_detection.preprocessing import RGBToMDD
from src.tasks.ball_detection.training.coordinate_compilation import (
    compile_coordinate_model,
    coordinate_compile_scope,
)
from src.tasks.ball_detection.training.coordinate_images import (
    coordinate_batches,
    jpeg_decoder_contract,
)
from src.tasks.ball_detection.training.coordinate_runtime import CoordinateRuntime
from src.tasks.ball_detection.training.heatmap_pretraining.checkpoint import (
    pretraining_config,
    transfer_encoder,
)
from src.tasks.ball_detection.training.heatmap_pretraining.objective import image_input
from src.tasks.ball_detection.training.posttraining.augmentation import (
    AugmentationConfig,
    VideoAugmenter,
)
from src.tasks.ball_detection.training.posttraining.checkpoint import (
    completed_pretraining,
)
from src.tasks.ball_detection.training.posttraining.paths import resolver
from src.utils.checksum import dual_sha256
from src.utils.configuration import (
    BoundaryPathField,
    NonHydraPathBoundary,
    PathDirection,
    PathKind,
    PathRole,
)

PATH_BOUNDARY = NonHydraPathBoundary(name="ball_detection.probe_mdd_posttraining", fields=(
    BoundaryPathField("manifest", PathRole.ARTIFACT, PathDirection.INPUT, PathKind.FILE, must_exist=True),
    BoundaryPathField("pretraining_run", PathRole.ARTIFACT, PathDirection.INPUT, PathKind.DIRECTORY, must_exist=True),
    BoundaryPathField("augmentation_config", PathRole.PROJECT, PathDirection.INPUT, PathKind.FILE, must_exist=True),
    BoundaryPathField("output", PathRole.OUTPUT, PathDirection.OUTPUT, PathKind.DIRECTORY),
))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("manifest", "pretraining-run", "augmentation-config", "output"):
        p.add_argument(f"--{name}", type=Path, required=True)
    p.add_argument("--image-prefetch", action="store_true")
    args = p.parse_args()
    names = ("manifest", "pretraining_run", "augmentation_config", "output")
    paths = PATH_BOUNDARY.validate({k: getattr(args, k) for k in names}, resolver=resolver(args.augmentation_config.parent,
        (args.manifest, args.pretraining_run), (args.output,), args.output.parent))
    for key in names:
        setattr(args, key, paths.declared(key).path)
    args.output.mkdir(parents=True, exist_ok=False)
    path, saved = completed_pretraining(args.pretraining_run, args.manifest)
    if saved["recipe"]["image_decode"] != jpeg_decoder_contract("nvjpeg"):
        raise ValueError("Probe decoder differs from pretraining")
    data = CoordinateWindowDataset(args.manifest, split="train", requires_pose=False, jpeg_decoder="nvjpeg")
    indices = [next(i for i, (_, w) in enumerate(data.windows) if w.frame_step == step) for step in (1, 2, 4)]
    # Verify/read the actual clips on CPU before CUDA startup.
    batches = [collate_coordinate_windows([data[i]]) for i in indices]
    runtime = CoordinateRuntime(precision="bf16", compile_mode="default", jpeg_decoder="nvjpeg")
    device = torch.device("cuda")
    runtime.configure(device)
    torch.manual_seed(42)
    model = DeepMDDQueryDetector(pretraining_config(saved))
    model.mdd = RGBToMDD.from_contract(saved["recipe"]["input_contract"])
    transfer_encoder(path, model)
    model.to(device)
    compile_coordinate_model(model, mode="default")
    optimizer = torch.optim.AdamW(model.parameters(), lr=.0001)
    augmenter = VideoAugmenter(AugmentationConfig.load(args.augmentation_config), 42)
    phases = []
    for phase in (0, 1):
        model.freeze_encoder(phase == 0)
        model.train()
        before = model.encoder.temporal[-1].conv.weight.detach().clone()
        losses, gradients = [], []
        begin = time.perf_counter()
        sequence = [batches[i % len(batches)] for i in range(16)]
        for batch in coordinate_batches(sequence, device, prefetch=args.image_prefetch):
            rgb, target, _ = augmenter(image_input(batch, device), batch, epoch=phase, profile="combined")
            optimizer.zero_grad(set_to_none=True)
            with coordinate_compile_scope(model):
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    output = model(rgb, batch["timestamps"].to(device))
                    loss = coordinate_loss(output, target["uv"].to(device), target["position_valid"].to(device))
                loss.backward()
            grad = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            if phase == 0 and any(p.grad is not None for p in model.encoder.parameters()):
                raise ValueError("Frozen CNN unexpectedly received gradients")
            if phase == 1 and (model.encoder.temporal[-1].conv.weight.grad is None or
                               not bool(model.encoder.temporal[-1].conv.weight.grad.abs().sum() > 0)):
                raise ValueError("Compiled unfreeze did not restore CNN gradients")
            optimizer.step()
            losses.append(float(loss.detach()))
            gradients.append(float(grad))
        changed = not torch.equal(before, model.encoder.temporal[-1].conv.weight)
        if changed != bool(phase):
            raise ValueError("CNN update does not match frozen/joint phase")
        phases.append(dict(frozen=not bool(phase), cnn_changed=changed, loss=losses, grad_norm=gradients,
                           seconds=time.perf_counter() - begin))
    report = dict(phase="posttraining_gpu_probe", precision="bf16", image_prefetch=args.image_prefetch, phases=phases,
                  peak_allocated_gib=torch.cuda.max_memory_allocated() / 2**30,
                  checkpoint=str(path), checkpoint_sha256=dual_sha256(path), steps=32,
                  manifest_sha256=dual_sha256(args.manifest), augmentation_sha256=dual_sha256(args.augmentation_config))
    (args.output / "PROBE_COMPLETED.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
