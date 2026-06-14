# dino_ssl — Tennis-domain DINOv3 fine-tuning with LoRA

Self-supervised fine-tuning of the vendored DINOv3 ViT backbone on web-derived
tennis imagery, so that downstream [`court_detection`](../court_detection) and
[`ball_detection`](../ball_detection) can build on tennis-adapted features.

Training reuses DINOv3's own self-distillation strategy — the DINO class-token
loss, the iBOT masked-patch loss, and the KoLeo regulariser, with an EMA teacher
— while keeping the run lightweight via **LoRA** adapters. The pretrained weights
stay frozen; only the LoRA adapters and the projection heads are learned, which
minimises catastrophic forgetting of DINOv3's general visual capability.

> The DINOv3 backbone, losses, and heads are loaded from `third_party/dinov3`
> and remain subject to the DINOv3 License Agreement.

## Pipeline

```
generate_dataset/  →  data/        →  models/              →  training/
collect web images    multi-crop      DINOv3 ViT + LoRA       DINO + iBOT + KoLeo
+ meta.json manifest   SSL dataset    student / EMA teacher   self-distillation
```

### 1. Collect a tennis image dataset

```bash
.venv/bin/python -m src.tasks.dino_ssl.scripts.collect
```

`configs/collector/tennis_sample.yaml` defines the sources. Three source types
are supported and degrade gracefully (a missing path / unreachable URL is logged
and skipped):

- `video_frames` — sample frames from a local **or remote** video. Web URLs
  (e.g. YouTube) are resolved to a direct stream via `yt-dlp`.
- `image_dir` — ingest images from a local directory tree.
- `image_urls` — download images from a list of URLs.

The default config is reproducible offline: it samples frames from
`data/samples/tennis_clip.mp4` and ingests `data/court/images`, and tops up with
deterministic synthetic images if real sources fall short of `min_images`. To
scale up domain coverage, add real web videos / image URLs to `sources`.

Output: `data/dino_ssl/<name>/images/*.jpg` plus a `meta.json` manifest.

### 2. Fine-tune with LoRA self-distillation

```bash
.venv/bin/python -m src.tasks.dino_ssl.scripts.train
# quick smoke run:
.venv/bin/python -m src.tasks.dino_ssl.scripts.train \
    model.dino.out_dim=2048 model.ibot.out_dim=2048 \
    data.augmentation.global_size=112 training.trainer.max_epochs=1 \
    +training.trainer.limit_train_batches=4
```

Logs `train/loss`, `train/dino`, `train/ibot`, `train/koleo`, plus the teacher
temperature / EMA-momentum schedules, to TensorBoard.

### 3. Export the backbone for downstream tasks

```bash
.venv/bin/python -m src.tasks.dino_ssl.scripts.export_backbone \
    --checkpoint outputs/dino_ssl/<run>/logs/version_0/checkpoints/last.ckpt \
    --config    outputs/dino_ssl/<run>/config.yaml \
    --output    outputs/dino_ssl/exported/dinov3_vitb16_tennis.pth
```

This merges the LoRA adapters into the base ViT and writes a `{"model": ...}`
checkpoint that loads cleanly into a fresh DINOv3 ViT-B (`strict=True`). Point a
downstream DINOv3 model's `checkpoint_path` at this file to use the
tennis-adapted backbone.

## Configuration highlights

| Key | Meaning |
|---|---|
| `model.lora.{r,alpha,target_modules}` | LoRA rank / scaling / which ViT linears to adapt (`qkv`, `proj`, `fc1`, `fc2`, and the patch-embed conv) |
| `model.dino.out_dim` / `model.ibot.out_dim` | prototype dimensions of the DINO / iBOT heads |
| `model.ibot.enabled` | toggle the masked-patch objective |
| `training.loss.{dino,ibot,koleo}_weight` | loss term weights |
| `training.schedule.*` | teacher-temperature warmup and EMA-momentum schedule |
| `data.augmentation.*` | multi-crop scales/sizes and iBOT mask ratio |

## Limitations of the current environment

- **Backbone scale.** Only the `dinov3_vitb16` checkpoint is populated; the
  `dinov3_vitl16` checkpoint file is present but empty, so ViT-L SSL is not yet
  runnable here.
- **Compute.** Self-distillation holds a student **and** an EMA teacher (~2× a
  ViT-B) in memory. The defaults (224px globals, 6 local crops, 65 536-dim heads)
  target a single GPU; reduce crop sizes / head dims / `local_crops_number` to
  fit smaller GPUs. The upstream DINOv3 trainer's FSDP/multi-node path is **not**
  wired in — this task runs single-process via PyTorch Lightning.
- **Data scale.** SSL benefits from large, diverse corpora. The shipped sample
  set is intentionally tiny (for a functioning, reproducible pipeline); real
  domain adaptation needs the web sources scaled up substantially.
- **Teacher centering.** The Sinkhorn-Knopp teacher is available in DINOv3 but
  this task uses softmax-centering for both DINO and iBOT, which is simpler and
  single-process friendly.

## Tests

`tests/tasks/test_dino_ssl_contracts.py` verifies the acceptance conditions
offline (no network, no 342 MB checkpoint): the collector writes a manifest, the
LoRA adapters receive gradients, the EMA teacher tracks the student, and the
combined loss decreases when overfitting a fixed batch.
