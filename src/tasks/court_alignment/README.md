# Ground-UV Court Alignment

This task learns court alignment from a single-channel ground-plane line-evidence
map. It provides two explicit backends: the original KP14 CNN predicts dense
keypoint heatmaps and centre votes, while the DINO backend predicts a set of
oriented court boxes. Neither backend consumes camera RGB or silently calls the
previous template-search alignment system.

## KP14 CNN contract

The input is `float32 [1, H, W]` in `[0, 1]`.  The initial procedural baseline
rasterizes court lines as one and background as zero.  A sample may contain more
than one court.  Output channel `c` contains every visible peak for semantic
keypoint `c`; overlapping Gaussian targets are combined with `max`, never by
addition.

Channels follow the shared immutable `GROUND_COURT_KP_NAMES` order:

| Channel | Semantic keypoint |
| ---: | --- |
| 0 | `far_doubles_left` |
| 1 | `far_doubles_right` |
| 2 | `near_doubles_left` |
| 3 | `near_doubles_right` |
| 4 | `far_singles_left` |
| 5 | `near_singles_left` |
| 6 | `far_singles_right` |
| 7 | `near_singles_right` |
| 8 | `far_service_left` |
| 9 | `far_service_right` |
| 10 | `near_service_left` |
| 11 | `near_service_right` |
| 12 | `far_service_t` |
| 13 | `near_service_t` |

Here `near`/`far` and `left`/`right` are the canonical ground-court axes used by
the procedural generator, not a camera-relative relabelling.  Samples retain
the per-instance `[N, 14, 2]` keypoints and visibility masks so this convention
can be checked at the dataset boundary.

Because an unlabelled court-line raster is unchanged by a 180-degree rotation,
the procedural rotation interval may span at most `pi` radians. With the
default seam margin `rotation_seam_margin_rad=0.05`, the configured interval is
`[0.05, pi - 0.05]`; using a full `2*pi` interval would give identical inputs
conflicting KP-channel targets.

The main model output is:

```text
heatmap_logits: [B, 14, H, W]
center_votes:   [B,  2, H, W]
```

The CNN is a four-down U-Net (full → 1/2 → 1/4 → 1/8 → 1/16) with
GroupNorm/SiLU blocks and full-resolution skip reconstruction.  Its bridge
receptive field is 221 input pixels, covering the maximum corner-to-centre
distance on the default 256×256 canvas.  Odd and rectangular input sizes are
upsampled to their corresponding skip shape, so the output always preserves
`H×W`.  The fourteen heatmap output biases start at a configurable prior of
`p=0.1` (logit ≈ -2.197); the two vote biases start at zero.

At every visible keypoint pixel the auxiliary vector points to the centre of
the same court.  During decoding, channel-local peaks vote for centres and are
clustered there.  This association is necessary: confidence rank in one KP
channel is not an instance identifier and cannot be paired with the same rank
in another channel.

## DINO oriented-box contract

The DINO preset uses the released four-scale Swin-L detector checkpoint. The
untouched 91-class COCO model is constructed and strict-loaded first; only then
is its classifier replaced by a one-class court head and LoRA is installed.
The pretrained backbone and transformer weights remain frozen. LoRA, the DINO
class/AABB heads, a court-specific head, and the optional input adapter are
trainable.

Each query returns the standard normalized DINO AABB plus three raw court
parameters:

```text
pred_logits:      [B, Q, 1]
pred_boxes:       [B, Q, 4]  # normalized AABB cx, cy, width, height
pred_court_boxes: [B, Q, 3]  # long-side logit, raw cos(2θ), raw sin(2θ)
```

The long-side axis is unoriented: θ and θ+π are the same rectangle. The short
side is fixed by the ITF doubles-court ratio `10.97 / 23.77`. This preserves a
continuous diagonal orientation that cannot be recovered from an axis-aligned
bbox alone. Multiple courts are separate DINO queries; decoding performs score
thresholding/top-k only and deliberately applies neither NMS nor peak
clustering.

Hungarian assignment combines focal classification, AABB L1/GIoU, logarithmic
long-side scale, and double-angle axis costs. Decoder auxiliary outputs, DINO
denoising outputs, and two-stage intermediate/encoder detections are supervised.
Targets are emitted only for courts whose four doubles-court corners are visible;
the DINO procedural preset constrains sampling so every generated court satisfies
that contract.

All DINO inputs use the released detector's evaluation resize: short side 800,
long side at most 1333. The current square procedural heatmap is therefore
800×800. The `384` in `swin_L_384_22k` identifies backbone pretraining and is not
the detector input size.

The input-mode ablation is applied before shared ImageNet normalization:

- `repeat_rgb`: `[x, x, x]`;
- `learnable_1x1`: trainable `Conv2d(1,3,1)`, initialized exactly as replication;
- `red_only`: `[x, 0, 0]`.

## Training

The normal run uses the shared Lightning runner and writes test predictions in
the repository prediction-bundle format.

```bash
.venv/bin/python -m src.tasks.court_alignment.scripts.train
.venv/bin/python -m src.tasks.court_alignment.scripts.train run.output_dir=court_alignment/my_run
```

The one-step CPU configuration is intended for integration checks:

```bash
.venv/bin/python -m src.tasks.court_alignment.scripts.train --config-name smoke
```

The four sigma experiments differ only in `data.sigma_px`.  Keep the seed,
split sizes, model, and training budget identical:

```bash
for sigma in 0.75 1.0 1.5 2.0; do
  .venv/bin/python -m src.tasks.court_alignment.scripts.train \
    data.sigma_px=${sigma} \
    run.output_dir=court_alignment/sigma_${sigma}
done
```

For DINO, initialize/build the pinned source as documented in
`src/submodules/README.md` and place the released checkpoint at
`ckpt/dino/checkpoint0029_4scale_swin.pth` (the multi-gigabyte file is not git
tracked). The runtime dependencies `addict` and `yapf` are declared in
`pyproject.toml`. Then select one input mode:

```bash
.venv/bin/python -m src.tasks.court_alignment.scripts.train \
  --config-name train_dino \
  model.device=cuda:0 \
  model.input_mode=repeat_rgb
```

Official DINO's denoising implementation currently selects `cuda:0` directly,
so this preset is single-GPU `cuda:0`; submit every local run through the shared
training queue. Keep all settings except `model.input_mode` and the output path
identical when comparing the three modes.

Every ablation uses the fixed `training.trainer.max_epochs=50` and
`training.steps_per_epoch=256` budget (the runner checks that steps per epoch
matches the configured training split and batch size).  Early stopping is
disabled by default.  Checkpoint selection is identical across runs: the top
two checkpoints by minimum `val/loss` are retained together with `last`, under
the run's `checkpoints/` directory.  The `last` checkpoint is a resume
artifact. When `test_after_fit` is enabled, the runner loads the single best
`val/loss` checkpoint (`ckpt_path=best`) before writing the test prediction
bundle; sigma comparisons therefore use the same selection rule in every run.

Local GPU runs must submit these commands through the shared training queue;
do not launch them directly.  Queue runs save the standard files below
`$TENNIS_REPRO_DIR/predictions/`:

- `pred_test.npz`: sample IDs, decoded top-K predicted peaks and their
  center-votes, plus predicted/GT instances and visibility masks;
- `metrics.json`: headline test metrics;
- `diagnostic_metrics.json`: secondary loss and association diagnostics.

The prediction bundle intentionally stores decoded outputs only.  Dense KP
logits are not saved, so changing the peak threshold, NMS kernel, or top-K
limit requires running `evaluate` again from the checkpoint; those decoder
settings cannot be retroactively applied to an old bundle.  Within the saved
top-K/threshold range, stored scores and coordinates remain available for
downstream analysis.

The shared runner also records `output_dir.txt` for checkpoint/reproduction
linkage when checkpointing is enabled.

To evaluate a checkpoint without fitting:

```bash
.venv/bin/python -m src.tasks.court_alignment.scripts.evaluate \
  evaluation.checkpoint_path=/absolute/path/to/model.ckpt
```

## Augmentation extension point

`data.augmentations` is an ordered typed list.  The baseline is explicit
identity:

```yaml
augmentations:
  - name: identity
    params: {}
```

Dataset construction validates the type and dispatches every item through the
task-local augmentation registry.  New corruptions should be added as a new
typed implementation and config entry, while leaving target geometry and
instance metadata unchanged or transforming them explicitly.  Planned
sim-to-real studies include line dropout, blur/threshold variation, width
variation, false line segments, partial crops, and small projection warps.
They are intentionally absent from the clean sigma baseline.

## Metrics and limitations

Headline metrics measure keypoint localization/recall and instance recovery;
diagnostic metrics retain heatmap/centre-vote loss components.  Sigma is in
output pixels, so changing output resolution changes its physical meaning.

The prototype assumes all generated court centres are inside the raster and
caps the number of courts per sample in data configuration.  Very close courts
can merge into one local maximum or one centre-vote cluster.  Small sigma makes
peaks sharper but also makes supervision sparse; it cannot remove pixel
quantization error.  Real detector heatmaps have missing lines, soft
probabilities, projection error, and distractors, so clean-synthetic accuracy
must not be presented as real-data alignment accuracy.  The sigma ablation is
the fixed parent experiment for subsequent one-factor augmentation studies.

## Schema summary

The model input is `image: float32 [B,1,H,W]` and the dense output is
`heatmap_logits: float [B,14,H,W]` plus `center_votes: float [B,2,H,W]`.
Each heatmap channel can contain multiple local peaks.  The decoder emits
`keypoints_px/scores/valid/center_votes_px: [B,14,K,(2)]`, then groups those
peaks into variable-length instances with `keypoints_px [N,14,2]`,
`valid [N,14]`, `scores [N,14]`, and `centers_px [N,2]`.  Ground-truth samples
carry padded `keypoints [B,N,14,2]`, `visibility [B,N,14]`, `centers [B,N,2]`,
and `num_courts [B]`.  Center-vote vectors use `(dx,dy)` pixels from the
keypoint pixel to its court centre.

For DINO the same batch geometry is converted at the Lightning boundary into a
variable-length target list with `labels [N]`, normalized AABB `boxes [N,4]`,
and `court_boxes [N,5] = (cx, cy, long_side, cos(2θ), sin(2θ))`. Decoded query
artifacts retain scores, query indices, AABBs, centres, long/short sides, axial
vectors, four corners, rotation modulo π, and scale in pixels per metre.
