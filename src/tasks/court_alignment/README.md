# Ground-UV Court Alignment (KP14)

This task learns court alignment from a single-channel ground-plane line-evidence
map.  The model predicts fourteen full-resolution keypoint heatmaps and a dense
two-channel centre-vote field.  It does not consume RGB images and it does not
silently call the previous template-search alignment system.

## Contract

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

## Training

The normal run uses the shared Lightning runner and writes test predictions in
the repository prediction-bundle format.

```bash
python -m src.tasks.court_alignment.scripts.train
python -m src.tasks.court_alignment.scripts.train run.output_dir=court_alignment/my_run
```

The one-step CPU configuration is intended for integration checks:

```bash
python -m src.tasks.court_alignment.scripts.train --config-name smoke
```

The four sigma experiments differ only in `data.sigma_px`.  Keep the seed,
split sizes, model, and training budget identical:

```bash
for sigma in 0.75 1.0 1.5 2.0; do
  python -m src.tasks.court_alignment.scripts.train \
    data.sigma_px=${sigma} \
    run.output_dir=court_alignment/sigma_${sigma}
done
```

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
python -m src.tasks.court_alignment.scripts.evaluate \
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
