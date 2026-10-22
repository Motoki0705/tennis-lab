# Court data scaling

This workflow trains on **synthetic images only**, with real images reserved for
video-disjoint validation and final testing. The default mixed-source training
and all `train_dinov3_size_ablation*` configurations retain their existing behavior.
Output paths follow [task output conventions](../../OUTPUTS.md).

## Freeze real evaluation once

Use the complete original TennisCourtDetector JPEG store, including its old
`train` and `val`. Nothing is written into that store. The known degenerate
annotation `QszoUKyCOHo_600` must occur exactly once and is excluded.

```bash
.venv/bin/python -m src.tasks.court_detection.scripts.freeze_real_evaluation \
  --root /absolute/data/court_detection/tennis_court_detector-v1 \
  --output "$PWD/outputs/court_detection/evaluate/real-video-split-v1/s42" \
  --version real-video-v1 --seed 42
```

`registry.json` contains ordered sample IDs, a version, seed, canonical SHA-256,
source descriptor/index hashes, image/video counts and old-split image counts.
An existing destination is refused. Copy the complete registry to durable
storage alongside the experiment inputs; subsequent consumers verify it against
the original store and recompute the assignment, rather than trusting its counts.

The final `_<frame>` suffix is removed to obtain the video ID. Sorted video IDs
are shuffled with an isolated Python RNG and divided 50:50 by **video count**
(rounding to the nearest integer). This balances independent evaluation units;
frame counts may differ. Equal halves give both monitoring and final reporting
substantial coverage without training on any real frame. `--val-fraction` is
explicitly configurable when creating a new registry version; it does not modify
an existing registry. Train/val provenance is retained even when one original
video spans both old splits. Validation and test have zero overlapping videos.
Validation controls monitoring/checkpoint selection; test is invoked separately
only for final reporting. The legacy `split_mapping` source is unchanged.

## Explicit synthetic publications

`data/source=synthetic_court_stores` accepts an **ordered, nonempty list** of
store roots under `paths.data_root`. Each root is an existing packed
`sample_store` publication (JPEG shards, sparse index and V3 target-court
geometry). It never searches `scenes/<scene>/datasets/court` implicitly. All
accepted records from every specified store become training records in memory,
including any historical validation/test records in a smoke-test input.
Stored labels and split fields are not rewritten. No synthetic evaluation split
is constructed. Sample IDs (`scene_id:sample_id`) must be unique across roots;
overlapping store lists fail instead of deduplicating samples.

Missing publications, incomplete/unsupported stores, malformed geometry,
corrupt indices/shards and duplicate IDs fail explicitly. Pose authority is
preflighted before model/worker startup. Accepted image counts come from the
actual source inventory, without a requested-count quota or replacement draws.
Epoch-end sampler padding may repeat records to fill a batch; it does not alter
the measured unique accepted image count.

`data_version` identifies the selected condition and `generation_seed` records
the declared generation experiment seed. `run.seed` is the independent training
seed. Descriptor/index hashes and the ordered training-ID hash pin the actual
store contents in `data-condition.json` and checkpoint hyperparameters.
Evaluation refuses input or registry drift relative to that checkpoint.

The generator owns the future condition-manifest schema. Its adapter should
validate the manifest and its referenced hashes/completion state, resolve the
exact ordered store list, then populate `store_roots`, `data_version` and
`generation_seed` here. There is no guessed manifest schema or implicit owner
fallback in this consumer.

## Train with optimizer-update counts

Enqueue the following command through the shared
[training queue](../../../../.agents/skills/training-queue/SKILL.md); run it from
the checkout containing the intended code. For a worktree, explicitly set
`TRAINING_QUEUE_DIR` to the main repository's `.training_queue` when enqueuing.
Use absolute roots for remote/local workloads.

```bash
.venv/bin/python -m src.tasks.court_detection.scripts.train \
  --config-name train_data_scaling \
  paths.data_root=/absolute/data paths.artifact_root=/absolute/outputs \
  'data.source.store_roots=[court_scale/condition-v1/B00,court_scale/condition-v1/B01]' \
  data.source.data_version=condition-v1 data.source.generation_seed=17 \
  mixed.sources.tennis_court_detector.registry=court_detection/evaluate/real-video-split-v1/s42/registry.json \
  training.step_schedule.max_steps=100 \
  training.step_schedule.validation_every_n_steps=20 \
  training.step_schedule.checkpoint_every_n_steps=20 \
  training.warmup_steps=10 run.seed=42 \
  run.output_dir=court_detection/train/data-scaling-condition-v1/s42
```

The update counts above are **illustrative**, not agreed production values.
`max_steps`, warmup, evaluation/checkpoint intervals, store list, version,
generation seed, registry and run output are mandatory configuration inputs.
The pilot must fix production counts before the main experiment.

The preset uses frozen DINOv3 ViT-B/16 without LoRA; Transformer width 768,
8 layers, 12 heads and FFN 2048; DPT 512 channels; residual dense heads with
hidden width 256 and depth 2; isotropic long-side 512 preprocessing. All four
dense heads and pose train. Batch **8 synthetic / 0 real** retains the previous
size-ablation total of 8 and its optimizer-update batch size. Any smaller smoke
batch must be reported as such. Validation uses the same explicit batch size and
stable registry order, since padding and DINO attention can depend on batch
composition. Compare results with matching preprocessing and evaluation batches.

The cosine learning-rate schedule uses `max_steps` independently of source
length. Warmup uses optimizer updates. Gradient accumulation must be 1. The
Trainer sets `max_epochs=-1`, `check_val_every_n_epoch=null` and an integer
`val_check_interval`, so validation crosses epoch boundaries at fixed updates.
The inherited epoch fields are required to stay at the inert value 1; epoch
warmup, a `steps_per_epoch` override, epoch checkpoint/qualitative callbacks,
early stopping, GAN and automatic post-fit test are rejected in this mode.
Resume requires identical recorded data, model, loss and training/schedule settings.
A longer schedule is a new independent run. Both intervals must divide `max_steps`, ensuring a final validation and exact
checkpoint. `update-00000100.ckpt` means exactly 100 optimizer updates. Every
scheduled checkpoint is retained. `best-val-*.ckpt` separately monitors
`val/loss` and must not replace the exact-update checkpoint in the main comparison.

## Evaluate and compare

Reuse the existing evaluator with explicit update, real split and curve-series
arguments. Checkpoint architecture/targets and strict state restoration remain
authoritative. Each invocation writes a new evaluation directory; val/test are
never combined into one score. Pose is trained but absent from these metrics.

```bash
.venv/bin/python -m src.tasks.court_detection.scripts.evaluate_ablation \
  --checkpoint /absolute/outputs/court_detection/train/data-scaling-condition-v1/s42/logs/version_0/checkpoints/update-00000100.ckpt \
  --expected-step 100 --split val --series A1 \
  --output "$PWD/outputs/court_detection/evaluate/data-scaling-condition-v1-val/s42"
```

Use a separate `--split test` invocation and output only at final reporting.
`--data-root`, `--checkpoint-root`, `--external-asset-root`, `--artifact-root` can relocate the
checkpoint's input roots; content and registry hashes must still match. The
checkpoint records its registry path under the ARTIFACT root; use
`--artifact-root` when moving a run and its registry together.

```bash
.venv/bin/python -m src.tasks.court_detection.scripts.compare_data_scaling \
  --evaluations /absolute/condition-a-val/evaluation.json /absolute/condition-b-val/evaluation.json \
  --output "$PWD/outputs/court_detection/report/data-scaling-A1/comparison"
```

The output includes Markdown/CSV per-run tables, a JSON receipt with evaluation
paths/hashes and seed aggregates, and `scaling.png`. The x-axis is logarithmic
and uses **measured accepted images**. Curves separate series, real split and
scene count; markers aggregate seeds at each observed image count (mean and
sample standard deviation; one run has zero deviation). This avoids averaging
scene-count comparisons into one point. The generator/pilot owns the meaning of
series labels such as A1/A2/B. A shared baseline may be evaluated with both A1
and A2 labels. Comparison rejects different optimizer updates, model/loss,
schedule, preprocessing, batch size, registry, sample order or target schema,
and rejects duplicate condition/seed results. It reports KP mean/median pixel
error, SEG mIoU, semantic LINE mIoU and LINE Dice.

To keep results reachable from knowledge run/group nodes, retain each run's
`data-condition.json`, exact evaluation receipt and comparison receipt as run
artifacts and link the report directory from the group. Register actual
experiment runs using the [knowledge workflow](../../../../.agents/skills/knowledge-control/SKILL.md).
This module does not create experiment claims or publish GitHub/Drive artifacts.
