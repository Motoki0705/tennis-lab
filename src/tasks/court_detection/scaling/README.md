# Court data scaling

This workflow trains on **synthetic images only**, with real images reserved for
video-disjoint validation and final testing. The default mixed-source training
and all `train_dinov3_size_ablation*` configurations retain their existing behavior.
Output paths follow [task output conventions](../../OUTPUTS.md).

## Ordered subsets and original validation

`data/source=synthetic_court_dataset_subset` selects explicit `manifest` and
`validation_manifest` paths under the DATA root. Both JSON files use
`schema: court_dataset_subset_v1`, `source_manifest` (relative to their parent,
without traversal), `source_manifest_sha256`, `selection_seed`, `split`
(`train` or `val`) and ordered `samples: [{scene_id, sample_id}]` with raw IDs.
Both files must reference the same complete scale publication and selection seed.
The full pool is verified by the generator before selecting records. Unknown or
duplicate IDs and train/validation trajectory overlap are errors. The original
pool remains unchanged; selected validation records are remapped only in memory.
Checkpoint provenance binds both selections, their order, training/validation
counts and the complete source publication. Resume/evaluation rejects drift.

To reproduce the original TennisCourtDetector validation population and order,
use `freeze_real_evaluation --policy original-val --version NAME` with the usual
root/output paths. Do not supply a seed or val fraction for this policy. Its
separate `court_real_original_val_evaluation_v1` schema exposes only validation;
it creates no test set and never includes original training images. The default
video-disjoint policy still requires an explicit seed and is unchanged.

`evaluate_ablation --expected-step N --split val --series NAME
--include-synthetic-val` evaluates synthetic validation and real validation in
separate passes, preserving each population's own order and metrics. Pose metrics
are reported only for synthetic data. The normal fit-time `val/loss` remains the
combined monitor, so it is not a domain-gap score; exact-update checkpoints and
the separate post-fit evaluations are authoritative. The image-count scaling
comparison command remains real-only. Experiments that vary synthetic validation
populations must make that difference explicit in their own comparison reports.

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

`train_data_scaling` selects `data/source=synthetic_court_dataset_version`.
Set `data.source.manifest` to an explicit `*.dataset.json` path under
`paths.data_root`. Set the root itself to an absolute path when needed; the
manifest value is role-relative, like other task inputs. The adapter calls
`scale.publication.load_dataset_version(..., verify_images=True)` before
model/worker startup. It verifies the frozen request, generation seed, recipe,
candidate geometry, exact accepted membership, QA ledgers, store/index hashes,
JPEG bytes and train-only split. D5 failures and geometry rejections never enter
the training Dataset. Version references resolve relative to the manifest's
parent; the adapter reads only its ordered scene/store inventory.

Scale stores have an explicit profile, stable-ID-derived sample indices and
`scale`/`qa` records. Their v3 KP14, visibility, camera/target and pose authority
pass unchanged to the shared online target pipeline. The generator's
[dataset-version contract](../../../synthetic_data_generation/dataset/court/README.md#dataset-version-consumer-contract)
owns publication semantics; no generation or renderer runs during training.

`data_version` and `generation_seed` in `data-condition.json` and checkpoint
hyperparameters come from the verified manifest. They cannot be supplied in
this source's configuration. Provenance also records manifest schema/hash,
request/plan/recipe hashes, method, actual accepted scene counts, store hashes
and the ordered training-ID hash. `run.seed` remains the independent training
seed. Resume and checkpoint evaluation revalidate the publication and refuse
input or registry drift.

For historical V3 stores, explicitly select
`data/source=synthetic_court_stores`. This accepts an **ordered, nonempty list** of
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

Only this historical source requires manually declared `data_version` and
`generation_seed`; it does not claim scale dataset-version validation. It
retains the existing strict V3 envelope and rejects scale stores. The source
discriminator selects exactly one contract, with no automatic detection or
fallback between manifests, historical stores and scene owners.

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
  data.source.manifest=court_scale/condition-v1/VERSION.dataset.json \
  mixed.sources.tennis_court_detector.registry=court_detection/evaluate/real-video-split-v1/s42/registry.json \
  training.step_schedule.max_steps=100 \
  training.step_schedule.validation_every_n_steps=20 \
  training.step_schedule.checkpoint_every_n_steps=20 \
  training.warmup_steps=10 run.seed=42 \
  run.output_dir=court_detection/train/data-scaling-condition-v1/s42
```

The update counts above are **illustrative**, not agreed production values.
`max_steps`, warmup, evaluation/checkpoint intervals, dataset-version manifest,
registry and run output are mandatory configuration inputs.
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
scheduled `update-*.ckpt` is retained, independently of
`training.checkpoint.save_top_k`. The separate `best-val-*.ckpt` callback monitors
`val/loss` (minimum) and uses that retention setting: `-1` retains every validation
checkpoint, `0` disables best-val saves, and a positive integer retains that many
lowest-loss checkpoints. Missing values, wrong types (including booleans), and
integers below `-1` fail configuration validation. The `train_data_scaling` preset
explicitly sets `save_top_k: -1` to retain all checkpoints for the main experiment.
With a publishing artifact store, finite top-k pruning also removes the evicted
files from the published store. `training.checkpoint.enabled=false` disables only
the shared epoch callback; the step callbacks above remain active. Their filenames,
monitor and mode are fixed. Best-val checkpoints must not replace the exact-update
checkpoint in the main comparison.

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
