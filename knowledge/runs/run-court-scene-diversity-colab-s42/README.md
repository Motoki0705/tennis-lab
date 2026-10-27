# Fixed-budget Court scene diversity on Colab L4

## Completed results (2026-10-27)

The four individual run records are
[scene1](../../nodes/court_detection/000051-run-court-scene-diversity-colab-scene-1-s42.md),
[scene2](../../nodes/court_detection/000052-run-court-scene-diversity-colab-scene-2-s42.md),
[scene3](../../nodes/court_detection/000053-run-court-scene-diversity-colab-scene-3-s42.md),
and [scene4](../../nodes/court_detection/000054-run-court-scene-diversity-colab-scene-4-s42.md).
The [comparison group](../../nodes/court_detection/000055-group-court-scene-diversity-colab-s42.md)
records Japanese findings and limitations. Each run retains its own resolved
configuration, complete evaluation metadata/ordered IDs, compact metrics, payload
verification and curves. Shared execution code is kept only in this bundle.

`comparison.{json,csv,png}` was produced by the original preparation `report.py`
from the four final evaluations. `final-acceptance.json`, `terminal-status.json`
and `continuation-completion.json` document the successful gates and terminal
managed command. `parent-verification.json` is the parent's independent review of
the retrieved results, including an identical comparison recomputation, rather
than another training/evaluation trial. `execution-receipt.json` identifies exact
execution commits, sessions, environment and durable Drive locations. Large
images/checkpoints and raw TensorBoard files remain in Drive, outside Git.

Scene2 resumed its own saved update26528 optimizer/scheduler/loop state and
finished at update33160 in `version_1`. The original job's terminal status remains
unknown. The last periodic observation was27112, while the final retained
TensorBoard contains update27150, so at least622 attempted updates were discarded.
The non-resumable mid-epoch DataLoader does not guarantee exact worker RNG/sampler
order continuation. The cause of the two earlier VM disappearances is unknown.
At final verification the owned session was already absent; official doctor
reported zero active assignments and usage0.00/hr (`gpu-release.json`). No stop
request was sent to a session that no longer existed.

### Fit-time curves

`kg_curves.py` was run for all four run IDs using the verified final TensorBoard
sources. The final `curves.json`/`curves.png` are then exported by the script below,
which clips scene2's old attempt at26528, excludes its discarded tail and adds a
visible resume boundary. The discarded old points remain separately labelled in
`curves.json`; they are never mixed into final-model history. Event steps are
zero-based, so optimizer updates are `event.step + 1`. Each history contains ten
validation observations at3316..33160. Validation curves are fit-time aggregate
synthetic500 + real2211; final evaluation is domain-separated in each run's
`evaluation.json`. All KP pixel units are post-preprocessing model coordinates
with longside512, rather than original-image coordinates.

The verified event source filenames, SHA-256 and Drive file paths are recorded in
`curve-sources.json`. Download those immutable files into the documented relative
layout under a local transfer directory, then run:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  knowledge/runs/run-court-scene-diversity-colab-s42/export_curves.py \
  --source-root /path/to/verified/colab-transfer
```

The original experiment protocol and execution commands follow.

The four immutable source configurations and input plan are in
[the preparation bundle](../run-court-scene-diversity-preparation-s42/README.md).
`colab_campaign.py` relocates their seven path roots to `/content/tennis-lab`,
uses separate `scene-N-s42` output roots, enables rclone persistence, and changes
checkpoint retention to `save_top_k=-1`. A field-by-field audit rejects any other
change. Model, loss, optimizer, seed, image subsets, batch size, preprocessing and
the 33,160-update final selection remain pinned to the preparation bundle.

The entire generated dataset subtree is transferred unchanged to
`gdrive:tennis_lab/data/court_scene_diversity_6000_v1`. Existing real images and
DINOv3-S weights are verified and reused at their regular `data/` and `ckpt/`
paths. The original 2,211-image real validation registry is transferred to its
`outputs/court_detection/campaign/scene-diversity-6000-v1/evaluation/registry.json`
path. No original assets, selected subsets, QA records or historical configs are
rewritten. Credential contents are never recorded in this bundle.

Use [tennis-colab](../../../.agents/skills/tennis-colab/SKILL.md) for session,
command and persistence management and
[tennis-drive](../../../.agents/skills/tennis-drive/SKILL.md) for input transfers.
Each managed Colab job runs in the foreground. Set
`PYTHONDONTWRITEBYTECODE=1` and `TORCHINDUCTOR_COMPILE_THREADS=1` explicitly.

After checkout of the execution commit and `uv sync --locked --python 3.11`,
initialize only `third_party/dinov3`, then download the verified Drive inputs.
Run the following inside the owned L4 checkout:

```bash
SCRIPT=knowledge/runs/run-court-scene-diversity-colab-s42/colab_campaign.py
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python "$SCRIPT" prepare
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python "$SCRIPT" preflight
```

Preflight checks the original weight and validation order, the full synthetic
publication through its existing validator, all eight subset hashes and counts,
and fixed train/val trajectory separation. Its config/evidence directory is
`outputs/court_detection/campaign/scene-diversity-6000-colab-v1`; persist it with
the Colab worker.

Before the main runs, invoke `probe` as a separate managed command, declaring
`--runner-output outputs/court_detection/train/scene-diversity-6000-s384-colab-probe/scene-1-s42`
and persisting the config/evidence directory. The probe uses the same model,
data, batch and compilation with three optimizer updates, validation/save at
update three, and zero warmup. These probe-only schedule changes are recorded.
It checks finite losses and gradients and saves a real Lightning checkpoint
through the normal publishing checkpoint I/O. Confirm the managed job's terminal
success and Drive save receipt before launching the first main condition.

```bash
PYTHONDONTWRITEBYTECODE=1 TORCHINDUCTOR_COMPILE_THREADS=1 .venv/bin/python "$SCRIPT" train --condition 1
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python "$SCRIPT" evaluate --condition 1
```

Run each of conditions 1–4 as an independent training job, declaring its matching
`--runner-output outputs/court_detection/train/scene-diversity-6000-s384-colab/scene-N-s42`.
Evaluation is a separate job with
`--persist outputs/court_detection/evaluate/scene-diversity-6000-s384-colab/scene-N-s42`.
It selects `update-00033160.ckpt` and reports synthetic validation and original
real validation separately. Confirm `returncode=0` and `drive_saved_at` after
every job. Only then advance to the next phase or condition.

For a verified interrupted run, download its complete saved output tree and pass
`train --condition N --resume /content/tennis-lab/outputs/.../update-XXXXXXXX.ckpt`.
The checkpoint must belong to that same condition/run. Lightning restores the
optimizer and scheduler; record the restart and any RNG/sampler continuation
limits. A completed condition is rejected by `train` and must be evaluated.

The existing preparation `report.py` compares the four completed evaluation JSONs
and writes CSV/JSON/PNG. Persist the comparison output with the worker. Keep
large checkpoints and full images on Drive, collect only small research evidence,
and stop the owned session after all jobs and saves are verified.

Local progress lives in
`outputs/court_detection/campaign/scene-diversity-6000-v1/colab-status.json`.
The parent checks it and resumes this same execution agent every 300 minutes;
there is no child timer or short-interval monitoring loop.

`continue_campaign.py` connects the remaining approved stages on one managed
Notebook command. It verifies scene 2's actual final checkpoint, permits only a
same-condition `run.resume` configuration difference, evaluates the two validation
domains, checks their exact counts and ordered IDs, and verifies Drive content
before starting each next condition. Scenes 3 and 4 start independently from the
original pretrained weight and seed. Their final checkpoint verification,
separate-domain evaluation and four-condition comparison follow serially; any
failed command or gate stops the sequence. The model, training and evaluation
source trees, lockfile, preparation bundle and original execution helper retain
their original Git object identities. This continuation changes orchestration
and evidence checks only. Declare both fresh run outputs with `--runner-output`
and persist the evaluation/comparison parent directories and audit directory.
