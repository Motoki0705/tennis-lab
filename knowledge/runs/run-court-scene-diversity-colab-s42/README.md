# Fixed-budget Court scene diversity on Colab L4

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
The parent checks it and resumes this same execution agent every 30 minutes;
there is no child timer or short-interval monitoring loop.
