# Checkpoint persistence and new-VM resume

This is a short continuity experiment, not an accuracy comparison or a training
budget selection. Findings and completion status belong to the
[knowledge node](../../nodes/ball_detection/000067-run-ball-colab-checkpoint-resume-20261024.md).
The driver is `resume_probe.py`; generic storage belongs to
[`RcloneSnapshotStore`](../../../src/utils/artifact_store/README.md).

Both VMs use `/content/tennis-lab`. Retrieve and verify the complete immutable
`ball-mix1800-720p-all-frame-train-val-v1` distribution at
`data/ball_detection/ball-mix1800-720p-all-frame-train-val-v1` with the preceding
distribution run's `colab_verify.py`. Use the exact Git bundle and the locked
Python 3.11 environment; preserve their receipts. Do not modify other sessions.

The input root for this proof is
`/content/tennis-lab/outputs/ball_detection/precompute/temporal_heatmap/resume-proof-20261024`.
It contains fresh T=32 manifests against the full distribution, an independently
versioned four-clip diagnostic store, its freshly generated train/evaluation
manifests, the full model configuration, and an explicit runtime/budget recipe.
The diagnostic store copies its own small RGB payloads without linking or
modifying the adopted distribution. Train clips cover TrackNet, Meiji and Chat;
validation uses Chat with observed and confirmed-negative teachers. Selection
uses smallest frame count then clip ID among native 1280x720 candidates satisfying
the declared teacher conditions. The driver records the selected IDs and counts.

Preparation publishes this entire input root as a verified immutable generation.
Restore that generation at the identical absolute path in VM B. Restore the
interrupted run at its original absolute output path. Do not regenerate manifests
or rewrite snapshot paths inside the recipe during same-run resume.

Run from the checkout with `PYTHONPATH=/content/tennis-lab`; the Colab worker
supplies its managed `RCLONE_CONFIG`. Actual arguments and receipts are retained
with the completed experiment. General command forms are:

```bash
.venv/bin/python knowledge/runs/run-ball-colab-checkpoint-resume-20261024/resume_probe.py prepare \
  --source-store <fixed-full-distribution> --root <fixed-input-root> \
  --remote-root gdrive:tennis_lab/outputs/ball_detection/precompute/temporal_heatmap/resume-proof-20261024/inputs \
  --receipts <new-preparation-receipts>
.venv/bin/python knowledge/runs/run-ball-colab-checkpoint-resume-20261024/resume_probe.py train \
  --inputs <fixed-input-root> --root <run-output> --mode baseline \
  --remote-root <new-control-snapshot-root> --receipts <new-control-receipts>
.venv/bin/python knowledge/runs/run-ball-colab-checkpoint-resume-20261024/resume_probe.py train \
  --inputs <fixed-input-root> --root <interrupted-run-output> --mode interrupted \
  --remote-root <same-run-snapshot-root> --receipts <new-interruption-receipts>
.venv/bin/python knowledge/runs/run-ball-colab-checkpoint-resume-20261024/resume_probe.py restore \
  --root <original-input-or-run-output> --remote-root <corresponding-snapshot-root> \
  --generation <verified-generation> --manifest-sha256 <pinned-marker-sha256> \
  --receipts <new-restore-receipts>
.venv/bin/python knowledge/runs/run-ball-colab-checkpoint-resume-20261024/resume_probe.py train \
  --inputs <fixed-input-root> --root <original-interrupted-run-output> --mode resume \
  --remote-root <same-run-snapshot-root> --receipts <new-resume-receipts>
.venv/bin/python knowledge/runs/run-ball-colab-checkpoint-resume-20261024/resume_probe.py compare \
  --reference <control-training-json> --actual <resumed-training-json> --output <comparison-json>
```

Declare each receipt directory using the skill's `--persist`. The synchronous
publisher owns the run and input snapshots; no worker-periodic mirror writes to
those same snapshot roots. Worker operation records preserve full command logs.
`--mode failure` injects loss of communication in the publisher at the first
checkpoint, after a real optimizer update. Its Colab job must fail and preserve
`failure.json`; it is not a successful training run.

The fixed diagnostic recipe is hybrid, T=32, 5ch, 1280x720, unmodified production
widths/depths, frame chunks of 4, activation checkpointing, BF16/nvJPEG, no
augmentation, seed 42, 128 input frames/update, one child window/microbatch,
12 updates, warmup 2, checkpoint interval 5. Full scalar values are in the saved
`recipe.json`; preparation and both VMs require exactly the same committed code
and lockfile. Interruption uses the runner's stop-after-update=5 and retains the
original 12-update schedule. The comparison checks all updates and sample/teacher
counts, cumulative input, LR, optimizer steps and loss with atol=1e-6, rtol=1e-4.
It additionally reports final model-state and CPU/CUDA RNG identity.

Each training snapshot explicitly selects the current checkpoint and weights plus
configuration, manifests and logs. Prior checkpoints remain in their own immutable
generations and are not copied again. Input snapshots include their whole tree.
The proof measures transfer/storage cost and does not prune any generation.
Only sessions created for this proof are stopped, with saved terminal-job and
session-stop receipts. Account-wide assignment zero is not a completion condition.
