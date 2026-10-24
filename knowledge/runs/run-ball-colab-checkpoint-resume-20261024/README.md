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

## Environment reconstruction

Both training arms use the commit recorded in `receipts/baseline-a.json`, not the
current branch tip. The input recipe checks that commit and the lockfile before
training. Git bundles carry local commits without publishing a branch. The setup
helper was added later; after running it, check out the training commit again.
The helper does not change training source or the lockfile.

VM A used uv 0.12.21 and CPython 3.11.16. VM B's default `uv sync --python 3.11`
selected uv 0.13.4 and CPython 3.11.17, despite identical locked package versions.
This attempt was rejected before data retrieval or training. Use
`setup_pinned_environment.py` with explicit versions and an isolated managed
Python directory. The helper checks the complete Python build string against
the control receipt. Follow the official
[versioned installer documentation](https://docs.astral.sh/uv/getting-started/installation/)
and [unmanaged installation contract](https://docs.astral.sh/uv/reference/installer/).
The image's `UV_INSTALL_DIR` overrides `UV_UNMANAGED_INSTALL`; the helper removes
that conflicting installer option. Fetching the installer through urllib failed
with HTTP 403, whereas the documented curl transport succeeded. Both failed jobs
and their final Drive-save receipts are retained as operational evidence.

```bash
UV_PYTHON_INSTALL_DIR=/content/tennis-tools/python-3.11.16 python3 \
  knowledge/runs/run-ball-colab-checkpoint-resume-20261024/setup_pinned_environment.py \
  --uv-version 0.12.21 --python-version 3.11.16 \
  --tool-dir /content/tennis-tools/uv-0.12.21 \
  --receipts outputs/ball_detection/colab_persistence/environment-b-pinned-isolated \
  --reference /content/baseline-training-reference.json
```

Select absent tool/receipt directories. The source helper revision and the final
training checkout are separate, explicit Git operations. Do not accept the
default Python patch selection as environment equality.

## Evidence map

| Receipt | Purpose |
| --- | --- |
| `receipts/git-bundles.json` | Exact training/setup commits, bundle prerequisites, byte hashes and Drive file IDs |
| `receipts/execution-plan-a.json`, `receipts/execution-plan-b.json`, `receipts/execution-plan-b-recovery.json` | Ordered command plans; session receipts identify which jobs actually ran. Each next job requires the preceding job's terminal state and Drive-save receipt |
| `receipts/environment-comparison.json` | Different VM boot IDs, equal Python build, uv version and principal package versions |
| `receipts/data-a.json`, `receipts/data-b.json` | Complete distribution retrieval on both VMs, independent SHA-256 and semantic sample checks |
| `receipts/preparation-a.json`, `receipts/full-manifest-comparison.json` | Fresh distribution manifests, diagnostic selection, input immutability and common-frame identities |
| `receipts/restore-references.json` | Pinned input/run generations, marker hashes, actual Drive marker IDs and verified byte counts |
| `receipts/baseline-a.json`, `receipts/interrupted-a.json`, `receipts/resumed-b.json` | Committed source, original schedule, optimizer steps, complete pack/loss records, RNG/model hashes and publication timings |
| `receipts/comparison-b.json`, `receipts/prefix-comparison-b.json` | All 12 updates within the predeclared loss tolerance, identical sample/teacher/LR sequence and RNG, and byte-identical committed five-line log prefix |
| `receipts/failure-injection-a.json` | Real update followed by a failing publisher, with no successful completion marker |
| `receipts/restore-inputs-b.json`, `receipts/restore-run-b.json` | Verified restoration at the original absolute paths |
| `receipts/notebook-recovery.json`, `receipts/notebook-transport-failure.log` | A connection-loss attempt that never started its user command, its saved cancellation and same-VM recovery |
| `receipts/local-verification.json`, `receipts/pytest.log` | Final CPU tests and static-check arguments/results; local home prefixes are normalized |
| `receipts/session-a.json`, `receipts/session-b.json` | Owned-session starts, saved terminal jobs, confirmed stops, and A stopping before B was created |
| `receipts/compute-units.json`, `receipts/doctor-*.json` | Owned-session cost estimate with assumptions, plus account-wide balance/rate observations |
| `receipts/input-inventory.json`, `receipts/run-inventory.json` | Final scoped Drive IDs, sizes and hashes; all 14 published markers matched their saved receipt IDs and MD5 values |
| `receipts/evidence-archive.json` | Verified Drive backup of the normalized receipt/driver bundle, with archive hash and actual Drive file ID |

Operation logs live under `gdrive:tennis_lab/operations/colab/<session>/<job-id>`.
Declared receipt trees live under `gdrive:tennis_lab/outputs/ball_detection/colab_persistence/`.
The run/input snapshots have their separate roots in the restore references.
Do not merge a periodic worker mirror into an immutable snapshot generation.
Only small summaries enter this bundle; checkpoint bytes and complete worker
logs remain on Drive. The four-clip recipe is specific to continuity diagnostics;
prepare a separate full-data recipe and budget for scientific training.

`advance_saved_jobs.py` preserves the local orchestration source used for these
plans. It accepts a plan JSON and an absent audit directory, submits sequential
managed jobs, and fails if the expected terminal result or Drive-save receipt is
missing. A null `argv` means to verify an already submitted job, not resubmit it.
Historical job/session IDs must not be reused as a fresh experiment.

After VM B's data job completed, the Notebook transport lost its connection while
preparing `restore-inputs`. The job remained pending without a worker start.
The original plan stopped after cancellation and its Drive save; reconnecting the
existing assignment preserved the boot ID and downloaded distribution. The input
destination and its old receipt directory were absent before recovery. The
recovery plan reuses the verified data job and a new restoration job/receipt ID.
It does not repeat a possibly running command. The
[runtime contract](../../../.agents/skills/tennis-colab/references/runtime.md)
defines cancellation and reconnection; the receipts record this occurrence.
