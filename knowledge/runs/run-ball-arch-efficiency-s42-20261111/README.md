# Ball architecture efficiency campaign

Issue #1859. The source/recipes of #1675 are read-only inputs. This campaign has
five models and 54,000 optimizer updates: T1 (MMMCCC) → T2 (CCCMMM) → C1 → C2 → C3.
Each query model has two connected 6k stages and re-freezes its CNN for the first
600 updates of each stage. CNN models train from scratch for 10k with the common
DPT's original 28k LR schedule. Seed is 42, microbatch is one window, and every
update consumes 128 frames. Main recipes are generated from the verified saved
recipes, not diagnostic settings.

## State and storage

Drive root: `tennis_lab/outputs/ball_detection/train/arch_efficiency/s42-20261111`.
The local root is the same path below `/content/tennis-lab/outputs`.
`plan.json` fixes the source/conditions; `level-decision.json` points to measured
4-level OOM and 3-level joint-gradient qualification. Both models use 3 levels
(/16,/32,/64), four points, chunk four, BF16 with the complete official MSDA in
FP32 and explicit compiler graph breaks. No gradient checkpointing is used.

`stage-status/<stage>/state.json` pins the latest immutable generation and its
manifest. Every publication pauses writers, verifies contents, restores the
snapshot independently and verifies all SHA-256 values before advancing that
pointer. `serial-status.json`, `persistence/`, stage logs and checkpoints retain
update/sample/LR/RNG/optimizer continuity. Drive generations are never pruned.
A stopped or failed process is not completion. OOM, nonfinite gradients and
persistence errors stop the queue command; later models are not skipped.

## First run

Use a worktree under `.claude/worktrees/ball-arch-efficiency`, with the repository
venv and a single idle L4. The mounted Drive must be available. `rclone` and a
0600 local-alias config `/tmp/ball-drive-mount.conf` point `mounted_drive:` at
`/content/drive/MyDrive/tennis_lab`; no OAuth secret is stored in this run.
Initialize `third_party/DINO` and follow the repository's CUDA build procedure.
The build command/ABI patches/toolchain are recorded in the build manifest.

From the worktree root:

```bash
RUN=knowledge/runs/run-ball-arch-efficiency-s42-20261111
.venv/bin/python "$RUN/fetch_data.py"
.venv/bin/python "$RUN/restore_inputs.py"
# Confirm or restore the measured level decision and explicitly selected binary.
export TENNIS_LAB_DINO_EXTENSION=/content/tennis-lab/outputs/ball_detection/train/arch_efficiency/s42-20261111/msda-build/lib/MultiScaleDeformableAttention.so
.venv/bin/python "$RUN/verify_msda.py"
.venv/bin/python "$RUN/serial.py" prepare
.venv/bin/python "$RUN/enqueue.py" smoke
.venv/bin/python "$RUN/enqueue.py" measure
.venv/bin/python "$RUN/enqueue.py" train
```

Wait for each prerequisite to succeed before the next enqueue command. The
common training queue uses `resource=all`, the main repo's `.training_queue`,
source CWD and idempotent attempt keys. Its worker is detached in its own session;
one serial queue command owns all five scientific jobs. Reusing a failed key
never silently retries it; inspect the failure, then explicitly use a new
`--attempt` after a justified repair.

Smoke is 20 real-data updates per model, stops at 10, verifies Drive bytes, then
resumes. It deliberately trains the CNN immediately and uses lazy shard checks
after the global complete snapshot verification. Those diagnostic settings are
recorded and never enter the main recipes. It verifies identical data and
augmentation draws within T1/T2 and within C1/C2/C3. Smoke updates do not count
as scientific training.

The untrained benchmark has 48 full/component/stage/block/spatial boundaries.
Each gets eval+inference_mode, cold start, at least 10 warmups and 30 measured
samples in each of three reordered blocks. Raw CUDA/wall phases, memory,
parameters, MAC terms/upper bounds, graph breaks and traces are saved to Drive.
It excludes input decoding/augmentation and makes no accuracy claim.

## Resume after VM loss

Only resume after the old VM is gone. Get the execution commit from the durable
`serial-status.json` (or `execution.json` before main startup), check out that pushed
version in the same worktree, mount Drive and recreate the compatible venv.
The Python/Torch/decoder/native ABI must match the recorded environment; do not
silently downgrade, use CPU, change shape, switch to weights-only or drop state.

```bash
RUN=knowledge/runs/run-ball-arch-efficiency-s42-20261111
.venv/bin/python "$RUN/fetch_data.py"
.venv/bin/python "$RUN/restore_inputs.py"
.venv/bin/python "$RUN/recover.py"
export TENNIS_LAB_DINO_EXTENSION=/content/tennis-lab/outputs/ball_detection/train/arch_efficiency/s42-20261111/msda-build/lib/MultiScaleDeformableAttention.so
.venv/bin/python "$RUN/verify_msda.py"
.venv/bin/python "$RUN/enqueue.py" train
```

After measurement, `.venv/bin/python "$RUN/summarize.py" --publish` produces the
aggregate tables/figure and verifies a second immutable Drive snapshot. This
operation is CPU-only. Raw samples and full traces stay on Drive.

The serial command restores each latest verified snapshot, skips durably
completed stages, resumes model/AdamW/RNG/sampler state, and advances only after
fixed validation (observed 85,473) and durable completion. A global `FAILED.json`
requires diagnosis and explicit recovery; archive its evidence rather than
clearing it to bypass an unexplained failure. Existing local uncommitted stage
suffixes are preserved as `.abandoned-*` before restoration.

Inspect with the common queue's `status`/`list`, `serial-status.json`,
`stage-status/` and `logs/`. Check finite losses/gradients/LRs, continuous
sample positions and verified 500-update generations. The Codex startup task
ends after T1 crosses update 600 and its first 500-update checkpoint has passed
Drive readback; completion of all five models is a later observation.
