# RTX 5060 Ti continuation of the architecture campaign

Current state: preparing attempt 5 with a verified NVMe dataset work copy after
the parent's 2026-11-14 additional instruction.
Attempt 4's closed failure remains in [attempt-4-storage-stop.json](attempt-4-storage-stop.json).
Canonical data, outputs, private environment and caches reside on the mounted
ext4 data volume at `/mnt/wsl/tennis-data`; original output aliases are symlinks.
The hot dataset is an independent copy at
`/home/kamimura/tennis-hot/ball-arch-s42-20261111-fixed-v1` on the WSL root/NVMe
filesystem. The E source is unchanged. The earlier E-only attempt-5 staging and
its interrupted recovery are retained; no production registration used that root.
The prior [monitoring-handoff.json](monitoring-handoff.json) still
describes the stopped attempt until the new startup acceptance is committed.

Refs #1859. The [2026-11-13 agreement](https://github.com/Motoki0705/tennis-lab/issues/1859#issuecomment-6840801232)
supersedes the original prohibition on checkpointing for T1/T2. The initial desktop migration started
T1 from tail update 5,500 (query cumulative 11,500). T1 has since completed with
verified validation and predictions. Attempt 4 resumes the verified T2 first-stage
update 1,000, then continues its remaining stages, C1, C2 and C3.
No completed stage is repeated. C1/C2/C3 retain their original recipe and do not
checkpoint activations. A different GPU cannot promise bitwise continuation.
Its timing/memory must not be compared with the historical untrained L4 benchmark.

Use the original campaign [README](../run-ball-arch-efficiency-s42-20261111/README.md)
for the scientific plan. This directory contains only desktop migration,
qualification, lifecycle and evidence; it does not replace historical generations.

## Inputs and safety

`$ROOT` is the recovery output `outputs/ball_detection/train/arch_efficiency/desktop-recovery-20261114-a5-nvme`
below the implementation worktree. `$SOURCE` is the immutable pushed runtime checkout.
`$ADAPTER` is this directory in that checkout. `$PY` is `$ROOT/env-original/bin/python`.
`$QUEUE` is the main repository's shared `.training_queue`; set `$RCLONE_CONFIG`
to the authenticated private config. Never print its contents.

- `rebuild_dataset.py` hardlinks every distribution payload from the verified
  desktop dataset. It reads and SHA-checks all 2,925 payloads and the distribution
  declaration; it copies no RGB data. The distribution is the original fixed
  train/validation snapshot, not a newly sampled dataset.
- For attempt 5, `copy_hot_dataset.py` instead copies this fixed distribution to
  the explicitly selected NVMe hot root, verifies every destination size/SHA,
  independent inode and exact inventory, and records copy/reuse bytes and time.
  `bootstrap_recovery.py --hot-dataset <D-root> --dataset-source <E-canonical>`
  verifies/reuses that work copy and points only its dataset alias at D. Other
  immutable inputs keep their E-side hardlinks. The existing path-only preparation
  and checkpoint migration regenerate manifest digests without changing content,
  optimizer, RNG, sampler, model tensors or the scientific recipe.
- `restore_inputs.py` verifies the T1 checkpoint against its immutable marker,
  the saved input receipt, and C0's complete snapshot. Source/inputs of #1675
  remain read-only. `state.py` explicitly relocates store paths and migrates the
  checkpointing/source metadata; tensor/AdamW/RNG/schedule/sample/log state is
  compared recursively before the result is accepted. `audit.py` also inspects
  what the real runner actually restores.
- The private environment uses the original Python 3.13.16 and Torch 2.14.1 CUDA
  13.0. The original freeze is the version source for its runtime dependencies.
  The recorded original environment itself has package metadata conflicts;
  restoring its exact installed versions uses explicit `--no-deps`, followed by
  import, decode, native numerical/gradient and actual-runner checks. Failed
  solver/network attempts are retained in output logs; they are not successful
  validation. The shared `.venv` is unchanged.
- `build_msda.py` verifies all seven DINO CUDA/C++ files against the previously
  official-verified manifest and runs the repository build with an explicit
  private interpreter and `TORCH_CUDA_ARCH_LIST=12.0`. It changes no kernel.
  sm_120 additionally uses `NVCC_APPEND_FLAGS=--maxrregcount=64`: the unbounded
  build exceeded the GPU's register-per-block limit in the original 1,024-thread
  double backward gradcheck. Compiler resource reports and both build manifests
  preserve this failed candidate and the passing correction.
- Every GPU workload is a shared-queue `resource=all` job. `guard.py` caps the
  CUDA allocator at 80%. Before each optimizer update it waits while host
  available RAM is below 6 GiB, without a wait deadline. The emergency observer
  stops strictly below 2 GiB (including during setup, persistence or waiting),
  or when either `/mnt/d` or `/mnt/wsl/tennis-data` statvfs available space falls
  below 10 GiB. The data volume must be mounted; failed measurements stop explicitly.
  Each update's resources and the host-guard record expose both capacities.
  Legacy filesystem
  growth is recorded as telemetry; it is not a stopping condition. This replaces
  the earlier 20 GB proxy by the parent's 2026-11-13 recovery instruction. systemd
  also bounds host memory. Shapes, batch, microbatch, precision, three MSDA levels, chunk
  four and compilation policy stay fixed.
- The original Drive tree is completely archived to E using tennis-drive
  `download-tree`, a precomputed trusted expected manifest and progress file.
  No desktop publication may begin until its closed-tree readback completes.
  The original `/mnt/e/tennis-lab-archive` remains storage only. From attempt 5,
  outputs/env/cache use the separate mounted ext4 data volume on E. The added
  parent instruction selects the independent D dataset work copy for training.

## Qualification and startup

The following qualification commands describe the initial setup, which is
already complete. Do not repeat them in the active recovery root. For another
recovery, follow the current handoff and recovery document. The private
interpreter is used throughout:

```bash
"$PY" "$ADAPTER/build_msda.py" --root "$ROOT" --assets "$MAIN_REPO"
"$PY" "$ADAPTER/desktop.py" prepare --root "$ROOT"
RCLONE_CONFIG="$RCLONE_CONFIG" "$PY" "$ADAPTER/launch.py" \
  --root "$ROOT" --kind qualification --session "$CODEX_SESSION_ID" --attempt 1
# Wait for native-qualification.json complete=true and a terminal successful job.
RCLONE_CONFIG="$RCLONE_CONFIG" "$PY" "$ADAPTER/launch.py" \
  --root "$ROOT" --kind probes --session "$CODEX_SESSION_ID" --attempt 1
# Inspect all candidates, checkpoint-policy.json, and probe-acceptance.json.
```

The qualification runs official forward and seven-width double gradchecks,
native dtype/padding/chunk tests, and actual Inductor forward/backward. The full
T64/720p equivalence reference applies the query feature VJP to stage-local
uncheckpointed CNN graphs to avoid the known oversized end-to-end reference.
All model parameter gradients and full-shape outputs are compared; this
component reference is a diagnostic, and production does not detach gradients.

`probe_matrix.py` tests two supported scopes on T1's real 5,500 full state and an
unfrozen diagnostic T2, plus unchanged C1. Probe updates are isolated and do not
enter scientific training. It chooses one common passing scope for T1/T2 and
records every candidate's peaks, cold sample and steady update samples. A
candidate allocator OOM is evidence; any other failure stops qualification.
A broader query-span/memory replay prototype failed the pinned Inductor
partitioner; it is excluded from the supported option and production matrix.
The two retained scopes compiled successfully without changing compiler policy.
The main T2 recipe still freezes the CNN for its first 600 updates per stage.
Earlier qualification a3/a4 also exposed an over-strict comparison between
differently fused BF16 compiled graphs. The final qualification separates exact
FP32 compiled replay and full-shape eager BF16 chain-rule equivalence from the
BF16 compiler's rounding differences. It requires close compiled BF16 outputs
and every finite parameter gradient; it does not claim elementwise compiled
BF16 gradient identity. Original failures and the numerical diagnostic remain
in the run outputs, and the accepted evidence is `native-qualification.json`.

After reviewing these records, CI Python success for the pushed source, complete
E archive, and the policy, start the single production command from `$SOURCE`:

```bash
RCLONE_CONFIG="$RCLONE_CONFIG" "$PY" "$ADAPTER/launch.py" \
  --root "$ROOT" --kind production --session "$CODEX_SESSION_ID" --attempt "$ATTEMPT"
```

`$ATTEMPT` must be fresh; use `recovery-bootstrap.json` and inspect existing
queue/unit registrations as described in recovery.md. Registration is idempotent
per attempt. A failed attempt is never silently
re-enqueued. A user systemd unit owns foreground queue `serve`; agent descendant
cleanup cannot stop it. The serial command uses a writer lock, invokes each
stage in a separate process, waits for validation and verified completion,
and fails immediately on OOM, nonfinite gradients or persistence errors.
The runtime checkout and the implementation/output worktree are git-worktree-locked
while the service uses them. Retain the initial `desktop-resume-20261113` root too:
the active root reuses its private environment, native binary and compiler cache.
The parent may unlock/remove these only after the terminal event and acceptance.

## State, persistence and continuation

The only desktop destination is Drive-root-relative
`outputs/ball_detection/train/arch_efficiency/s42-20261111/desktop-attempts/20261113-a1/`.
Each 500-update full checkpoint, configuration, log, validation and per-frame
prediction is published as an immutable snapshot. Every snapshot is independently
restored and fully compared before its pointer advances. Drive retains every
scientific generation. A local retirement ledger records obsolete local files
removed after durable verification, keeping local storage bounded. The ledger
`production/<stage>/retired-weights.json` names the exact generation, digest,
size and source commit for each evicted tensor. Diagnostic snapshots and their
`probe-receipts/` are also read back before local probe tensors are evicted.

Inspect `production-registration-a4.json`, `serial-status.json`,
`production-status.json`, `stage-status/*/state.json`, `persistence/`, logs,
and the recorded service/queue identities. The systemd terminal event is written
once when the queue exits; no periodic follow-up or post-acceptance progress
polling is registered. Full environment/build and local paths remain output
metadata; versioned summaries link the evidence.

Worker acceptance is at least ten finite continuous post-compile updates in
the first running stage after recovery. Allocated and reserved VRAM must remain
below the allocator cap, host available RAM at least the 2 GiB emergency floor
(with at least 6 GiB at each admission), and D: available space at least 10 GiB. The worker leaves a committed `monitoring-handoff.json`
with the actual unit, job, process identities and authoritative evidence paths.
The parent then monitors T1 tail 6,000/12k validation and prediction readback,
T2's 500 checkpoint and 600-to-601 unfreeze, and every condition's completion.
No periodic follow-up is registered by this worker.
If recovery selects a fully verified terminal T1 checkpoint at update 6,000,
the normal runner publishes its completion without repeating T1 updates; the
worker then verifies the first ten post-compile T2 updates. The same allocator,
RAM and D: available-space limits apply.
An interrupted process or an existing output directory never proves completion.
Do not repeat `prepare` or delete a failed attempt. See [recovery](recovery.md).

## Desktop qualification and attempt history

The selected scope is `encoder_msda`: replay each CNN stage and each official
MSDA frame chunk. Ordinary CA, temporal attention and FFN are not checkpointed.
T1/T2 use the same policy; C1/C2/C3 keep checkpointing off. See
[probe-summary.json](probe-summary.json) for exact bytes, all samples and OOMs.

| Exact-shape probe | Allocated / reserved GB | Cold first update (s) | Following-two median (s) |
| --- | ---: | ---: | ---: |
| T1, encoder + MSDA | 10.804 / 12.595 | 234.39 | 3.77 |
| T2, encoder + MSDA, CNN unfrozen | 10.804 / 12.533 | 93.61 | 3.16 |
| C1, no checkpointing | 9.397 / 10.475 | 130.11 | 1.81 |

GB means 1,000,000,000 bytes. Allocator cap is 13,682,291,507 bytes.
The MSDA-only candidates OOM'd (T1 allocated/reserved 13.534/13.652 GB;
T2 13.509/13.590 GB). These are three-update qualifications, not an untrained
benchmark or a comparison with L4. C2/C3 have not been probed on this GPU.

Attempt 1 stopped during the initial 5,500 checkpoint readback, before a new
optimizer update. The whole-filesystem growth guard observed 19.20 GB (stop
threshold 19 GB); host available was 19.76 GB. Three interrupted private-env
wheel download directories were identified by timestamp, checked for absent
process descriptors and single links, then removed, reclaiming 1.54 GB. Its
failed job and uploaded terminal event remain intact. The initial full state was
independently reverified, restored to the fresh active root, and compared exactly
before attempt 2. Obsolete local attempt-1 tensors were evicted only after the
verified Drive snapshot and recovery. No optimizer reset, scientific update
replay, scope change or reduction of shape/precision was used. Evidence is in
[attempt-1-storage-recovery.json](attempt-1-storage-recovery.json) and
[local-tensor-retirement.json](local-tensor-retirement.json).

## Parent monitoring

The committed [monitoring-handoff.json](monitoring-handoff.json) is the current
identity/path/acceptance record. Its output root is the active attempt; the old
root is retained input/environment/failure evidence. The execution checkout is
fixed at the pushed commit recorded there, while this README and handoff JSON
belong to the reporting branch. The Drive handoff receipt locates an independently
verified copy of these reporting files.

| Parent milestone | Authoritative evidence |
| --- | --- |
| T1 tail 6,000 / query 12k | `stage-status/T1-q06000-q12000/state.json`: event completed, update 6000, receipt/readback verified, checkpoint_state_equal true. `validation-update-00006000.json`: observed 85473 and matching target/teacher/frame identities; its `.frames.jsonl` must be in that verified snapshot. |
| T2 first 500 checkpoint | Retained `persistence/T2-q00000-q06000/*.json` entry: event checkpoint, update 500, verified receipt/readback and checkpoint_state_equal true. The mutable latest pointer may already be later. |
| T2 CNN unfreeze | Its `train.jsonl` row 600 has encoder_frozen true, row 601 false, subsequent updates/samples continuous with finite loss/gradient/LR. |
| Each condition completed | Matching stage completed pointer and local COMPLETED marker, exact budget (T 6000 per stage, C 10000), fixed validation/prediction readback; controller moves on only afterward. |
| Whole serial process ended | Root COMPLETED.json plus every completed stage, queue done, and the recorded terminal event state done / returncode 0. A service exit alone does not prove completion. |

For source coordinates, report `metrics.mean_error_px`, `median_error_px`,
`p95_error_px`, `within_distance_rate` with `distance_threshold_source_px=20`.
The parent owns these future milestones; the worker does not wait for them or
register periodic follow-ups. The terminal event is written locally by service
exit/ExecStopPost, then uploaded and SHA-read back. A missing event alongside an
inactive unit needs investigation. Failure paths and the precise fresh-root
recovery commands are in the handoff and [recovery.md](recovery.md).

The historical attempt-2 startup prefix is retained in `startup-acceptance.json`
(update 5511 / sample 21511, execution `cb3243ed4`). Its later storage failure
is recorded in [attempt-2-storage-recovery.json](attempt-2-storage-recovery.json).
The historical attempt-3 acceptance prefix remains in
`attempt-3-startup-acceptance.json`; the current handoff identifies attempt 4.

## Interrupted 6,000 checkpoint recovery

Attempt 2 completed T1 tail update 6,000 and fixed validation, but its checkpoint
readback crossed the old 19 GB whole-WSL growth threshold. Host RAM was adequate.
The parent replaced this storage proxy with physical D: available space below
10 GiB; whole-WSL growth remains telemetry only. CPU guard and existing adapter
tests passed 23 cases in 3.34 seconds, including 10 new guard cases.

The uploaded `checkpoint-00006000-8586c177` was independently restored with
pinned marker SHA-256 `2719458b1e83187eb55acd14f5e23bc71db98a0e1a21fcbc29efc4f1159e85e0`.
All 11 files (550,906,644 bytes), including checkpoint, committed log, validation
and predictions, matched the stopped local output. Full model/AdamW/RNG/LR/
sampler state and the weights artifact matched exactly; the strict resume reader
accepted update 6,000 / sample 22,000. Its verified checkpoint pointer was
published and independently read back. The normal runner, rather than this
verification, publishes the completed-stage marker. No T1 optimizer updates or
validation need to be repeated.

T1 query cumulative 12k validation has observed 85,473 frames, mean 35.3367 px,
median 8.9393 px, P95 176.7366 px and <=20 px rate 0.741907 in original/source
coordinates. This is a validation result, not a deployment decision. Full metrics,
identities, exact SHA-256 and recovery evidence are in the linked attempt-2 record.

Guard implementation: `6ddf48cfa`; CI-compatible tests and production source:
`2cb49a907241bdfecd718c71a14c1da93965c57f`. Its CI Python succeeded in run
[43374423329](https://github.com/Motoki0705/tennis-lab/actions/runs/43374423329).
The initial CI test-double failure and its correction are retained in the recovery
record; the fixed tests also passed with xdist and the real CI measurement plugin.
[attempt-3-local-cleanup.json](attempt-3-local-cleanup.json) records 1,476,173,824
WSL allocated bytes released from exact Drive-protected stopped-attempt tensors
and the independent readback cache. This does not claim a matching VHDX shrink.
The initial root was inventoried; its env/native/compiler resources were retained,
and the unused cloned env has no single-link file blocks to reclaim.

## Attempt-3 acceptance

T1 was restored at 6,000 / sample 22,000, including actual model/AdamW/RNG/
sampler checks, and completed with zero replayed updates. Its fixed validation
and predictions are in the independently restored completed snapshot. T2 then
recorded updates 1–11 / samples 10,001–10,011 with finite loss, gradient norm
and learning rates. After the first compile update, the next ten updates had
median 1.8348 seconds. Peak allocated/reserved was 7,147,210,752 / 9,737,076,736
bytes; the CNN is still frozen at this acceptance prefix. Host available RAM
minimum was 17,030,041,600 bytes and D: available minimum 33,458,446,336 bytes,
inside the 4 GiB / 10 GiB lower bounds. See
[attempt-3-startup-acceptance.json](attempt-3-startup-acceptance.json).

The historical attempt-3 unit was `tennis-ball-arch-desktop-production-20261113-a3.service`; it later failed at the old RAM floor.
The historical prefix precedes its later RAM failure. The current handoff records
attempt 4, its terminal-event path and fresh-root attempt-5 recovery commands.
T2's first checkpoint and unfreeze were verified in attempt 3; remaining stage
completions belong to the parent. No periodic follow-up or further live progress
polling is registered by this worker.

## RAM admission after attempt 3

Attempt 3 stopped in T2 backward near update 1,283 when global host available
RAM fell below its old 4 GiB floor. Numerical training was finite; the latest
verified full checkpoint is `checkpoint-00001000-e2f35d3f`. Parent observations
include concurrent pytest workers and filesystem inventory activity. Without
training-process RSS history the cause cannot be fully attributed.

The revised policy waits before each optimizer update below 6 GiB, resumes at
6 GiB or more, and retains a strict 2 GiB emergency stop plus the unchanged
D: 10 GiB stop. Waits do not consume updates, sampler draws or RNG and are outside
the measured update seconds. More than 30 minutes produces one notice; there
is no maximum wait duration.

Each stage writes `production/<stage>/ram-waits.jsonl`. Events are
`ram_wait_started`, `ram_wait_resumed`, `ram_wait_over_30_minutes` and
`ram_wait_interrupted`; keys include `update`, `timestamp`, `started_at`,
`elapsed_seconds`, `host_available_bytes`, `process_tree_rss_bytes`,
`parent_rss_bytes`, `children_rss_bytes` and `process_tree_pids`. The sum includes
shared pages and is RSS, not PSS. Every new train.jsonl row contains `resources`
with the same memory fields, `ram_wait_seconds`, `admitted_host_available_bytes`
and `admitted_at`. Historical rows stay unchanged. The wait log joins checkpoint
snapshots when present; later wait events remain local until the next snapshot.

A generic optional `UpdateObserver` in the temporal runner supplies admission
and telemetry. The default performs no callbacks and changes no recipe. The
strict resume reader is retained. Only the pinned old/new runner source pair in
`resource-observer-migration.json` may be adapted during explicit recovery; every
other unexpected source change is rejected, and model/AdamW/RNG/schedule/sampler
and committed logs are compared completely before adoption.

## Attempt-4 acceptance

The verified T2 generation `checkpoint-00001000-e2f35d3f` was independently
restored. All model/AdamW/RNG/schedule/sampler state and the committed 1,000-row
log prefix matched exactly. The real runner restored 381 model keys and 375
optimizer states, sample position 11,000, with no new state keys. Only the
approved observer source transition was adapted. Attempt 3's uncommitted
1,001–1,283 suffix was discarded. T1's completed stage and validation were
retained without replay. See [attempt-3-ram-recovery.json](attempt-3-ram-recovery.json).

The pushed execution source is `15cdc684ff60a210cb1d8209f393e60490ca8f44`;
CI Python succeeded in run
[43392201904](https://github.com/Motoki0705/tennis-lab/actions/runs/43392201904).
The detached unit is `tennis-ball-arch-desktop-production-20261113-a4.service`,
with one shared-queue `resource=all` serial command. Actual unit/job/PID/PGID and
terminal-event paths are in [monitoring-handoff.json](monitoring-handoff.json).
The frozen runtime checkout stays at this execution source while reporting
commits advance in the implementation worktree.

Updates 1,001–1,011 and sample positions 11,001–11,011 were continuous, with finite
loss, gradient norm and learning rates. The CNN was unfrozen. Data draws,
augmentation and learning rates matched the same attempt-3 prefix. The first
update took 37.7527 seconds including compilation; the following ten had median
3.4532 seconds. Peak allocated/reserved was 10,803,609,600 / 12,763,267,072 bytes,
below the 13,682,291,507-byte allocator cap. Host available minimum was
8,412,168,192 bytes, admission minimum 8,616,722,432, and D: available minimum
16,144,359,424. All stayed within the 2 GiB emergency, 6 GiB admission and 10 GiB
disk bounds. Maximum parent-plus-child RSS sum was 13,527,920,640 bytes, including
shared-page duplication. There were no RAM wait events in this prefix; indefinite
waiting, one long-wait notice, and emergency interruption were checked in CPU
tests. The acceptance evidence is [attempt-4-startup-acceptance.json](attempt-4-startup-acceptance.json).

The selected guard/adapter/real-model resume suite passed 36 cases in 20.66
seconds. Twelve added cases cover admission/RSS, pinned source migration and
real-model observer behavior. Mutations that lower admission to 4 GiB or include
waiting in update timing failed the intended assertions; sources were restored
before the passing suite. Normal pre-commit and CI passed. The initial observation
helper ran before launch registration existed and failed; only that read-only
helper was restarted after registration. Its original failure log is retained.
The production job was never duplicated or restarted for this helper race.

Worker observation ends at this accepted prefix. The parent owns future RAM waits,
T2 validation and all remaining condition completions. Wait events and every new
update's RSS are exposed in the handoff. No periodic follow-up is registered.

## Attempt-4 storage stop after the accepted prefix

While the reporting HEAD's CI was completing, attempt 4 later stopped during T2
forward because D: available fell below 10 GiB. The guard recorded
10,708,541,440 bytes, below 10,737,418,240, with host available minimum
8,395,161,600 bytes; RAM was not the trigger. The failed unit has MainPID 0,
the queue job is in `failed/`, and its terminal event was uploaded and independently
SHA-read back. No next stage was started. Both the execution source and the first
reporting HEAD passed CI Python; this subsequent runtime failure is separate.

Local updates 1,001–1,313 / sample positions 11,001–11,313 remained continuous
and finite. They have no newer verified checkpoint and must not be adopted.
The latest verified pointer remains update 1,000,
`initialized-00001000-120a3ef2`, with matching full-state/readback evidence.
The successful startup proof remains a historical prefix; it does not assert
current service liveness or worker completion. The current handoff sets
`worker_acceptance_complete=false` and records the storage block.

At the stop investigation D: available was 6,279,356,416 bytes, already below
the limit. No additional cleanup or restart was performed. The evidence does
not attribute D: consumption to a particular writer. Restore adequate capacity
before following the fresh-root/unused-attempt-5 recovery commands. Original
private environment/native/cache and stopped-attempt evidence remain intact.
