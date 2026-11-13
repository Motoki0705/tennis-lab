# RTX 5060 Ti continuation of the architecture campaign

Refs #1859. The [2026-11-13 agreement](https://github.com/Motoki0705/tennis-lab/issues/1859#issuecomment-6840801232)
supersedes the original prohibition on checkpointing for T1/T2. The initial desktop migration started
T1 from tail update 5,500 (query cumulative 11,500). The current recovery adopts
its fully verified update 6,000, then continues T2, C1, C2 and C3.
No completed stage is repeated. C1/C2/C3 retain their original recipe and do not
checkpoint activations. A different GPU cannot promise bitwise continuation.
Its timing/memory must not be compared with the historical untrained L4 benchmark.

Use the original campaign [README](../run-ball-arch-efficiency-s42-20261111/README.md)
for the scientific plan. This directory contains only desktop migration,
qualification, lifecycle and evidence; it does not replace historical generations.

## Inputs and safety

`$ROOT` is the active run output `outputs/ball_detection/train/arch_efficiency/desktop-recovery-20261113-a3`
below the implementation worktree. `$SOURCE` is the immutable pushed runtime checkout.
`$ADAPTER` is this directory in that checkout. `$PY` is `$ROOT/env-original/bin/python`.
`$QUEUE` is the main repository's shared `.training_queue`; set `$RCLONE_CONFIG`
to the authenticated private config. Never print its contents.

- `rebuild_dataset.py` hardlinks every distribution payload from the verified
  desktop dataset. It reads and SHA-checks all 2,925 payloads and the distribution
  declaration; it copies no RGB data. The distribution is the original fixed
  train/validation snapshot, not a newly sampled dataset.
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
  CUDA allocator at 80%, requires at least 4 GiB host available RAM, and stops
  when `/mnt/d` statvfs available space falls below 10 GiB. WSL filesystem
  growth is recorded as telemetry; it is not a stopping condition. This replaces
  the earlier 20 GB proxy by the parent's 2026-11-13 recovery instruction. systemd
  also bounds host memory. Shapes, batch, microbatch, precision, three MSDA levels, chunk
  four and compilation policy stay fixed.
- The original Drive tree is completely archived to E using tennis-drive
  `download-tree`, a precomputed trusted expected manifest and progress file.
  No desktop publication may begin until its closed-tree readback completes.
  E is storage only; training never reads/writes its tensors or data there.

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

Inspect `production-registration-a3.json`, `serial-status.json`,
`production-status.json`, `stage-status/*/state.json`, `persistence/`, logs,
and the recorded service/queue identities. The systemd terminal event is written
once when the queue exits; no periodic follow-up or post-acceptance progress
polling is registered. Full environment/build and local paths remain output
metadata; versioned summaries link the evidence.

Worker acceptance is at least ten finite continuous post-compile updates in
the first running stage after recovery. Allocated and reserved VRAM must remain
below the allocator cap, host available RAM at least 4 GiB and D: available
space at least 10 GiB. The worker leaves a committed `monitoring-handoff.json`
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
The attempt-3 identities and acceptance prefix are in the current handoff.

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

The active unit is `tennis-ball-arch-desktop-production-20261113-a3.service`.
The handoff contains the actual queue identity, PID/PGID, terminal-event path
and fresh-root attempt-4 recovery commands. T2 500 persistence, 600→601 unfreeze
and remaining completions belong to the parent. No periodic follow-up or
further live progress polling is registered by this worker.
