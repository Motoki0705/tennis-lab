# RTX 5060 Ti continuation of the architecture campaign

Refs #1859. The [2026-11-13 agreement](https://github.com/Motoki0705/tennis-lab/issues/1859#issuecomment-6840801232)
supersedes the original prohibition on checkpointing for T1/T2. This run continues
T1 from tail update 5,500 (query cumulative 11,500), then T2, C1, C2 and C3.
No completed stage is repeated. C1/C2/C3 retain their original recipe and do not
checkpoint activations. A different GPU cannot promise bitwise continuation.
Its timing/memory must not be compared with the historical untrained L4 benchmark.

Use the original campaign [README](../run-ball-arch-efficiency-s42-20261111/README.md)
for the scientific plan. This directory contains only desktop migration,
qualification, lifecycle and evidence; it does not replace historical generations.

## Inputs and safety

`$ROOT` is the run output `outputs/ball_detection/train/arch_efficiency/desktop-resume-20261113`
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
  before filesystem growth reaches the 20 GB allowance. systemd also bounds
  host memory. Shapes, batch, microbatch, precision, three MSDA levels, chunk
  four and compilation policy stay fixed.
- The original Drive tree is completely archived to E using tennis-drive
  `download-tree`, a precomputed trusted expected manifest and progress file.
  No desktop publication may begin until its closed-tree readback completes.
  E is storage only; training never reads/writes its tensors or data there.

## Qualification and startup

Prepare the fully verified inputs once, then use the private interpreter:

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
  --root "$ROOT" --kind production --session "$CODEX_SESSION_ID" --attempt 1
```

Registration is idempotent per attempt. A failed attempt is never silently
re-enqueued. A user systemd unit owns foreground queue `serve`; agent descendant
cleanup cannot stop it. The serial command uses a writer lock, invokes each
stage in a separate process, waits for validation and verified completion,
and fails immediately on OOM, nonfinite gradients or persistence errors.
The runtime checkout is additionally git-worktree-locked while its service uses
it. The parent may unlock/remove it only after the terminal event and acceptance.

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

Inspect `production-registration-a1.json`, `serial-status.json`,
`production-status.json`, `stage-status/*/state.json`, `persistence/`, logs,
and the recorded service/queue identities. The systemd terminal event is written
once when the queue exits; no periodic follow-up or post-acceptance progress
polling is registered. Full environment/build and local paths remain output
metadata; versioned summaries link the evidence.

The revised worker acceptance is at least ten finite continuous updates after
T1's cold compilation, resuming the original tail 5,500 full state. Allocated and
reserved VRAM must remain below the allocator cap, and host available RAM above
the guard threshold. The worker leaves a committed `monitoring-handoff.json`
with the actual unit, job, process identities and authoritative evidence paths.
The parent then monitors T1 tail 6,000/12k validation and prediction readback,
T2's 500 checkpoint and 600-to-601 unfreeze, and every condition's completion.
No periodic follow-up is registered by this worker.
An interrupted process or an existing output directory never proves completion.
Do not repeat `prepare` or delete a failed attempt. See [recovery](recovery.md).
