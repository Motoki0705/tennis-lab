# Separate-test evaluation contract verification

Findings, acceptance status and remaining work belong to the
[knowledge node](../../nodes/ball_detection/000069-run-ball-separate-test-contract-20261024.md).
The permanent evaluation interface is documented in the
[training README](../../../src/tasks/ball_detection/training/README.md#trainvalidationとは別配布版のtest).

This run checks data correspondence without real test predictions, GPU work or
Colab sessions. Source snapshots and existing train/validation distributions are
read-only. Preparation and hashing do not copy or hardlink RGB payloads.
Large evaluation manifests and raw operation logs stay below the active
checkout's ignored `tmp/`; small identities and receipts are retained here.

## Reproduce the local contract

Use the source repository pinned by the protocol's `inputs.json`, the two
unchanged distribution stages from the portable-distribution run, and its
prepared train/validation evaluation manifests. All paths are explicit; nothing
searches for substitutes. Run from the active checkout with its virtualenv:

```bash
export PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$PWD" CUDA_VISIBLE_DEVICES=""
SOURCE_REPO_ROOT=<absolute-source-repository>
TRAINING_DISTRIBUTION=<absolute-train-val-distribution-stage>
TEST_DISTRIBUTION=<absolute-test-distribution-stage>
TRAINING_MANIFESTS=<absolute-prepared-train-val-manifests>
.venv/bin/python knowledge/runs/run-ball-separate-test-contract-20261024/prepare_contract.py \
  --source-repo-root "$SOURCE_REPO_ROOT" \
  --training-distribution "$TRAINING_DISTRIBUTION" --test-distribution "$TEST_DISTRIBUTION" \
  --training-manifests "$TRAINING_MANIFESTS" --output "$PWD/tmp/separate-test-contract-repro"
```

If the explicit train/validation manifests need regeneration, use the all-frame
preparation entry documented in `data/windows/README.md` with the original
distribution stage and `--distribution-source-store`. Keep the same T and window
parameters. This does not change the adopted distribution.

The script checks protocol clip IDs against source indices and portable indices,
source/group/split/FPS, complete common intervals, frame/teacher identities and
each T's owner hash/counts. Contract generation additionally verifies both
distributions' metadata, annotations and every RGB payload against the shared
original snapshot, and compares regenerated GT/PTS/teachers/packs/owners.
Before/after inode, size, mtime and ctime must agree for existing snapshot files.

The first recorded attempt completed all test manifests, then stopped because
the experiment script compared a four-field protocol key with three values.
After correcting that lookup, a new output was used with the explicit
`--test-manifests <first-output>/test-manifests` option. That attempt rechecks all
identities and source bytes; its `preparation_seconds=0` describes reuse, not a
fresh preparation. The first attempt's full elapsed preparation time was not
captured. The failed output and logs remain separate from the successful receipt.

The executed data-verification source is pinned by `execution.json` and
`executed-source.patch` against its recorded base. The patch was reconstructed
and checked against every source-file digest in the preparation plan, before
later API rejection checks were added. It is historical evidence, not the final
implementation diff. The final implementation has its own ordinary test record.

After the additional rejection guards, `check_contract_mapping.py` checked the
final mapping validator against the real 1,537-clip training distribution and
225-clip test distribution. Its explicit arguments are `--source-store`,
`--training-distribution`, `--test-distribution`, `--contract`,
`--contract-sha256` and a new `--output`. The matching semantic/mapping digests and
module identity are retained in `current-contract-check.json`. That check reuses
the prior RGB-byte verification and does not claim another full payload hash.
`final-contract-check.json` additionally records the final guard that binds each
copied original annotation to its clip's annotation digest. The prior mapping
receipt is retained as a separate observation.
`accepted-contract-check.json` records the completed implementation, including
the training-code identity check aligned with the existing resume contract.

## Ordinary verification

`validation.json` records the exact lock-wrapped pytest command, environment,
source digests and result: 628 passed. `synthetic-cli.json` records all six real
CLI conditions and unchanged validation calibration, using 24×24 CPU fixtures.
The six-condition eight-frame comparison is kept at
`tmp/separate-test-review/comparison-wide.mp4`; its small receipt is `video.json`.
These are functional checks, not real test accuracy.

One existing prefetch regression is explicitly deselected from that passing
batch. It waits for DataLoader data with `num_workers=1`. The original combined
batch was interrupted after 609 passes; isolated and explicit-spawn diagnostic
attempts both reached the watchdog. The spawn trial was reverted. An unchanged
code copy of the branch's base reproduced the wait with a 45-second watchdog.
`prefetch-regression.json` and `baseline-source.json` retain these observations.
The original regression remains in the test file. Both architectures' inline
loading resume checks pass, including exact model/optimizer/schedule/RNG/sample
state comparisons and source-change refusal. The production training runner and
runtime were not changed to work around the unresolved prefetch wait.

## Initial Drive transfer and stop condition

The preflight and terminal observations are retained in
[transfer-failure.json](transfer-failure.json). The authenticated configuration
was selected by path without exposing its contents. The preflight found enough
quota and no existing test destination. The portable-distribution transfer
supervisor/driver was then invoked with the pinned test stage, source snapshot,
new test path and a new receipt directory:

```bash
export RCLONE_CONFIG=<authenticated-rclone-configuration>
.venv/bin/python knowledge/runs/run-ball-portable-distribution-20261023/transfer_supervisor.py \
  --detach --state "$PWD/tmp/test-transfer-supervisor.json" \
  --stdout "$PWD/tmp/test-transfer.stdout.log" --stderr "$PWD/tmp/test-transfer.stderr.log" -- \
  .venv/bin/python knowledge/runs/run-ball-portable-distribution-20261023/transfer_data.py \
  --distribution "$TEST_DISTRIBUTION" --source-store <absolute-pinned-source-snapshot> \
  --drive-path data/ball_detection/ball-mix1800-720p-all-frame-test-v1 \
  --receipts "$PWD/tmp/test-distribution-transfer"
```

Do not repeat this command against the current destination. Four concurrent first
uploads were followed by three same-name `shards` folders. All four upload CLI
calls returned failure on ambiguous path verification; inventory observed four
partial payloads. No metadata or final distribution manifest was published.
The supervisor exited with return code 1 and both recorded processes were gone
when checked. This is a failed upload, not an available test dataset.

Existing assets were not moved, trashed or overwritten. The child's instructions
require stopping on duplicate/existing content, so no repair, retry or receipt
archive upload was attempted. The raw receipts remain in local `tmp/`.
The parent must decide how to repair the duplicate folders or authorize another
destination. Serial creation of destination subdirectories before concurrent
payload uploads is a proposed driver correction; it was not applied to the
historical driver or presented as a verified recovery.

Receipt location strings use `<worktree>`, `<distribution-worktree>`,
`<source-repo>` and `<home>` placeholders. Counts, IDs, hashes, timestamps and
failure statuses retain the observed values. The JSON records contain no tokens.
