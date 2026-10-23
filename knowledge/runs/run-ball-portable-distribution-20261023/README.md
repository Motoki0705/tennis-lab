# Portable ball frame-store preparation

Findings and completion status belong to the
[knowledge node](../../nodes/ball_detection/000064-run-ball-portable-distribution-20261023.md).
This bundle records local preparation and the receipt-backed Drive/Colab checks.
The historical runtime preflight is retained in [preflight.json](preflight.json).
The parent subsequently authorized L4 for verification in
[runtime-decision.json](runtime-decision.json); the skill itself is unchanged.

The permanent API is documented in
[data/distribution](../../../src/tasks/ball_detection/data/distribution/README.md).
It prepares small files and hashes original payloads in place. Neither preparation
nor local verification copies or hardlinks the RGB shards.

## Reproduce local preparation

Run from the active checkout. The source repository must contain the exact
snapshot and original annotations pinned by the
[protocol input identity](../run-ball-all-frame-protocol-20261022/inputs.json).
Use a new output directory; existing output is never overwritten.

```bash
export PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$PWD"
SOURCE_REPO_ROOT=<absolute-source-repository>
DISTRIBUTION_OUTPUT="$PWD/tmp/ball-portable-distribution-repro"
.venv/bin/python knowledge/runs/run-ball-portable-distribution-20261023/prepare_data.py \
  --source-repo-root "$SOURCE_REPO_ROOT" --output-root "$DISTRIBUTION_OUTPUT"
.venv/bin/python knowledge/runs/run-ball-portable-distribution-20261023/check_prior_payloads.py \
  --source-repo-root "$SOURCE_REPO_ROOT" \
  --distribution "$DISTRIBUTION_OUTPUT/train-val" \
  --distribution "$DISTRIBUTION_OUTPUT/test" \
  --output "$DISTRIBUTION_OUTPUT/prior-payloads.json"
```

The recorded execution used `tmp/ball-portable-distribution` instead of the fresh
reproduction path above. Preparation time includes source hashing and small-file
construction. The subsequent whole-distribution verification and sample checks
have separate timings. Every source payload, source index/metadata and original
Chat annotation is checked for unchanged inode/size/mtime/ctime across the run.
The before/after free-space measurement also includes other activity on this host;
`local_generated_bytes` isolates the files produced by this operation.

`train-val/` and `test/` are separate explicitly versioned stores. Each contains
only the selected clips, retaining every native frame within those clips. Window
inputs restrict use to the protocol's common intervals. Each `distribution.json`
is the complete size/SHA-256 file manifest and explicit source/new-index mapping.
It deliberately references absent local payload destinations: upload the original
shards using the mapping, rather than trying to open the staging store normally.

Protocol catalog/interval/pack indices remain **source snapshot indices**. Join
them by clip ID through `distribution.json`; never use them as new store indices.
The original protocol files are not silently reindexed. Only the catalog is
filtered by the distribution's clip membership. Test is not uploaded with training
data and is not used for threshold/checkpoint selection.

Sample receipts compare T=32/64/128 windows for every source and selected split.
They include JPEG/RGB, native PTS, complete labels/masks and original Chat notes
through the checked annotation content hash. They do not introduce a new teacher
policy. The window implementation owns note-based negative-mask interpretation.

The complete prepared stores remain under the ignored output directory. The
small local receipts are retained here; do not remove the staging directory before
upload. Reproduction reconstructs the complete manifests from the pinned source.

## Upload and resume on Drive

Follow the
[Drive skill and storage policy](../../../.agents/skills/tennis-drive/SKILL.md).
Select the authenticated configuration by path without displaying its contents.
The target is a new `data/ball_detection/<metadata.version>` directory. Existing
assets are never moved, trashed, adopted into a new plan or silently overwritten.

```bash
export PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$PWD"
export RCLONE_CONFIG="$HOME/.config/tennis-lab/rclone-drive.conf"
DISTRIBUTION_OUTPUT="$PWD/tmp/ball-portable-distribution"
SOURCE_STORE=<absolute-snapshot-path-from-protocol-inputs.json>
DISTRIBUTION_RECEIPTS="$PWD/tmp/ball-distribution-transfer"
DISTRIBUTION_VERSION=ball-mix1800-720p-all-frame-train-val-v1
distribution_args=(
  --distribution "$DISTRIBUTION_OUTPUT/train-val"
  --source-store "$SOURCE_STORE"
  --drive-path "data/ball_detection/$DISTRIBUTION_VERSION"
  --receipts "$DISTRIBUTION_RECEIPTS"
)
.venv/bin/python knowledge/runs/run-ball-portable-distribution-20261023/transfer_data.py \
  "${distribution_args[@]}" --interrupt-largest
.venv/bin/python knowledge/runs/run-ball-portable-distribution-20261023/transfer_data.py \
  "${distribution_args[@]}" --max-files 3
DIST_TRANSFER_STAMP=$(date -u +%Y%m%dT%H%M%S)
.venv/bin/python knowledge/runs/run-ball-portable-distribution-20261023/transfer_supervisor.py \
  --detach --state "$DISTRIBUTION_RECEIPTS/supervisor-$DIST_TRANSFER_STAMP.json" \
  --stdout "$PWD/tmp/distribution-upload-resume-$DIST_TRANSFER_STAMP.stdout.log" \
  --stderr "$PWD/tmp/distribution-upload-resume-$DIST_TRANSFER_STAMP.stderr.log" -- \
  .venv/bin/python knowledge/runs/run-ball-portable-distribution-20261023/transfer_data.py \
  "${distribution_args[@]}"
```

Before restarting, check that the previous supervisor and its command have exited;
inspect the recorded boot ID, PID and process start token to avoid PID reuse.
The original full attempt was interrupted when the model session ended, after
962 payloads had logged successful verification. The resumed transfer uses a
separate OS session and file-backed standard streams. Its supervisor keeps an
atomic terminal receipt with the actual command return code. A running receipt
whose process vanished is an interrupted attempt, not a successful completion.
The dataset is complete only when the transfer's final `verified` attempt and
full inventory also match; a supervisor return code alone is insufficient.

The first invocation bounds one attempted upload with the Drive CLI's timeout.
Its receipt distinguishes interruption from completion before the timeout; do not
claim an interruption unless the observed receipt supports it. The second pauses
between completed files. Resume always computes fresh local SHA-256/MD5 values,
checks a new Drive inventory, retains matching files and rejects corrupt, extra,
duplicate or unverifiable entries. File-level resume restarts an incomplete file;
it does not promise continuation of an abandoned HTTP upload's byte offset.

Up to four independent file CLI uploads run concurrently. Small metadata files
are copied into a local metadata-only upload tree; this never includes RGB.
Metadata is uploaded after payloads, and `distribution.json` is published only
after every other file passes inventory size/hash comparison. Every CLI operation
has stdout/stderr/timing receipts. Preserve the full receipt directory under a new
`operations/drive/<session>/<job>` path and reference its verified inventory/ID
from the completed knowledge record. Do not put all per-file logs into Git.

The same driver accepts the separate test store and its own version/destination.
Its upload is reserved for final evaluation.

The observed completed upload and the verified Drive archive containing all
transfer receipts and the frozen Colab source bundle are pinned in
[drive-upload.json](drive-upload.json). The archive deliberately excludes the
metadata-only upload stage. Individual durations use a monotonic clock; UTC
timestamps are separate observations because their deltas differ on this host.
The overall UTC span starts with the first payload attempt, includes interruptions
and subsequent source rehashes, and excludes the first source-hash phase.

## Verify in two Colab VMs

Use the [Colab skill](../../../.agents/skills/tennis-colab/SKILL.md). The current
owned-session CLI accepts only GPU runtimes and requires `nvidia-smi` during
bootstrap. The parent explicitly authorized L4 after the host reboot. Allocate
only after the complete input has passed Drive inventory verification, and stop
each VM after its terminal jobs have been saved.

Use an exact committed code revision. The child must not push it. The parent can
publish the commit before `start --ref`, or bootstrap a reachable base and transfer
a small Git bundle through the skill's upload operation, then fetch/checkout the
exact local commit with `exec`. Record both bootstrap and executed source commits;
do not describe a locally patched checkout as the original bootstrap revision.

Run `python3 knowledge/runs/run-ball-portable-distribution-20261023/setup_environment.py
--receipts <repo-relative-receipts>` as a persisted skill job. It builds the locked
environment with `uv sync --locked --python 3.11 --no-cache` and records elapsed
time, disk usage and the resulting package versions separately. Python 3.11 matches the locally
validated independent `_sha256` checksum provider. A different Python/default
Colab environment must not silently replace that provider or its verification.
The recorded execution obtains the small sample specification and reference
receipt from the frozen Git commit; both JSON files are tracked in that bundle.
The exact bootstrap/execution commits and archived bundle are referenced by
`source_plan` in [drive-upload.json](drive-upload.json). Do not overwrite those
references after fixing the execution revision. Large input files use the Drive
CLI.

Run `colab_verify.py` from the repository with `PYTHONPATH` set to its root. Its
required arguments are `--drive-path`, a fresh `--destination`, a new `--receipts`
output, `--windows`, `--reference`, `--manifest-sha256` and `--expected-size-bytes`.
Read the last two values from `train-val-local-verification.json`; do not transcribe
hash values into Markdown. Declare the receipt output with the skill's `--persist`.

For VM A, select `--interrupt-download` to interrupt a fresh download with the
Drive CLI timeout, record partial files and resume with verification. VM B uses a
new session, a fresh checkout/disk and a new destination, without that flag, to
measure re-download from Drive. If a real retrieval fails, inspect its saved
receipts before using a new job/receipt directory with `--resume-existing`.
Never interpret a failed download as a verified dataset.

If retrieval completed with CLI verification but a later check failed, preserve
that attempt and its timings. After committing the fix, `--verify-existing` with
new receipts verifies the pinned manifest, every file and all samples without
another transfer. Its `download_seconds=0` is verification-only work; report the
original retrieval separately and never count this as a fresh-VM download.

The script checks available capacity before retrieval.
It pins the downloaded manifest to the local content identity, verifies the exact
file set and every SHA-256/size, then compares complete local/remote sample receipts.
The default four hash workers use the existing independent-provider verifier.
Recorded rclone temporary files are removed only after their final target is
verified and their own content still matches the interruption receipt.
It records retrieval, whole-file verification, sample checks and free disk space
separately. Failure preserves evidence and does not count as verification success.

Confirm terminal job status, returncode and `drive_saved_at`, retrieve the small
receipts and stop the session. Save and check the stop result. A never-started
session is recorded as `not_started`, not as a stopped VM or successful retrieval.

## Ordinary validation

Run pytest through the campaign's shared `PYTEST_LOCK`, with at most four workers:

```bash
flock "$PYTEST_LOCK" .venv/bin/python -m pytest \
  tests/unit/tasks/ball_detection/data/test_distribution.py \
  knowledge/runs/run-ball-portable-distribution-20261023/test_transfer_data.py \
  knowledge/runs/run-ball-portable-distribution-20261023/test_colab_verify.py -n 4 -q
.venv/bin/python -m ruff check src/tasks/ball_detection/data/distribution \
  src/tasks/ball_detection/data/store.py tests/unit/tasks/ball_detection/data/test_distribution.py \
  knowledge/runs/run-ball-portable-distribution-20261023
.venv/bin/python -m mypy --follow-imports=skip src/tasks/ball_detection/data/distribution \
  src/tasks/ball_detection/data/store.py tests/unit/tasks/ball_detection/data/test_distribution.py \
  knowledge/runs/run-ball-portable-distribution-20261023
```

The local record also includes the existing store builder/datamodule/pipeline
regression tests. Unit tests for resume decisions are separate from the live
interruption/resume receipts. The node links the observed remote verification
and checked stop results.

## Consumer integration

Resolve original Chat annotations with `store.annotation_path(clip)`. The new
metadata declares `annotation_path_base="store"`; directly interpreting its
`annotation_path` string as a host-absolute path is invalid. The parallel window
implementation owns this caller update and note-based teacher policy. This run
compares the complete teacher inputs and original notes, without implementing
that policy or generating the final window manifests.
