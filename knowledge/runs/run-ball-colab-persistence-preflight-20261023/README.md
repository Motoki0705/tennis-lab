# Colab persistence preflight

The [knowledge node](../../nodes/ball_detection/000066-run-ball-colab-persistence-preflight-20261023.md)
records the failed integration preflight. This bundle contains read-only CPU
evidence, not a completed Colab save/resume experiment. The original training
source revision is pinned by `code_commit` in the probe receipt. The diagnostic
script was added locally; no uncommitted training implementation was executed.

## Reproduce the boundary check

Run from the active checkout. `DISTRIBUTION_STAGE` is the small prepared train/val
store from the preceding distribution run. `SOURCE_STORE` is its pinned source
snapshot. Both inputs remain read-only. Select a fresh receipt path.

```bash
export PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$PWD"
DISTRIBUTION_STAGE=<absolute-prepared-train-val-directory>
SOURCE_STORE=<absolute-pinned-source-snapshot>
.venv/bin/python knowledge/runs/run-ball-colab-persistence-preflight-20261023/probe_annotation_paths.py \
  --distribution "$DISTRIBUTION_STAGE" --source-store "$SOURCE_STORE" \
  --output tmp/colab-persistence-preflight/annotation-path-probe.json
```

The observed exit code is 2 with `status="blocked"`. Other unexpected errors
propagate. `source_payloads` explicitly checks the source snapshot identity and
maps the selected clip IDs; it is not a search or fallback to an arbitrary store.
The probe uses pinned RGB hashes for manifest construction and does not rehash,
copy or hardlink the large payloads. It checks file versions before and after.
For a fully transferred distribution, omit `--source-store`; that mode lacks the
original-snapshot control and was not executed in this preflight.

The resolver finds the original annotation and verifies its checksum. The same
clip passes the legacy consumer in the source snapshot. The two portable
consumers fail:

| Consumer | Observed result | Required upstream repair |
| --- | --- | --- |
| `all_frame_preparation.py`, lines 75–89 | The unresolved relative annotation fails `file_version` | Resolve with `store.annotation_path(clip)` before collecting input file versions |
| `hybrid.py:reviewed_empty_evidence`, lines 74–80 | Rejects the relative annotation as not absolute | Use the same resolver, retaining checksum/frame/notes validation |
| `build_all_frame_manifests` | Propagates the latter error | Add relocated-distribution-to-window integration coverage |

The `preparation_file_version` field reproduces the first failing operation; it
does not claim to have run full preparation on the metadata-only source view.
The actual manifest builder is called separately. Fixing only preparation's
stat collection cannot fix the independent absolute-path guard in the teacher
consumer. Existing published data does not require an in-place rewrite.

## Fixed VM placement for the resumed experiment

Use the skill's checkout root `/content/tennis-lab` on both VMs. Keep the following
absolute paths identical across attempts:

| Role | VM path |
| --- | --- |
| Immutable distribution | `/content/tennis-lab/data/ball_detection/ball-mix1800-720p-all-frame-train-val-v1` |
| Generated train/evaluation manifests | `/content/tennis-lab/outputs/ball_detection/precompute/temporal_heatmap/all-frame-train-val-v1` |
| One run | `/content/tennis-lab/outputs/ball_detection/train/temporal_heatmap/<run-id>` |

Restore the original manifests with the run; do not regenerate different bytes
and claim the same run. Generate the initial manifests against the distribution
at the fixed VM path, never against the old snapshot. The existing resume
identity remains authoritative, including its absolute snapshot path. There is
no content-only path override. Fixed placement is an implementation decision;
successful new-VM resume at these paths has not yet been measured.

## Remaining execution

After the parent integrates the consumer repair and its integration tests:

1. Re-run the boundary check; require `boundary_passed` and unchanged input
   files. Generate fresh all-frame manifests against the full distribution.
2. Implement checkpoint-completion persistence with verified size/hash receipts
   and fail-closed communication/save failure injection. Preserve checkpoint,
   `latest.json`, `run.json`, both manifests, `train.jsonl` and logs coherently.
   The generic storage implementation belongs to existing tooling or
   `src/utils`; experiment orchestration belongs in a knowledge run.
3. Commit and transfer the exact Git revision using the preceding distribution
   run's Git-bundle procedure. Reuse its locked Python 3.11 environment and data
   retrieval/verification scripts. Record new setup and transfer measurements.
4. Run the authorized short 720p hybrid T=32 diagnostic with explicit memory
   controls, compare uninterrupted versus interrupted/new-VM-resumed updates,
   sample sequence and loss, and retain save/restore receipts and timings.
5. Confirm terminal jobs and Drive persistence before stopping each owned
   session; record stop and assignment observations.

Checkpoint saving, failure injection, L4 training, new-VM restoration and locked
environment reconstruction remain unexecuted. Their durations are unknown.
Do not reuse the preceding data-only VM checks as training-resume evidence.
`runtime-preflight.json` records a pre-existing account-wide assignment; this
task allocated no session and did not stop that unidentified assignment. It
therefore provides neither an account-wide zero-assignment receipt nor a
training compute-unit measurement.

## Ordinary verification

The following existing tests passed: 44 tests in 22.88 seconds. Their fixtures
exercise distribution and windows separately; they do not cover the failing
portable-annotation consumer boundary.

```bash
PYTHONDONTWRITEBYTECODE=1 flock "$PYTEST_LOCK" .venv/bin/python -m pytest \
  tests/unit/tasks/ball_detection/data/test_distribution.py \
  tests/integration/tasks/ball_detection/test_all_frame_windows.py -n 4 -q
.venv/bin/python -m ruff check knowledge/runs/run-ball-colab-persistence-preflight-20261023
.venv/bin/python -m mypy --follow-imports=skip knowledge/runs/run-ball-colab-persistence-preflight-20261023
```

The probe's original annotation-path receipt and account preflight are preserved
as JSON without local home/repository/worktree paths or credential contents.
