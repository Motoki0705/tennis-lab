# Bounded temporal-heatmap performance pilot

The pilot has exactly six conditions: DPT/hybrid × T=32/64/128. Each condition
has at most ten optimizer updates and twenty minutes, including evaluation and
checkpoint publication. One L4 session has a four-hour allocation-to-stop limit;
work stops after three and a half hours. `session_guard.py` runs locally and
enforces that deadline independently of the conversation. No scientific training
budget is authorized by running this diagnostic.

The source is committed locally, transferred in a Git bundle, and checked out at
its exact commit on Colab. GitHub is not written. The setup and full-distribution
download reuse the preceding [persistence proof](../run-ball-colab-checkpoint-resume-20261024/README.md)
and [distribution verifier](../run-ball-portable-distribution-20261023/README.md).
The ordered execution plan and job/save receipts accompany the completed record.

The full 157 GB train/validation distribution is retrieved and verified afresh.
The pinned input snapshot from the persistence proof is restored at its original
path, checked against the downloaded index/metadata, and its original manifest
content hash is retained. Three new manifests use the same independently
versioned diagnostic store: three training clips (TrackNet, Meiji, Chat), one
Chat validation clip, native 1280×720 throughout. They preserve the original
common-frame, teacher and pack identities. This limits evaluation to 162 unique
frames. Full-data validation time and sustained random-read behavior are not
measured by this pilot. Checkpoint accuracy on these clips cannot select a model.

`pilot.py prepare` restores those verified inputs, prepares all three T values,
freezes the production model configurations and publishes the small pilot inputs.
`pilot.py matrix` starts fresh child processes sequentially. T=32 starts directly
with seed 42; T=64/128 each start independently from that architecture's ten-update
**pilot** artifact, with a fresh optimizer and sampler. Each condition has ten
updates of its own; no additional training precedes the six conditions. These
artifacts are not approved ancestors for scientific fine-tuning.

All conditions use 128 input frames/update, one window/microbatch (four/two/one
accumulations), BF16, nvJPEG, compile off, activation checkpointing on, frame chunks
of four, no augmentation, upfront input verification, zero loader workers, and
the original production widths. The recipe stores every optimizer/objective value.
Updates 5/10 publish checkpoint generations synchronously through
`RcloneSnapshotStore`; final completion has a separate small publication. Each
generation is new and verified; nothing is pruned or overwritten.
Restore a checkpoint generation when resuming; the small completion generation
contains only `COMPLETED.json` and `latest.json`, and its checkpoint receipt is
recorded separately.

The driver times the existing evaluation and serialization/hash functions without
changing their behavior. `train.jsonl` supplies update/data-wait times and peak
allocated/reserved VRAM. Separate child processes release memory after each
condition. Data wait is the runner's host wall time around the next batch and
image-input call; it does not isolate every asynchronous GPU decoding operation.
OOM and timeout remain failed measurements; later eligible conditions
continue. Missing T=32 pilot ancestors make the corresponding longer conditions
unmeasured. A storage or unexpected execution failure stops the matrix. No
resolution, width, sequence-length or device fallback is performed.

`summarize.py --receipts <downloaded-conditions> --output <new-summary.json>`
checks the six-condition matrix, update limits, publication receipts, common
model/runtime/budget and matching sample/teacher/LR sequences. Missing quantities
remain null. Costs extrapolated to future runs must state their assumptions and
remain separate from the measured table. The current VM does not guarantee future
VRAM, disk capacity, throughput or CU rate.

## Completed evidence

Findings and measured tables belong to the
[knowledge node](../../nodes/ball_detection/000068-run-ball-temporal-colab-pilot-20261024.md).
The node separates completed pilot measurements from proposals awaiting human
agreement. Historical session/job IDs and timestamps must not be reused for a new
allocation; use an absent output root and the new session's actual creation time.

| Receipt | Meaning |
| --- | --- |
| `receipts/executed-requests.json` | Exact managed argv for all five jobs, including the pinned setup and bounded matrix |
| `receipts/source-bundle.json`, `receipts/source-inventory.json` | Committed pilot source, bundle prerequisite/hash and actual Drive ID |
| `receipts/environment*.json`, `receipts/final-environment.json` | Package/build pins, GPU/driver/CUDA and observed disk capacity |
| `receipts/data.json`, `receipts/preparation.json` | Complete distribution verification, sample equality, restored input identity and fixed diagnostic coverage |
| `receipts/conditions/` | Final matrix and complete per-update, timing and publication records for each condition |
| `receipts/measurements.json` | Validated rounded-table inputs; no missing-value estimates |
| `receipts/inventory-comparison.json`, `receipts/*-inventory.json` | Published marker IDs/hashes, exact file counts and total sizes checked against Drive inventory |
| `receipts/terminal-jobs.json`, `receipts/session-stop.json`, `receipts/doctor-stopped.json`, `receipts/guard.json` | Saved terminal jobs, successful owned stop, assignment zero and the armed local guard exiting after stop |
| `receipts/compute-units.json` | Observed balance change and a separately labelled constant-rate estimate |
| `receipts/budget-proposal.json` | Unapproved future-cost projections and explicit assumptions; generated by `estimate_budget.py` |
| `receipts/upstream-test-contract.json`, `receipts/test-asset-inspection.json` | Prior-scope test dataset/calibration integration gap; no workaround was applied |
| `receipts/local-verification.json`, `receipts/*-final.log` | Locked CPU tests, explicit ruff/mypy checks and source content hashes |

Recompute the local summaries without starting a GPU or writing existing assets:

```bash
RUN=knowledge/runs/run-ball-temporal-colab-pilot-20261024
.venv/bin/python -B "$RUN/summarize.py" --receipts "$RUN/receipts/conditions" --output tmp/temporal-pilot-measurements.json
.venv/bin/python -B "$RUN/estimate_budget.py" --receipts "$RUN/receipts/conditions" \
  --preparation "$RUN/receipts/preparation.json" --environment "$RUN/receipts/environment.json" \
  --data "$RUN/receipts/data.json" --doctor "$RUN/receipts/doctor-running.json" --output tmp/temporal-pilot-budget.json
```

Host home/worktree prefixes are normalized in retained receipts. Remote Colab
paths are preserved because the input and resume contracts require them. The
large raw stop inventory is backed up under the recorded `operations/colab/`
closure path; its small Git receipt retains hashes, counts and the empty source
diff. Checkpoint bytes and full worker logs remain on Drive. Main training,
full-data accuracy evaluation and convergence extensions were not run.

`receipts/evidence-archive.json` identifies the verified Drive backup of the
normalized run bundle and node. The archive excludes that self-reference receipt.
