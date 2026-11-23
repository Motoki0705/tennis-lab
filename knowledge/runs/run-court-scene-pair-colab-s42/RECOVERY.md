# Closed evaluation recovery adapter

`recovery_bootstrap.sh` is a separately versioned operational launcher for the
fixed #1581/#1585 campaign. The original `bootstrap.sh`, `campaign.py`,
`runtime.py`, `src`, lockfile and DINO gitlink remain the scientific source.
Deploy the added launcher and adapter from one exact operational commit into a
checkout of the original scientific commit. Record both identities and the
deployed byte digests in the private operation receipt. The current approved
Drive CLI is supplied as an explicit versioned path, with its dependencies.

Only an already closed evaluation stage can use the adapter. It verifies the
original marker and input identity, resolves the unique remote ancestry,
inventories exact subtree paths/IDs/sizes/MD5, downloads one directory with the
existing CLI's `--verify`, and checks every local SHA against the original
marker. Existing local directories must already be complete and valid; they
are never silently repaired. The original backend still verifies the final
checkpoint lineage, report schema, configuration, all IDs/counts/order and all
artifact SHAs. A second remote inventory must match before returning. The
cache is discarded even on failure. It never intercepts refresh, checkpoint,
input, mutable, write or other-stage paths. No scientific/sample RNG is used.

All five expensive per-VM input/pose/provenance checks remain in the original
`prepare`. The optimization does not implement shared input validation or
complete #1801. It changes no seed, batch size, update budget, metric or dataset.

Set `COURT_RECOVERY_DRIVE_CLI`, `COURT_RECOVERY_OPERATIONAL_REF` and
`COURT_RECOVERY_ADAPTER_SHA256` from the deployment receipt. Optional
`COURT_RECOVERY_EVIDENCE_DIR` records bounded stage milestones. Launch only
through the one owned, root-supervised queue/Notebook transport, after old
writers have stopped and quota/GPU identity have been verified. The shell
launcher preserves locked Python 3.11, Torch 2.14.1+cu130, torchvision
0.29.1+cu130 and the original compile-thread setting.

CPU verification and fault coverage were designed in
[RECOVERY_TEST_PLAN.md](RECOVERY_TEST_PLAN.md). Run:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest \
  knowledge/runs/run-court-scene-pair-colab-s42/test_recovery_adapter.py \
  knowledge/runs/run-court-scene-pair-colab-s42/test_campaign.py -n 0 --durations=0
```

Representative real-Drive qualification, invocation limits, operational
identities, numerical latest-checkpoint selection, restored optimizer/RNG
provenance and real save receipts belong in the owned private operations tree.
Existing scientific outputs and completion markers are never manufactured or
overwritten by this adapter. Parent acceptance and #1586 remain separate.
