# Scoped closed-evaluation recovery adapter

This operational adapter is for #1585 recovery; it does not complete #1801.
The original scientific campaign, runtime, bootstrap, source, lockfile and DINO
gitlink stay unchanged. All five per-VM input/pose checks still run.

Before implementation, the test contract is:

- Only a closed evaluation whose original completion marker matches the current
  input identity may use one temporary, exact-file cache. Checkpoint fetches,
  writes, refreshes and all other paths use the original guards.
- Reject ambiguous ancestors/subtrees, duplicate paths/IDs, missing or extra
  members, wrong size/MD5/SHA, mutation during transfer or original verification,
  local symlinks and path escapes. Compare remote membership/ID/content metadata
  before and after; compare every local SHA with the original completion marker.
- Run the original `ColabBackend.verify` and `check_evaluation`: wrong input,
  checkpoint lineage, report identity/schema/count/order/metrics must fail.
- Preserve Python/NumPy/Torch random state and protected scientific source.

Reuse `test_campaign.py` (39 existing cases), particularly
`test_checkpoint_adapter_rejects_newest_invalid_without_fallback`,
`test_evaluation_adapter_checks_pinned_order_and_unweighted_mean`, and the
VM-loss/state-machine failure gates. Add CPU tests in `test_recovery_adapter.py`
with a small fake transport but the actual original stage/report validators.
The transport substitutes only external Drive I/O and checkpoint bytes; it does
not replace the original report validator. Faults use asymmetric scene IDs and
independent byte digests. Keep each parameterized function below 50 cases.

Measure collection count and `--durations=0`; these fixtures contain only small
files, need no GPU/network, and target under five seconds per case. Separately
measure a representative read-only real-Drive directory qualification before
allocating a replacement L4. No scientific/sample RNG is used for temporary
names or qualification. Record the exact operational commit and deployment
bytes separately from the frozen scientific identity.
