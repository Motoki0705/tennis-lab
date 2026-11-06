# CI repair design (#1674, before implementation)

- Configuration contract: every CandidateGMMConfig field is required in Python;
  defaults belong solely to the existing Hydra composition. Remove the evaluation
  adapter's implicit settings instance too. Tests and replay explicitly receive
  composed settings. Preserve all YAML values and every numerical kernel.
  Reuse `tests/unit/utils/configuration/` source-discovery/default-ownership
  tests and the public configuration/component integration. No audit bypass.
- Baseline contract: positions, valid, inliers, accepted pixels and rejection
  codes remain exact against the frozen main arrays. Only the diagnostic
  reprojection residual may differ by at most four float32 ULPs; keep zero
  residuals exact. The fixture's largest four-ULP bound is below 6e-11 px.
  Investigate float64 fit.x versus exported float32 positions and CPU/BLAS
  execution paths in sequential subprocesses. Record hashes proving the default
  reconstruction/assembly/numerical sources are unchanged from main.
- Evidence: compare old Python default values to newly composed values exactly,
  then characterize prediction impact. No parameter tuning or new evaluation
  split. Keep candidate_gmm_v2 immutable. Report whether evidence is a real-data
  rerun or code/config equivalence, rather than claiming an unperformed replay.
- Conditional DINO fix: inspect failed jobs. Apply exactly the single-file
  a8982a7e0 change only if the named isolation failure is observed.
- Cost: extend existing cases, no new heavy/video fixtures. pytest -n 0 (at most
  two workers), sequential CPU-heavy checks. Record collect-only and all phase
  durations for configuration and related tests. Run ruff/mypy/knowledge checks;
  push normally, inspect every latest-head CI job and report after all succeed.
