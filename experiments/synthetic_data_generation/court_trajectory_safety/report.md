# B00 Court trajectory safety decision report

Status: **complete**. The quality-only decision is **quality_only_rejected**; geometry remains authoritative.

## Blind pilot evidence

- Reviewer A: 128 records, 11 artifact-heavy.
- Reviewer B: 128 records, 3 artifact-heavy.
- Disagreements/adjudications: 10; consensus positives: 5 (calibration 2, held-out 3).

## Frozen quality-only result

Calibration evaluated 800 adjacent-midpoint/operator candidates, but no threshold family passed all frozen calibration gates. No rule, threshold, recall, precision, or other predictive metric was selected or reported. The explicit rejection reasons are `no_calibration_threshold_family_passes_frozen_gates, insufficient_held_out_positive_labels`.

Held-out safe V4 candidates were 0/9 artifact-heavy; held-out legacy views were 2/12.

## Geometry and final route

- Frozen V4 plan: 2016 frames, 39 trajectory groups, 0 support violations, and group-disjoint splits=True.
- Source authority: `data/synthetic_data_generation/raw/B00.mp4` matches the immutable ingested copy at SHA-256 `6a4387e6061b4d81fabd5e99a6f0814953469138ed52eed7aec665b37c50962c`; the stale recorded `tennis_court.mp4` path is not used.
- Final V4 dataset: complete, 2016/2016 accepted frames, 39 trajectory groups, 8 shards, 0 split leaks, 0 safety violations, and 0 renderer errors across 8 invocations.
