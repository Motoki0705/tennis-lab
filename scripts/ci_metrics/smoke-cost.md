# Model smoke cost decision

[日本語](smoke-cost.ja.md) · [简体中文](smoke-cost.zh-CN.md)

## Decision

For [#1404](https://github.com/Motoki0705/tennis-lab/issues/1404), retain the
existing SLCS training/evaluation and MDD structure tests. The measured model work
is small; reducing their integration coverage or adding meta-device setup is not
justified by this sample. Prioritize merge fixture preparation and CPU scheduling.

## Observation, 2026-10-25

Run the current measurement entry point with two workers and the two file paths:

```bash
.venv/bin/python -m scripts.ci_metrics run --output /absolute/new/smoke --workers 2 -- \
  tests/integration/tasks/slcs/test_training_smoke.py \
  tests/unit/tasks/ball_detection/models/mdd/test_cnn.py
```

On the local Linux host, with native thread limits set to one and CUDA hidden,
all 12 tests passed. This is one subset observation, not a full CPU baseline or
a measurement of GitHub-hosted runner performance. The source and environment,
inventory, per-phase events, JUnit and completeness result were retained together.
The previously cancelled measurements were not reused.

| Work | Cases | Summed call seconds |
|---|---:|---:|
| SLCS training smoke, including normal fit/callback | 5 | 4.582 |
| MDD CNN tests | 7 | 1.054 |
| Shipped MDD configuration structure check, included above | 1 | 0.099 |

The command took 24.853 monotonic seconds; summed call time was 5.636 seconds.
All workers finished collection after 14.347 seconds, and the last phase report
preceded command exit by 7.248 seconds. Summed worker time overlaps and must not
be added to command wall time. The maximum individual pytest process RSS was
958,972 KiB; it is not aggregate memory or memory attributable to one model.

## Preserved contracts and reconsideration

No tests were removed, moved or consolidated. Retain all frames 6/18 × context
off/on cases, real adapter/model/loss backward and context gradients, optimizer
updates, checkpoint reload, and all four evaluation modes with target/mask/frame
identity preservation. Retain normal fit, the monitored checkpoint callback and
actual TensorBoard events. The MDD test still constructs the production module
tree and checks convolution counts and SwiGLU width.

Consequently no removed-case mapping or mutation comparison is needed for this
decision. Reconsider if a future complete measurement makes these tests a material
contributor, or production configuration growth changes their initialization cost.
The broader measurement and comparison contracts remain in [README](README.md).
