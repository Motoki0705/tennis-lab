# MDD + pose: pre-training review (2026-10-04)

CPU-only proposal report; no real-data training or GPU model run was performed.

- [Timeline overview](timelines.png): green = play proposal, orange = excluded/non-play proposal;
  purple = supervised 32-frame window coverage; blue = presence evidence; black = observed position labels.
- `example-00.jpg` through `example-08.jpg`: actual training-clip frames, with clip/frame references in [summary.json](summary.json).
  Yellow circles show existing located annotations, including reference-only frames. Orange/excluded does not certify semantic non-play.
- [manifest.json](manifest.json): frozen approved subset, exact pose/review identities, JPEG shard SHA256, all play/excluded intervals and window starts.
- [pose-audit.json](pose-audit.json): full approved-pose checksum/timeline/missing-mask audit.
- [model-smoke.json](model-smoke.json): an untrained default encoder/model on one actual 720p, 32-frame window.

Algorithm and reproduction command: [play-window documentation](../../../src/tasks/ball_detection/data/PLAY_INTERVALS.md).
Architecture and model input contract: [MDD+pose documentation](../../../src/tasks/ball_detection/models/mdd_pose/README.md).

The shown thresholds are proposals for human review, not approved training settings.
