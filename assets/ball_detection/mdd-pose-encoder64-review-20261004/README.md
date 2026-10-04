# Encoder /64 revision — CPU smoke, no training

[model-smoke.json](model-smoke.json) records the updated /16 spatial stem and two
shared 2D→2D→3D blocks on one real 32-frame 720p window. It records shapes and
finite output, not detection accuracy or a controlled performance benchmark.

The [frozen play-window manifest](../mdd-pose-review-20261004/manifest.json) and
[play/non-play visualizations](../mdd-pose-review-20261004/README.md) are unchanged.
The old folder's `model-smoke.json` belongs to the earlier /8 encoder proposal.

Current implementation and input/output contract:
[models/mdd_pose/README.md](../../../src/tasks/ball_detection/models/mdd_pose/README.md).
