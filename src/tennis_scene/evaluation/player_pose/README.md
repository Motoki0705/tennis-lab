# Player detection and COCO17 evaluation

Use the published manifest SHA-256 to load the common 2D reference and the fixed body recovery requests.

```python
from src.tennis_scene.evaluation.player_pose.bundle import (
    load_body_input_manifest,
    load_reference_frames,
)
from src.tennis_scene.evaluation.player_pose.scoring import (
    FrameKey,
    Predictions,
    score_players,
)

references = load_reference_frames(bundle_path, expected_sha256=manifest_sha256)
predictions = {
    FrameKey("video_000/clip_003", "cam0", 120): Predictions(
        boxes_xyxy,
        detection_scores,
        pose_result.keypoints.detach().cpu().float().numpy(),
    )
}
report = score_players(references, predictions)
body_inputs = load_body_input_manifest(bundle_path, expected_sha256=manifest_sha256)
```

Each `Predictions` holds one image's person boxes in native-pixel xyxy, one detection score per box and optional aligned `(N,17,3)` native-pixel x/y/confidence rows. Empty or omitted predictions count as misses. The loader returns every cohort image, including images with no eligible players. Prediction keys outside that cohort are rejected. Provide the same image-to-prediction mapping for all variants. Pass `box_margin` explicitly to the loader for sensitivity analyses with fixed eligibility masks.

## Metric definitions

- Each image uses one-to-one IoU matching. The assignment maximizes the count at or above the cutoff before maximizing summed IoU. Every assignment uses the same candidate order, including assignments after a detection score cutoff. Candidates sort by descending detection score, ascending lexicographic `(x_min, y_min, x_max, y_max)`, then ascending lexicographic COCO17 x/y/confidence in joint order, with NaNs last. Among candidates with the same assigned IoU for a player, the one-to-one assignment prefers earlier candidates in that order.
- Recall at 0.50 and 0.75 is matched eligible players divided by all eligible players. Each score cutoff filters boxes and reruns matching. Matched IoU, miss rate and candidate-minus-reference xyxy errors use IoU 0.50. Extra or duplicate detections have unknown label status. Precision and AP are unavailable for this partial reference.
- Pose follows the unfiltered IoU 0.50 assignment. A predicted joint needs finite x/y/confidence and confidence above the policy cutoff (zero by default). OKS uses the [COCOeval sigmas and kernel](https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocotools/cocoeval.py) with the pseudo-box area, and averages across scoreable people. PCK divides each distance by the square root of that area and pools scoreable joints at thresholds 0.10 and 0.20. Both metrics treat missing joints as failures. Pixel distances have a measurement only where both joints exist. Their sample counts and the fraction of predicted eligible joints accompany each summary.
- The coverage report includes every scoreable joint of every eligible person. The matched-only report includes those with an associated box. Player IDs are scoped to clips. Size strata use pseudo-box height `<80`, `[80,160)`, and `≥160` native pixels by default.

See the [Meiji run bundle](../../../../knowledge/runs/run-meiji-multiview-player-evaluation/README.md) for reference generation, prerequisites and the quality gate.
