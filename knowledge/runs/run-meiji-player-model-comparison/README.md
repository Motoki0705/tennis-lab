# Meiji 4-camera player model comparison

This comparison uses the six clips, frame indices and SMPL-derived references from [the fixed player benchmark](../run-meiji-multiview-player-evaluation/README.md). The references are partial. Other people in each image have unknown label status, so player recall is available and precision and AP are N/A.

## Stored DINO and ViTPose-H baseline

The saved [full-video run](../run-i2363-trt-player-video/README.md) contains full-frame DINO detections and ViTPose-H crops for every accepted detection on the same source media. Its upstream 2D outputs remain available when a later 3D step failed. The scorer reads the TensorRT FP16 artifacts before tracking and selection, checks the media hashes and pose-to-detection dependencies, and scores the frame indices in the fixed evaluation manifest.

Run from the repository root with the manifest SHA-256 from #2495 and the recorded full-video run's dataset root.

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONPATH=. \
  .venv/bin/python -B knowledge/runs/run-meiji-player-model-comparison/baseline.py \
  --benchmark data/tennis_multiview/processed/meiji_4cam/benchmarks/multiview-player-evaluation-v1 \
  --manifest-sha256 "$EVALUATION_MANIFEST_SHA256" \
  --dataset "$MEIJI_TRT_RUN/inputs/dataset" \
  --output outputs/player-model-comparison
```

The stored baseline covers 6,844 images and 26,276 eligible player boxes. DINO's Recall@0.50 is 96.67% and Recall@0.75 is 47.07%. For DINO + ViTPose-H, coverage-inclusive mean OKS is 0.7631 and PCK@0.10 is 77.95%. Joint coverage is 429,996/437,714. The [accuracy receipt](baseline-trt16.json) records the counts, size strata and raw output hashes. The full accuracy reports contain camera, clip and player strata and the archived artifact identities. These results describe agreement with the SMPL-derived reference, which uses some of the same 2D observations during reconstruction.

The archived DINO input is 750 by 1,333 pixels, batch 1, person confidence at least 0.30, and duplicate merging disabled. ViTPose-H uses 256 by 192 pixels, batch up to 8 crops before flip testing, flip testing enabled and a 1.2 box enlargement. Both models use TensorRT FP16 on the RTX 5060 Ti. Speed and VRAM for comparison will come from the shared #2496 protocol. PT32 and PT16 accuracy conditions are pending.

## Candidate evidence

| Candidate | Accuracy evidence on the fixed cohort | Shared speed and VRAM |
|---|---|---|
| RF-DETR-L | PT32, PT16 and TRT16 on all 6,844 images. Recall@0.50 is 96.81%, 96.82% and 96.81%. | Pending |
| RF-DETR-2XL | All three backend conditions excluded by the user's license decision. | Excluded |
| ECDet-L/X, COCO and O365 | Six-weight EdgeCrafter smoke includes four ECDet release checkpoints. Fixed-cohort scores pending. | Pending |
| ECPose-L/X, O365 | Both pose releases produced native-pixel COCO17 and aligned boxes on the smoke frames. Fixed-cohort scores pending. | Pending |
| RT-DETRv4-L/X | Both release weights and FP16 ONNX exports pinned. Fixed-cohort scores pending. | Pending |

The [RF-DETR-L accuracy record](../run-meiji-rfdetr-large-accuracy/results.json) contains the complete accuracy, candidate-versus-PT32 parity and box-margin sensitivity results. [RF-DETR licensing and reproduction](rfdetr.md) · [EdgeCrafter evidence](EDGECRAFTER.md) · [RT-DETRv4 evidence](rtdetrv4.md)

The five measured baseline and RF-DETR-L conditions were rescored from their pinned predictions with the #2495 equal-IoU ordering rule. Every metric and stratum was identical to its previous value. The [rescoring receipt](accuracy-recheck.json) identifies the scorer commit and the fresh full-score files.

## ECPose recipe contract

The [ECPose configuration and output contract](../../../src/tennis_scene/pipeline/ECPOSE.md) uses the [shared recipe design](https://github.com/Motoki0705/tennis-lab/issues/2491#issuecomment-7129950015). The tests run the real decoder with a fake model and verify rows through the component store.

The published ECPose-L checkpoint ran through the pipeline component on CPU for the first three consecutive source frames of `clip_003/cam0`. Batch size 2 exercised the partial final batch. With confidence 0.30 and box margin 0.10, each frame produced two positive-area boxes with corresponding COCO17 rows. [Component execution receipt](ecpose-recipe-smoke.json)

The recipe tests use the existing pipeline runtime and component store on CPU. A fake network produces two differently placed people, an empty frame and a trailing partial batch. The actual decoder and compatibility components must preserve source-pixel coordinates, the shared frame boundaries and detection rows through persisted artifacts, with one inference pass. The active graph must keep appearance, tracking and body recovery on their existing input contracts. The tests reject a different recipe or decoding policy in the cache and verify prior standard publication and node identities with the archived FP32 fixture. The small fixture requires no weights or GPU. Run the component tests and the archived-cache integration test separately and measure their durations.

## Test design

- Reuse the fixed-cohort scorer's matching, missing-keypoint coverage and checksum tests. The comparison-specific CPU tests check the serialized row contract with two distinguishable people and an empty frame. Duplicate, missing or extra frame keys and a candidate from a different manifest must fail before scoring.
- Check archived row alignment with detection source IDs that begin at a nonzero offset. Pose features with a different source row or frame boundary must fail even when the box and pose counts agree. The real-data run also checks media and dependency pins for all 24 videos and uses the original source frame numbers.
- Run these small unit fixtures without checkpoints or a GPU. Report test counts and durations from pytest. Record the real-data coverage and artifact hashes from the generated scoring receipts.
