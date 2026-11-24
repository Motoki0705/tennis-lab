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

## ECPose recipe contract

The [shared recipe design](https://github.com/Motoki0705/tennis-lab/issues/2491#issuecomment-7129950015) selects the route through `recipe=ecpose`. One full-frame model call supplies detections and aligned COCO17 rows. The same source row IDs and frame offsets flow to the two existing artifact contracts and on to appearance, tracking and selection. The ECPose store and publication namespace carry a distinct identity, with the official checkpoint, source, backend and preprocessing in the producing node's identity.

The recipe tests will verify the active dependency graph, current standard publication and node identities, and reuse of the existing detection and pose artifacts by downstream code. Empty frames and multiple detections require one shared frame and row axis. A changed pose model, weight, or decode setting requires a distinct cache key. The ECPose runtime and full-scene qualification depend on the official adapter and the new downstream route.

## Test design

- Reuse the fixed-cohort scorer's matching, missing-keypoint coverage and checksum tests. The comparison-specific CPU tests check the serialized row contract with two distinguishable people and an empty frame. Duplicate, missing or extra frame keys and a candidate from a different manifest must fail before scoring.
- Check archived row alignment with detection source IDs that begin at a nonzero offset. Pose features with a different source row or frame boundary must fail even when the box and pose counts agree. The real-data run also checks media and dependency pins for all 24 videos and uses the original source frame numbers.
- Run these small unit fixtures without checkpoints or a GPU. Report test counts and durations from pytest. Record the real-data coverage and artifact hashes from the generated scoring receipts.
