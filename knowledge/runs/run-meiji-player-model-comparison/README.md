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

The archived DINO input is 750 by 1,333 pixels, batch 1, person confidence at least 0.30, and duplicate merging disabled. ViTPose-H uses 256 by 192 pixels, batch up to 8 crops before flip testing, flip testing enabled and a 1.2 box enlargement. Both models use TensorRT FP16 on the RTX 5060 Ti. PT32 and PT16 full-cohort accuracy conditions are pending.

To score PT32 and PT16, run `baseline_pytorch.py` once per backend through the shared GPU queue. It checks the #2496 batch-1 receipt, CUDA extension and checkpoint digests, DINO source revision, original media and the fixed scorer. Each image is detected once, all accepted boxes are passed to the existing ViTPose crop path, and the same detection rows feed both accuracy conditions. Both jobs disable TF32. The PT16 job uses CUDA autocast with FP32 deformable attention in DINO. Accuracy files are written to the backend directories with their prediction hashes.

```bash
run=knowledge/runs/run-meiji-player-model-comparison
media=data/tennis_multiview/processed/meiji_4cam/benchmarks
evidence="$run/comparison-evidence.json"

# Repeat for backend=pt16. Enqueue this command with resource all.
backend=pt32
speed_receipt_sha=$(jq -er --arg backend "$backend" \
  '.conditions[] | select(.model_id == "dino-swin-l-vitpose-h" and .backend == $backend) | .timing.sha256' "$evidence")
env PYTHONPATH=. OMP_NUM_THREADS=4 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  TENNIS_LAB_DINO_EXTENSION="$TENNIS_LAB_DINO_EXTENSION" \
  .venv/bin/python -B -u "$run/baseline_pytorch.py" \
  --backend "$backend" --benchmark "$media/multiview-player-evaluation-v1" \
  --source "$media/multiview-v1" --output outputs/player-model-comparison \
  --manifest-sha256 "$(jq -er '.evaluation.sha256' "$evidence")" \
  --scorer-commit "$(jq -er '.evaluation.scorer_commit' "$evidence")" \
  --speed-root "$PERSON_SPEED_ROOT" --external-assets "$PERSON_EXTERNAL_ASSETS" \
  --speed-manifest-sha256 "$(jq -er '.speed_input.sha256' "$evidence")" \
  --speed-receipt-sha256 "$speed_receipt_sha"
```

`PERSON_SPEED_ROOT` contains #2496's `inputs` and `results` directories. `PERSON_EXTERNAL_ASSETS` contains its pinned DINO checkout. The CUDA extension path must match the receipt. The CLI accepts `--check-assets` for a CPU check of these pins before submitting the job.

## Accuracy and speed snapshot

The [condition table](conditions.csv) covers the 30 candidate and six baseline conditions. Full-cohort results include RF-DETR-L at PT32/PT16/TRT16 and the four ECDet weights at PT32/PT16. ECDet-L O365 has 98.19% Recall@0.50 in both PyTorch conditions. The ECDet-L COCO TRT16 attempt has 74.68% Recall@0.50 and failed PT32 parity. At score 0.25, only 66.40% of boxes matched in both directions and absolute score error P95 was 0.313. The three RF-DETR-2XL rows record the user's license exclusion. The table keeps each model's inference-time cutoff and includes Recall@0.50 with a common extra cutoff of score 0.50.

The #2496 speed manifest has 96 ordered native BGR images. All image IDs, native shapes and file hashes were checked against the fixed evaluation cohort. The speed record now covers all six batch-1 DINO and DINO + ViTPose-H conditions. DINO PT32 and PT16 measured median stage wall times of 289.97 ms/image and 167.68 ms/image, with sampled NVML peak increases of 2,309 MiB and 2,406 MiB. The joint PT32 case measured 590.33 ms/image (P95 890.03) and 4,730 MiB. DINO TRT16 and joint PT16/TRT16 failed the shared PT32 parity gate. The joint PT16 replay over 96 IDs had two missing and two extra detection rows, with a matched joint error of 0.125 px P95 and 30.64 px maximum. Their speed reports have no eligible timing aggregates. Batch 2 has the same pass/fail split and uses the separate `*-b2` reports in the #2496 output tree.

Pareto assessment minimizes batch-1 stage wall P50 and maximizes Recall@0.50 for detection or coverage-inclusive mean OKS for detection plus pose. At this snapshot, paired conditions with qualifying speed and accuracy are still pending. The comparison command produces a plot per workload as qualifying points arrive. [Artifact and manifest pins](comparison-evidence.json)

The [RF-DETR-L accuracy record](../run-meiji-rfdetr-large-accuracy/results.json) contains the complete accuracy, candidate-versus-PT32 parity and box-margin sensitivity results. [RF-DETR licensing and reproduction](rfdetr.md) · [EdgeCrafter evidence](EDGECRAFTER.md) · [RT-DETRv4 evidence](rtdetrv4.md)

The five measured baseline and RF-DETR-L conditions were rescored from their pinned predictions with the #2495 equal-IoU ordering rule. Every metric and stratum was identical to its previous value. The [rescoring receipt](accuracy-recheck.json) identifies the scorer commit and the fresh full-score files.

## Representative frames

Each grid uses the preselected cam0 frame from one representative clip and the same source pixels for the SMPL-derived reference, DINO + ViTPose-H TRT16, RF-DETR-L TRT16 and ECDet-L O365 PT32. Model overlays show detections with score at least 0.30. Unlabelled people can appear in model panels. [Six image paths, counts and hashes](representative-cam0-index.json)

A second grid for each of the same six images replaces the RF-DETR-L panel with a PT32 CPU preview from the official ECPose-L O365 checkpoint. ECPose retained one row in five previews and two rows in one preview at the display cutoff. The reference has four eligible boxes in each image. Full-cohort GPU scores and synchronized video remain pending. [Preview frame keys, weights and hashes](representative-cam0-ecpose-preview-index.json)

```bash
run=knowledge/runs/run-meiji-player-model-comparison
artifacts="$(dirname "$(git rev-parse --path-format=absolute --git-common-dir)")/.worktree-outputs"
media=data/tennis_multiview/processed/meiji_4cam/benchmarks
PYTHONPATH=. .venv/bin/python -B "$run/compare_conditions.py" \
  --artifacts "$artifacts" --evidence "$run/comparison-evidence.json" \
  --output outputs/player-model-comparison/comparison
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONPATH=. \
  .venv/bin/python -B "$run/render_comparison.py" \
  --artifacts "$artifacts" --evidence "$run/comparison-evidence.json" \
  --reference "$media/multiview-player-evaluation-v1" --source "$media/multiview-v1" \
  --model dino-swin-l-vitpose-h/trt16 --model rfdetr-l/trt16 --model ecdet-l-o365/pt32 \
  --output outputs/player-model-comparison/representative-cam0

OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=3 MKL_NUM_THREADS=2 PYTHONPATH=. \
  .venv/bin/python -B "$run/render_comparison.py" \
  --artifacts "$artifacts" --evidence "$run/comparison-evidence.json" \
  --reference "$media/multiview-player-evaluation-v1" --source "$media/multiview-v1" \
  --model dino-swin-l-vitpose-h/trt16 --model ecdet-l-o365/pt32 \
  --preview-ecpose-assets "$EDGECRAFTER_ASSETS" \
  --output outputs/player-model-comparison/representative-cam0-ecpose-preview
```

## ECPose recipe contract

The [ECPose configuration and output contract](../../../src/tennis_scene/pipeline/ECPOSE.md) uses the [shared recipe design](https://github.com/Motoki0705/tennis-lab/issues/2491#issuecomment-7129950015). The tests run the real decoder with a fake model and verify rows through the component store.

The published ECPose-L checkpoint ran through the pipeline component on CPU for the first three consecutive source frames of `clip_003/cam0`. Batch size 2 exercised the partial final batch. With confidence 0.30 and box margin 0.10, each frame produced two positive-area boxes with corresponding COCO17 rows. [Component execution receipt](ecpose-recipe-smoke.json)

The ECPose-L TRT16 plan and B=1/B=2 smoke receipts were verified with the same engine reader used by the recipe. On `clip_003/cam0` source frame 0, the plan produced two detections at score 0.30 or above in B=1 and zero in B=2. The EdgeCrafter worker queued L/X jobs that retrace the ONNX graph with batch 2, check batch-one PT32 parity, and score the full cohort when the speed parity gate passes. This experiment is shared with the EdgeCrafter worker. [TRT16 plan and smoke receipts](ecpose-trt16-smoke.json)

The recipe tests use the existing pipeline runtime and component store on CPU. A fake network produces two differently placed people, an empty frame and a trailing partial batch. The actual decoder and compatibility components must preserve source-pixel coordinates, the shared frame boundaries and detection rows through persisted artifacts, with one inference pass. The active graph must keep appearance, tracking and body recovery on their existing input contracts. The tests reject a different recipe or decoding policy in the cache and verify prior standard publication and node identities with the archived FP32 fixture. The small fixture requires no weights or GPU. Run the component tests and the archived-cache integration test separately and measure their durations.

## Downstream multiview validation

The #1739 validation depends on the integrated #2449, #2469, #2461 and #2470 work. The first stage in #2449 checks `detector=dino`, requires an empty detection dependency map, and requires a pose dependency named `detections` pointing to that exact DINO artifact. The last stage also checks the exact observation references and pose dependency. The ECPose views bind to a common ECPose producer. The downstream readers need an explicit ECPose observation route that verifies both views against the same producer and keeps the existing DINO lineage check. [3D candidate input gates](https://github.com/Motoki0705/tennis-lab/blob/bb9f36ccde8f6f89b7a3e5a4ab5f6a7288424ada/src/tennis_scene/multiview_pose/inputs.py#L195) · [Scene assembly input gates](https://github.com/Motoki0705/tennis-lab/blob/4d2773a05923916e84123f78a38da3665d826af4/src/tennis_scene/multiview_scene/inputs.py#L353)

For the comparison, reuse the v1 camera matrices, court-side decision and ball input. Run the 3D candidates, tracking and placement on the original contiguous camera frames in a separate ECPose store. Check both players per side, measured 3D joint coverage, track/source row lineage and valid body placement against the DINO control. Use the pinned evaluation frame indices for the coverage comparison.

## Test design

- Reuse the fixed-cohort scorer's matching, missing-keypoint coverage and checksum tests. The comparison-specific CPU tests check the serialized row contract with two distinguishable people and an empty frame. Duplicate, missing or extra frame keys and a candidate from a different manifest must fail before scoring.
- Check archived row alignment with detection source IDs that begin at a nonzero offset. Pose features with a different source row or frame boundary must fail even when the box and pose counts agree. The real-data run also checks media and dependency pins for all 24 videos and uses the original source frame numbers.
- Run these small unit fixtures without checkpoints or a GPU. Report test counts and durations from pytest. Record the real-data coverage and artifact hashes from the generated scoring receipts.
- For the comparison, use CPU fixtures with an equal-latency tie, an equal-quality tie, a failed parity check and an incomplete condition. A faster joint model must stay in the detector-plus-pose workload. Validate timing reports against the pinned order of native images, batch size and common protocol. Representative panels use the previously chosen frame per clip and share the same unmodified source pixels and display cutoff.
- The PT baseline fixture checks that two asymmetric detection boxes create the original square crops in the same row order, then checks a missing joint and an empty image. Existing ViTPose frame-sequence and person feature tests cover crop batching and UDP decode. The six official CPU preview frames exercise the render path with actual ECPose outputs.
