# Meiji 4-camera player model comparison

This comparison uses the six clips, frame indices and SMPL-derived references from [the fixed player benchmark](../run-meiji-multiview-player-evaluation/README.md). The references are partial. Other people in each image have unknown label status, so player recall is available and precision and AP are N/A.

## DINO and ViTPose-H accuracy baseline

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

The archived DINO input is 750 by 1,333 pixels, batch 1, person confidence at least 0.30, and duplicate merging disabled. ViTPose-H uses 256 by 192 pixels, batch up to 8 crops before flip testing, flip testing enabled and a 1.2 box enlargement. Both models use TensorRT FP16 on the RTX 5060 Ti.

The PT16 queue run completed both accuracy conditions over the same 6,844 images and 26,276 eligible boxes. All frame keys and the prediction SHA-256 values match their receipts. The two paths retain identical box and score rows. Recall@0.50 is 96.69% and Recall@0.75 is 46.99%. DINO + ViTPose-H reaches coverage-inclusive mean OKS 0.76313, PCK@0.10 77.99% and joint coverage 98.24%. The PT32 accuracy job was cancelled under the 2026-11-25 scope decision. Previously measured PT32 candidate scores and parity records remain available. [Queue and prediction pins](comparison-evidence.json)

The PT16 reproduction command uses the shared GPU queue. `baseline_pytorch.py` checks both checkpoint digests, the DINO CUDA extension and source revision, the original media and the fixed scorer. It uses the earlier #2496 batch-1 receipt to identify assets and settings. Each image is detected once and the existing ViTPose crop path processes every accepted box. The run uses CUDA FP16 autocast, FP32 deformable attention in DINO, and IEEE precision for FP32 matmul and convolution. The CLI writes the detection-only and joint prediction and score artifacts with their hashes.

```bash
run=knowledge/runs/run-meiji-player-model-comparison
media=data/tennis_multiview/processed/meiji_4cam/benchmarks
evidence="$run/comparison-evidence.json"

# Enqueue this command with resource all.
speed_receipt_sha=$(jq -er \
  '.conditions[] | select(.model_id == "dino-swin-l-vitpose-h" and .backend == "pt16") | .historical_timing.sha256' "$evidence")
env PYTHONPATH=. OMP_NUM_THREADS=4 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  TENNIS_LAB_DINO_EXTENSION="$TENNIS_LAB_DINO_EXTENSION" \
  .venv/bin/python -B -u "$run/baseline_pytorch.py" \
  --backend pt16 --benchmark "$media/multiview-player-evaluation-v1" \
  --source "$media/multiview-v1" --output "$BASELINE_REPRO_OUTPUT" \
  --manifest-sha256 "$(jq -er '.evaluation.sha256' "$evidence")" \
  --scorer-commit "$(jq -er '.evaluation.scorer_commit' "$evidence")" \
  --speed-root "$PERSON_SPEED_ROOT" --external-assets "$PERSON_EXTERNAL_ASSETS" \
  --speed-manifest-sha256 "$(jq -er '.speed_input.sha256' "$evidence")" \
  --speed-receipt-sha256 "$speed_receipt_sha"
```

`PERSON_SPEED_ROOT` is #2496's initial `person-model-speed-baseline-v1` directory containing `inputs` and `results`. `PERSON_EXTERNAL_ASSETS` contains its pinned DINO checkout. The CUDA extension path must match the receipt. Use a fresh directory for `BASELINE_REPRO_OUTPUT`. The CLI accepts `--check-assets` for a CPU check of these pins before submitting the job.

## Accuracy and speed comparison

The [condition table](conditions.csv) contains 36 rows, with full-cohort accuracy and synchronized speed for 22 PT16/TRT16 conditions. Five additional PT32 accuracy results are historical. RF-DETR-2XL is outside the comparison under the user's license decision. The table includes small-player recall and recall at a common minimum score of 0.50. The main recall column uses each model's saved inference cutoff.

The two plots compare batch-1 stage wall P50 with Recall@0.50 for detection or coverage-inclusive mean OKS for detection and pose. The blue line connects the empirical Pareto frontier. The measured RT-DETRv4 TRT16 plans appear as crosses and were excluded from the frontier after their same-input diagnostic. A square denotes two PyTorch CPU threads. DINO, DINO + ViTPose-H and the repeated RF-DETR-L cases used four; EdgeCrafter and RT-DETRv4 used two. All were timed on the same 96 ordered native BGR images and RTX 5060 Ti with the synchronized boundary, 30 warm-ups and three blocks of 100 repetitions. Fine latency differences across thread settings do not establish a speed ranking under a matched host configuration. [Detection plot](pareto-detection.png) · [Detection and pose plot](pareto-detection_and_pose.png) · [Artifact and schedule hashes](comparison-evidence.json)

In matched CPU-thread settings, RF-DETR-L TRT16 took 26.43 ms/image P50 versus 94.79 for DINO TRT16. NVML peak increase was 142 versus 984 MiB. Recall@0.50 was 96.81% versus 96.67%, and Recall@0.75 was 42.09% versus 47.07%. RF-DETR-L is the fastest measured detection condition with recall close to DINO at IoU 0.50. The stricter localization result warrants checking its true predicted crops with ViTPose-H before a downstream substitution. [Matched RF-DETR-L measurements](../run-meiji-rfdetr-large-speed/comparison.csv)

RT-DETRv4-L PT16 reached 98.12% Recall@0.50 at 52.52 ms/image. ECDet-L O365 PT16 reached 98.19% at 61.46 ms/image. ECDet and RT-DETRv4 retain positive-scoring person rows from their postprocessing, while DINO and RF-DETR-L use minimum scores of 0.30 and 0.25, respectively. At a common minimum score of 0.50, recall was 95.31% for RT-DETRv4-L PT16 and 95.79% for ECDet-L O365 PT16. The ECDet-L COCO and RT-DETRv4-L PT16 P50 values both round to 52.52 ms/image. [EdgeCrafter scores](edgecrafter-accuracy.json) · [RT-DETRv4 measurements and diagnosis](rtdetrv4-results.json)

ECPose-L PT16 measured 55.11 ms/image and mean OKS 0.2908, compared with 266.30 ms/image and 0.7631 for DINO + ViTPose-H PT16. The measured ECPose-L TRT16 plan recorded 33.49 ms/image and 0.2751, compared with 148.58 ms/image and 0.7631 for the joint TRT16 baseline. For ECPose-L PT16 small-player Recall@0.50 was 10.99%, compared with 95.17% for the joint PT16 baseline. This cohort shows a substantial quality cost for the measured ECPose paths. SMPL placement uses observations from the current models, so this comparison measures agreement with the derived reference. [Scores and timing](conditions.csv) · [Reference construction and quality mask](../run-meiji-multiview-player-evaluation/README.md)

ECDet TRT16 recorded 63.02–70.29% Recall@0.50, and ECPose-X TRT16 recorded mean OKS 0.0534. These EdgeCrafter timings describe the exact plans used for scoring. In a separate instrumented ECDet-L COCO plan, swapped inputs agreed through the sampled Gather and diverged at the final logits. The responsible operation or tactic in the scoring plans remains unconfirmed. RT-DETRv4-L/X TRT16 scored 0.00% and 0.03% Recall@0.50. Their diagnosis reproduced the output loss before shared postprocessing. All these results keep their measured speed and accuracy. [EdgeCrafter raw-head and batch-order diagnosis](EDGECRAFTER.md#diagnosis-of-the-batch-2-traced-plans) · [RT-DETRv4 same-input diagnosis](rtdetrv4.md#same-input-tensorrt-diagnosis)

For the current timing conditions, PT32 parity is explicitly unmeasured. Historical candidate parity reports remain pinned. The first ECDet-L COCO TRT16 plan scored 74.68% Recall@0.50 and failed the historical PT32 comparison. Its outputs and parity remain recorded in `historical_conditions`. The initial PT32 stage wall P50 was 289.97 ms/image for DINO and 590.33 ms/image for DINO + ViTPose-H. Those timings use the earlier synchronization boundary. The five RF-DETR-L and archived baseline results in the rescoring record retained every metric after the #2495 equal-IoU ordering change. [Historical timing](../run-person-model-speed-baseline/README.md) · [Rescoring record](accuracy-recheck.json) · [RF-DETR licensing](rfdetr.md)

## Joint DINO and ViTPose-H timing

The additional `suite joint` workload uses the synchronized #2497 harness merged at `59faa32a28d2f54e6492abdba2429846fa0397e3`. It runs PT16 and TRT16 with image batches 1 and 2 on the original ordered 96 BGR frames, using 30 warm-ups and 3 blocks of 100 measured repetitions. Each of the four conditions gets a fresh process. The report includes the full interval from DINO preprocessing through ViTPose decode and final device synchronization. The pinned scope decision sets parity to `PT32未計測`.

DINO uses the standard full-frame settings, resizing with short side 800 and long-side cap 1,333 to produce 750×1,333 inputs. Every person row at score ≥0.30 goes to ViTPose in detection order. Duplicate merging is disabled. The standard square crop uses 1.2 times the box's longer side. ViTPose-H uses 256×192 input and the standard flip test. The inference batch is up to 8 person crops before flipping, as in the initial joint run and the archived TensorRT recipe. Both stages load the initial run's weights and TRT plans. The measured workload generates its own crops from the same backend's DINO output each time. [Initial baseline and fixed inputs](../run-person-model-speed-baseline/README.md)

All four processes exited successfully with 300 measured repetitions over the same 96 images. The image order for each batch size matched the fixed timing schedule. The checkpoint and PT16 extension hashes matched the full-cohort accuracy run. The TRT16 plan hashes matched the archived accuracy artifacts and the initial timing run.

| Backend | Image batch | Stage wall P50 (ms/image) | Stage wall P95 (ms/image) | Images/s | NVML peak increase (MiB) |
|---|---:|---:|---:|---:|---:|
| PT16 | 1 | 266.30 | 359.32 | 3.51 | 6,484 |
| PT16 | 2 | 270.61 | 352.13 | 3.52 | 7,618 |
| TRT16 | 1 | 148.58 | 194.14 | 6.41 | 2,292 |
| TRT16 | 2 | 145.65 | 182.93 | 6.62 | 2,284 |

At batch 2, per-image percentiles divide each batch duration by two. Throughput is total images divided by total measured wall time. NVML peak increase is relative to the pre-load reading. The [case summaries](joint-speed/cases.json) retain the model-only percentiles, environment, asset hashes and report digests. The [case table](joint-speed/cases.csv) and per-repetition timing CSVs for [PT16](joint-speed/measurements/dino-vitpose-pt16.csv) and [TRT16](joint-speed/measurements/dino-vitpose-trt16.csv) were generated from measurement commit `65d6349a5bb1b6c4f88f4fbbc894cc8e6f5b480a`. The [queue receipt and batch-1 report pins](comparison-evidence.json) connect these data to the comparison table.

Run this command through the shared GPU queue with `--resource all`. Set `JOINT_SPEED_OUTPUT` to a fresh directory. The [baseline setup](../run-person-model-speed-baseline/README.md#再現) describes the pinned input and asset paths.

```bash
baseline=knowledge/runs/run-person-model-speed-baseline
env PYTHONPATH=. OMP_NUM_THREADS=4 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  TENNIS_LAB_DINO_EXTENSION="$TENNIS_LAB_DINO_EXTENSION" \
  .venv/bin/python -B -u "$baseline/run.py" --output "$JOINT_SPEED_OUTPUT" \
  --input-dir "$PERSON_FIXED_INPUTS" --trt-root "$PERSON_TRT_ROOT" \
  --external-assets "$PERSON_EXTERNAL_ASSETS" suite joint
```

`invocation-joint.json` records the measurement HEAD and inputs. `attempts-joint.json` tracks each subprocess. The raw `results/dino-vitpose/<backend>-b<batch>/report.json` contains per-repetition timing samples, model hashes, environment and VRAM. Use the same baseline bundle's record command to verify the four reports and reproduce the tables. Set `JOINT_RECORD_OUTPUT` to a fresh directory.

```bash
.venv/bin/python -B knowledge/runs/run-person-model-speed-baseline/record.py \
  "$JOINT_SPEED_OUTPUT" "$JOINT_RECORD_OUTPUT" --input-dir "$PERSON_FIXED_INPUTS" --group joint
```

## Representative frames

Each grid uses the preselected cam0 frame from one representative clip and the same source pixels for the SMPL-derived reference, DINO + ViTPose-H TRT16, RF-DETR-L TRT16 and ECDet-L O365 PT32. Model overlays show detections with score at least 0.30. Unlabelled people can appear in model panels. [Six image paths, counts and hashes](representative-cam0-index.json)

A second grid for each of the same six images replaces the RF-DETR-L panel with a PT32 CPU preview from the official ECPose-L O365 checkpoint. ECPose retained one row in five previews and two rows in one preview at the display cutoff. The reference has four eligible boxes in each image. [Preview frame keys, weights and hashes](representative-cam0-ecpose-preview-index.json)

The GPU comparison uses those same six cam0 source frames, with the pseudo-reference, DINO + ViTPose-H TRT16, RF-DETR-L TRT16 and ECPose-L PT16. At the shared display cutoff of 0.30, ECPose-L shows one box in five frames and two in the remaining frame. The checked images are in `outputs/player-model-comparison/representative-cam0-gpu-candidates`. The versioned index provides the original video, prediction and overlay SHA-256 values. [GPU comparison frame index](representative-cam0-gpu-candidates-index.json)

```bash
run=knowledge/runs/run-meiji-player-model-comparison
artifacts="$(dirname "$(git rev-parse --path-format=absolute --git-common-dir)")/.worktree-outputs"
media=data/tennis_multiview/processed/meiji_4cam/benchmarks
PYTHONPATH=. .venv/bin/python -B "$run/compare_conditions.py" \
  --artifacts "$artifacts" --evidence "$run/comparison-evidence.json" \
  --output outputs/player-model-comparison/comparison-synchronized-v2
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

OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONPATH=. \
  .venv/bin/python -B "$run/render_comparison.py" \
  --artifacts "$artifacts" --evidence "$run/comparison-evidence.json" \
  --reference "$media/multiview-player-evaluation-v1" --source "$media/multiview-v1" \
  --model dino-swin-l-vitpose-h/trt16 --model rfdetr-l/trt16 --model ecpose-l-o365/pt16 \
  --output outputs/player-model-comparison/representative-cam0-gpu-candidates
```

## ECPose recipe contract

The [ECPose configuration and output contract](../../../src/tennis_scene/pipeline/ECPOSE.md) uses the [shared recipe design](https://github.com/Motoki0705/tennis-lab/issues/2491#issuecomment-7129950015). The tests run the real decoder with a fake model and verify rows through the component store.

The published ECPose-L checkpoint ran through the pipeline component on CPU for the first three consecutive source frames of `clip_003/cam0`. Batch size 2 exercised the partial final batch. With confidence 0.30 and box margin 0.10, each frame produced two positive-area boxes with corresponding COCO17 rows. [Component execution receipt](ecpose-recipe-smoke.json)

The ECPose-L TRT16 plan and B=1/B=2 smoke receipts were verified with the same engine reader used by the recipe. On `clip_003/cam0` source frame 0, the plan produced two detections at score 0.30 or above in B=1 and zero in B=2. The subsequent EdgeCrafter jobs traced ONNX with batch 2, scored the full cohort and timed the exact resulting plans. Those results use the 2026-11-25 PT32 scope decision. They describe the candidate plans in the comparison above. [Initial recipe smoke](ecpose-trt16-smoke.json) · [Retraced plan diagnosis](EDGECRAFTER.md#diagnosis-of-the-batch-2-traced-plans)

The recipe tests use the existing pipeline runtime and component store on CPU. A fake network produces two differently placed people, an empty frame and a trailing partial batch. The actual decoder and compatibility components must preserve source-pixel coordinates, the shared frame boundaries and detection rows through persisted artifacts, with one inference pass. The active graph must keep appearance, tracking and body recovery on their existing input contracts. The tests reject a different recipe or decoding policy in the cache and verify prior standard publication and node identities with the archived FP32 fixture. The small fixture requires no weights or GPU. Run the component tests and the archived-cache integration test separately and measure their durations.

## Downstream constraints

ECPose is an opt-in scene recipe and can be used for 2D comparison. The #1739 3D path explicitly rejects its artifacts. The first stage in #2449 checks `detector=dino`, requires an empty detection dependency map, and requires a pose dependency named `detections` pointing to that exact DINO artifact. The scene assembly stage also verifies the exact observation references and pose dependency. The ECPose views depend on a shared producer. Adopting ECPose for #1739 will require an approved change to the downstream readers that verifies both views against the ECPose producer. [3D candidate input gates](https://github.com/Motoki0705/tennis-lab/blob/bb9f36ccde8f6f89b7a3e5a4ab5f6a7288424ada/src/tennis_scene/multiview_pose/inputs.py#L195) · [Scene assembly input gates](https://github.com/Motoki0705/tennis-lab/blob/4d2773a05923916e84123f78a38da3665d826af4/src/tennis_scene/multiview_scene/inputs.py#L353)

When that route is enabled, validate it with the same v1 camera matrices, court-side decision and ball input. Check two players per side, measured 3D joint coverage, track and source-row lineage, and valid body placement against the DINO control.

## Test design

- Reuse the fixed-cohort scorer's matching, missing-keypoint coverage and checksum tests. The comparison-specific CPU tests check the serialized row contract with two distinguishable people and an empty frame. Duplicate, missing or extra frame keys and a candidate from a different manifest must fail before scoring.
- Check archived row alignment with detection source IDs that begin at a nonzero offset. Pose features with a different source row or frame boundary must fail even when the box and pose counts agree. The real-data run also checks media and dependency pins for all 24 videos and uses the original source frame numbers.
- Run these small unit fixtures without checkpoints or a GPU. Report test counts and durations from pytest. Record the real-data coverage and artifact hashes from the generated scoring receipts.
- For the comparison, use CPU fixtures with an equal-latency tie, an equal-quality tie, a failed parity check and an incomplete condition. A faster joint model must stay in the detector-plus-pose workload. Measured TensorRT diagnostic points stay visible in the table and plot and are excluded from the frontier. A timing report with raw IDs can be checked against the schedule when it lacks an embedded order digest. Reject a conflicting digest or sample order. For RT-DETRv4, the accuracy receipt pins the scorer verified against the job's execution commit, since its raw report has no scorer field. Representative panels use the previously chosen frame per clip and share the same unmodified source pixels and display cutoff.
- The PT baseline fixture checks that two asymmetric detection boxes create the original square crops in the same row order, then checks a missing joint and an empty image. Existing ViTPose frame-sequence and person feature tests cover crop batching and UDP decode. The six official CPU preview frames exercise the render path with actual ECPose outputs.
- The joint timing fixture follows all 12 fake detection rows through an eight-crop batch and a partial batch, including an image with no rows, and checks restored pixel coordinates and order. The baseline suite fixture checks that the joint group schedules only the four PT16/TRT16 image cases.
