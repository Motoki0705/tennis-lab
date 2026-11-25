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

## Accuracy and speed snapshot

The [condition table](conditions.csv) covers the 30 candidate and six baseline conditions. Full-cohort results include RF-DETR-L at PT32/PT16/TRT16 and the four ECDet weights at PT32/PT16. ECDet-L O365 has 98.19% Recall@0.50 in both PyTorch conditions. The ECDet-L COCO TRT16 attempt has 74.68% Recall@0.50 and failed PT32 parity. At score 0.25, only 66.40% of boxes matched in both directions and absolute score error P95 was 0.313. The three RF-DETR-2XL rows record the user's license exclusion. The table keeps each model's inference-time cutoff and includes Recall@0.50 with a common extra cutoff of score 0.50.

Current speed uses #2497's synchronized harness on 96 ordered native BGR images. All image IDs, native shapes and file hashes were checked against the fixed evaluation cohort. At image batch 1, DINO PT16 measured stage wall P50/P95 165.69/170.71 ms/image and 2,406 MiB of NVML peak increase. DINO TRT16 measured 94.79/101.53 ms/image and 984 MiB with the same plan used by the archived accuracy condition. Both report `PT32未計測` for parity. The 22 synchronized baseline cases include DINO, standalone ViTPose-H and body recovery. The joint DINO + ViTPose-H workloads remain pending for PT16 and TRT16. [Synchronized baseline records](../run-person-model-speed-low-precision/README.md)

The initial speed and parity results remain in the historical receipts. The initial PT32 stage wall P50 was 289.97 ms/image for DINO and 590.33 ms/image for DINO + ViTPose-H. The synchronized harness measures through device synchronization after the adapter returns. The provisional comparison uses timing under that boundary and records the earlier receipt hashes for traceability. [Initial baseline records](../run-person-model-speed-baseline/README.md)

Pareto assessment minimizes batch-1 stage wall P50 and maximizes Recall@0.50 for detection or coverage-inclusive mean OKS for detection plus pose. So far the two DINO detection conditions have both a fixed-cohort score and a synchronized speed result. Candidate speed runs and the joint baseline are pending. The comparison command can render a plot per workload once the comparisons are ready. [Artifact, manifest and timing-schedule pins](comparison-evidence.json)

The [RF-DETR-L accuracy record](../run-meiji-rfdetr-large-accuracy/results.json) contains the complete accuracy, candidate-versus-PT32 parity and box-margin sensitivity results. [RF-DETR licensing and reproduction](rfdetr.md) · [EdgeCrafter evidence](EDGECRAFTER.md) · [RT-DETRv4 evidence](rtdetrv4.md)

The five measured baseline and RF-DETR-L conditions were rescored from their pinned predictions with the #2495 equal-IoU ordering rule. Every metric and stratum was identical to its previous value. The [rescoring receipt](accuracy-recheck.json) identifies the scorer commit and the fresh full-score files.

## Joint DINO and ViTPose-H timing

The additional `suite joint` workload uses the synchronized #2497 harness merged at `59faa32a28d2f54e6492abdba2429846fa0397e3`. It runs PT16 and TRT16 with image batches 1 and 2 on the original ordered 96 BGR frames, using 30 warm-ups and 3 blocks of 100 measured repetitions. Each of the four conditions gets a fresh process. The report includes the full interval from DINO preprocessing through ViTPose decode and final device synchronization. The pinned scope decision sets parity to `PT32未計測`.

DINO uses the standard full-frame settings, resizing with short side 800 and long-side cap 1,333 to produce 750×1,333 inputs. Every person row at score ≥0.30 goes to ViTPose in detection order. Duplicate merging is disabled. The standard square crop uses 1.2 times the box's longer side. ViTPose-H uses 256×192 input and the standard flip test. The inference batch is up to 8 person crops before flipping, as in the initial joint run and the archived TensorRT recipe. Both stages load the initial run's weights and TRT plans. The measured workload generates its own crops from the same backend's DINO output each time. [Initial baseline and fixed inputs](../run-person-model-speed-baseline/README.md)

Run this command through the shared GPU queue with `--resource all`. Set `JOINT_SPEED_OUTPUT` to a fresh directory. The [baseline setup](../run-person-model-speed-baseline/README.md#再現) describes the pinned input and asset paths.

```bash
baseline=knowledge/runs/run-person-model-speed-baseline
env PYTHONPATH=. OMP_NUM_THREADS=4 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  TENNIS_LAB_DINO_EXTENSION="$TENNIS_LAB_DINO_EXTENSION" \
  .venv/bin/python -B -u "$baseline/run.py" --output "$JOINT_SPEED_OUTPUT" \
  --input-dir "$PERSON_FIXED_INPUTS" --trt-root "$PERSON_TRT_ROOT" \
  --external-assets "$PERSON_EXTERNAL_ASSETS" suite joint
```

`invocation-joint.json` records the measurement HEAD and inputs. `attempts-joint.json` tracks each subprocess. The raw `results/dino-vitpose/<backend>-b<batch>/report.json` contains per-repetition timing samples, model hashes, environment and VRAM. The same baseline bundle's `record.py --group joint` can verify and summarize this subset of reports.

## Representative frames

Each grid uses the preselected cam0 frame from one representative clip and the same source pixels for the SMPL-derived reference, DINO + ViTPose-H TRT16, RF-DETR-L TRT16 and ECDet-L O365 PT32. Model overlays show detections with score at least 0.30. Unlabelled people can appear in model panels. [Six image paths, counts and hashes](representative-cam0-index.json)

A second grid for each of the same six images replaces the RF-DETR-L panel with a PT32 CPU preview from the official ECPose-L O365 checkpoint. ECPose retained one row in five previews and two rows in one preview at the display cutoff. The reference has four eligible boxes in each image. Full-cohort GPU scores and synchronized video remain pending. [Preview frame keys, weights and hashes](representative-cam0-ecpose-preview-index.json)

```bash
run=knowledge/runs/run-meiji-player-model-comparison
artifacts="$(dirname "$(git rev-parse --path-format=absolute --git-common-dir)")/.worktree-outputs"
media=data/tennis_multiview/processed/meiji_4cam/benchmarks
PYTHONPATH=. .venv/bin/python -B "$run/compare_conditions.py" \
  --artifacts "$artifacts" --evidence "$run/comparison-evidence.json" \
  --output outputs/player-model-comparison/comparison-synchronized --tables-only
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

## Downstream constraints

ECPose is an opt-in scene recipe and can be used for 2D comparison. The #1739 3D path explicitly rejects its artifacts. The first stage in #2449 checks `detector=dino`, requires an empty detection dependency map, and requires a pose dependency named `detections` pointing to that exact DINO artifact. The scene assembly stage also verifies the exact observation references and pose dependency. The ECPose views depend on a shared producer. Adopting ECPose for #1739 will require an approved change to the downstream readers that verifies both views against the ECPose producer. [3D candidate input gates](https://github.com/Motoki0705/tennis-lab/blob/bb9f36ccde8f6f89b7a3e5a4ab5f6a7288424ada/src/tennis_scene/multiview_pose/inputs.py#L195) · [Scene assembly input gates](https://github.com/Motoki0705/tennis-lab/blob/4d2773a05923916e84123f78a38da3665d826af4/src/tennis_scene/multiview_scene/inputs.py#L353)

When that route is enabled, validate it with the same v1 camera matrices, court-side decision and ball input. Check two players per side, measured 3D joint coverage, track and source-row lineage, and valid body placement against the DINO control.

## Test design

- Reuse the fixed-cohort scorer's matching, missing-keypoint coverage and checksum tests. The comparison-specific CPU tests check the serialized row contract with two distinguishable people and an empty frame. Duplicate, missing or extra frame keys and a candidate from a different manifest must fail before scoring.
- Check archived row alignment with detection source IDs that begin at a nonzero offset. Pose features with a different source row or frame boundary must fail even when the box and pose counts agree. The real-data run also checks media and dependency pins for all 24 videos and uses the original source frame numbers.
- Run these small unit fixtures without checkpoints or a GPU. Report test counts and durations from pytest. Record the real-data coverage and artifact hashes from the generated scoring receipts.
- For the comparison, use CPU fixtures with an equal-latency tie, an equal-quality tie, a failed parity check and an incomplete condition. A faster joint model must stay in the detector-plus-pose workload. Validate timing reports against the pinned order of native images, batch size and common protocol. Representative panels use the previously chosen frame per clip and share the same unmodified source pixels and display cutoff.
- The PT baseline fixture checks that two asymmetric detection boxes create the original square crops in the same row order, then checks a missing joint and an empty image. Existing ViTPose frame-sequence and person feature tests cover crop batching and UDP decode. The six official CPU preview frames exercise the render path with actual ECPose outputs.
- The joint timing fixture follows all 12 fake detection rows through an eight-crop batch and a partial batch, including an image with no rows, and checks restored pixel coordinates and order. The baseline suite fixture checks that the joint group schedules only the four PT16/TRT16 image cases.
