# EdgeCrafter models for Meiji 4-camera player evaluation

| Model ID | Task training and transfer | Model zoo parameters | Released bytes |
|---|---|---:|---:|
| `ecdet-l-coco` | COCO | 33 M | 132,351,848 |
| `ecdet-x-coco` | COCO | 49 M | 200,304,616 |
| `ecdet-l-o365` | Objects365 pretraining, then COCO detection | 33 M | 132,405,122 |
| `ecdet-x-o365` | Objects365 pretraining, then COCO detection | 49 M | 200,357,826 |
| `ecpose-l-o365` | Objects365 detection transfer, then COCO keypoints | 34 M | 138,759,372 |
| `ecpose-x-o365` | Objects365 detection transfer, then COCO keypoints | 51 M | 206,712,140 |

The COCO detection weights use the `edgecrafterv1` release. The O365 variants use the `o3652coco` files in `edgecrafterv1_o365`, with a COCO head. The paper describes distillation of the L/X detectors on ImageNet-1K and COCO images using a DINOv3-B teacher adapted on COCO detection. It reports 31 M parameters for ECDet-L in Table 2. The model zoo reports 33 M. The release asset size, SHA-256, matching COCO configuration and pinned source revision are in [releases.py](../../../src/submodules/models/ecpose/releases.py). [Model zoo](https://github.com/Intellindust-AI-Lab/EdgeCrafter/blob/d8752b75c9beaecfcdd0e5cd207bae3b6946ae23/README.md#-model-zoo) · [Paper](https://arxiv.org/html/2603.18739v2) · [COCO release](https://github.com/capsule2077/edgecrafter/releases/tag/edgecrafterv1) · [Objects365-to-COCO release](https://github.com/capsule2077/edgecrafter/releases/tag/edgecrafterv1_o365)

The official release assets are public. The pinned EdgeCrafter source license grants non-commercial benchmarking and requires a separate commercial license for commercial production. It permits non-commercial redistribution under its attribution, license-copy and modification-notice terms and preserves applicable third-party terms. The release host repository has a separate Apache-2.0 notice. This evaluation uses the current source license's non-commercial terms. Confirm the terms applicable to each artifact and its third-party material before commercial deployment or redistribution. [Pinned EdgeCrafter license](https://github.com/Intellindust-AI-Lab/EdgeCrafter/blob/d8752b75c9beaecfcdd0e5cd207bae3b6946ae23/LICENSE.md) · [Release host notice](https://github.com/capsule2077/edgecrafter/blob/edgecrafterv1_o365/LICENSE)

## Image and output contracts

- Convert native BGR to RGB, resize to `640×640` by PIL bilinear interpolation with antialiasing, and use ImageNet mean and standard deviation. The resize stretches both axes independently. Decode normalized x with native width and y with native height. Use a single full-image forward and direct top-k decoding. [Official ECDet preprocessing](https://github.com/Intellindust-AI-Lab/EdgeCrafter/blob/d8752b75c9beaecfcdd0e5cd207bae3b6946ae23/ecdetseg/tools/inference/torch_inf.py) · [Official ECPose preprocessing](https://github.com/Intellindust-AI-Lab/EdgeCrafter/blob/d8752b75c9beaecfcdd0e5cd207bae3b6946ae23/ecpose/tools/inference/torch_inf.py)
- ECDet emits `300×80` class logits and `300×4` normalized `cxcywh` boxes. The official flattened sigmoid top-300 uses class 0 for person in deploy mode. Clip person boxes to the image and require finite positive area. [Official ECDet postprocessing](https://github.com/Intellindust-AI-Lab/EdgeCrafter/blob/d8752b75c9beaecfcdd0e5cd207bae3b6946ae23/ecdetseg/engine/edgecrafter/postprocessor.py)
- ECPose emits `60×2` class logits and `60×17×2` normalized COCO17 coordinates. Use the official flattened sigmoid top-60 and COCO person class 1. Coordinates and score travel together by query index. A joint's `1` marks an available coordinate and `0` represents a missing joint stored as `[0,0,0]`. The head learns one score per person. [Official ECPose head](https://github.com/Intellindust-AI-Lab/EdgeCrafter/blob/d8752b75c9beaecfcdd0e5cd207bae3b6946ae23/ecpose/engine/edgecrafter/detrpose_transformer.py#L749) · [Official ECPose postprocessing](https://github.com/Intellindust-AI-Lab/EdgeCrafter/blob/d8752b75c9beaecfcdd0e5cd207bae3b6946ae23/ecpose/engine/edgecrafter/detrpose_postprocesses.py)
- Select a joint-envelope margin explicitly for ECPose. The adapter uses the envelope of finite in-image joints with at least two joints and positive extent on both axes. It expands each side by the selected fraction of the corresponding extent, then clips to the native image. The comparison uses `0.10`, the explicit margin in the ECPose recipe smoke and the fixed pseudo-reference. This proxy measures agreement with joint-envelope boxes and affects the IoU matching used for pose scoring.
- Keep all positive-scoring person rows in the official top-k for the benchmark. The shared scorer computes recall at fixed score cutoffs. The production path can set its own explicit person cutoff.

## Reproduction

Run from the repository root. The cohort arguments pin the shared 6-clip evaluation and the command checks the original camera video checksums. GPU evaluation runs through the shared training queue with an exclusive GPU reservation.

```bash
export PYTHONPATH=.
runner=knowledge/runs/run-meiji-player-model-comparison/edgecrafter.py
manifest_sha=3dc496513380a39d4511ff7f8711f4192eababa5109f6a0506f88a60baa70580
output=outputs/player-model-comparison

.venv/bin/python "$runner" assets
```

The evaluation records `predictions.jsonl` with every fixed `FrameKey` and `metrics.json` with the cohort, backend, settings, artifact checksums and shared scorer results. Use `make_benchmark_adapter` in `edgecrafter.py` with the #2496 harness for fixed speed inputs. Each timed invocation receives already decoded native BGR images and measures resize and normalization, H2D, forward, decode and D2H through the harness.

`edgecrafter_run.sh` runs one model's full-cohort accuracy cases in an exclusive queue reservation. ECPose runs PT16 and TRT16. ECDet reuses its recorded PT16 accuracy and runs TRT16 with the batch-2 trace. The batch-2-traced TensorRT attempt writes both conversion receipts and the batch-1 full-cohort accuracy to `trt16-trace2` for each model. A successful engine build advances to scoring. The initial batch-1 trace and its outputs are recorded under `trt16`.

```bash
# Register this command in the shared training queue from the worker checkout.
bash knowledge/runs/run-meiji-player-model-comparison/edgecrafter_run.sh \
  ecpose-l-o365 "$manifest_sha"
```

For TensorRT, run export and build as separate Python processes in the training queue. The exporter traces the official deployed network under CUDA FP16 autocast, with an FP32 image binding and dynamic batch axis. It records the graph's FP16 and FP32 initializer and cast counts. Use an example batch of 2 to trace ECDet's anchor replication. A batch-1 trace dropped that branch and TensorRT warned about a batch extent of 1 in the decoder `GatherElements` at the maximum profile shape. Postprocessing uses FP32 and the same implementation for all backends. TensorRT uses a strongly typed graph and an explicit workspace and batch profile. `--engine` points to the profile to use during evaluation.

```bash
conversion=knowledge/runs/run-meiji-player-model-comparison/edgecrafter_trt.py
condition="$output/ecdet-l-coco/trt16-trace2"

.venv/bin/python "$conversion" export --model ecdet-l-coco \
  --engine "$condition/engine" --receipt "$condition/onnx-export.json"
.venv/bin/python "$conversion" build --max-batch 2 \
  --engine "$condition/engine" --receipt "$condition/engine-build.json"
.venv/bin/python "$runner" evaluate --model ecdet-l-coco --backend trt16 \
  --parity not_measured --manifest-sha256 "$manifest_sha" \
  --engine "$condition/engine" --output "$condition"
```

PT16 uses CUDA FP16 autocast with FP32 weights and input tensors. Following the 2026-11-25 decision, new accuracy runs require `--parity not_measured` and record `PT32未計測` in the metrics. All conditions use the same partial-label pseudo-reference for accuracy. The saved ECDet PT32 v1 results and existing parity reports remain historical evidence. The replacement TensorRT trace addresses the dynamic batch profile.

## Accuracy on the fixed cohort

The six queue jobs at `83507bdd57097ac5b37892b5b1d6db3af0666af2` produced eight new scores and six TensorRT engines. Including the four retained ECDet PT16 results, all 12 PT16/TRT16 conditions contain the same 6,844 frame-and-camera rows and 26,276 eligible boxes. The receipt records the per-condition metrics, prediction digests, queue logs and verified engine artifacts. The raw outputs are under `outputs/player-model-comparison/` in the EdgeCrafter worker checkout. [Accuracy receipts](edgecrafter-accuracy.json)

ECDet's PT16 recall at IoU 0.50 is 97.58–98.19%, and its TRT16 recall is 63.02–70.29%. ECPose-L's coverage-inclusive PCK@0.20 is 38.56% on PT16 and 36.53% on TRT16. ECPose-X reaches 33.08% and 6.91%, respectively. All conditions completed scoring. The large ECPose-X difference is measured against the pseudo-reference, and its cause remains open.

## Fixed-input speed cases

`edgecrafter_speed_run.sh` runs PT16 and TRT16 at batch 1 and 2 in separate processes. `edgecrafter_speed.py` verifies the #2496 manifest, timing protocol and harness revision. It uses the ordered set of 96 native BGR images, 30 warm-ups and three blocks of 100 iterations on a dedicated CUDA stream. TRT16 loads the engine from the full-cohort accuracy run. The shared harness records `parity_status=not_evaluated` with reason `PT32未計測`. The previous six speed jobs were cancelled before execution while the common timing contract was revised.

```bash
# Register this command in the shared training queue for each model.
bash knowledge/runs/run-meiji-player-model-comparison/edgecrafter_speed_run.sh \
  ecpose-l-o365 "$fixed_speed_inputs" 13e508cb78c46663fbe58ad8be03bcf9c96816a6
```

`fixed_speed_inputs` is the published baseline input directory. Receipts are written to `outputs/player-model-comparison/<model>/speed-13e508cb7/<backend>/speed-b<batch>.json`. See the baseline documentation for the input location and schedule. [Baseline inputs and protocol](../run-person-model-speed-baseline/README.md#固定入力と初回の成果物)

## Rescoring stored predictions

`edgecrafter_rescore.py` verifies the original prediction file against its digest, scores all cohort frames using the checked-out common scorer, and writes the revised result beside the original metrics. It records which aggregates changed and preserves the hash of the first result. The script uses the shared `prediction_rows.py` from the primary comparison branch. On a worker branch without that file, extract its pinned version before running the same command.

```bash
run=knowledge/runs/run-meiji-player-model-comparison
helper="$output/rescore-tools"
mkdir -p "$helper"
git show bdbbb2127f8a842ae007ae42b0518e00f77d9de2:"$run/prediction_rows.py" > "$helper/prediction_rows.py"
mapfile -t conditions < <(find "$output" -path '*/ecdet-*/*/metrics.json' | sort)
PYTHONPATH=.:"$helper" .venv/bin/python "$run/edgecrafter_rescore.py" \
  --cohort data/tennis_multiview/processed/meiji_4cam/benchmarks/multiview-player-evaluation-v1 \
  --manifest-sha256 "$manifest_sha" \
  --summary "$output/edgecrafter-rescore.json" "${conditions[@]}"
```

All nine previously measured ECDet conditions retained their numeric aggregates after the `0d4bf0990` equal-IoU scoring change. Each report covers 6,844 images and 26,276 eligible boxes. The receipt contains paths and hashes for the original predictions, updated accuracy reports and historical parity reports from the original diagnostic. [Rescoring receipt](edgecrafter-rescore.json)

On the same saved frames, the shared parity rules passed the two COCO-only PT16 variants. ECDet-L O365 PT16 exceeded the missing and extra fractions at 1.39% and 1.14%. ECDet-X O365 PT16 also fell below the IoU threshold with P05 0.9767. These historical comparisons cover all saved person rows from the full accuracy cohort. [Shared parity on the full cohort](edgecrafter-shared-parity.json)

## First TensorRT precision check

The original ECDet-L COCO TensorRT plan ran all 6,844 images at batch 1 on the RTX 5060 Ti. The initial diagnostic at score ≥0.25 matched 66.4% of the person rows in both directions against PT32. The matched IoU P05 was 0.884 and absolute score error P95 was 0.313. For original frames 0 and 2 of `clip_003/cam0`, ONNX Runtime CPU running the same exported graph matched 24 of 24 PT32 rows. The TensorRT plan matched 17 of 24 on those inputs. The dynamic batch warning and the batch-1 precision mismatch are separate observations.

## Diagnosis of the batch-2-traced plans

The diagnosis uses four published #2496 inputs. It selects one image per clip in descending order of the total person detections at score ≥0.25 across ECDet-L COCO, ECPose-L and ECPose-X PT16. The diagnostic records both single-image inputs and both image orders for each pair. It passes one contiguous preprocessed tensor per case to the PT16 forward and the TensorRT runner, then records each raw head's shape, dtype, finite range and NaN/Inf counts. The report compares outputs by source image ID and checks context strides against the PyTorch buffers. Decoded boxes use the same top-k and score settings as accuracy, with no NMS.

```bash
diagnose=knowledge/runs/run-meiji-player-model-comparison/edgecrafter_diagnose.py
baseline=knowledge/runs/run-person-model-speed-baseline

# Run the alternate interpreter on CPU.
PYTHONPATH=".:$baseline" OMP_NUM_THREADS=4 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  .venv/bin/python -B "$diagnose" --inputs "$fixed_speed_inputs" \
  --output outputs/player-model-comparison/trt16-diagnosis-v1 --onnx-cpu

# Submit this command through the training queue with resource all.
PYTHONPATH=".:$baseline" OMP_NUM_THREADS=4 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  .venv/bin/python -B "$diagnose" --inputs "$fixed_speed_inputs" \
  --output outputs/player-model-comparison/trt16-diagnosis-v1
```

On `clip_028/cam3` frame 513, ONNX Runtime CPU reproduced all 68 archived PT16 ECDet-L boxes and all 60 archived PT16 ECPose-X boxes at IoU ≥0.50. The same ONNX files yielded 30 and 11 matching boxes through their archived TRT16 plans. The gap on this image points to the TensorRT path. The differing compiler operation or runtime behavior remains under investigation. [ONNX CPU comparison](edgecrafter-trt-onnx-cpu.json)

## CPU integration check

On the adapter revision `bf9cf828f6508c48da90e220743e06b330be47ff`, the detached local job `20261124T113947Z_a94715bdcac9_edgecrafter-assets-and-cpu-smoke` exited successfully. All six downloaded weights matched the official release byte sizes and digests. A two-image CPU batch used the original frames 0 and 2 from `video_000/clip_003`, `cam0`. ECPose used margin `0.10` for this contract check. Each model wrote `outputs/player-model-comparison/<model_id>/smoke-pt32.json` in the worker worktree. [CPU result paths and checksums](edgecrafter-cpu-smoke.json)

| Model ID | Person rows at source frame 0 | Person rows at source frame 2 | COCO17 joints per row |
|---|---:|---:|---:|
| `ecdet-l-coco` | 90 | 91 | — |
| `ecdet-x-coco` | 88 | 88 | — |
| `ecdet-l-o365` | 72 | 70 | — |
| `ecdet-x-o365` | 69 | 69 | — |
| `ecpose-l-o365` | 59 | 59 | 17 |
| `ecpose-x-o365` | 60 | 60 | 17 |

## Test coverage

Four new CPU tests use small asymmetric image and model-output fixtures. They check native width and height, color channel order, the distinct person labels, selected-query order, positive-area boxes, aligned left and right wrist coordinates, the joint availability flag and the empty-result shape. The joint-envelope policy must be explicit. Existing `test_exact_backend_selection_never_loads_another_backend`, `test_forward_cannot_recover_a_failed_trt_call_with_pt`, `test_open_keeps_empty_images_and_rejects_changed_files` and `test_pose_reports_missing_person_and_missing_joint_in_coverage_denominators` exercise the shared harness and scorer boundaries.

The parity tests use two rearranged detection rows, perturbations and missing joints on a single pose row, and two tiny JSONL files with different source frames. They check the pixel and score difference calculations, reject missing coverage and mismatched input identity, and reuse `test_build_time_is_separate_from_trt_load_and_inference` and `test_profile_bounds_include_tail_and_reject_oversize` for the conversion and engine boundaries.

The 2026-11-25 scope change adds five CPU cases. The queue entry-point tests substitute a command recorder for the GPU processes and check that ECPose requests PT16 and TRT16, ECDet requests TRT16, and the TensorRT conversion precedes scoring. The CLI cases reject PT32 and a missing parity setting. A parity receipt test rejects comparison when the candidate declares `PT32未計測`.

The speed entry-point tests check the four half-precision backend and batch combinations, ordered images, harness revision validation and the explicit parity argument. The accuracy wrapper tests check that a failed command or missing engine receipt is reported against the corresponding condition and preserves a successful result from the other backend.
