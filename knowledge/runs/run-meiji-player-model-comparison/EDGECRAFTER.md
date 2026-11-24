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
- Select a joint-envelope margin explicitly for ECPose. The adapter uses the envelope of finite in-image joints with at least two joints and positive extent on both axes. It expands each side by the selected fraction of the corresponding extent, then clips to the native image. `0.10` is the proposed operating point for comparison because the fixed pseudo-reference uses a 10% envelope margin. This proxy measures agreement with joint-envelope boxes and affects the IoU matching used for pose scoring.
- Keep all positive-scoring person rows in the official top-k for the benchmark. The shared scorer computes recall at fixed score cutoffs. The production path can set its own explicit person cutoff.

## Reproduction

Run from the repository root. The cohort arguments pin the shared 6-clip evaluation and the command checks the original camera video checksums. The smoke command runs one batch and writes a JSON receipt. GPU evaluation runs through the shared training queue with an exclusive GPU reservation.

```bash
export PYTHONPATH=.
runner=knowledge/runs/run-meiji-player-model-comparison/edgecrafter.py
manifest_sha=3dc496513380a39d4511ff7f8711f4192eababa5109f6a0506f88a60baa70580
output=outputs/player-model-comparison

.venv/bin/python "$runner" assets
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python "$runner" smoke \
  --model ecpose-l-o365 --backend pt32 --manifest-sha256 "$manifest_sha" \
  --pose-box-margin 0.10 --output "$output/ecpose-l-o365/smoke-pt32.json"

# GPU command for the shared training queue
.venv/bin/python "$runner" evaluate --model ecdet-l-coco --backend pt32 \
  --manifest-sha256 "$manifest_sha" --output "$output/ecdet-l-coco/pt32"
```

The evaluation records `predictions.jsonl` with every fixed `FrameKey` and `metrics.json` with the cohort, backend, settings, artifact checksums and shared scorer results. Use `make_benchmark_adapter` in `edgecrafter.py` with the #2496 harness for fixed speed inputs. Each timed invocation receives already decoded native BGR images and measures resize and normalization, H2D, forward, decode and D2H through the harness.

For TensorRT, run export and build as separate Python processes in the training queue. The exporter traces the official deployed network under CUDA FP16 autocast, with an FP32 image binding and dynamic batch axis. It records the graph's FP16 and FP32 initializer and cast counts. Postprocessing uses FP32 and the same implementation for all backends. TensorRT uses a strongly typed graph and an explicit workspace and batch profile. `--engine` points to the profile to use during evaluation.

```bash
conversion=knowledge/runs/run-meiji-player-model-comparison/edgecrafter_trt.py
condition="$output/ecdet-l-coco/trt16"

.venv/bin/python "$conversion" export --model ecdet-l-coco \
  --engine "$condition/engine" --receipt "$condition/onnx-export.json"
.venv/bin/python "$conversion" build --max-batch 16 \
  --engine "$condition/engine" --receipt "$condition/engine-build.json"
.venv/bin/python "$runner" evaluate --model ecdet-l-coco --backend trt16 \
  --manifest-sha256 "$manifest_sha" --engine "$condition/engine" --output "$condition"

.venv/bin/python knowledge/runs/run-meiji-player-model-comparison/edgecrafter_parity.py \
  --pt32 "$output/ecdet-l-coco/pt32" --candidate "$condition"
```

CUDA PT32 fixes matmul and cuDNN convolution to IEEE precision. PT16 uses CUDA FP16 autocast with FP32 weights and input tensors. The separate PT32 parity receipt aligns the exact same source frames and verifies prediction hashes and evaluation settings. It uses one-to-one IoU 0.50 matching for rows with score ≥0.25, then requires at least 98% matches in both directions, matched IoU P05 ≥0.90, and absolute score error P95 ≤0.025. Pose parity adds ≥98% joint availability agreement and paired joint error P95 ≤8 native pixels. All precision conditions use the same partial-label pseudo-reference for accuracy.

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
