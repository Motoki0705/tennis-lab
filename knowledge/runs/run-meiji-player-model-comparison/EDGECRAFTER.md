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
export PYTHONPATH=. PYTHONDONTWRITEBYTECODE=1
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

### Queue log retention

The 26 queue logs referenced in the EdgeCrafter receipts are stored at `.worktree-outputs/parent-2492-issue-2487-edgecrafter/player-model-comparison/queue-logs/` relative to the main checkout. In the worker checkout, use `outputs/player-model-comparison/queue-logs/`. Each log matches the original SHA-256 recorded in its receipt. Run `sha256sum --check SHA256SUMS` in that directory to verify the archive.

When a new job completes, copy its log into that directory and verify its SHA-256 against the source. Record the retained path and hash in the receipt and add the log to `SHA256SUMS`.

## Accuracy on the fixed cohort

The six queue jobs at `83507bdd57097ac5b37892b5b1d6db3af0666af2` produced eight new scores and six TensorRT engines. Including the four retained ECDet PT16 results, all 12 PT16/TRT16 conditions contain the same 6,844 frame-and-camera rows and 26,276 eligible boxes. The receipt records the per-condition metrics, prediction digests, queue logs and verified engine artifacts. The raw outputs are under `outputs/player-model-comparison/` in the EdgeCrafter worker checkout. [Accuracy receipts](edgecrafter-accuracy.json)

ECDet's PT16 recall at IoU 0.50 is 97.58–98.19%, and its TRT16 recall is 63.02–70.29%. ECPose-L's coverage-inclusive PCK@0.20 is 38.56% on PT16 and 36.53% on TRT16. ECPose-X reaches 33.08% and 6.91%, respectively. All conditions completed scoring. The large ECPose-X difference is measured against the pseudo-reference, and its cause remains open.

## Fixed-input speed cases

`edgecrafter_speed_run.sh` runs the selected backend at batch 1 and 2 in separate processes. The optional fourth argument accepts `pt16`, `trt16` or `both` and defaults to `both`. `edgecrafter_speed.py` verifies the #2496 manifest, timing protocol and harness revision. It uses the ordered set of 96 native BGR images, 30 warm-ups and three blocks of 100 iterations on a dedicated CUDA stream. It sets and records four PyTorch host threads and one OpenCV thread, matching the DINO baseline. TRT16 loads the engine from the full-cohort accuracy run. The shared harness records `parity_status=not_evaluated` with reason `PT32未計測`. The previous six speed jobs were cancelled before execution while the common timing contract was revised.

```bash
# Register this command in the shared training queue for each model.
bash knowledge/runs/run-meiji-player-model-comparison/edgecrafter_speed_run.sh \
  ecpose-l-o365 "$fixed_speed_inputs" 13e508cb78c46663fbe58ad8be03bcf9c96816a6 pt16
```

`fixed_speed_inputs` is the published baseline input directory. The runner writes receipts to `outputs/player-model-comparison/<model>/speed-13e508cb7-threads4-1/<backend>/speed-b<batch>.json`. The first measurements are stored under `speed-13e508cb7`. [Baseline inputs and protocol](../run-person-model-speed-baseline/README.md#固定入力と初回の成果物)

The first 24 speed cases used two PyTorch host threads at checkout `3bb026d5de12dbe980bc17e3d0a97c751b58fd4b`. The original TRT16 timings identify the plan SHA-256 used for scoring and remain provisional. [Initial speed and VRAM receipts](edgecrafter-speed.json)

The PT16 rerun at `60af57ee662aa7d307cd1f49ae069b45aaa65446` completed all 12 cases with the same host thread settings, RTX 5060 Ti, GPU power limit, image order and timing protocol as the PT16 baselines. The 3,600 timing samples reproduced the recorded summaries. NVML peak increase is relative to device usage before loading the model. The batch 2 numbers describe timing and VRAM. Cohort accuracy uses batch 1. [Matched PT16 measurements, raw report hashes and accuracy references](edgecrafter-pt16-speed-matched.json)

| PT16 model | Batch 1 wall P50 / P95 (ms/image) | Batch 2 wall P50 / P95 (ms/image) | Batch 2 images/s | NVML peak increase B1 / B2 (MiB) |
|---|---:|---:|---:|---:|
| ECDet-L COCO | 53.61 / 62.97 | 39.23 / 42.59 | 25.18 | 324 / 372 |
| ECDet-X COCO | 52.45 / 59.56 | 39.89 / 43.47 | 24.70 | 432 / 532 |
| ECDet-L O365 | 52.91 / 63.14 | 39.19 / 42.22 | 25.28 | 324 / 372 |
| ECDet-X O365 | 50.75 / 55.68 | 41.19 / 46.33 | 23.87 | 432 / 532 |
| ECPose-L O365 | 56.89 / 66.94 | 41.86 / 45.13 | 23.69 | 362 / 458 |
| ECPose-X O365 | 56.06 / 65.74 | 42.86 / 50.79 | 22.76 | 472 / 592 |

ECDet-X O365 recorded the lowest batch 1 wall P50 among the four ECDet weights. Its 50.75 ms/image compares with 165.69 for DINO. Its pseudo-reference Recall@0.50 was 97.97% versus DINO's 96.69%, and Recall@0.75 was 46.62% versus 46.99%. The ECPose variants measured 56.06–56.89 ms/image versus 266.30 for DINO plus ViTPose-H. Their coverage-inclusive OKS was 0.259–0.291 versus 0.763. [Synchronized PT16 comparison](edgecrafter-pt16-speed-matched.json)

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

The diagnosis uses four published #2496 inputs. It selects one image per clip in descending order of the total person detections at score ≥0.25 across ECDet-L COCO, ECPose-L and ECPose-X PT16. The diagnostic records both single-image inputs and both image orders for each pair. It passes one contiguous preprocessed tensor per case to the PT16 forward and the TensorRT runner, then records each raw head's shape, dtype, finite range and NaN/Inf counts. The report compares outputs by source image ID and checks context strides against the PyTorch buffers. Decoded boxes use the accuracy run's top-k and score settings, with no NMS.

```bash
diagnose=knowledge/runs/run-meiji-player-model-comparison/edgecrafter_diagnose.py
baseline=knowledge/runs/run-person-model-speed-baseline

# Run the alternate interpreter on CPU.
PYTHONPATH=".:$baseline" OMP_NUM_THREADS=4 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  .venv/bin/python -B "$diagnose" --inputs "$fixed_speed_inputs" \
  --output outputs/player-model-comparison/trt16-diagnosis-v1 --onnx-cpu

# Submit one model per process through the training queue with resource all.
PYTHONPATH=".:$baseline" OMP_NUM_THREADS=4 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  .venv/bin/python -B "$diagnose" --inputs "$fixed_speed_inputs" \
  --model ecdet-l-o365 --output outputs/player-model-comparison/trt16-diagnosis-v2/ecdet-l-o365
```

On `clip_028/cam3` frame 513, ONNX Runtime CPU reproduced all 68 archived PT16 ECDet-L boxes and all 60 archived PT16 ECPose-X boxes at IoU ≥0.50. The same ONNX files yielded 30 and 11 matching boxes through their archived TRT16 plans. The gap on this image points to the TensorRT path. [ONNX CPU comparison](edgecrafter-trt-onnx-cpu.json)

The first CUDA diagnostic produced raw results for five weights. The upstream YAML loader shares its mutable default dictionary across calls. After loading ECDet-X, the same process tried to construct ECDet-L O365's MLP with width 2304 for a checkpoint with width 1536. The revised command uses one model per process and repeats the original batch order after the swapped execution. The prior speed and accuracy cases already used separate processes.

Every captured binding stride matched its PyTorch buffer and the sampled output heads were finite. All five PT16 variants reproduced their raw outputs when the two batch slots were swapped and aligned back to image IDs. All five original TRT16 plans changed their decoded box sets. ONNX Runtime CPU also changed raw query order when the slots were swapped for ECDet-L COCO and ECPose-X. For ECDet-L COCO, small observed differences grew from an encoder score P95 of 0.0078 to a final logit P95 of 1.305 after TopK changed selected indices. The CPU interpreter still matched all 154 decoded boxes at IoU ≥0.50, with matched IoU P05 above 0.997 for each image. The TRT16 plans' much larger decoded disagreement remains unexplained. [CUDA and CPU batch-order evidence](edgecrafter-trt-cuda-diagnosis.json)

The ECDet-L O365 single-process run completed all ten cases. Swapping the two images and aligning them by ID preserved all 254 PT16 person boxes. The original TRT16 plan matched 199 of 243 person boxes at IoU ≥0.50 across the two pairs. Repeating the original order produced identical raw heads on both backends. All captured heads were finite and every binding stride matched its buffer. [Requeued job evidence](edgecrafter-trt-diagnosis-v2.json)

The ECDet-L COCO probe exposed ten intermediate tensors in a separate plan. The intermediates, including the sampled TopK values, indices and Gather output, matched exactly after the batch swap. The final logits had an absolute difference P95 of 0.5664 and the normalized box coordinates had a P95 of 0.00211. The instrumented plan matched 93 of the original plan's 241 decoded boxes at IoU ≥0.50 on the same batch. The final discrepancy therefore appears downstream of the sampled Gather in this build. Extra outputs can affect fusion, and the exact cause in the original plan remains open. [Requeued job evidence](edgecrafter-trt-diagnosis-v2.json), [TensorRT engine and debug tools](https://docs.nvidia.com/deeplearning/tensorrt/10.x.x/inference-library/engine-tools.html)

```bash
# Submit through the training queue with resource all.
PYTHONPATH=".:$baseline" OMP_NUM_THREADS=4 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  .venv/bin/python -B knowledge/runs/run-meiji-player-model-comparison/edgecrafter_trt_probe.py \
  --inputs "$fixed_speed_inputs" --archive outputs/player-model-comparison/trt16-diagnosis-v1 \
  --output outputs/player-model-comparison/trt16-diagnosis-v2/ecdet-l-coco-probe
```

The follow-up builds a separate ECDet-L COCO plan with only the four decoder self-attention score MatMuls and their Softmax operations in FP32. Both output bindings, the input profile and the weights match the original graph. The diagnostic compares raw heads and decoded boxes with the archived PT16 heads on the same input bytes at batch 1, batch 2 and reversed batch order. A matching result would support testing the change across the full cohort. Engine rebuilding can change tactic selection, so further controls would be needed to attribute the difference to those operations alone.

The ONNX CPU check verified all four widened regions and preserved the original binding dtypes, shapes and weight digest. On `clip_028/cam3` frame 513 the rewritten graph matched 68 of 68 archived PT16 boxes at IoU ≥0.50, with matched IoU P05 of 0.993.

On the RTX 5060 Ti the rebuilt plan improved agreement with the saved PT16 predictions in all three cases. The table gives the number of boxes matched at IoU ≥0.50, with the number of eligible PT16 rows as the denominator. It shows both the full set of decoded person rows and the shared minimum score of 0.25. [Build, raw heads and verification](edgecrafter-trt-diagnosis-v3.json)

| Fixed input | Original plan | FP32 attention plan | Original plan at score ≥0.25 | FP32 attention plan at score ≥0.25 |
|---|---:|---:|---:|---:|
| `clip_028/cam3` frame 513, batch 1 | 30/68 | 56/68 | 3/12 | 10/12 |
| Both images, batch 2 A–B | 88/154 | 134/154 | 17/25 | 22/25 |
| Both images, batch 2 B–A | 71/154 | 137/154 | 12/25 | 24/25 |

The rebuilt plan matched 122 of 158 boxes after the batch swap, with final logit absolute difference P95 of 2.3712. Repeating the original order reproduced the raw tensors exactly. Against the fixed pseudo-reference at score ≥0.25 and IoU ≥0.50, the rebuilt plan matched 3/4 players in batch 1 and 6/8 and 8/8 in the two batch orders. Saved PT16 matched 4/4 and 8/8 in both orders. Two images provide a diagnostic sample. These results show a plan-level improvement with a residual batch-order effect. [Sample pseudo-reference scores and batch-order checks](edgecrafter-trt-diagnosis-v3.json)

```bash
# Submit through the training queue with resource all.
PYTHONPATH=".:$baseline" OMP_NUM_THREADS=4 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  .venv/bin/python -B knowledge/runs/run-meiji-player-model-comparison/edgecrafter_trt_probe.py \
  --inputs "$fixed_speed_inputs" --archive outputs/player-model-comparison/trt16-diagnosis-v1 \
  --output outputs/player-model-comparison/trt16-diagnosis-v3/ecdet-l-coco-fp32-attention \
  --fp32-decoder-attention
```

## Full-cohort evaluation of FP32 decoder attention

The `trt16-fp32attn` condition uses FP32 for the four decoder self-attention score MatMuls and their Softmax operations. Casts restore each probability tensor's original FP16 type. The four ECDet exports have the same target pairs. The checked ONNX artifacts preserve the original weights, other nodes and I/O contracts. ECDet-L COCO uses the verified plan from the diagnosis and the three other ECDet weights use new exports. [ONNX and plan checksums](edgecrafter-trt16-fp32attn-preflight.json)

ECPose-L and ECPose-X O365 are inapplicable to this exact rewrite. Both graphs have separate `within_attn` and `across_attn` modules in each decoder layer, which process different query axes. The ECDet path with one self-attention module per layer has no direct correspondence in those graphs. [ECPose decoder](https://github.com/Intellindust-AI-Lab/EdgeCrafter/blob/d8752b75c9beaecfcdd0e5cd207bae3b6946ae23/ecpose/engine/edgecrafter/detrpose_transformer.py#L179)

Use the original batch 1 accuracy protocol and shared scorer for all 6,844 cohort images. The accuracy and prediction headers identify the condition as `trt16-fp32attn`. Speed uses the common warm-up and measurement schedule at image batches 1 and 2 with PyTorch 4 threads and OpenCV 1 thread. The speed receipt uses `<model>/trt16-fp32attn` as its case name and records the plan SHA-256 in `engines`. Compare cohort accuracy with batch 1 timing because the ECDet-L COCO diagnosis still found batch-order sensitivity. The reference output for the diagnosis is the saved PT16 run.

Run the rewrite on CPU for each new export. The queue command uses the resulting digest and engine directory for both scoring and timing.

```bash
model=ecdet-x-coco
run=knowledge/runs/run-meiji-player-model-comparison
original="$output/$model/trt16-trace2/engine"
variant="$output/$model/trt16-fp32attn"
engine="$variant/engine"
PYTHONPATH=. .venv/bin/python -B "$run/edgecrafter_trt.py" rewrite-attention \
  --model "$model" --base-engine "$original" --engine "$engine" \
  --receipt "$variant/onnx-rewrite.json"
base_onnx_sha=$(jq -er .base_onnx_sha256 "$variant/onnx-rewrite.json")

# Queue one model per job with resource all.
bash "$run/edgecrafter_run.sh" "$model" "$manifest_sha" "$engine" "$base_onnx_sha" && \
  bash "$run/edgecrafter_speed_run.sh" "$model" "$fixed_speed_inputs" \
    13e508cb78c46663fbe58ad8be03bcf9c96816a6 trt16-fp32attn "$engine" "$base_onnx_sha"
```

To run ECDet-L COCO with the diagnosed plan, use the `engine_directory` and `base_onnx.sha256` recorded for that model and start with the queue command. Separate ONNX conversion and plan build timings are unavailable for that diagnostic.

All four ECDet jobs completed at `3acc97cbb78e6018a424e426bc76451d74933967`. Rescoring the 6,844 prediction rows for each model reproduced the saved results for all 26,276 eligible boxes. Each model's accuracy and both speed reports reference the same verified plan SHA-256. The eight speed cases reproduced their summaries from 2,400 timing samples and the fixed input order. [Results, plan checksums and raw reports](edgecrafter-trt16-fp32attn-results.json)

The original `trt16` and `trt16-fp32attn` measurements have the status “conversion defect, cause unresolved”. The accuracy–speed Pareto comparison uses only PT16 for EdgeCrafter.

| Model | Recall@0.50 PT16 / original TRT16 / FP32 attention (%) | Recall@0.75 PT16 / original TRT16 / FP32 attention (%) | FP32 attention batch 1 wall P50 / P95 (ms/image) | FP32 attention batch 2 wall P50 / P95 (ms/image) | Batch 2 images/s | NVML peak increase B1 / B2 (MiB) |
|---|---:|---:|---:|---:|---:|---:|
| ECDet-L COCO | 97.61 / 64.88 / 83.05 | 41.60 / 17.85 / 33.47 | 45.78 / 97.98 | 30.19 / 43.97 | 30.20 | 172 / 172 |
| ECDet-X COCO | 97.58 / 70.29 / 78.25 | 39.79 / 21.22 / 27.16 | 32.97 / 36.94 | 29.84 / 33.11 | 32.99 | 210 / 210 |
| ECDet-L O365 | 98.19 / 63.02 / 81.04 | 44.88 / 21.53 / 32.56 | 31.48 / 34.10 | 29.34 / 35.74 | 32.86 | 172 / 172 |
| ECDet-X O365 | 97.97 / 65.88 / 45.46 | 46.62 / 23.87 / 16.01 | 32.11 / 34.74 | 30.07 / 31.77 | 33.01 | 210 / 210 |

The FP32 attention plans left a batch 1 Recall@0.50 gap of 14.56–52.51 percentage points to saved PT16. Both L weights and X COCO improved over their original TRT16 plans. X O365 lost 20.43 points. ECDet-L COCO's batch 1 preprocessing P95 was 82.62 ms in this run. The other timing stages and per-block medians are in the result record. Batch 2 cohort accuracy is unmeasured. [Accuracy and timing details](edgecrafter-trt16-fp32attn-results.json)

## 残課題

The TensorRT investigation ended after the full-cohort evaluation. If EdgeCrafter becomes a deployment candidate, compare intermediate outputs on the same images to identify the first operation where TensorRT diverges from PT16 and determine how the divergence depends on image order within a batch.

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

The speed entry-point tests check explicit PT16 selection, the four default backend and batch combinations, the host thread settings, ordered images, harness revision validation and the explicit parity argument. The accuracy wrapper tests check that a failed command or missing engine receipt is reported against the corresponding condition and preserves a successful result from the other backend.

The decoder precision test uses a small typed ONNX graph to verify FP32 MatMul inputs and scores, FP32 Softmax and restoration of the original FP16 tensor at the next consumer. It checks that the source graph stays intact.
