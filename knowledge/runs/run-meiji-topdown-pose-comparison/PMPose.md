# PMPose-L and PMPose-H

## Assets and input contract

Use PMPose 1.0.0 with the pinned [BBoxMaskPose source](https://github.com/MiraPurkrabek/BBoxMaskPose/tree/49a070a6f8396147323b1c4959474077dbe2ce8e) and the [official weight release](https://huggingface.co/vrg-prague/BBoxMaskPose/tree/cfc89052cf16655a52889f1f4e1b866e74a6c19c/PMPose). The source, weight revision, filenames, sizes and SHA-256 pins are in `pmpose_assets.py`.

The author’s [BBox demo](https://huggingface.co/spaces/purkrmir/PMPose-demo/blob/7d76e51513a55b8bde5f032ddec9ff712dcfcace/app.py) creates a binary rectangle from each BBox. Use the same floor/ceil clipping and external contour as the demo and the upstream `inference_topdown` implementation. The rectangle uses only the fixed input BBox. The official L/H configuration applies background alpha 0.25, adds padding 1.25 to the box, fixes the aspect ratio, warps with UDP to 256×192 (height × width), converts BGR to RGB and normalizes with the configured mean and standard deviation on the 0–255 scale. The official inference codec decodes 64×48 maps using DARK UDP and maps coordinates back to native pixels. Both configurations use horizontal flip TTA with two forwards and no heatmap shift.

The [upstream 23-joint schema](https://github.com/MiraPurkrabek/BBoxMaskPose/blob/49a070a6f8396147323b1c4959474077dbe2ce8e/mmpose/configs/_base_/datasets/merged_COCO_AIC_MPII_mergable.py) preserves the names, order and left/right pairs of COCO17 at indices 0–16. Channels 17–22 encode MPII and AIC thorax, pelvis, neck and head top. The L/H configuration selects expected OKS as `keypoint_scores`. The adapter preserves these finite scores and provides the native 23-joint coordinates, presence, visibility, expected OKS, heatmap peak and the error-head value divided by the heatmap diagonal. The configuration freezes the error head during training. Its output is diagnostic. The shared scoring policy accepts all finite scores.

## License

The upstream [source license](https://github.com/MiraPurkrabek/BBoxMaskPose/blob/49a070a6f8396147323b1c4959474077dbe2ce8e/LICENSE) and [PMPose weight license](https://huggingface.co/vrg-prague/BBoxMaskPose/blob/cfc89052cf16655a52889f1f4e1b866e74a6c19c/README.md) specify GPL-3.0. Section 2 permits local execution for evaluation. Distribution of GPL-covered source or binaries is subject to sections 4–6, including the applicable license, notice and corresponding-source obligations. Production adoption depends on the intended integration and distribution arrangements, including any packaged weights or converted engine. The experiment uses an external checkout and downloads the weights directly from the publisher.

## Reproduction

Run from the repo root with its existing PyTorch environment. The download and native dependency builds run on the CPU. Install the extra packages in a private import directory that can be used alongside the shared PyTorch and NumPy. MMPretrain 1.2.0 imports utilities from the legacy Transformers BERT API during registration. Its compatible Transformers build is pinned in `pmpose-requirements.txt`.

```bash
run=knowledge/runs/run-meiji-topdown-pose-comparison
assets=outputs/pmpose
PYTHONPATH="$PWD" .venv/bin/python -B "$run/pmpose_assets.py" "$assets"
bash "$run/pmpose-dependencies.sh" "$assets/dependencies"

export PYTHONPATH="$PWD:$PWD/$run:$PWD/$assets/dependencies"
model=pmpose-l
checkpoint="$assets/PMPose-${model##*-}-1.0.0.pth"
source_root="$assets/BBoxMaskPose"
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 .venv/bin/python -B "$run/pmpose_tools.py" smoke \
  --model "$model" --source-root "$source_root" --checkpoint "$checkpoint" \
  --receipt "$assets/$model-official-parity.json"
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 .venv/bin/python -B "$run/pmpose_tools.py" export \
  --model "$model" --source-root "$source_root" --checkpoint "$checkpoint" \
  --directory "$assets/$model-engine" --receipt "$assets/$model-export.json"
```

For the GPU commands use the shared training queue’s exclusive reservation. To build the TRT16 engine, pass the output directory from `export` to `pmpose_tools.py build`. Set `--opt-batch`, `--max-batch` and `--workspace-mib` for the chosen speed and accuracy workloads. Use `pmpose_tools.py inspect --batch N` to record TensorRT layer metadata on the same GPU. Use the engine directory as `--engine` for accuracy and speed. All forwards use the same input shape per model and two TTA branches. PT16 and TRT16 use FP16 weights and inputs with the stock Sparsemax projection in FP32. Flip restoration and DARK UDP use FP32 outputs. The model records an explicit failure when a required engine or upstream asset is unavailable.

Use the [common accuracy runner](README.md#accuracy-command) with `--model pmpose-l` or `pmpose-h`, the matching checkpoint and SHA-256 from `pmpose_assets.py`, and `--source-root "$source_root"`. The historical accuracy records include both the scores against the pinned #2495 bundle and coordinate differences from that model’s PT32 predictions.

## Synchronized speed protocol

`pmpose_speed.py` uses the #2496 [fixed inputs and timing order](../run-person-model-speed-baseline/README.md) through the [shared timing harness](../../../src/utils/inference_benchmark.md). The cohort contains 384 BGR crop rows. The runner uses native `xyxy` boxes and the two official flip forwards per batch. Run PT16 and TRT16 for both models at batch 1, 4 and 8. Each condition uses 30 warmup iterations, then 3 blocks of 100 iterations on the dedicated CUDA stream. The synchronized measurements require a fresh output directory for each case. TRT16 reuses the plans from the accuracy runs.

Use the repo’s training queue to run the following per-case command with an exclusive reservation. Use the shared fixed speed input directory as `inputs` and choose a new `output` for the series.

```bash
run=knowledge/runs/run-meiji-topdown-pose-comparison
baseline=knowledge/runs/run-person-model-speed-baseline
assets=outputs/pmpose
repo=$(dirname "$(git rev-parse --path-format=absolute --git-common-dir)")
inputs="$repo/.worktree-outputs/parent-2492-issue-2496-primary/person-model-speed-baseline-v1/inputs"
output="$assets/speed-pt16-trt16-sync"
PYTHONPATH="$PWD:$PWD/$run:$PWD/$baseline:$PWD/$assets/dependencies" \
  OMP_NUM_THREADS=4 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  .venv/bin/python -B "$run/pmpose_speed.py" --inputs "$inputs" --assets "$assets" \
  --output "$output" --model pmpose-l --backend PT16 --batch 1
```

Each `report.json` records the execution and harness commits, input pins, model artifacts, environment, synchronized timing samples and memory. `infeasible` reports carry the failure phase and message. The speed reports set `parity_not_evaluated_reason` to `PT32計測はスコープ外（ユーザー判断 2026-11-25）` under the November 25 scope. Use the saved pseudo-GT scores for accuracy and retain the earlier PT32 scores and numerical comparisons as historical results.

## CPU preflight

The shared PyTorch `2.13.0+cu130` environment completed the two-row real-checkpoint comparison against upstream `inference_topdown` for each model. The COCO17 and extra axes were checked, including the 23 native coordinates. Presence, visibility, expected OKS and heatmap peak agreed exactly. The maximum difference for the normalized error head was `2.33e-11`. Both FP16 ONNX files passed the ONNX checker and expose `pixels` as FP16 and `heatmaps` as FP32, with a dynamic batch axis.

| Model | Parameters | Maximum absolute native-coordinate difference | ONNX export wall time |
| --- | ---: | ---: | ---: |
| PMPose-L | 421,922,675 | 0.000007183 px | 78.333 s |
| PMPose-H | 814,320,243 | 0.000007487 px | 152.827 s |

The verified run used commit `0860ce5e12c953dc5520f02a502cc50cc2b3381b` and local job `20261124T123234Z_03845a48192c_i2490-pmpose-parity-export`. Under `outputs/pmpose/`, `*-official-parity.json` contains the CPU comparison, `*-export.json` the conversion timing, and `*-engine/export.json` the verified checkpoints, ONNX hashes and tensor bindings. The import warning about an optional CPU 4-bit GEMM kernel appeared before the successful FP32 smoke and FP16 export.

## GPU accuracy and rescoring

All six conditions produced 26,276 native-pixel COCO17 rows at batch 8 on the RTX 5060 Ti. Each array contained 446,692 finite joint predictions. The TensorRT plans used version `10.14.1.48.post1`, profile batches 1/4/8 and 1,536 MiB builder workspace. Build times were 85.092 s for L and 109.279 s for H. The batch-8 TensorRT inspector shows FP32 bindings on the Sparsemax sort, cumulative sum and threshold computation, and on portions of layer normalization. The original score files and the plan hashes are in the per-condition receipts under `outputs/pmpose/`.

The canonical matcher reproduced every numeric score and count for all six conditions. The score files gained the `policy.candidate_order_for_iou_ties` field. PCK pools 437,714 scoreable joints, and mean OKS averages 25,811 scoreable people. The [accuracy record](pmpose_accuracy.json) pins the rescored files, raw predictions, execution commit, queue commands, engine manifests and layer inspector reports.

| Model | Backend | Mean OKS | PCK@0.10 | PCK@0.20 | Pixel error P95 |
| --- | --- | ---: | ---: | ---: | ---: |
| PMPose-L | PT32 | 0.776631 | 0.787578 | 0.966796 | 11.804 px |
| PMPose-L | PT16 | 0.776622 | 0.787562 | 0.966777 | 11.805 px |
| PMPose-L | TRT16 | 0.776621 | 0.787523 | 0.966784 | 11.804 px |
| PMPose-H | PT32 | 0.780841 | 0.794747 | 0.970426 | 11.550 px |
| PMPose-H | PT16 | 0.780835 | 0.794766 | 0.970424 | 11.549 px |
| PMPose-H | TRT16 | 0.780849 | 0.794752 | 0.970435 | 11.546 px |

Across all 446,692 output joints, the PT32-relative coordinate P95 was 0.0121/0.0312 px for L and 0.0128/0.0296 px for H in PT16/TRT16 order. The largest differences were 1,926.6/1,325.3 px for L and 88.3/98.7 px for H. The maximum cases were outside the pseudo-GT eligibility mask. Among scoreable joints, the maxima were 71.5/116.2 px for L and 85.5/86.3 px for H. The largest L difference is at the right knee of player 0 in `clip_028`, `cam1`, frame 1005. The PT32 x-coordinate is 2,273.9 px for a box spanning x=778.5–795.2 px. Keep these outputs for the backend comparison. The upstream DARK UDP refinement has no bound on its coordinate update.

Rescore the original queue outputs through the canonical matcher from #2498. Supply the six job filenames from this accuracy run.

```bash
run=knowledge/runs/run-meiji-topdown-pose-comparison
media=data/tennis_multiview/processed/meiji_4cam/benchmarks
queue="$(dirname "$(git rev-parse --path-format=absolute --git-common-dir)")/.training_queue"
CUDA_VISIBLE_DEVICES='' PYTHONPATH="$PWD:$PWD/$run" .venv/bin/python -B "$run/pmpose_results.py" \
  --queue "$queue" --assets outputs/pmpose \
  --bundle "$media/multiview-player-evaluation-v1" \
  --bundle-sha256 3dc496513380a39d4511ff7f8711f4192eababa5109f6a0506f88a60baa70580 \
  --output outputs/pmpose/accuracy-rescored "${job_ids[@]}"
```

The script writes new score files, retains the original scores for verification, and summarizes coordinate and confidence differences against the matching PT32 predictions. Speed-protocol parity uses the common speed input and the shared thresholds.

### Auxiliary scores

On scoreable PT32 joints, the mean expected OKS was 0.680 for L and 0.695 for H. The mean per-joint pseudo-GT OKS under the common normalization was 0.777 and 0.781. For ranking PCK@0.10 successes, the expected OKS ROC AUC was 0.584 and 0.586. The [diagnostic record](pmpose_diagnostics.json) also reports the AUC for presence, visibility and heatmap peak. The pseudo-GT validity mask gates the analysis. The cohort had zero missing model outputs and provides no independent labels for true joint visibility.

```bash
CUDA_VISIBLE_DEVICES='' PYTHONPATH="$PWD:$PWD/$run" .venv/bin/python -B "$run/pmpose_diagnostics.py" \
  --accuracy-record "$run/pmpose_accuracy.json" \
  --bundle "$media/multiview-player-evaluation-v1" \
  --output outputs/pmpose/pmpose_diagnostics.json
```

## Tests

Use small CPU fixtures with no weight or GPU requirement. Compare the rectangle contour against the demo’s raster-plus-OpenCV calculation, including fractional and edge-clipped boxes. An asymmetric pose with two boxes from one frame exercises row order, native-pixel decoding, flip restoration of eyes and limbs, the full 23-joint auxiliary axis and a missing heatmap. Inject distinct scores including zero to detect silent score filtering. Run the same assertion through the accuracy stages and the shared timing stages to check that both count the two real forwards. Loader tests exercise explicit rejection of an absent source, wrong checkpoint and absent TRT engine. Reuse `tests/unit/utils/test_inference_benchmark.py::test_exact_backend_selection_never_loads_another_backend` and the shared pose scorer tests for backend selection and fixed-denominator coverage. Measure the focused tests with `--durations=0 --durations-min=0`. The real-asset smoke will compare the adapter’s PT32 output against the upstream API on the same images, rectangles and weights.

For rescore verification, check every saved prediction hash, ordered row address and input BBox against the published cohort. Keep the original score files to compare with the canonical matcher. Use a small CPU case with an exchanged person order and a mismatched expected OKS to verify that the archive checks reject incorrect input assignments. The full six-condition run verifies the real queue receipts and engine artifacts.

For the auxiliary-score diagnostic, reuse the common scorer’s PCK normalization and COCO sigmas. Check ROC AUC against a hand-counted set with a tied positive and negative score. Check the PCK success totals against the canonical scorer on both real PT32 archives. Evaluate the largest backend differences against the existing reference validity mask.

For the speed runner, reuse the shared timing tests in `tests/unit/utils/test_inference_benchmark.py` and the input verification tests in `tests/unit/utils/test_inference_benchmark_inputs.py`. A CPU CLI test with two synthetic crop rows checks that the runner follows the pinned order, uses raw `xyxy` coordinates and passes only the requested adapter and the explicit parity reason to the shared harness. Check that the CLI rejects PT32 and that the adapter reports TensorRT context creation failures while preserving CUDA OOM. Check all real crop inputs and batch schedules on the CPU before queueing. Keep these tests independent of upstream dependencies, model weights and CUDA.
