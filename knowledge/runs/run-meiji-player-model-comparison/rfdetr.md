# RF-DETR detection comparison

RF-DETR-L is the RF-DETR checkpoint in the [#2487 comparison](https://github.com/Motoki0705/tennis-lab/issues/2487). It uses the 6,844 native images pinned by [#2495](https://github.com/Motoki0705/tennis-lab/pull/2498). The images belong to six Meiji 4cam clips at saved GVHMR inference frames. Eligible players have 26,276 reference boxes derived from placed SMPL. The scorer reports player recall and IoU matching on partial labels, with precision and AP unavailable.

## Checkpoints and licenses

| Model ID | Comparison scope | Official checkpoint | Parameters | Input | Published evaluation |
| --- | --- | --- | ---: | --- | --- |
| `rfdetr-l` | Included | `rf-detr-large-2026.pth`, 135,954,129 bytes | 33,933,218 loaded | 704 × 704 | COCO AP50 75.1 and AP50:95 56.5 |
| `rfdetr-2xl` | Excluded for licensing reasons | `rf-detr-xxlarge.pth` | 126.9M published | 880 × 880 | COCO AP50 78.5 and AP50:95 60.1 |

Roboflow identifies both releases as COCO detection checkpoints. The paper reports Objects365 pretraining with SAM2 pseudo-masks before COCO fine-tuning and describes the DINOv2-S (L) and DINOv2-B (2XL) configurations. The figures in the table are publisher COCO results. [Pinned model zoo](https://github.com/roboflow/rf-detr/blob/f29e2a8bbb7cdcb8920bf74a335062f88a35046d/README.md) · [ICLR paper](https://openreview.net/forum?id=qHm5GePxTh)

- **L** uses `rfdetr==1.12.1` from commit `f29e2a8bbb7cdcb8920bf74a335062f88a35046d`. The downloaded bytes match Roboflow's MD5, and the adapter pins their SHA-256. Apache 2.0 permits local evaluation and commercial self-hosting. Redistribution is permitted under the license's notice and attribution conditions. [Official weight registry](https://github.com/roboflow/rf-detr/blob/f29e2a8bbb7cdcb8920bf74a335062f88a35046d/src/rfdetr/assets/model_weights.py) · [Apache 2.0](https://github.com/roboflow/rf-detr/blob/f29e2a8bbb7cdcb8920bf74a335062f88a35046d/LICENSE)
- **2XL** was excluded from PT32, PT16 and TRT16 by the user on 2026-11-24. PML 1.0 requires accepting and complying with a Roboflow platform agreement. For offline use without built-in tracking, the user is responsible for tracking and reporting usage under that agreement. Commercial self-hosting requires the Enterprise add-on. [User decision in #2492](https://github.com/Motoki0705/tennis-lab/issues/2492) · [PML 1.0](https://github.com/roboflow/rf-detr_plus/blob/31474876e519dacef3ba320bc8977f1483d69b34/LICENSE) · [Deployment terms](https://roboflow.com/model-licenses/rf-detr)

## Inference and scoring

Decode source video frames as BGR uint8, convert to RGB, resize directly to 704 × 704 with antialias-free bilinear interpolation in float32, and use ImageNet normalization. The upstream postprocessor selects the highest 300 query/class pairs by sigmoid score and restores and clips xyxy in the native image. The adapter retains positive-area boxes in sparse COCO category 1 with score ≥0.25. The cutoff is the lowest configured scorer cutoff. The output includes all detected persons above that cutoff and has no pose. [Pinned upstream postprocessor](https://github.com/roboflow/rf-detr/blob/f29e2a8bbb7cdcb8920bf74a335062f88a35046d/src/rfdetr/models/postprocess.py)

The recorded PT32 accuracy run used eager CUDA PyTorch with TF32 disabled. PT16 uses CUDA FP16 autocast over the same eager model with FP32 weights. The PyTorch autocast policy keeps eligible numerically sensitive operations such as layer normalization and softmax in FP32. Preprocessing and decoding use FP32. TRT16 uses the pinned vendor ONNX exporter (opset 17) and the vendor's FP16 ONNX cast with FP32 I/O, through the shared typed TensorRT builder and runtime. The plan fixes batch size and input resolution. [PyTorch autocast policy](https://docs.pytorch.org/docs/stable/amp.html#cuda-op-specific-behavior) · [Pinned vendor FP16 conversion](https://github.com/roboflow/rf-detr/blob/f29e2a8bbb7cdcb8920bf74a335062f88a35046d/src/rfdetr/export/_tensorrt/exporter.py)

`benchmark_adapter()` supplies the per-batch callable for the [shared speed harness](../../../src/utils/inference_benchmark.md). It measures RGB conversion, resize and normalization as preprocessing, tensor copies as H2D, the declared backend as forward, and upstream sigmoid, top-k and coordinate projection as postprocessing. D2H covers result transfer and conversion to the person-only prediction contract. Use the #2496 speed manifest and its published order when measuring.

Each accuracy invocation writes the primary worker's JSONL with one row for each source frame/camera pair, including empty frames. The receipt records the manifest and prediction hashes, runtime and engine, and shared pseudo-GT score. The completed accuracy run includes PT32 and the original parity measurements. Its exact commands, source commit and artifact hashes are recorded with its results. The 2026-11-25 decision limits new `evaluate` and `speed` invocations to PT16 and TRT16 and requires `--parity not_measured`. New accuracy output goes to `rfdetr-l/accuracy/<backend>/`. This speed campaign uses the saved accuracy scores. [User decision](https://github.com/Motoki0705/tennis-lab/issues/2492) · [Run findings](../../nodes/player_detection/000009-run-meiji-rfdetr-large-accuracy.md) · [Artifact inventory and metrics](../run-meiji-rfdetr-large-accuracy/results.json)

## Reproduction

Run from the repository root. The isolated Python overlay uses the existing project torch, CUDA and TensorRT runtime. Obtain the official L checkpoint once. Keep data and weights outside Git.

```bash
out=outputs/player-model-comparison
tooling=$out/tooling
mkdir -p "$tooling"
uv pip install --target "$tooling/python" --no-deps \
  rfdetr==1.12.1 supervision==0.29.1 pyDeprecate==0.9.0 \
  defusedxml==0.7.1 onnxconverter-common==1.16.0
curl -fL https://repo.roboflow.com/rfdetr/rf-detr-large-2026.pth \
  -o "$tooling/rf-detr-large-2026.pth"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 HF_HUB_OFFLINE=1
export PYTHONPATH="$tooling/python:."
```

## Fixed speed comparison

The six jobs submitted with the earlier timing program were cancelled before execution on 2026-11-25 JST. The current protocol comes from #2497, merged as `59faa32a28d2f54e6492abdba2429846fa0397e3`. It uses the same 96 BGR images, batch 1 and 2, a dedicated CUDA stream, 30 warmups and three blocks of 100 synchronized timed batches. The loader verifies the fixed assets and batch-specific timing order. Reports label parity as `PT32未計測` in the comparison and record the reason in `parity_not_evaluated_reason`. [Cancellation receipts](rfdetr-speed-cancellations.json) · [Fixed inputs](../run-person-model-speed-baseline/inputs/manifest.json) · [Timing protocol](../run-person-model-speed-baseline/timing-protocol.json) · [DINO speed results](../run-person-model-speed-low-precision/cases.csv)

Use a fresh `speed_out`. Submit each speed condition to the shared training queue with the complete GPU reservation. Reuse the validated batch-1 plan and run export, build and speed as separate processes for batch 2. The batch-2 speed process requires a successful build.

```bash
checkout_root=$(dirname "$(git rev-parse --path-format=absolute --git-common-dir)")
speed_inputs="$checkout_root/.worktree-outputs/parent-2492-issue-2496-primary/person-model-speed-baseline-v1/inputs"
speed_out=outputs/player-model-comparison/speed-synchronized-v1
rf=(.venv/bin/python -B -u knowledge/runs/run-meiji-player-model-comparison/rfdetr.py
    --weights "$tooling/rf-detr-large-2026.pth" --output-root "$speed_out")
common=(--speed-inputs "$speed_inputs" --parity not_measured)
for batch in 1 2; do
  "${rf[@]}" speed --backend pt16 --batch-size "$batch" "${common[@]}"
done
"${rf[@]}" speed --backend trt16 --batch-size 1 "${common[@]}" \
  --engine-dir "$out/rfdetr-l/trt16/batch-1/engine"
"${rf[@]}" export --batch-size 2 && "${rf[@]}" build --batch-size 2 &&
  "${rf[@]}" speed --backend trt16 --batch-size 2 "${common[@]}"
```

Reports at `rfdetr-l/<backend>/speed-b<batch>/report.json` include case status, timings, VRAM, per-batch input IDs, and the measured Git HEAD and harness merge. Preflight failures have a `speed-failure.json` in the condition directory. Export and build receipts are under `rfdetr-l/trt16/batch-<batch>/` in the corresponding asset root.

## CPU preflight

- The official L checkpoint loaded on CPU with 33,933,218 parameters, 300 queries and 91 sparse COCO logit slots. At `video_000/clip_003`, `cam0`, source frame 120, the adapter's 10 person boxes and scores exactly matched the vendor API at cutoff 0.25.
- The CPU export generated a 61,233,732-byte FP16 graph in 13.68 s measured by the shared conversion timer. ONNX structural validation passed. The SHA-256 is `9cbf28254313462f421cfd7050bec08c18ca903029013a55d33102e2f5fdd21e`. The cast logged clamping of subnormal FP32 constants. GPU build and numeric parity are separate checks.
- The L-only update passed 39 targeted CPU tests in 6.54 s. The seven adapter cases cover RGB and original size, person boxes, paired output comparison, hashed JSONL, model selection, checkpoint pinning, and legacy L plan metadata. The measured plan and all 6,844 PT32 rows loaded successfully, and the ten boxes and scores on the CPU preflight frame exactly matched the measured revision.
