# RF-DETR detection comparison

This adapter evaluates the two official COCO detection checkpoints specified in [#2487](https://github.com/Motoki0705/tennis-lab/issues/2487). It uses the 6,844 native images pinned by [#2495](https://github.com/Motoki0705/tennis-lab/pull/2498). The images belong to six Meiji 4cam clips at saved GVHMR inference frames. Eligible players have 26,276 reference boxes derived from placed SMPL. The scorer reports player recall and IoU matching on partial labels, with precision and AP unavailable.

## Checkpoints and licenses

| Model ID | Official checkpoint | Parameters | Input | Published evaluation |
| --- | --- | ---: | --- | --- |
| `rfdetr-l` | `rf-detr-large-2026.pth`, 135,954,129 bytes | 33,933,218 loaded | 704 × 704 | COCO AP50 75.1 and AP50:95 56.5 |
| `rfdetr-2xl` | `rf-detr-xxlarge.pth`, size and SHA-256 pending authorized acquisition | 126.9M published | 880 × 880 | COCO AP50 78.5 and AP50:95 60.1 |

Roboflow identifies both releases as COCO detection checkpoints. The paper reports Objects365 pretraining with SAM2 pseudo-masks before COCO fine-tuning and describes the DINOv2-S (L) and DINOv2-B (2XL) configurations. The figures in the table are publisher COCO results. [Pinned model zoo](https://github.com/roboflow/rf-detr/blob/f29e2a8bbb7cdcb8920bf74a335062f88a35046d/README.md) · [ICLR paper](https://openreview.net/forum?id=qHm5GePxTh)

- **L** uses `rfdetr==1.12.1` from commit `f29e2a8bbb7cdcb8920bf74a335062f88a35046d`. The downloaded bytes match Roboflow's MD5, and the adapter pins their SHA-256. Apache 2.0 permits local evaluation and commercial self-hosting. Redistribution is permitted under the license's notice and attribution conditions. [Official weight registry](https://github.com/roboflow/rf-detr/blob/f29e2a8bbb7cdcb8920bf74a335062f88a35046d/src/rfdetr/assets/model_weights.py) · [Apache 2.0](https://github.com/roboflow/rf-detr/blob/f29e2a8bbb7cdcb8920bf74a335062f88a35046d/LICENSE)
- **2XL** uses the same core plus `rfdetr-plus==1.3.0` from commit `31474876e519dacef3ba320bc8977f1483d69b34`. PT32, PT16 and TRT16 are pending user confirmation of PML 1.0 and a Roboflow account/plan permitting local evaluation. After confirmation, supply the official local checkpoint, its SHA-256 and `--pml-accepted`. The adapter verifies the release MD5 and forwards explicit acceptance to the vendor constructor. Commercial deployment on Roboflow is included in eligible plans and commercial self-hosting requires the Enterprise add-on. Any redistribution of the checkpoint or derived engine is subject to the platform agreement and PML notices. [Account](https://app.roboflow.com/) · [Plus release](https://github.com/roboflow/rf-detr_plus/tree/31474876e519dacef3ba320bc8977f1483d69b34) · [Weight registry](https://github.com/roboflow/rf-detr_plus/blob/31474876e519dacef3ba320bc8977f1483d69b34/src/rfdetr_plus/assets/model_weights.py) · [PML 1.0](https://github.com/roboflow/rf-detr_plus/blob/31474876e519dacef3ba320bc8977f1483d69b34/LICENSE) · [Deployment terms](https://roboflow.com/model-licenses/rf-detr)

## Inference and scoring

Decode source video frames as BGR uint8, convert to RGB, resize directly to the variant's square input with antialias-free bilinear interpolation in float32, and use ImageNet normalization. The upstream postprocessor selects the highest 300 query/class pairs by sigmoid score and restores and clips xyxy in the native image. The adapter retains positive-area boxes in sparse COCO category 1 with score ≥0.25. The cutoff is the lowest configured scorer cutoff. The output includes all detected persons above that cutoff and has no pose. [Pinned upstream postprocessor](https://github.com/roboflow/rf-detr/blob/f29e2a8bbb7cdcb8920bf74a335062f88a35046d/src/rfdetr/models/postprocess.py)

PT32 uses eager CUDA PyTorch with TF32 disabled. PT16 uses CUDA FP16 autocast over the same eager model with FP32 weights. The PyTorch autocast policy keeps eligible numerically sensitive operations such as layer normalization and softmax in FP32. Both use FP32 preprocessing and decoding. TRT16 uses the pinned vendor ONNX exporter (opset 17) and the vendor's FP16 ONNX cast with FP32 I/O, through the shared typed TensorRT builder and runtime. The plan fixes batch size and input resolution. [PyTorch autocast policy](https://docs.pytorch.org/docs/stable/amp.html#cuda-op-specific-behavior) · [Pinned vendor FP16 conversion](https://github.com/roboflow/rf-detr/blob/f29e2a8bbb7cdcb8920bf74a335062f88a35046d/src/rfdetr/export/_tensorrt/exporter.py)

`benchmark_adapter()` supplies the per-batch callable for the [shared speed harness](../../../src/utils/inference_benchmark.md). It measures RGB conversion, resize and normalization as preprocessing, tensor copies as H2D, the declared backend as forward, and upstream sigmoid, top-k and coordinate projection as postprocessing. D2H covers result transfer and conversion to the person-only prediction contract. Use the #2496 speed manifest and its published order when measuring.

Each accuracy invocation writes the primary worker's JSONL with one row for each source frame/camera pair. The accuracy receipt records the manifest and prediction hashes, runtime and engine, and the full shared pseudo-GT score. PT16 and TRT16 also compare separately to the same variant's checksummed PT32 predictions. The parity calculation matches boxes by one-to-one IoU ≥0.50. The pass criteria are ≥98% matched on each side, matched IoU P05 ≥0.90 and absolute score delta P95 ≤0.05. It reports counts, pixel and score errors even if the gate fails. Empty frames are retained.

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
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 HF_HUB_OFFLINE=1
export PYTHONPATH="$tooling/python:."
rf=(.venv/bin/python -B knowledge/runs/run-meiji-player-model-comparison/rfdetr.py
    --weights "$tooling/rf-detr-large-2026.pth" --output-root "$out")
"${rf[@]}" export
```

Run the GPU commands below through the shared training queue, starting with PT32. Keep the same output root so the other backends can read its checked predictions. The queue stores the commands and measured Git HEAD in its reproduction bundles.

```bash
benchmarks=data/tennis_multiview/processed/meiji_4cam/benchmarks
common=(--source "$benchmarks/multiview-v1"
        --evaluation-bundle "$benchmarks/multiview-player-evaluation-v1"
        --manifest-sha256 3dc496513380a39d4511ff7f8711f4192eababa5109f6a0506f88a60baa70580)
"${rf[@]}" evaluate --backend pt32 "${common[@]}"
"${rf[@]}" evaluate --backend pt16 "${common[@]}"
"${rf[@]}" build
"${rf[@]}" evaluate --backend trt16 "${common[@]}"
```

The accuracy artifacts live under `rfdetr-l/<backend>/`. The checked ONNX, engine and conversion receipts live under `rfdetr-l/trt16/batch-1/`. Use a fresh output root to reproduce a conversion from scratch.

## CPU preflight

- The official L checkpoint loaded on CPU with 33,933,218 parameters, 300 queries and 91 sparse COCO logit slots. At `video_000/clip_003`, `cam0`, source frame 120, the adapter's 10 person boxes and scores exactly matched the vendor API at cutoff 0.25.
- The CPU export generated a 61,233,732-byte FP16 graph in 13.68 s measured by the shared conversion timer. ONNX structural validation passed. The SHA-256 is `9cbf28254313462f421cfd7050bec08c18ca903029013a55d33102e2f5fdd21e`. The cast logged clamping of subnormal FP32 constants. GPU build and numeric parity are separate checks.
- The CPU verification passed 23 tests in 4.73 s, including 18 existing shared harness and evaluation tests. The five new cases use no upstream checkpoint and cover RGB and original size, sparse person label and positive-area boxes, paired output comparison independent of row order, hashed JSONL including empty frames, and 2XL consent gating.
