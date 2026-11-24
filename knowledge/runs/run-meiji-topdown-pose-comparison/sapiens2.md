# Sapiens2 fixed-box pose evaluation

Use the two Meta checkpoint revisions in [sapiens2.py](sapiens2.py). The
expected SHA-256 values come from Hugging Face's LFS metadata. The adapter checks
the local file size and digest before loading. [Official pose releases](https://github.com/facebookresearch/sapiens2/blob/7e5bae88456ac418ff0e58e74106c9fe192055d4/docs/POSE.md)

| Model | Hugging Face release | Checkpoint bytes | Stored F32 elements | Model parameters |
|---|---|---:|---:|---:|
| Sapiens2-1B | [facebook/sapiens2-pose-1b](https://huggingface.co/facebook/sapiens2-pose-1b/tree/f5fed8b97b99698d5eea1d14ff0855d0b4c3f000) | 6,079,194,752 | 1,519,780,164 | 1,519,780,148 |
| Sapiens2-5B | [facebook/sapiens2-pose-5b](https://huggingface.co/facebook/sapiens2-pose-5b/tree/ada1f29aa1fd454ca28665c700923a0101b6b24f) | 20,480,899,148 | 5,120,198,663 | 5,120,198,644 |

Both downloaded files matched the pinned SHA-256. All 696 state tensors for
1B and 968 for 5B matched the official model's names and shapes. The CPU
preflight also verified the official 1024 × 768 crop, center, padding and
COCO17 flip permutation. The check constructed model tensors on the PyTorch
meta device and read the downloaded checkpoint headers.

The model cards list pose estimation and human-centric vision research as
intended uses. On 2026-11-24, the user accepted the pinned Meta Sapiens2 License
for this evaluation, including conversion to ONNX and TensorRT. Both Hugging
Face repositories returned `gated=false` and an unauthenticated redirect
for the pinned checkpoint name.
[User decision in #2492](https://github.com/Motoki0705/tennis-lab/issues/2492)
[1B metadata](https://huggingface.co/api/models/facebook/sapiens2-pose-1b?blobs=true)
[5B metadata](https://huggingface.co/api/models/facebook/sapiens2-pose-5b?blobs=true)

### Production adoption

1. Determine whether the prohibition on biometric processing covers this use of pose estimation. [License §1(b)(vi)](https://github.com/facebookresearch/sapiens2/blob/7e5bae88456ac418ff0e58e74106c9fe192055d4/LICENSE.md#1-license-rights-and-redistribution)
2. Include a copy of the Sapiens2 License when redistributing the materials or derivatives under its terms, including derived ONNX models and TensorRT engines. [License §1(b)(i)](https://github.com/facebookresearch/sapiens2/blob/7e5bae88456ac418ff0e58e74106c9fe192055d4/LICENSE.md#1-license-rights-and-redistribution)
3. Acknowledge use of Sapiens2 when publishing the research results. [License §1(b)(ii)](https://github.com/facebookresearch/sapiens2/blob/7e5bae88456ac418ff0e58e74106c9fe192055d4/LICENSE.md#1-license-rights-and-redistribution)

## Input and output

The pinned upstream test pipeline takes a single xyxy box, expands it by 1.25,
fits it to width:height 3:4 about its center, and uses an OpenCV UDP affine crop
at 1024 × 768 (H × W). The image preprocessor switches BGR to RGB and applies
the released mean and standard deviation in FP32 on the host. Both models use
two sequential forwards per batch, original and horizontal flip. The adapter
selects the COCO17 channels, restores left/right and width order, averages in
FP32, and downloads those 17 heatmaps for the upstream UDP/DARK decoder. UDP
uses the heatmap corner endpoints so the decoded points map back to native
pixels via the padded crop scale and center. [1B config](https://github.com/facebookresearch/sapiens2/blob/7e5bae88456ac418ff0e58e74106c9fe192055d4/sapiens/pose/configs/keypoints308/shutterstock_goliath_3po/sapiens2_1b_keypoints308_shutterstock_goliath_3po-1024x768.py)
[5B config](https://github.com/facebookresearch/sapiens2/blob/7e5bae88456ac418ff0e58e74106c9fe192055d4/sapiens/pose/configs/keypoints308/shutterstock_goliath_3po/sapiens2_5b_keypoints308_shutterstock_goliath_3po-1024x768.py)
[Pose demo](https://github.com/facebookresearch/sapiens2/blob/7e5bae88456ac418ff0e58e74106c9fe192055d4/sapiens/pose/tools/vis/vis_pose.py)

The upstream COCO-WholeBody mapping has an exact named channel for every
COCO17 joint. It uses the coarse nose, eyes and ears at channels 0–4 and maps
the left and right wrists to 62 and 41. The adapter verifies the label names
and all 17 flip destinations before inference. The output stores the raw
averaged heatmap maximum in the confidence column. Upstream's nonpositive
peak sentinel and other unusable coordinates appear as NaN in x/y. The
comparison scorer accepts every finite confidence attached to a valid position.
[Official joint definitions](https://github.com/facebookresearch/sapiens2/blob/7e5bae88456ac418ff0e58e74106c9fe192055d4/sapiens/pose/configs/_base_/keypoints308.py#L3818-L3844)

PT32 holds the model on CUDA in FP32 and uses the common IEEE precision
settings. All three backends set the official backbone's `pos_embed_rope_dtype`
to `fp32`. This also fixes the positional coordinate and rotation arithmetic
for the half-precision conditions. PT16 uses explicit FP16 casts for the
other model tensors and for the input. TRT16 uses a strongly typed engine
with FP16 inputs and outputs. FP32 heatmap averaging, CPU UDP decoding and the
two test views are shared across backends. The build retains TensorRT's
detailed layer inspector metadata and adapter inference disables NVTX.
The adapter's load time includes
source and checkpoint verification. [Official RoPE configuration](https://github.com/facebookresearch/sapiens2/blob/7e5bae88456ac418ff0e58e74106c9fe192055d4/sapiens/backbones/sapiens2.py#L520-L577)

## Reproduction

Use the pinned source and official checkpoint names. The source package
declares Python 3.12 or newer. Create an isolated environment for the upstream
model, common evaluator and TensorRT export. The shared benchmark environment
provides the package constraints, including PyTorch 2.13.0 with CUDA 13.0 and
TensorRT 10.14.1.48.post1. Keep the resolved package list with the run.
[Upstream requirements](https://github.com/facebookresearch/sapiens2/blob/7e5bae88456ac418ff0e58e74106c9fe192055d4/pyproject.toml)

```bash
run="$PWD/knowledge/runs/run-meiji-topdown-pose-comparison"
assets="$PWD/outputs/sapiens2"
media="$PWD/data/tennis_multiview/processed/meiji_4cam/benchmarks"
export PYTHONPATH="$PWD:$run"
mkdir -p "$assets"
git clone https://github.com/facebookresearch/sapiens2.git "$assets/upstream"
git -C "$assets/upstream" checkout --detach 7e5bae88456ac418ff0e58e74106c9fe192055d4
uv pip freeze --python .venv/bin/python --exclude-editable --exclude tennis-lab \
  > "$assets/runtime-constraints.txt"
uv venv --python 3.12 "$assets/runtime"
uv pip install --python "$assets/runtime/bin/python" \
  --constraint "$assets/runtime-constraints.txt" \
  "$assets/upstream" av einops omegaconf psutil pydantic pytorch-lightning smplx \
  onnx onnxscript nvidia-ml-py tensorrt-cu13
uv pip freeze --python "$assets/runtime/bin/python" > "$assets/runtime-freeze.txt"
.venv/bin/hf download facebook/sapiens2-pose-1b sapiens2_1b_pose.safetensors \
  --revision f5fed8b97b99698d5eea1d14ff0855d0b4c3f000 --local-dir "$assets/weights"
.venv/bin/hf download facebook/sapiens2-pose-5b sapiens2_5b_pose.safetensors \
  --revision ada1f29aa1fd454ca28665c700923a0101b6b24f --local-dir "$assets/weights"
```

For TRT16, export on the CPU using the local job runner. Build the engine in a
separate process with the shared training queue's exclusive reservation. Both
phases emit a conversion receipt for the shared harness. Set the build profile
and workspace to the recorded values for the condition. The commands below
show a batch 1 build.

```bash
size=1b
py="$assets/runtime/bin/python"
checkpoint="$assets/weights/sapiens2_${size}_pose.safetensors"
engine="$assets/trt-$size-fp32-rope-b1"
"$py" "$run/sapiens2_engine.py" onnx_export --model "sapiens2-$size" \
  --checkpoint "$checkpoint" --source-root "$assets/upstream" \
  --directory "$engine" --receipt "$assets/export-$size-fp32-rope-b1.json"
"$py" "$run/sapiens2_engine.py" engine_build --device cuda:0 \
  --optimum-batch 1 --maximum-batch 1 --workspace-mib 2048 \
  --directory "$engine" --receipt "$assets/build-$size-fp32-rope-b1.json"
```

For accuracy, use the common evaluator with the published #2495 manifest
digest. The example selects PT32 at batch 1. Set `--backend PT16` for half
precision, or `--backend TRT16 --engine "$engine"` for the built profile.
Run each CUDA condition through the shared training queue and retain
`receipt.json`, `scores.json`, and `predictions.npz` from the output path.

```bash
"$py" "$run/evaluate.py" \
  --bundle "$media/multiview-player-evaluation-v1" \
  --bundle-sha256 3dc496513380a39d4511ff7f8711f4192eababa5109f6a0506f88a60baa70580 \
  --media-root "$media/multiview-v1" \
  --model "sapiens2-$size" --backend PT32 --device cuda:0 --batch-size 1 \
  --checkpoint "$checkpoint" --source-root "$assets/upstream" \
  --output "${TENNIS_REPRO_DIR:?}/predictions/sapiens2-$size-pt32"
```

Use the published #2496 speed manifest and the [shared harness](../../../src/utils/inference_benchmark.md)
for latency, memory, and comparison against the same weight's PT32 output by
row ID. If 5B PT32 fails on the local GPU, 5B PT16 and TRT16 can still be scored
against the pseudo-reference. Retain their timing samples with the harness's
parity status and the reason the PT32 reference is unavailable.

## Evidence available before execution

The isolated runtime preflight passed with Python 3.12.3, PyTorch
2.13.0+cu130 (CUDA 13.0), TensorRT 10.14.1.48.post1 and ONNX Script 0.7.2.
The comparison exports passed the ONNX checker with FP16 input and output
bindings. Graph inspection found only FP32, FP16 and INT64 cast targets.
The 1B export took 138.81 s and the 5B export took 686.88 s.
[Conversion receipts and SHA-256 values](sapiens2_conversion.json)

| Model | PT32 | PT16 | TRT16 |
|---|---|---|---|
| Sapiens2-1B | Pending GPU run | Pending GPU run | ONNX checked. Pending engine build and GPU run |
| Sapiens2-5B | Pending CUDA allocation attempt | Pending GPU run | ONNX checked. Pending engine build and GPU run |

The 5B checkpoint contains 20,480,794,652 bytes (19.074 GiB) of FP32
tensors. The RTX 5060 Ti reported 16,311 MiB. The checkpoint therefore
requires about 3,221 MiB more than the installed VRAM before CUDA runtime,
activations and scratch space. Record the actual PT32 load failure and peak
allocator figures after the local test.

The CPU tests exercise the named source joints and left/right reflection
using asymmetric predictions, the crop inverse and short batch ordering,
nonpositive peaks with preserved confidence, strict checkpoint binding, and
TRT availability without a plan. They use synthetic inputs and weights.
The official model forward, TensorRT builds and fixed-cohort measurements
are pending.
