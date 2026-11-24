# RTMPose-M and RTMPose-X

Both checkpoints use the official Body7 training mixture, the 700-epoch Halpe26 recipe and a 384×288 image (height × width). Body8 in the configuration name refers to the evaluation mixture, which adds OCHuman. The M and X recipes specify training batches of 512 and 256 per device, respectively. The model IDs, config filenames and checkpoint SHA-256 values are in [rtmpose.py](rtmpose.py). The downloaded files are 56,436,913 bytes (M) and 200,397,852 bytes (X). [Official model zoo](https://github.com/open-mmlab/mmpose/blob/5408bc76f5b848cf925a0d1857899011d8c5b497/configs/body_2d_keypoint/rtmpose/body8/rtmpose_body8-halpe26.md)

The first 17 Halpe26 channels have a one-to-one correspondence with COCO17, including the nose, eyes, ears and the left/right limb pairs. Both configs map COCO channels `0..16` to the same Halpe indices. The adapter checks the checkpoint metadata against the upstream Halpe definition before loading weights. [Halpe26 definition](https://github.com/open-mmlab/mmpose/blob/7bea17cdf362cdd731fc5e79b982e4b79be5cca9/configs/_base_/datasets/halpe26.py) · [RTMPose-M config](https://github.com/open-mmlab/mmpose/blob/7bea17cdf362cdd731fc5e79b982e4b79be5cca9/configs/body_2d_keypoint/rtmpose/body8/rtmpose-m_8xb512-700e_body8-halpe26-384x288.py) · [RTMPose-X config](https://github.com/open-mmlab/mmpose/blob/7bea17cdf362cdd731fc5e79b982e4b79be5cca9/configs/body_2d_keypoint/rtmpose/body8/rtmpose-x_8xb256-700e_body8-halpe26-384x288.py)

MMPose code is Apache-2.0. The official zoo supplies both checkpoints by public download, and a maintainer gave general permission for Body7 model use in response to a commercial-use question. This evaluation uses local copies of the released artifacts. Commercial deployment and redistribution of these exact checkpoints or converted engines remain unconfirmed. The checkpoint-specific licensing and training-data questions are open upstream. [Code license](https://github.com/open-mmlab/mmpose/blob/7bea17cdf362cdd731fc5e79b982e4b79be5cca9/LICENSE) · [Maintainer response](https://github.com/open-mmlab/mmpose/issues/3062#issuecomment-2136896681) · [Checkpoint terms discussion](https://github.com/open-mmlab/mmpose/issues/3275)

## Test design

- Small CPU unit fixtures exercise two distinct boxes, asymmetric left/right joints and unequal x/y SimCC peaks. The checks cover native-pixel reprojection, input order, the model's missing-coordinate sentinel and preservation of its raw score. A malformed Halpe keypoint order must be rejected. These cases require no model files and fit the unit-test timing limit.
- Reuse `test_pose_reports_missing_person_and_missing_joint_in_coverage_denominators` for the treatment of NaN coordinates after scoring, and `test_open_keeps_empty_images_and_rejects_changed_files` for the pinned reference loader. The adapter tests exercise the RTMPose-specific transformations and backend boundaries that those tests do not call.
- Compare the staged CPU FP32 adapter with the official `inference_topdown` on two real pseudo-BBoxes from one pinned camera frame for each checkpoint. Verify both native positions and raw scores. This covers affine cropping, RGB normalization, flip fusion and the output slice. Report GPU parity separately against CUDA PT32 on the fixed comparison inputs.

## Inference and export settings

The adapter uses MMPose's `GetBBoxCenterScale` with padding 1.25 and `TopdownAffine` with width:height 3:4, bilinear interpolation and zero borders. It converts the crop from BGR to RGB and normalizes in FP32 with mean `[123.675, 116.28, 103.53]` and standard deviation `[58.395, 57.12, 57.375]` on the 0–255 scale. It applies the official horizontal flip test in every backend. Each batch executes two forwards. After swapping the flipped view's Halpe left/right channels and reversing the x bins, it averages the logits in FP32. The CPU SimCC decoder divides peak indices by 2, takes the smaller x/y peak as the raw score and restores coordinates using the crop center and scale. It maps the decoder's missing-coordinate sentinel to NaN and retains finite raw scores.

PT32 uses explicit FP32, with TF32 disabled by the CUDA evaluation and benchmark runners. PT16 uses explicit FP16 weights and inputs with autocast disabled. The strongly typed TRT16 graph has FP16 parameters, inputs and SimCC outputs. Flip fusion and decoding use the same code outside all three model graphs. Engine inspector output records the realized layer formats when CUDA runs. The TensorRT context disables NVTX markers during inference. [Engine inspector](https://docs.nvidia.com/deeplearning/tensorrt/10.x.x/inference-library/engine-tools.html)

The experiment uses the MMPose revision that includes OpenMMLab's CSPNeXt port and precedes the EDPose import. It explicitly selects MMPose's CSPNeXt in the official configs and loads each checkpoint state dict strictly. Local `mmcv-lite` suffices for this path. The older `xtcocotools` wheel was compiled for NumPy 1, so the setup builds it against the repo's NumPy 2 in a worktree-specific package overlay. [CSPNeXt port](https://github.com/open-mmlab/mmpose/pull/2620) · [MMPose source](https://github.com/open-mmlab/mmpose/tree/7bea17cdf362cdd731fc5e79b982e4b79be5cca9)

## Reproduce

Prepare the pinned source and dependencies from the checkout root. The package overlay uses the current repo environment for Python, Torch, NumPy and OpenCV.

```bash
run=knowledge/runs/run-meiji-topdown-pose-comparison
artifact="$PWD/outputs/rtmpose"
source_root="$artifact/vendor/mmpose"
overlay="$artifact/vendor/python"
pin=7bea17cdf362cdd731fc5e79b982e4b79be5cca9
mkdir -p "$artifact/vendor" "$artifact/weights"
git init "$source_root"
git -C "$source_root" fetch --depth=1 https://github.com/open-mmlab/mmpose.git "$pin"
git -C "$source_root" checkout --detach FETCH_HEAD
uv pip install --python .venv/bin/python --target "$overlay" --no-deps \
  mmcv-lite==2.1.0 mmengine==0.10.7 addict==2.4.0 yapf==0.40.2 \
  json-tricks==3.17.3 munkres==1.1.4 Cython==3.1.3 wheel==0.45.1
PYTHONPATH="$overlay" uv pip install --python .venv/bin/python --target "$overlay" \
  --no-binary xtcocotools --no-build-isolation --no-deps xtcocotools==1.14.3
export PYTHONPATH="$PWD/$run:$overlay:$source_root:$PWD"
```

Fetch each official checkpoint and export its raw SimCC graph. Repeat the commands with `model=rtmpose-x` for the X condition.

```bash
model=rtmpose-m
read -r filename sha256 < <(.venv/bin/python -B -c \
  'import sys; from rtmpose import MODELS; spec = MODELS[sys.argv[1]]; print(spec.checkpoint_name, spec.sha256)' "$model")
weight="$artifact/weights/$filename"
curl --fail --location "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/$filename" \
  --output "$weight.part"
printf '%s  %s\n' "$sha256" "$weight.part" | sha256sum --check
mv -T "$weight.part" "$weight"
engine="$artifact/engines/$model"
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=2 \
  .venv/bin/python -B "$run/rtmpose_convert.py" export --model "$model" \
  --checkpoint "$weight" --source-root "$source_root" --engine "$engine" \
  --receipt "$artifact/engines/$model-export.json"
```

Use the published #2495 accuracy bundle and digest. The commands below reproduce the CPU smoke and, inside an exclusive training queue allocation, the TRT16 measurements of pose accuracy. The evaluation command selects PT32 or PT16 through `--backend`. The `--engine` argument belongs to TRT16. The build targets batches 1–16 with an optimum of 8 and uses the current scene TensorRT builder's 1536 MiB workspace setting. The #2496 speed measurement and PT32 parity use the published speed inputs when available.

```bash
media=data/tennis_multiview/processed/meiji_4cam/benchmarks
bundle="$media/multiview-player-evaluation-v1"
bundle_sha=3dc496513380a39d4511ff7f8711f4192eababa5109f6a0506f88a60baa70580
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=2 \
  .venv/bin/python -B "$run/rtmpose_smoke.py" --bundle "$bundle" \
  --bundle-sha256 "$bundle_sha" --media-root "$media/multiview-v1" \
  --weights "$artifact/weights" --source-root "$source_root"

.venv/bin/python -B "$run/rtmpose_convert.py" build --engine "$engine" \
  --opt-batch 8 --max-batch 16 --workspace-mib 1536 --device cuda:0 \
  --receipt "$TENNIS_REPRO_DIR/build.json"
.venv/bin/python -B "$run/evaluate.py" --bundle "$bundle" \
  --bundle-sha256 "$bundle_sha" --media-root "$media/multiview-v1" \
  --model "$model" --backend TRT16 --device cuda:0 --batch-size 8 \
  --checkpoint "$weight" --source-root "$source_root" --engine "$engine" \
  --output "$TENNIS_REPRO_DIR/predictions"
.venv/bin/python -B "$run/rtmpose_convert.py" inspect --model "$model" \
  --checkpoint "$weight" --source-root "$source_root" --engine "$engine" \
  --batch 8 --device cuda:0 --receipt "$TENNIS_REPRO_DIR/layers-b8.json"
```

## CPU verification

On `video_000/clip_003`, `cam0`, source frame 120, both models gave an exact match to the official CPU FP32 API for the native coordinates and raw scores of the two fixed boxes after normalizing the missing-coordinate convention. The last box gave identical coordinates alone and in the two-box batch. The maximum raw score difference was `8.34465e-7` for M and `2.980232e-7` for X. Each invocation made two forwards. The strict-load parameter counts were 14,056,356 and 49,998,772.

Both FP16 ONNX exports passed the ONNX checker under PyTorch 2.13.0+cu130 and ONNX 1.21.0. M took 13.47 seconds and X took 28.54 seconds in separate processes that overlapped on the host. These timings include config and checkpoint loading. Both graphs have FP16 initializers and declare dynamic batch length.

| Model | ONNX SHA-256 |
|---|---|
| RTMPose-M | `8ff280c64e2ce02419ed28e5f010589f310732719bfea98d47b2af958f0bbacf` |
| RTMPose-X | `0a374c720fc1551fb35fce8fb240cd467511844a4de569202708b1e1c0c6df54` |

The focused CPU suite passed 24 tests in 8.60 seconds. That includes three new RTMPose tests and 21 existing tests for the comparison, reference scoring, bundle and timing contracts.
