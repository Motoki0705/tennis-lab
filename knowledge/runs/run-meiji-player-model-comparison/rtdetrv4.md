# RT-DETRv4-L and RT-DETRv4-X

Both are official COCO2017 detector checkpoints at source commit `55fefaaed7efe2a5f72d0a18fd4e05965e35c292`. The configurations use PP-HGNetV2 B4/B5 with the published stage-1 backbone initialization and DINOv3 ViT-B/16 distillation (`LVD-1689M`). The teacher is only used during training. The released detector graph can run with the supplied checkpoint alone. The parameter counts below are from the official training logs. [Model zoo](https://github.com/AIMIA-PKU/RT-DETRv4/blob/55fefaaed7efe2a5f72d0a18fd4e05965e35c292/README.md#-performance) · [RT-DETRv4 paper](https://arxiv.org/html/2510.25257v2) · [Backbone initialization](https://github.com/AIMIA-PKU/RT-DETRv4/blob/55fefaaed7efe2a5f72d0a18fd4e05965e35c292/engine/backbone/hgnetv2.py)

| Model ID | Parameters | Published checkpoint | Bytes |
| --- | ---: | --- | ---: |
| `rtdetrv4-l` | 30,967,267 | [RTv4-L-hgnet.pth](https://drive.google.com/file/d/1shO9EzZvXZyKedE2urLsN4dwEv8Jqa_8/view) | 505,400,981 |
| `rtdetrv4-x` | 61,991,235 | [RTv4-X-hgnet.pth](https://drive.google.com/file/d/19gnkMTgFveJsrOvSmEPQXCTG6v9oQHN3/view) | 1,010,474,984 |

The code carries Apache-2.0. The model zoo links to public, ungated downloads for both COCO checkpoints. The linked checkpoint files and model-zoo text provide no separate terms for the weight binaries. The experiment uses them for local evaluation. Production use and redistribution of the weight binaries remain unverified. The upstream code license permits commercial use and redistribution subject to its terms. [Pinned license](https://github.com/AIMIA-PKU/RT-DETRv4/blob/55fefaaed7efe2a5f72d0a18fd4e05965e35c292/LICENSE) · [Official weight listing](https://github.com/AIMIA-PKU/RT-DETRv4/blob/55fefaaed7efe2a5f72d0a18fd4e05965e35c292/README.md#-performance)

## Inference contract

- Decode the source frame in BGR and convert to RGB. Stretch the whole native image to 640 × 640 using PIL bilinear interpolation, then convert to `[0,1]` without mean or standard-deviation normalization. The original width and height restore native box coordinates. [Upstream inference](https://github.com/AIMIA-PKU/RT-DETRv4/blob/55fefaaed7efe2a5f72d0a18fd4e05965e35c292/tools/inference/torch_inf.py)
- Strictly restore the official EMA detector and call its deployment conversion with backbone auto-download disabled. Use the upstream focal-loss postprocessor in FP32 for all backends. Its scores are sigmoid probabilities for the global top 300 query/class pairs. Retain the internal class 0 (COCO person/category 1), clip native `xyxy` to the image, and remove boxes with zero area after clipping. Retain the scores for the shared scorer's cutoff sweep. The upstream visualization draws scores above 0.4. The inference graph applies no NMS. [Upstream postprocessor](https://github.com/AIMIA-PKU/RT-DETRv4/blob/55fefaaed7efe2a5f72d0a18fd4e05965e35c292/engine/rtv4/postprocessor.py)
- PT16 uses CUDA float16 autocast for the detector forward with TF32 disabled, including the PyTorch autocast policy for operations that require FP32. TRT16 runs the exported detector graph using typed FP16 internals and FP32 input/output bindings. The same GPU Torch FP32 postprocessor follows both forwards. Trace ONNX with two images to capture the decoder's batch-dependent anchor expansion. Its batch axis remains variable with fixed 640 × 640 spatial dimensions.

## Local setup

From the repository root, use a worktree-local asset directory. The two extra Python packages satisfy imports of the pinned upstream package in this environment. Install them into a separate target and add it to `PYTHONPATH` for these commands.

```bash
assets=outputs/player-model-comparison/upstream
mkdir -p "$assets"
git clone https://github.com/AIMIA-PKU/RT-DETRv4.git "$assets/RT-DETRv4"
git -C "$assets/RT-DETRv4" checkout --detach 55fefaaed7efe2a5f72d0a18fd4e05965e35c292
uv pip install --python .venv/bin/python --target "$assets/python-deps" --no-deps \
  faster-coco-eval==1.8.1 calflops==0.3.2
curl --fail --location --output "$assets/RTv4-L-hgnet.pth" \
  'https://drive.usercontent.google.com/download?id=1shO9EzZvXZyKedE2urLsN4dwEv8Jqa_8&export=download&confirm=t'
curl --fail --location --output "$assets/RTv4-X-hgnet.pth" \
  'https://drive.usercontent.google.com/download?id=19gnkMTgFveJsrOvSmEPQXCTG6v9oQHN3&export=download&confirm=t'
```

The [adapter](rtdetrv4.py) pins both published file digests under `WEIGHT_SHA256`. The original files use PyTorch's pre-ZIP serialization, and the adapter loads them directly with `weights_only=True` and strict state-dict matching. On `video_000/clip_003`, `cam0`, source frame 120, the CPU smoke found 73 L detections and 64 X detections among the global top 300, with 10 and 9 respectively above 0.4. This verifies checkpoint loading and the deployed detector's output contract on a native video frame.

The batch-one accuracy command runs PT16 or TRT16 in the shared training queue. `COHORT_SHA256` comes from #2495. Select `--parity-not-evaluated-reason PT32未計測` explicitly for each condition. For TRT16, give `--engine-dir` for the same checkpoint's built plan.

```bash
bench=data/tennis_multiview/processed/meiji_4cam/benchmarks
PYTHONPATH=".:$assets/python-deps" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  .venv/bin/python -B knowledge/runs/run-meiji-player-model-comparison/rtdetrv4.py \
  --model-id rtdetrv4-l --upstream "$assets/RT-DETRv4" \
  --weights "$assets/RTv4-L-hgnet.pth" \
  accuracy --backend pt16 --device cuda:0 --parity-not-evaluated-reason PT32未計測 \
  --source "$bench/multiview-v1" --benchmark "$bench/multiview-player-evaluation-v1" \
  --manifest-sha256 "$COHORT_SHA256" --output outputs/player-model-comparison/rtdetrv4-l/pt16
```

For TensorRT preparation, run the CPU `export` step first. It creates the ONNX graph, the binding manifest and a timing or failure receipt. Run `build` in the shared training queue with the profile chosen for the speed sweep.

```bash
trt=outputs/player-model-comparison/rtdetrv4-l/trt16/trace-b2
PYTHONPATH=".:$assets/python-deps" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  .venv/bin/python -B knowledge/runs/run-meiji-player-model-comparison/rtdetrv4.py \
  --model-id rtdetrv4-l --upstream "$assets/RT-DETRv4" --weights "$assets/RTv4-L-hgnet.pth" \
  export --engine-dir "$trt/engine" --receipt "$trt/onnx-export.json"
```

Use `build --device cuda:0 --engine-dir "$trt/engine" --receipt "$trt/engine-build.json" --opt-batch 2 --max-batch 2 --workspace-gib 1` after the same model and asset arguments, in the queue. This sets a 1 GiB workspace limit and the same min/opt/max image batches of 1/2/2 as the #2496 DINO engine. This plan is shared by the batch-one accuracy and both speed cases.

`accuracy.json` records agreement with the fixed SMPL-derived reference. Following the 2026-11-25 comparison decision, new accuracy and speed reports pair `parity_status: not_evaluated` with `parity_not_evaluated_reason: PT32未計測`.

## ONNX conversion evidence

Both initial batch-one traces completed at `9fe7b3987e52b7dd0f0d255e633bb310b0120861`. The common TensorRT exporter checked the graphs. A second read verified the per-file SHA-256 digests in `export.json`, ONNX opset 19, 18 GridSample nodes per network and FP16 for all floating-point initializers. The manifest below pins both the graph and its external weight file under `outputs/player-model-comparison/<model-id>/trt16/engine/`.

| Model | ONNX export wall time | `export.json` SHA-256 | Local job |
| --- | ---: | --- | --- |
| L | 23.265 s | `83d24e3485ef95e1d8ad9358ad0a0b6e487837df73cf361487f2bae8b4411e44` | `20261124T114215Z_5ea907a8d63a_rtdetrv4-l-onnx` |
| X | 32.411 s | `6b29778f5a14d3a57aa8405b03d0777afce586531496610af8caf5659f7ae64b` | `20261124T114215Z_9940a8133486_rtdetrv4-x-onnx` |

The initial GPU attempt at `fb0c76e6f7a0ca6afa035f995b500c18c8ff1c95` stopped at the first PT32 forward in each model. The upstream encoder stores position tensors as plain CPU attributes. The adapter now registers those existing tensors as nonpersistent buffers after strictly loading the EMA state, so they follow the model to CUDA and keep their values. The initial TensorRT build selected a fixed batch-one binding from a symbolic ONNX input and failed the shared engine validation. The retry traces batch two and uses a profile spanning batches 1 and 2, keeping its conversion receipts under `trt16/trace-b2/`. PT16 and TRT16 accuracy in the initial attempt stopped at their missing prerequisites.

## Shared speed input

The [fixed baseline bundle](../run-person-model-speed-baseline/README.md) publishes `meiji_4cam/person-model-speed-v1`, SHA-256 `64c74d75e5f1afe869911f1620b9c8e8a793fc4c4e4ac830f4abd97b21fb1a62`. Use its 96 decoded full frames and their order for PT16 and TRT16, with image batches 1 and 2. The suite uses #2496's checksummed timing protocol, a dedicated CUDA stream, and separate processes per timing case. `speed-b1.json` and `speed-b2.json` retain the shared harness reports with raw iteration timing, memory, environment and the PT32 parity status. Full-cohort detections for pseudo-GT scoring are in each condition's `predictions.jsonl`.

For each variant, give the fixed speed bundle's `inputs/` directory and both manifest digests to the following command in the shared GPU queue. The accuracy manifest SHA-256 is `3dc496513380a39d4511ff7f8711f4192eababa5109f6a0506f88a60baa70580`.

```bash
bash knowledge/runs/run-meiji-player-model-comparison/rtdetrv4_run.sh \
  rtdetrv4-l "$SHARED_PERSON_SPEED_INPUTS" "$SPEED_SHA256" "$COHORT_SHA256"
```

## Test design

- The CPU unit fixtures check source width/height order and input channel scaling on unequal native image sizes. A stage-level fixture with two images and hand-computed pixel boxes checks alignment and the declared forward backend.
- A two-image fixture checks that COCO person rows retain their score order and valid clipped coordinates, including a score below the demo threshold. The fixture also checks an empty result. A separate malformed output catches non-finite boxes becoming finite through clipping.
- A fixture saved in the release's old format reproduces the original load failure and verifies that the EMA state is selected. The encoder test checks that the same cached tensors enter nonpersistent buffers and move with `.to()`.
- The command tests check that accuracy and speed require an explicit parity status and reject PT32. A stubbed queue suite checks that both FP16 backends reach accuracy and both speed batches, and that TensorRT conversion still runs. The shared harness tests cover the explicit opt-out and its default required-parity behavior.
- Reuse `test_exact_backend_selection_never_loads_another_backend` for the shared benchmark's backend routing, `test_one_detection_cannot_match_two_players_and_an_unlabelled_box_has_no_fp` for partial-label scoring, and `test_pose_rows_keep_both_people_and_zero_detection_frames` for JSONL rows. The fixtures run on CPU with generated data. Measure with `pytest --durations=0 --durations-min=0.01` and qualify the upstream adapter separately with the actual L/X checkpoints.
