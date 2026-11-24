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
- PT32 uses CUDA with TF32 disabled. PT16 uses CUDA float16 autocast for the detector forward, including the PyTorch autocast policy for operations that require FP32. TRT16 runs the exported detector graph using typed FP16 internals and FP32 input/output bindings. The same GPU Torch FP32 postprocessor follows all three forwards. ONNX export uses a variable batch axis and fixed 640 × 640 input.

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

Use the recorded weight hash for `--weight-sha256`. The CLI checks the source commit and weights before loading the model. This is the batch-one accuracy entry point for use in the shared training queue. `COHORT_SHA256` comes from #2495.

```bash
bench=data/tennis_multiview/processed/meiji_4cam/benchmarks
PYTHONPATH=".:$assets/python-deps" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  .venv/bin/python -B knowledge/runs/run-meiji-player-model-comparison/rtdetrv4.py \
  --model-id rtdetrv4-l --upstream "$assets/RT-DETRv4" \
  --weights "$assets/RTv4-L-hgnet.pth" --weight-sha256 "$L_WEIGHT_SHA256" \
  accuracy --backend pt32 --device cuda:0 \
  --source "$bench/multiview-v1" --benchmark "$bench/multiview-player-evaluation-v1" \
  --manifest-sha256 "$COHORT_SHA256" --output outputs/player-model-comparison/rtdetrv4-l/pt32
```

## Test design

- The CPU unit fixtures check source width/height order and input channel scaling on unequal native image sizes. A stage-level fixture with two images and hand-computed pixel boxes checks alignment and the declared forward backend.
- A two-image fixture checks that COCO person rows retain their score order and valid clipped coordinates, including a score below the demo threshold. The fixture also checks an empty result. A separate malformed output catches non-finite boxes becoming finite through clipping.
- Reuse `test_exact_backend_selection_never_loads_another_backend` for the shared benchmark's backend routing, `test_one_detection_cannot_match_two_players_and_an_unlabelled_box_has_no_fp` for partial-label scoring, and `test_pose_rows_keep_both_people_and_zero_detection_frames` for JSONL rows. The four new cases require neither checkpoints nor CUDA. Measure with `pytest --durations=0 --durations-min=0.01` and qualify the upstream adapter separately with the actual L/X checkpoints.
