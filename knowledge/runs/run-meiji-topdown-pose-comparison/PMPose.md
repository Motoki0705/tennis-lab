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

Use the [common accuracy runner](README.md#accuracy-command) with `--model pmpose-l` or `pmpose-h`, the matching checkpoint and SHA-256 from `pmpose_assets.py`, and `--source-root "$source_root"`. Record both the scores against the pinned #2495 bundle and coordinate differences from that model’s PT32 predictions.

## Tests

Use small CPU fixtures with no weight or GPU requirement. Compare the rectangle contour against the demo’s raster-plus-OpenCV calculation, including fractional and edge-clipped boxes. An asymmetric pose with two boxes from one frame exercises row order, native-pixel decoding, flip restoration of eyes and limbs, the full 23-joint auxiliary axis and a missing heatmap. Inject distinct scores including zero to detect silent score filtering. Run the same assertion through the accuracy stages and the shared timing stages to check that both count the two real forwards. Loader tests exercise explicit rejection of an absent source, wrong checkpoint and absent TRT engine. Reuse `tests/unit/utils/test_inference_benchmark.py::test_exact_backend_selection_never_loads_another_backend` and the shared pose scorer tests for backend selection and fixed-denominator coverage. Measure the focused tests with `--durations=0 --durations-min=0`. The real-asset smoke will compare the adapter’s PT32 output against the upstream API on the same images, rectangles and weights.
