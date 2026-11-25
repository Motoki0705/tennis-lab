# Top-down COCO17 pose comparison on Meiji 4cam

The six frozen clips and masks from [the player evaluation bundle](../run-meiji-multiview-player-evaluation/README.md) are the accuracy reference. Each condition receives the same native BGR images and eligible pseudo boxes. The boxes enclose projected joints from placed GVHMR SMPL. The model input carries no joint coordinates or masks. The reference has 1,711 source frames across six clips, 6,844 camera images and 26,276 eligible person boxes.

The reference inherits biases from ViTPose observations, body placement, GVHMR and SMPL. Agreement measures how closely a prediction follows this reference. It cannot establish accuracy against independent human annotations. See the shared bundle for the overlay, exclusions and mask definitions.

## Accuracy command

From the checkout root, use the published manifest digest. Run CUDA commands through the shared training queue with the `all` reservation.

```bash
run=knowledge/runs/run-meiji-topdown-pose-comparison
media=data/tennis_multiview/processed/meiji_4cam/benchmarks
PYTHONPATH="$PWD:$PWD/$run" .venv/bin/python -B "$run/evaluate.py" \
  --bundle "$media/multiview-player-evaluation-v1" \
  --bundle-sha256 3dc496513380a39d4511ff7f8711f4192eababa5109f6a0506f88a60baa70580 \
  --media-root "$media/multiview-v1" \
  --model vitpose-h --backend PT32 --device cuda:0 --batch-size 8 \
  --checkpoint ckpt/vitpose/vitpose-h-multi-coco.pth \
  --checkpoint-sha256 50e33f4077ef2a6bcfd7110c58742b24c5859b7798fb0eedd6d2215e0a8980bc \
  --output outputs/pose-comparison/vitpose-h-pt32
```

For FP16 autocast, select `PT16`. For `TRT16`, pass `--engine` with a validated scene engine directory for the selected flip setting and batch range. Select `vitpose-h-no-flip` for the supplemental single-view forward. The ViTPose-H accuracy adapter uses the same square padding factor 1.2, half-resolution image crop, 256×192 model input, flip restoration and UDP decoding as the scene model. PyTorch FP16 uses production autocast, with FP32 heatmap transfer and UDP decoding. The two flip inputs share one forward.

The current checkpoint was distributed for GVHMR as `vitpose-h-multi-coco.pth`. The licensed source and training data terms are documented in [LICENSES.md](../../../LICENSES.md#third-party-weights-and-body-model-assets). See the [pinned ViTPose model zoo](https://github.com/ViTAE-Transformer/ViTPose/blob/c050ed29112da7704797cc1a65af0234b525010d/README.md) and [pinned GVHMR distribution guide](https://github.com/zju3dv/GVHMR/blob/ee960bb6e2ea2d381aa97f08e9b71ef320b624b1/docs/INSTALL.md).

Each successful condition writes `receipt.json`, `scores.json` and checksum-pinned `predictions.npz`. The last file holds one row per `(clip_id, camera_id, source_frame, player_id)`, input bbox, native-pixel COCO17 `keypoints_xyc` and optional `extra__NAME` arrays aligned with that axis. `scores.json` uses the shared `score_players` on the input boxes. For distinct boxes the exact IoU 1.0 matches recover the known input association. The fixed-denominator pose metrics include missing output joints. `finite_pairs` uses the same OKS and PCK definitions on eligible finite pairs. Every finite raw confidence is eligible, including zero and negative values. A joint with any nonfinite field is missing. Box matches and their placeholder scores only support association.

Use the [#2496 harness](../../../src/utils/inference_benchmark.md) and its published fixed speed input for candidate latency and memory. The ViTPose-H speed rows use the [#2496 baseline](https://github.com/Motoki0705/tennis-lab/issues/2496). This accuracy runner has no latency measurements.

## Replay retained accuracy

`replay_accuracy.py` verifies each completed queue receipt against the frozen cohort, original file hashes, BBoxes and person addresses. It re-scores the saved COCO17 rows with the current common scorer and records which metric sections changed, the two Recall values, and coordinate and raw-score differences from the same model's PT32 outputs. The PT32 comparison reports observed differences for the accuracy cohort. The shared timing harness uses its own fixed input and parity policy.

```bash
run=knowledge/runs/run-meiji-topdown-pose-comparison
media=data/tennis_multiview/processed/meiji_4cam/benchmarks
queue_root="$(dirname "$(git rev-parse --path-format=absolute --git-common-dir)")/.training_queue"
mapfile -t VITPOSE_ACCURACY_JOBS < <(jq -r '.conditions[][].job_id' "$run/vitpose_accuracy.json")
PYTHONPATH="$PWD:$PWD/$run" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  .venv/bin/python -B "$run/replay_accuracy.py" --queue "$queue_root" \
  --bundle "$media/multiview-player-evaluation-v1" \
  --bundle-sha256 3dc496513380a39d4511ff7f8711f4192eababa5109f6a0506f88a60baa70580 \
  --output outputs/pose-comparison/vitpose-replay --retain "${VITPOSE_ACCURACY_JOBS[@]}"
```

Supply the six complete `.job` IDs in `VITPOSE_ACCURACY_JOBS`. `--retain` preserves the original receipt, predictions, scores and log in the queue's retained `repro/<job>/predictions/`. The replay writes its own report directory and keeps the prior scores available for comparison.

## Review synchronized overlays

Each panel uses the same native frame, player-box geometry and frame label. The SMPL panel draws the eligible reference joints. Model panels draw their finite predictions. Use the original camera frame number for a PNG, or omit `--frame` and set an MP4 output to render the full evaluation timeline of that camera.

```bash
mapfile -t vi_jobs < <(jq -r '.conditions["vitpose-h"] | .PT32.job_id, .TRT16.job_id' "$run/vitpose_accuracy.json")
PYTHONPATH="$PWD:$PWD/$run" .venv/bin/python -B "$run/render_overlays.py" \
  --queue "$queue_root" --bundle "$media/multiview-player-evaluation-v1" \
  --bundle-sha256 3dc496513380a39d4511ff7f8711f4192eababa5109f6a0506f88a60baa70580 \
  --media-root "$media/multiview-v1" --clip video_000/clip_003 --camera cam2 \
  --frame 66 --player 6 --output outputs/pose-comparison/clip_003-cam2-66.png "${vi_jobs[@]}"
```

The five-model PT32 video for `clip_003/cam2` uses the PT32 job IDs from `vitpose_accuracy.json`, `rtmpose_accuracy.json` and `pmpose_accuracy.json`. Select `vitpose-h` as the ViTPose variant. The frame count and hashes of both previews are recorded in the [knowledge node](../../nodes/tennis_scene/000086-run-meiji-topdown-pose-comparison.md).

## Adapter and tests

The adapter contract binds one explicit backend, input order and COCO17 axis. Each adapter declares its forward calls with `stages.measure_forward(backend, operation, name="pose")` and may return person-aligned supplementary outputs. Renderer and analysis code can use row addresses to join with the frozen source. [Input and output contract](https://github.com/Motoki0705/tennis-lab/issues/2490#issuecomment-7129915413) · [Forward method](https://github.com/Motoki0705/tennis-lab/issues/2490#issuecomment-7131482779)

The focused CPU tests use unequal boxes, nonconsecutive frames, a partially empty cohort and a short final batch to expose reordering, skipped rows, mismatched backends and confidence-induced false misses. Replay tests reject a reordered player column and recover a retained copy after the temporary source is removed. The overlay test checks exact display pixels from a known source frame and respects the reference mask. The tests reuse the shared scorer's reference dataclasses and the production ViTPose frame sequence. The focused cases use tiny arrays and an injected frame stream, with no external weights, video or GPU. [Coverage test](../../../tests/unit/tennis_scene/evaluation/player_pose/test_scoring.py) · [ViTPose decoding test](../../../tests/unit/submodules/models/vitpose/test_pose2d.py)
