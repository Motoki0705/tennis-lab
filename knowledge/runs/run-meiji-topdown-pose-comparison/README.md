# Top-down COCO17 pose comparison on Meiji 4cam

The six frozen clips and masks from [the player evaluation bundle](../run-meiji-multiview-player-evaluation/README.md) are the accuracy reference. Each condition receives the same native BGR images and eligible pseudo boxes. The boxes enclose projected joints from placed GVHMR SMPL. The model input carries no joint coordinates or masks. The reference has 1,711 source frames across six clips, 6,844 camera images and 26,276 eligible person boxes.

The reference inherits biases from ViTPose observations, body placement, GVHMR and SMPL. Agreement measures how closely a prediction follows this reference. It cannot establish accuracy against independent human annotations. See the shared bundle for the overlay, exclusions and mask definitions.

The [accuracy table](accuracy.csv) contains 15 primary conditions for ViTPose-H, RTMPose-M/X and PMPose-L/H, plus three ViTPose-H conditions without flip. The six Sapiens2 rows preserve the attempts behind the November 25 decision to exclude both sizes due to GPU time. [Sapiens2 run evidence](sapiens2.md#conversion-and-accuracy-evidence)

## Accuracy command

From the checkout root, use the published manifest digest. Run CUDA commands through the shared training queue with the `all` reservation. For any new accuracy runs, use PT16 or TRT16 under the November 25 scope decision. The saved PT32 accuracy and backend differences remain available for reading and CPU replay. [Scope decision](https://github.com/Motoki0705/tennis-lab/issues/2492)

```bash
run=knowledge/runs/run-meiji-topdown-pose-comparison
media=data/tennis_multiview/processed/meiji_4cam/benchmarks
PYTHONPATH="$PWD:$PWD/$run" .venv/bin/python -B "$run/evaluate.py" \
  --bundle "$media/multiview-player-evaluation-v1" \
  --bundle-sha256 3dc496513380a39d4511ff7f8711f4192eababa5109f6a0506f88a60baa70580 \
  --media-root "$media/multiview-v1" \
  --model vitpose-h --backend PT16 --device cuda:0 --batch-size 8 \
  --checkpoint ckpt/vitpose/vitpose-h-multi-coco.pth \
  --checkpoint-sha256 50e33f4077ef2a6bcfd7110c58742b24c5859b7798fb0eedd6d2215e0a8980bc \
  --output outputs/pose-comparison/vitpose-h-pt16
```

For FP16 autocast, select `PT16`. For `TRT16`, pass `--engine` with a validated scene engine directory for the selected flip setting and batch range. Select `vitpose-h-no-flip` for the supplemental single-view forward. The ViTPose-H accuracy adapter uses the same square padding factor 1.2, half-resolution image crop, 256×192 model input, flip restoration and UDP decoding as the scene model. PyTorch FP16 uses production autocast, with FP32 heatmap transfer and UDP decoding. The two flip inputs share one forward.

The current checkpoint was distributed for GVHMR as `vitpose-h-multi-coco.pth`. The licensed source and training data terms are documented in [LICENSES.md](../../../LICENSES.md#third-party-weights-and-body-model-assets). See the [pinned ViTPose model zoo](https://github.com/ViTAE-Transformer/ViTPose/blob/c050ed29112da7704797cc1a65af0234b525010d/README.md) and [pinned GVHMR distribution guide](https://github.com/zju3dv/GVHMR/blob/ee960bb6e2ea2d381aa97f08e9b71ef320b624b1/docs/INSTALL.md).

Each successful condition writes `receipt.json`, `scores.json` and checksum-pinned `predictions.npz`. The last file holds one row per `(clip_id, camera_id, source_frame, player_id)`, input bbox, native-pixel COCO17 `keypoints_xyc` and optional `extra__NAME` arrays aligned with that axis. `scores.json` uses the shared `score_players` on the input boxes. For distinct boxes the exact IoU 1.0 matches recover the known input association. The fixed-denominator pose metrics include missing output joints. `finite_pairs` uses the same OKS and PCK definitions on eligible finite pairs. Every finite raw confidence is eligible, including zero and negative values. A joint with any nonfinite field is missing. Box matches and their placeholder scores only support association.

## Speed and memory

For ViTPose-H, reuse the synchronized PT16/TRT16 results with and without flip at batch 1/4/8. The baseline records include raw timings, the input order digest, and hashes of the weights and engines. [Published speed and memory](../run-person-model-speed-low-precision/README.md)

Use the common harness and pinned crop schedule for new candidate runs. Pass `parity_not_evaluated_reason="PT32計測はスコープ外（ユーザー判断 2026-11-25）"` and record the comparison label `PT32未計測`. Keep the original provisional PT32 measurements as historical data. [Harness and opt-out contract](../../../src/utils/inference_benchmark.md) · [Published dataset and timing protocol](../run-person-model-speed-baseline/README.md#固定入力と初回の成果物)

For the primary comparison, join fixed-denominator pseudo-GT accuracy to synchronized speed by model and backend. The CSV records `speed_batch_size` for timing and VRAM and `accuracy_batch_size` for the predictions used for scoring. All accuracy rows use saved batch 8 predictions. Accuracy at batches 1 and 4 was unmeasured. Compare each speed row to ViTPose-H with flip at the same speed batch and backend. Compute the Pareto frontier separately for each speed batch by maximizing mean OKS and minimizing wall P50 per person and sampled NVML device memory. The plot shows the batch 1 costs against batch 8 accuracy and marks membership in the three-metric frontier. Identical points share frontier membership. Supplemental ViTPose-H results without flip are available in the original speed and accuracy tables.

The published comparison contains all 30 synchronized speed cases and their fixed-box accuracy. The CSV reports latency and sampled-memory reductions versus the same backend and batch of ViTPose-H with flip. Positive reductions indicate lower time or memory. Throughput gain uses persons/s. Accuracy differences are percentage points, and the original accuracy table retains the historical PT32 values and output differences. [Comparison CSV](comparison.csv) · [Batch 1 Pareto plot](pareto-batch1.png) · [Original accuracy table](accuracy.csv)

Run the comparison on the CPU from the checkout root and choose a new output directory. The CLI checks the bytes it reads against the 24 case hashes and the baseline summary hash in `comparison_inputs.json`. If a future case ends before writing a report, pass `--omissions` with a JSON list of entries containing `model`, `backend`, `batch_size`, `reason`, and `evidence` pointing to the retained log. Record the terminal job and its log hash in the comparison index. The log must contain that exact case's `infeasible` status or nonzero process exit from the RTMPose sweep. A case with a pinned report must use that report. [Comparison inputs and queue records](comparison_inputs.json)

```bash
set -euo pipefail
run=knowledge/runs/run-meiji-topdown-pose-comparison
baseline=knowledge/runs/run-person-model-speed-baseline
root="$(dirname "$(git rev-parse --path-format=absolute --git-common-dir)")"
fixed="$root/.worktree-outputs/parent-2492-issue-2496-primary/person-model-speed-baseline-v1/inputs"
mapfile -t reports < <(jq -r --arg root "$root" '.reports[] | "\($root)/\(.path)"' "$run/comparison_inputs.json")
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$PWD:$PWD/$run:$PWD/$baseline" \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  .venv/bin/python -B "$run/compare_speed.py" --inputs "$fixed" \
  --output outputs/pose-comparison/final-speed-reproduced "${reports[@]}"
```

The comparison tests reuse `test_pinned_schedule_rejects_reordered_loaded_batches` and `test_explicit_parity_opt_out_records_speed_without_extra_inference` from the shared harness. The comparison-specific CPU tests check timed input order, Pareto membership and baseline improvements. They also verify both batch labels, rejection of a modified report summary and rejection of missing or unrelated terminal logs.

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

Supply the six complete `.job` IDs in `VITPOSE_ACCURACY_JOBS`. `--retain` preserves the original receipt, predictions, scores and log in the queue's retained `repro/<job>/predictions/`. For jobs that write their predictions into `repro` directly, run `--retain` after completion to preserve `queue.log` while the live queue log is still available. The ViTPose-H, RTMPose and PMPose accuracy records pin the retained logs. The replay writes its own report directory and keeps the prior scores available for comparison.

## Difficulty and temporal diagnostics

The diagnostic groups are fixed by the reference. A small projected person has a box under 80 native pixels high. Overlap means pseudo-box IoU greater than 0.10 in the same image. An edge case has a box within 10 native pixels of an image edge. The occlusion group has at least one joint flagged by the reference's inter-player occlusion heuristic. An incomplete reference has fewer than 17 scoreable joints.

Fast wrist motion is the top decile of the larger available left/right wrist speed on the placed SMPL in court metres. The cutoff comes from unique clip/player/time samples before repetition across cameras. Speed uses consecutive inferred samples at most two source frames apart and their elapsed time at the original source FPS. This group is a proxy for swings. Temporal jitter is the magnitude of the centered second difference of the signed model-minus-reference residual in native pixels across three samples spaced two source frames apart. The samples must belong to the same player in the same view, and all three joint references and predictions must be valid.

The reports give reference exclusions by clip and joint, valid and missing prediction counts, joint error distributions, PCK at 0.10/0.20 of the square root of the box area, and the fraction of instances with OKS at least 0.50/0.75. OKS requires at least one eligible joint. The occlusion and incomplete-reference results score the eligible joints of affected players. The excluded joints remain in the quality counts.

```bash
mapfile -t accuracy_jobs < <(jq -r '.conditions[][].job_id' "$run"/{vitpose_accuracy,rtmpose_accuracy,pmpose_accuracy}.json)
PYTHONPATH="$PWD:$PWD/$run" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  .venv/bin/python -B "$run/pose_difficulty.py" --queue "$queue_root" \
  --bundle "$media/multiview-player-evaluation-v1" \
  --bundle-sha256 3dc496513380a39d4511ff7f8711f4192eababa5109f6a0506f88a60baa70580 \
  --output outputs/pose-comparison/difficulty-v1.json "${accuracy_jobs[@]}"
```

The script verifies the original output digest and checks the all-row counts, PCK and mean OKS against the shared score for each condition. The full report is stored with the artifacts in the [knowledge node](../../nodes/tennis_scene/000091-run-meiji-topdown-pose-comparison.md). [PT32 results by group and joint](difficulty_pt32.csv)

The added CPU tests use known 3D wrist displacement, gaps and irregular source timestamps, then use asymmetric 2D residuals and a missing joint to verify temporal alignment and the PCK denominator. They reuse the fixed COCO17 definitions and row addresses. The fixtures use a few rows in memory and require no media, GPU or checkpoint.

## Review synchronized overlays

Each panel uses the same native frame, player-box geometry and frame label. The SMPL panel draws the eligible reference joints. Model panels draw their finite predictions. Use the original camera frame number for a PNG, or omit `--frame` and set an MP4 output to render the full evaluation timeline of that camera. When the grid is irregular, the MP4 uses native camera FPS and holds the most recent evaluated panel until the next sampled frame. The frame label always identifies the actual sample used.

```bash
mapfile -t vi_jobs < <(jq -r '.conditions["vitpose-h"] | .PT32.job_id, .TRT16.job_id' "$run/vitpose_accuracy.json")
PYTHONPATH="$PWD:$PWD/$run" .venv/bin/python -B "$run/render_overlays.py" \
  --queue "$queue_root" --bundle "$media/multiview-player-evaluation-v1" \
  --bundle-sha256 3dc496513380a39d4511ff7f8711f4192eababa5109f6a0506f88a60baa70580 \
  --media-root "$media/multiview-v1" --clip video_000/clip_003 --camera cam2 \
  --frame 66 --player 6 --output outputs/pose-comparison/clip_003-cam2-66.png "${vi_jobs[@]}"
```

```bash
mapfile -t pt32_jobs < <(jq -r '.conditions | to_entries[] | select(.key != "vitpose-h-no-flip") | .value.PT32.job_id' \
  "$run"/{vitpose_accuracy,rtmpose_accuracy,pmpose_accuracy}.json)
PYTHONPATH="$PWD:$PWD/$run" .venv/bin/python -B "$run/render_overlays.py" \
  --queue "$queue_root" --bundle "$media/multiview-player-evaluation-v1" \
  --bundle-sha256 3dc496513380a39d4511ff7f8711f4192eababa5109f6a0506f88a60baa70580 \
  --media-root "$media/multiview-v1" --representatives \
  --output outputs/pose-comparison/representatives-v1 "${pt32_jobs[@]}"
```

This writes the 24 source frames selected by the fixed bundle. The manifest lists the source addresses and PNG hashes. Use the same five PT32 jobs with `--clip video_000/clip_003 --camera cam2` for the first clip video or with `--clip video_000/clip_004 --camera cam1 --frame 386 --player 4` for the fast-motion example. The variable-spacing video for `clip_004/cam2` uses the ViTPose-H and PMPose-H PT32 jobs. [Representative manifest](representatives.json) · [Playback and artifact hashes](../../nodes/tennis_scene/000091-run-meiji-topdown-pose-comparison.md)

## Adapter and tests

The adapter contract binds one explicit backend, input order and COCO17 axis. Each adapter declares its forward calls with `stages.measure_forward(backend, operation, name="pose")` and may return person-aligned supplementary outputs. Renderer and analysis code can use row addresses to join with the frozen source. [Input and output contract](https://github.com/Motoki0705/tennis-lab/issues/2490#issuecomment-7129915413) · [Forward method](https://github.com/Motoki0705/tennis-lab/issues/2490#issuecomment-7131482779)

The focused CPU tests use unequal boxes, nonconsecutive frames, a partially empty cohort and a short final batch to expose reordering, skipped rows, mismatched backends and confidence-induced false misses. Replay tests reject a reordered player column and recover a retained copy after the temporary source is removed. The overlay test checks exact display pixels from a known source frame and respects the reference mask. The tests reuse the shared scorer's reference dataclasses and the production ViTPose frame sequence. The focused cases use tiny arrays and an injected frame stream, with no external weights, video or GPU. [Coverage test](../../../tests/unit/tennis_scene/evaluation/player_pose/test_scoring.py) · [ViTPose decoding test](../../../tests/unit/submodules/models/vitpose/test_pose2d.py)
