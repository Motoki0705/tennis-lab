# RTMPose-M and RTMPose-X

Both checkpoints use the official Body7 training mixture, the 700-epoch Halpe26 recipe and a 384×288 image (height × width). Body8 in the configuration name refers to the evaluation mixture, which adds OCHuman. The M and X recipes specify training batches of 512 and 256 per device, respectively. The model IDs, config filenames and checkpoint SHA-256 values are in [rtmpose.py](rtmpose.py). The downloaded files are 56,436,913 bytes (M) and 200,397,852 bytes (X). [Official model zoo](https://github.com/open-mmlab/mmpose/blob/5408bc76f5b848cf925a0d1857899011d8c5b497/configs/body_2d_keypoint/rtmpose/body8/rtmpose_body8-halpe26.md)

The first 17 Halpe26 channels have a one-to-one correspondence with COCO17, including the nose, eyes, ears and the left/right limb pairs. Both configs map COCO channels `0..16` to the same Halpe indices. The adapter checks the checkpoint metadata against the upstream Halpe definition before loading weights. [Halpe26 definition](https://github.com/open-mmlab/mmpose/blob/7bea17cdf362cdd731fc5e79b982e4b79be5cca9/configs/_base_/datasets/halpe26.py) · [RTMPose-M config](https://github.com/open-mmlab/mmpose/blob/7bea17cdf362cdd731fc5e79b982e4b79be5cca9/configs/body_2d_keypoint/rtmpose/body8/rtmpose-m_8xb512-700e_body8-halpe26-384x288.py) · [RTMPose-X config](https://github.com/open-mmlab/mmpose/blob/7bea17cdf362cdd731fc5e79b982e4b79be5cca9/configs/body_2d_keypoint/rtmpose/body8/rtmpose-x_8xb256-700e_body8-halpe26-384x288.py)

MMPose code is Apache-2.0. The official zoo supplies both checkpoints by public download, and a maintainer gave general permission for Body7 model use in response to a commercial-use question. This evaluation uses local copies of the released artifacts. Commercial deployment and redistribution of these exact checkpoints or converted engines remain unconfirmed. The checkpoint-specific licensing and training-data questions are open upstream. [Code license](https://github.com/open-mmlab/mmpose/blob/7bea17cdf362cdd731fc5e79b982e4b79be5cca9/LICENSE) · [Maintainer response](https://github.com/open-mmlab/mmpose/issues/3062#issuecomment-2136896681) · [Checkpoint terms discussion](https://github.com/open-mmlab/mmpose/issues/3275)

## Test design

- Small CPU unit fixtures exercise two distinct boxes, asymmetric left/right joints and unequal x/y SimCC peaks. The checks cover native-pixel reprojection, input order, the model's missing-coordinate sentinel and preservation of its raw score. A malformed Halpe keypoint order must be rejected. These cases require no model files and fit the unit-test timing limit.
- Reuse `test_pose_reports_missing_person_and_missing_joint_in_coverage_denominators` for the treatment of NaN coordinates after scoring, and `test_open_keeps_empty_images_and_rejects_changed_files` for the pinned reference loader. The adapter tests exercise the RTMPose-specific transformations and backend boundaries that those tests do not call.
- Compare the staged CPU FP32 adapter with the official `inference_topdown` on two real pseudo-BBoxes from one pinned camera frame for each checkpoint. Verify both native positions and raw scores. This covers affine cropping, RGB normalization, flip fusion and the output slice. Report GPU parity separately against CUDA PT32 on the fixed comparison inputs.
- Verify the six accuracy receipts against the published row order and fixed boxes before pairing outputs. Use an asymmetric CPU fixture with a 3–4–5 pixel displacement and missing joints to check that the backend comparison measures Euclidean distance and counts missing predictions separately. Check the shared benchmark's two declared forwards in an actual timing case.
- Exercise the common `measure_forward` API with a CPU fixture that has two unequal crop columns. The two model calls must receive the original and then the flipped crop, with both calls registered under the chosen backend. The real-image smoke test continues to verify the official forward and decoder.
- Re-score all six saved conditions with the merged #2495 scorer. Compare the complete metric payloads with the original files and report the pose metrics and fixed-box recall. A small CPU fixture checks that a pose change is detected even if recall and the top-level sections stay the same.
- Exercise the speed CLI on CPU with a tiny crop fixture and an injected benchmark. Check that PT32 is rejected at argument parsing and that PT16 and TRT16 pass the scheduled crop order, the dedicated stream and the explicit unmeasured-parity reason to the harness. Reuse `test_explicit_parity_opt_out_records_speed_without_extra_inference` for the real harness call count and `test_pinned_schedule_rejects_reordered_loaded_batches` for input integrity. Check the real published inputs separately by loading them on CPU.

## Inference and export settings

The adapter uses MMPose's `GetBBoxCenterScale` with padding 1.25 and `TopdownAffine` with width:height 3:4, bilinear interpolation and zero borders. It converts the crop from BGR to RGB and normalizes in FP32 with mean `[123.675, 116.28, 103.53]` and standard deviation `[58.395, 57.12, 57.375]` on the 0–255 scale. It applies the official horizontal flip test in every backend. Each batch executes two forwards. After swapping the flipped view's Halpe left/right channels and reversing the x bins, it averages the logits in FP32. The CPU SimCC decoder divides peak indices by 2, takes the smaller x/y peak as the raw score and restores coordinates using the crop center and scale. It maps the decoder's missing-coordinate sentinel to NaN and retains finite raw scores.

The historical PT32 accuracy runs used explicit FP32 with TF32 disabled. PT16 uses explicit FP16 weights and inputs with autocast disabled. The strongly typed TRT16 graph has FP16 parameters, inputs and SimCC outputs. Flip fusion and decoding use the same code outside all three model graphs. Engine inspector output records the realized layer formats when CUDA runs. The TensorRT context disables NVTX markers during inference. [Engine inspector](https://docs.nvidia.com/deeplearning/tensorrt/10.x.x/inference-library/engine-tools.html)

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

Use the published #2495 accuracy bundle and digest. The CPU smoke command records how the earlier FP32 input and decode verification ran. Reproduce the TRT16 accuracy measurement inside an exclusive training queue allocation. For a PT16 reproduction, select `--backend PT16` and omit `--engine`. The build targets batches 1–16 with an optimum of 8 and uses the scene TensorRT builder's 1536 MiB workspace setting.

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

## GPU accuracy on the #2495 reference

RTX 5060 Ti、CUDA 13.0、PyTorch 2.13.0+cu130、TensorRT 10.14.1.48.post1 で3 backendを実行した。Batch 8の全条件で固定BBox 26,276件のキーと座標が一致した。有効な採点関節は437,714点。pixel L2は出力のある点の統計で、OKS・PCKは固定母数で欠測を失敗として扱う。実行HEAD、重み・engine・予測・採点・生ログのSHA-256と格納場所を記録した。 [実行記録](rtmpose_accuracy.json)

| Model | Backend | Mean OKS | PCK@0.10 | PCK@0.20 | L2 mean (px) | L2 median (px) | L2 P95 (px) |
|---|---|---:|---:|---:|---:|---:|---:|
| M | PT32 | 0.75205 | 0.76122 | 0.94517 | 4.995 | 3.627 | 13.690 |
| M | PT16 | 0.75193 | 0.76119 | 0.94509 | 4.995 | 3.628 | 13.688 |
| M | TRT16 | 0.75181 | 0.76116 | 0.94488 | 4.999 | 3.627 | 13.712 |
| X | PT32 | 0.76835 | 0.77804 | 0.95660 | 4.720 | 3.473 | 12.776 |
| X | PT16 | 0.76831 | 0.77813 | 0.95664 | 4.720 | 3.471 | 12.779 |
| X | TRT16 | 0.76826 | 0.77801 | 0.95651 | 4.722 | 3.473 | 12.788 |

PT32でXはMより疑似GTの平均OKSが0.01630高く、平均L2が0.275 px小さかった。各モデル内の平均OKSの変化は最大0.000242。M TRT16で採点関節1点が欠測し、他の5条件は全採点関節を出力した。疑似GTの生成には既存モデルの観測と身体モデルを用いているため、この差は独立した人手GTでの優劣を表さない。 [疑似GTの作成と限界](../run-meiji-multiview-player-evaluation/README.md)

### Output differences from the same checkpoint's PT32

固定26,276行すべてで、同じ行・同じCOCO17関節の座標と生スコアを比較した。M TRT16の欠測1点は差分の母数から除いた。各条件ともP95は0.34 px未満で、5 pxを超える変位も観測された。最大変位のclip・camera・frame・player・関節を記録した。 [同一重みの出力差](rtmpose_accuracy.json)

| Model | Backend | Finite pairs | XY P95 (px) | XY maximum (px) | Above 5 px | Score absolute P95 |
|---|---|---:|---:|---:|---:|---:|
| M | PT16 | 446,692 | 0.272 | 157.363 | 720 | 0.00714 |
| M | TRT16 | 446,691 | 0.336 | 119.325 | 1,066 | 0.00932 |
| X | PT16 | 446,692 | 0.212 | 72.167 | 171 | 0.00222 |
| X | TRT16 | 446,692 | 0.292 | 93.559 | 277 | 0.00408 |

TensorRTのbuildはM 55.12秒、X 64.74秒。planは30,040,084／102,454,292 bytes。Batch 8のEngine InspectorでMの315件、Xの469件のlayer入力・出力形式はすべて`Half`だった。tactic名に`f32`を明示するGEMMもM 4件、X 5件あり、channel attention・MLP・GAUで使用された。該当layer名とtactic名を保存した。 [変換記録](rtmpose_accuracy.json)

### Re-scoring with the merged #2495 scorer

2026-11-25にM/Xの6条件を、保存済みの予測各26,276行からCPUで再採点した。採点器は #2498 でマージされた `0d4bf0990f82ae0b8ebf2afe08d3209e583d2364` の版。OKS・PCK・pixel誤差、欠測および全層別集計が再採点前と一致した。固定BBoxの照合は全条件26,276/26,276件で、Recall@0.50とRecall@0.75はともに1.0だった。JSONに追加されたのは同一IoU時の正準順序を示す `candidate_order_for_iou_ties`。元の予測・採点ファイルと再採点ファイルのSHA-256を照合した。 [#2498](https://github.com/Motoki0705/tennis-lab/pull/2498) · [再採点記録](rtmpose_accuracy.json)

再現には前述の`run`・`bundle`・`bundle_sha`と、共有queueのパス`queue_dir`を設定する。`--rescore-dir`には新規ディレクトリを渡し、完全な再採点ファイルを保存する。実行HEADは `920c9aaff24639e6b461c8985a254d80ba3a1385`。

```bash
mapfile -t jobs < <(jq -r '.conditions[][] | .job_id' "$run/rtmpose_accuracy.json")
PYTHONPATH="$PWD/$run:$PWD" .venv/bin/python -B "$run/rtmpose_accuracy.py" \
  --queue "$queue_dir" --bundle "$bundle" --bundle-sha256 "$bundle_sha" \
  --rescore-dir outputs/rtmpose/rescore-reproduced \
  --output outputs/rtmpose/accuracy-reproduced.json "${jobs[@]}"
```

## Speed on the #2496 workload

The shared #2496 input `meiji_4cam/person-model-speed-v1` supplies predecoded native BGR frames and 384 fixed DINO crops on 96 frames. RTMPose reads the native `xyxy` before applying its official aspect ratio and padding. The harness merged in `59faa32a28d2f54e6492abdba2429846fa0397e3` verifies the manifest and timing protocol SHA-256 and the timed crop IDs. The shared sweep uses batches 1, 4 and 8, a dedicated CUDA stream, 30 warm-up iterations and three blocks of 100 timed iterations ending after CUDA synchronization. The loader supplies 384 full-batch crop rows at each size. The timed loop consumes the first 300 rows at batch 1 and cycles the full crop pool at batches 4 and 8. Flip testing makes two model forwards per batch. [Shared protocol](../run-person-model-speed-baseline/timing-protocol.json) · [Baseline measurements](../run-person-model-speed-low-precision/README.md)

The 2026-11-25 scope uses PT16 and TRT16 with `parity_not_evaluated_reason` set to `PT32計測はスコープ外（ユーザー判断 2026-11-25）`. The comparison uses the previously recorded pseudo-GT accuracy and reports the parity label `PT32未計測` for these speed runs.

On 2026-11-25, the parent cancelled the two original speed sweeps before execution while #2497 was being revised. Both queue entries ended as `cancelled` with no log, repro directory or measurements. [#2497 review](https://github.com/Motoki0705/tennis-lab/pull/2497#issuecomment-7138771687)

- M job `0-8204471182921321101_2026-11-24T14-00-17-078678898Z_2310221_i2490-rtmpose-m-speed-sweep.job`
- X job `0-8204471182614046695_2026-11-24T14-00-17-385953304Z_2310299_i2490-rtmpose-x-speed-sweep.job`

Run each case in its own process in the exclusive training queue. Use batches 1, 4 and 8 for each model and backend and supply a fresh output root. Each condition writes a `report.json` that carries the source HEAD, harness merge commit, input pins, artifact hashes, timing and status. Add `--engine "$engine"` for TRT16.

```bash
speed=knowledge/runs/run-person-model-speed-baseline
prep_job=0-8204477064386375676_2026-11-24T12-22-15-613624323Z_1489359_person-speed-prepare
speed_inputs="$(cat "$queue_dir/repro/$prep_job/output_dir.txt")/inputs"
speed_sha256=64c74d75e5f1afe869911f1620b9c8e8a793fc4c4e4ac830f4abd97b21fb1a62
PYTHONPATH="$PWD/$speed:$PYTHONPATH" OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  .venv/bin/python -B "$run/rtmpose_speed.py" --inputs "$speed_inputs" \
  --input-sha256 "$speed_sha256" --model "$model" --backend PT16 --batch 1 \
  --checkpoint "$weight" --source-root "$source_root" --device cuda:0 \
  --output "$TENNIS_REPRO_DIR/measurements"
```
