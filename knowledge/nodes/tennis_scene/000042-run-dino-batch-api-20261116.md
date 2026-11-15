---
id: run-dino-batch-api-20261116
type: run
task: tennis_scene
sequence: 42
recorded_at: '2026-11-16'
title: 'DINO正式複数入力API: Meiji 4cameraの8画像で単一推論との一致・速度・VRAMを確認'
issue: 1750
provider: codex
date: '2026-11-16'
status: done
config:
  checkpoint: ckpt/dino/checkpoint0029_4scale_swin.pth
  confidence: 0.3
  short_side: 800
  max_long_side: 1333
  dtype: float32
  batch_sizes:
  - 1
  - 2
  - 4
  score_atol: 0.0001
  box_atol_original_pixels: 0.1
  rtol: 0
  device: NVIDIA GeForce RTX 5060 Ti
  resource: all
  timed_repeats: 3
metrics:
  images: 8
  detections: 58
  load_seconds: 31.157817505998537
  cpu_tests_passed: 62
  batch1_score_max_abs_diff: 0.0
  batch1_box_max_abs_diff_px: 0.0
  batch1_median_seconds: 2.6768068020028295
  batch1_frames_per_second: 2.9886355616005877
  batch1_peak_allocated_bytes: 1866030592
  batch1_peak_reserved_bytes: 2281701376
  batch2_score_max_abs_diff: 1.7881393432617188e-06
  batch2_box_max_abs_diff_px: 0.000244140625
  batch2_median_seconds: 2.543829119000293
  batch2_frames_per_second: 3.1448653292969397
  batch2_peak_allocated_bytes: 2777832448
  batch2_peak_reserved_bytes: 3477078016
  batch4_score_max_abs_diff: 1.1920928955078125e-06
  batch4_box_max_abs_diff_px: 0.000244140625
  batch4_median_seconds: 2.4941279729973758
  batch4_frames_per_second: 3.207533890246143
  batch4_peak_allocated_bytes: 4609745408
  batch4_peak_reserved_bytes: 5924454400
repro:
  commit: f738a1918f966c6a9bc679f3a8e0432843f9ee62
  branch: codex/dino-batch-api
  remote: git@github.com:Motoki0705/tennis-lab.git
  command: timeout -k 10s 900s env PYTHONPATH=. TENNIS_LAB_DINO_EXTENSION="$TENNIS_LAB_DINO_EXTENSION"
    OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 .venv/bin/python outputs/tennis_scene/evaluate/dino_batch_api_20261116/compare.py
    outputs/tennis_scene/evaluate/dino_batch_api_20261116
artifacts:
  run_dir: knowledge/runs/run-dino-batch-api-20261116
  results: knowledge/runs/run-dino-batch-api-20261116/results.json
  inputs: knowledge/runs/run-dino-batch-api-20261116/inputs.json
  protocol: knowledge/runs/run-dino-batch-api-20261116/protocol.json
  comparison_script: knowledge/runs/run-dino-batch-api-20261116/compare.py
  cpu_validation: knowledge/runs/run-dino-batch-api-20261116/cpu-validation.json
  gpu_log: knowledge/runs/run-dino-batch-api-20261116/gpu.log
  local_output: outputs/tennis_scene/evaluate/dino_batch_api_20261116
  raw_predictions: .training_queue/repro/0-8205244284021426274_2026-11-15T15-15-15-978573725Z_209883_dino-batch-api-20261116/predictions/dino_batch.npz
  publication_provenance: knowledge/runs/run-dino-batch-api-20261116/publication-provenance.json
  publication_audit: knowledge/runs/run-dino-batch-api-20261116/publication-audit.json
parents: []
relations: []
papers: []
tags:
- dino
- batch-inference
- parity
- benchmark
- gpu
summary_impact: none
---

## 結果と適用範囲

#1750（親 #1734、C7）の正式API `predict_batch(requests, *, batch_size=1)` を検証した。
前処理後の形状が一致する連続入力だけを上限付きでstackし、入力順で結果を返す。
shape変更で分割し、padding・resize・閾値・重複統合設定・精度は変更しない。
実装・契約の正本は [submodules README](../../../src/submodules/README.md)。

`summary_impact: none`。標準pipelineのcheckpoint・既定設定・採用方式を変えないAPIの受入記録であり、
[summaryの標準pipeline判断](../../summary.md#標準pipelinetennis_scene)は維持する。
C8のclip/camera集約、scene全体の速度・3D精度は未検証。

## 固定条件と比較

書出し済み `meiji_4cam` の `video_000/clip_000` を読み取り専用で使用。
4cameraを先頭から全356frame復号し、manifestのframe数と一致、読取中のmedia変更なしを確認した。
各cameraのframe 0・120、計8枚（各1920×1080）を固定し、前処理後は全て750×1333。
[入力manifest](../../runs/run-dino-batch-api-20261116/inputs.json)と
[実行前protocol](../../runs/run-dino-batch-api-20261116/protocol.json)を残した。

FP32、COCO person checkpoint、confidence 0.30、800/1333を全条件で共用した。
PyTorch 2.13.0+cu130、CUDA 13.0、RTX 5060 Ti（16311 MiB）。
matmul precision=highest、matmul TF32=false、cuDNN TF32=trueは既定状態を記録し、精度設定は変更していない。
training queueのresource=all、allocator上限は総VRAMの72%とした。

比較の基準は変更前commit `e561377ce` の元の単一画像wrapper。
[そのまま保存した参照実装](../../runs/run-dino-batch-api-20261116/scalar_reference.py)と新APIが
同じ実モデル・重みを共用する。旧wrapperは `preprocess_frame(...).to(device).unsqueeze(0)` をそのまま実行する。
許容差は実行前にscore絶対差1e-4、box絶対差0.1元画像px、相対差0とした。

全8画像・計58検出で、batch=1/2/4すべて合格した。
raw queryのscore閾値mask、score降順のquery row順、画像ごとの検出数、
重複統合off/onのkept/dropped rowは厳密一致した。
実画像では重複統合によるdropが0件だったため、実際のdropありの同一性はCPUの非対称入力テストで確認した。

| batch | score最大絶対差 | box最大絶対差（元px） | 8画像の中央値（秒） | 画像/秒 | batch=1比 | peak allocated / reserved（GiB） |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 0 | 0 | 2.676807 | 2.989 | 1.000 | 1.738 / 2.125 |
| 2 | 1.7881393e-6 | 0.000244140625 | 2.543829 | 3.145 | 1.052 | 2.587 / 3.238 |
| 4 | 1.1920929e-6 | 0.000244140625 | 2.494128 | 3.208 | 1.073 | 4.293 / 5.518 |

warmup各1回後の3反復。時間にはCPU前処理・H2D・GPU forward・復号・D2Hを含み、
動画復号・重みload・warmup・診断hook・保存は除く。初回model loadは31.158秒。
VRAMはPyTorch allocatorのpeakであり、GPU全体の使用量ではない。
4はこの小入力で確認した値で、任意入力での最大安全batchの保証ではない。
速度差は約5%/7%にとどまり、scene全体の高速化率へ外挿しない。
詳細は [results.json](../../runs/run-dino-batch-api-20261116/results.json)。

## CPU検証

既存 `test_person_detector.py` の5caseを拡張し、23case追加して28case。
fakeの小さいCPU modelと実前処理を使い、batch軸、元サイズ、threshold境界、同点row、空検出、
shape分割、最大件数と端数、batch=1、load/no_grad、重複統合off/onを検証した。
関連62caseは12.40秒で成功。DINO 28caseは5.50秒（import込み）、最長call 0.07秒、setup表示はすべて0.01秒未満。
collect-onlyは28case、単一関数の最大展開は6case。

batch indexを0へ固定する変異と、元サイズを先頭画像のものへ固定する変異を、別CPU processのメモリ内だけへ注入した。
対応するassertがそれぞれ90px・4.92805pxの不一致で失敗し、元ファイルを変更しない通常実行で両nodeが成功することを確認した。
pre-commitのruff/mypyは成功。コマンド・時間・変異のnodeと失敗ログは
[cpu-validation.json](../../runs/run-dino-batch-api-20261116/cpu-validation.json)から参照できる。

## 資産と再現

このworktreeの未初期化 `third_party/DINO` は `git submodule update --init third_party/DINO` で
固定gitlinkの版を取得した。submodule source・pointerは変更していない。
CUDA拡張は既存のクリーン導入検証で生成したbinaryを明示選択した。
`validate_dino_extension()` のCPU forward/backward dispatchと実GPU推論が成功し、
#1088で問題になったCUDA拡張の読み込み不良には当たらなかった。再build・共有venv変更は不要だった。

実装commitと公開用commandテンプレートは [run.json](../../runs/run-dino-batch-api-20261116/run.json)、
[比較script](../../runs/run-dino-batch-api-20261116/compare.py)、
[公開用queue log](../../runs/run-dino-batch-api-20261116/gpu.log)に残した。
公開コピーから会話識別子を除き、環境固有のパスはrepo rootからの相対表記にした。
binaryの位置は明示的に設定する `$TENNIS_LAB_DINO_EXTENSION` として表し、実際のbinaryのhashは保持する。
原本はGit外の非公開領域へ保存し、[公開来歴](../../runs/run-dino-batch-api-20261116/publication-provenance.json)
に原本と公開コピーのhashを記録した。`results.json` の `inputs_sha256` は原本の入力manifestに対応し、
公開用manifestのhashは公開来歴から照合できる。数値結果・入力画像・checkpoint・binaryの検証値は変更していない。
[queue commandの公開コピー](../../runs/run-dino-batch-api-20261116/queue-repro-original.sh)も同じ処理を施した。
再実行には保存済みscriptを参照する [repro.sh](../../runs/run-dino-batch-api-20261116/repro.sh) を使う。
この入口はshared queue経由の起動と実装の一致を検査し、新しい出力先へ入力を再作成して比較する。
比較scriptの保存時変更はimport整形のみで、[来歴](../../runs/run-dino-batch-api-20261116/script-provenance.json)に記録した。
再実行用の入力作成は [prepare.py](../../runs/run-dino-batch-api-20261116/prepare.py) を使う。
repo rootから次を実行し、CUDA拡張を導入READMEに従って明示選択した新しいqueue jobで比較する。

```bash
.venv/bin/python knowledge/runs/run-dino-batch-api-20261116/prepare.py \
  --clip data/tennis_multiview/processed/meiji_4cam/dataset/videos/video_000/clips/clip_000 \
  --output outputs/tennis_scene/evaluate/dino_batch_api_replay
# 下記commandを共有training queueへresource=allで登録する
# TENNIS_LAB_DINO_EXTENSIONには導入READMEに従って検証済みbinaryの絶対pathを指定する
PYTHONPATH=. OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 \
  .venv/bin/python knowledge/runs/run-dino-batch-api-20261116/compare.py \
  outputs/tennis_scene/evaluate/dino_batch_api_replay
```

予測配列はGitへ入れず、共有queueの再現bundle `predictions/dino_batch.npz` と
ローカル出力の `predictions.npz` に保持する。動画・抽出画像・重みもGitへコピーしない。
