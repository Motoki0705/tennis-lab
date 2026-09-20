---
task: ball_detection
sequence: 17
recorded_at: 2026-09-15
date_source: experiment_date
papers: []
id: run-ball-detection-ui-gpu-smoke-20260915
type: run
title: Ball Detection Web UIのGPU推論スモーク確認
provider: codex
session: 01a09e9c-5c30-7bf3-8e7b-9d2f9d6d3bd8
date: '2026-09-15'
status: done
config:
  model: conv_next_unet
  data: youtube::video_000001/clip_000001
  checkpoint: ckpt/ball_detection/run-i618-convnext-v2-ft-epoch13.ckpt
metrics:
  http_status: 200
  predicted_frames: 8
  precision: 0.0
  recall: 0.0
  f1: 0.0
repro:
  commit: 782fc137d85c054d0f5ef775fe5edbf8cf52f00f
  branch: codex/dataset-scene-review
  remote: git@github.com:Motoki0705/tennis-lab.git
  command: /home/kamimura/projects/tennis-lab/.venv/bin/python -m src.tasks.base.visualization.inference_queue
    /home/kamimura/projects/tennis-lab/.training_queue/ui_requests/ball_detection-915azon0/request.json
artifacts:
  run_dir: knowledge/runs/run-ball-detection-ui-gpu-smoke-20260915
  log: /home/kamimura/projects/tennis-lab/.training_queue/logs/1789478846683415821_949228_ball_detection-web-inference.log
parents: []
relations: []
tags: [ball-detection, web-ui, inference, smoke]
---

## 考察 / Findings

### 要約
ブラウザからの実推論要求が共有GPUキューを通ってHTTP 200で完了し、8frameの結果をUIへ表示できた。学習や精度benchmarkではなく、実データ・実checkpointを使う接続確認。

### アーキテクチャ詳細
保存されたConvNeXtモデルとadapterを既存factoryから構築し、model weightsをstrictに復元。元画像を保存解像度へresizeして[0,1] RGBで入力する。CUDAモデルはqueue workerだけが保持し、終了時に解放する。

### メトリクスの解釈
YouTube clip先頭0..7frame、しきい値0.5、距離閾値4pxの限定確認で、一致検出は0件だった。学習は実施していないため収束曲線はない。実行時のcanonical metricは一致0件の平均距離を0と返したが、UI側はその後、N/Aとする修正と回帰テストを追加した。

### アーキテクチャ⇄メトリクスの因果考察
この少数sampleから精度や原因を一般化しない。HTTP、model-I/O、queue終了、画像座標表示の成立を確認した。

### 既存実験との比較
他runとの精度比較は実施していない。requestとresultは共有`.training_queue/ui_requests/ball_detection-915azon0/`に残る。repro patchは未追跡の新規ソースを含まないため、再現には本Web UIのworktree変更一式が必要。

### 次に有効な実験
学習精度の評価は既存のevaluation manifestによる固定split評価で別途行う。
