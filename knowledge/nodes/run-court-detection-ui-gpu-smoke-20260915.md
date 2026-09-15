---
id: run-court-detection-ui-gpu-smoke-20260915
type: run
title: Court Detection Web UIの4ヘッドGPU推論スモーク確認
provider: codex
session: 01a09e9c-5c30-7bf3-8e7b-9d2f9d6d3bd8
date: '2026-09-15'
status: done
config:
  model: court_hierarchical
  data: tennis_court_detector/val::EF-hx40Q4Mg_700
  checkpoint: mixed-source/semantic-line-frozen-b00-b03-b8-e20-s42/logs/version_0/checkpoints/court-detection-epoch=05.ckpt
metrics:
  http_status: 200
  kp_scored_points: 14
  kp_mean_error_px: 1.9285729813980659
  seg_mean_iou: 0.9633933275226668
  line_iou: 0.5273136786902881
  semantic_line_mean_iou: 0.5536626073424399
repro:
  commit: 782fc137d85c054d0f5ef775fe5edbf8cf52f00f
  branch: codex/dataset-scene-review
  remote: git@github.com:Motoki0705/tennis-lab.git
  command: /home/kamimura/projects/tennis-lab/.venv/bin/python -m src.tasks.base.visualization.inference_queue
    /home/kamimura/projects/tennis-lab/.training_queue/ui_requests/court_detection-l0ni6oby/request.json
artifacts:
  run_dir: knowledge/runs/run-court-detection-ui-gpu-smoke-20260915
  log: /home/kamimura/projects/tennis-lab/.training_queue/logs/1789479192229301717_960324_court_detection-web-inference.log
parents: []
relations: []
tags: [court-detection, web-ui, inference, smoke]
---

## 考察 / Findings

### 要約
ブラウザのcheckpoint候補選択から共有GPUキュー経由の4ヘッド推論がHTTP 200で完了した。GTと予測KP/seg/line/semantic_lineの原画像上の重ね表示を確認。学習や精度benchmarkではなく、単一画像による接続確認。

### アーキテクチャ詳細
保存configとtarget_bundle_stateを厳密に復元する現行Lightning loaderを使い、1回のforwardで4ヘッドをdecodeする。legacy bundle欠落や古い必須configを推測で補わず、カタログで非対応理由を示す。

### メトリクスの解釈
TennisCourtDetector valの1枚でKP14の平均誤差1.93px、seg mIoU0.963。448x256の予測gridを1280x720へ再標本化したことをUI warningで明示。学習は行っておらず収束曲線はない。

### アーキテクチャ⇄メトリクスの因果考察
単一sampleから汎化性能や精度差の原因を一般化しない。保存schema、原画像座標変換、共有キュー終了、ブラウザ表示が一貫していることを確認した。

### 既存実験との比較
他runとの精度比較は未実施。request/resultは共有`.training_queue/ui_requests/court_detection-l0ni6oby/`に残る。repro patchは未追跡の新規ソースを含まないため、再現には本Web UIのworktree変更一式が必要。

### 次に有効な実験
精度評価は固定されたtest split全体で別途行い、実写とsyntheticを分けて確認する。
