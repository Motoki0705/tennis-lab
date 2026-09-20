---
task: slcs
sequence: 111
recorded_at: 2026-09-19
date_source: experiment_date
papers: []
id: run-slcs-meiji-v9-observation-reuse-v3
type: run
title: ViTPose再取得後の観測再利用がDINO固定SHA照合で公開前停止
provider: codex
date: '2026-09-19'
status: failed
config:
  device: cpu
  cuda_visible_devices: ''
  omp_mkl_threads: 4
  driver: knowledge/runs/run-slcs-meiji-v9-observation-reuse-v1/reuse.py
  source_observations: /home/kamimura/projects/tennis-lab/outputs/tennis_scene/precompute/meiji_dino_vitpose/s42-004
  target_observations: /home/kamimura/projects/tennis-lab/outputs/tennis_scene/precompute/meiji_dino_vitpose/s42-005
  automatic_retry: false
metrics:
  elapsed_seconds: 40.00238067100145
  attempts: 1
  camera_plans: 0
  published_raw_cameras: 0
  published_people_cameras: 0
  reused_cameras: 0
  recompute_cameras: null
  recompute_clip_ids: null
  input_files: 928
  unchanged_input_files: 927
  inputs_before_after_equal: false
  dino_initial_pin_matches: false
  dino_exception_audit_pin_matches: true
  vitpose_pin_check: not_reached
  target_observation_counts:
    '*_detections.npz': 0
    '*_people.npz': 0
    '*_people.reuse.json': 0
artifacts:
  run_dir: knowledge/runs/run-slcs-meiji-v9-observation-reuse-v3
  output_dir: /home/kamimura/projects/tennis-lab/outputs/tennis_scene/analyze/meiji_v9_observation_reuse/s42-003
  log: knowledge/runs/run-slcs-meiji-v9-observation-reuse-v3/stderr.log
parents:
- run-slcs-meiji-v9-observation-reuse-v2
- run-slcs-vitpose-redownload-v1
relations: []
tags:
- slcs
- meiji
- integrity
- cpu
repro:
  commit: 754c7201ec9540b3176d227b3d00d8e99b90faf2
  branch: codex/slcs-real-rgb
  command: env PYTHONPATH=. CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
    .venv/bin/python knowledge/runs/run-slcs-meiji-v9-observation-reuse-v1/reuse.py
    --source /home/kamimura/projects/tennis-lab/outputs/tennis_scene/precompute/meiji_dino_vitpose/s42-004
    --target /home/kamimura/projects/tennis-lab/outputs/tennis_scene/precompute/meiji_dino_vitpose/s42-005
    --output /home/kamimura/projects/tennis-lab/outputs/tennis_scene/analyze/meiji_v9_observation_reuse/s42-003
session: 01a0b314-89b5-7902-b0a1-3141a9323ea9
---

## 考察 / Findings

### 要約
ViTPoseを固定revisionから再取得した後、未変更のCPU観測再利用driverを1回実行した。DINO checkpointの初回固定SHA照合で失敗し、人物選択・公開前に停止した。再試行は行っていない。

### アーキテクチャ詳細
v1 driver、既存src/configとpinを維持し、CUDAを無効化、OMP/MKLを4 threadで実行した。予定は56 clip・168 cameraのraw再利用と、6選択配列のdtype/shape/value完全一致時のみpeople再利用だった。今回はcheckpoint検査で停止したため比較に到達していない。

### メトリクスの解釈
CPU実行のwall elapsedは40.002秒、exit codeは1。計画・公開・再利用は0。targetのdetections/people/reuse receiptも0を確認した。26 cameraの再計算予定とclip setは今回検証されておらず、nullとした。入力928件中927件は例外後監査でも一致した。DINOの初回hashと固定pinは不一致で、未変更driverの例外handlerによるafter監査は固定pinに一致した。具体的path・digestはintegrity.jsonを正とする。ViTPoseの照合には未到達。学習なしのため収束曲線は対象外。

### アーキテクチャ⇄メトリクスの因果考察
今回の直接の停止条件はDINO固定SHA不一致であり、ViTPose再取得の成否を実使用で再確認できたわけではない。前後のhash差だけで原因を断定しない。追加の手動checkpoint読取・原因追求・自動再試行は実施していない。

### 既存実験との比較
v2はViTPose初回照合で停止したが、本runは先行するDINO照合で停止した。再取得runのViTPose完全性検査の結果と、本runの未到達を区別する。旧読取異常の原因究明を継続条件へ戻すことはせず、この新規不一致を親へ直ちに報告した。

### 次に有効な実験
今回のDINO不一致を踏まえ、親タスクで次の対応を判断する。成功するまでの再試行はしない。観測再利用と26 cameraの再計算は未完了である。
