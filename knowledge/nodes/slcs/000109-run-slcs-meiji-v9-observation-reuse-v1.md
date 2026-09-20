---
task: slcs
sequence: 109
recorded_at: 2026-09-19
date_source: experiment_date
papers: []
id: run-slcs-meiji-v9-observation-reuse-v1
type: run
title: Meiji v9観測の再利用前照合でViTPose SHA不一致を検出
provider: codex
date: '2026-09-19'
status: failed
config:
  device: cpu
  cuda_visible_devices: ''
  source_observations: tennis_scene/precompute/meiji_dino_vitpose/s42-004
  target_observations: tennis_scene/precompute/meiji_dino_vitpose/s42-005
  checkpoint_pins: build_slcs_dataset.yaml
  selection_comparison: exact dtype/shape/value in six production selection arrays
metrics:
  planned_cameras: 168
  identical_selection_candidates: 142
  selection_recompute_candidates: 26
  published_cameras: 0
  input_files: 2058
  prepublication_mismatches: 1
  after_failure_mismatches: 0
artifacts:
  run_dir: knowledge/runs/run-slcs-meiji-v9-observation-reuse-v1
  output_dir: /home/kamimura/projects/tennis-lab/outputs/tennis_scene/analyze/meiji_v9_observation_reuse/s42-001
  failure: knowledge/runs/run-slcs-meiji-v9-observation-reuse-v1/failure.json
parents:
- run-slcs-meiji-v9-court-v1
- run-slcs-meiji-v8-observe-repair-v1
relations: []
tags:
- slcs
- meiji
- court
- observation-reuse
session: 01a0b314-89b5-7902-b0a1-3141a9323ea9
repro:
  commit: 67aadd96
  branch: codex/slcs-real-rgb
  command: env PYTHONPATH=. CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
    .venv/bin/python knowledge/runs/run-slcs-meiji-v9-observation-reuse-v1/reuse.py
    --source /home/kamimura/projects/tennis-lab/outputs/tennis_scene/precompute/meiji_dino_vitpose/s42-004
    --target /home/kamimura/projects/tennis-lab/outputs/tennis_scene/precompute/meiji_dino_vitpose/s42-005
    --output /home/kamimura/projects/tennis-lab/outputs/tennis_scene/analyze/meiji_v9_observation_reuse/s42-001
---

## 考察 / Findings

### 要約
Court v9の採用前比較を通過後、旧・新Hで人物選択を再計算した。168camera中142は6配列が完全一致し26は再計算候補だったが、公開前の2058入力再照合でViTPose重みだけSHAが変わり停止した。公開は0camera。候補の一致を採用完了とは扱わない。

### アーキテクチャ詳細
CPU・CUDA_VISIBLE_DEVICES空、OMP/MKL4で実forwardなし。旧Courtと人物receipt、固定DINO/ViTPose pins、raw schema、旧選択の再現を確認し、新Hのproduction選択結果のboxes・track IDs・observed/support masks・source detection IDs・sample indicesを比較する。公開前に全入力を再照合し、通過時だけrawをコピーし、入力が一致するposeと新Hのreceiptを公開する設計。今回は公開ループ前に停止した。

### メトリクスの解釈
計画は142camera一致・26camera差。2058入力のprepublication比較でViTPoseの1件だけ不一致、他2057は一致した。beforeは固定pin50e33f4077ef2a6bcfd7110c58742b24c5859b7798fb0eedd6d2215e0a8980bc、prepublicationはeac3c5606ef52f2d10ba3a09cab4eb02cbbe4ce9e5dd7efc84803bd2c34acdb4。例外処理のafter記録は全2058件でbeforeと一致した。後の一致で失敗を取消さず、入力の採用は保留する。学習曲線は無い。

### アーキテクチャ⇄メトリクスの因果考察
GPU forwardが無いCPU処理でも固定期待値との間欠的な不一致を観測した。ファイルの実際の更新、読取byte、計算状態、ハードウェア等の根本原因は未確定。不一致が発生した読取byte列そのものはこのdriverに保存されていないため、hash記録だけで差分位置や原因を決められない。

### 既存実験との比較
Courtの150完全一致と入力110件の安定性は別の成功根拠である。本runでは旧新Court配布・people provenanceの検証を全56clipへ拡大し、公開前ゲートが不一致を拒否した。過去のcheckpoint/動画SHA不一致や、その後の限定対照の成功と同様に、1回の成功を一般的な正常性とは扱わない。

### 次に有効な実験
過去の診断を再読し、同じ対照を繰り返さず、不一致時の読取byteを保持する実験で読取とdigest状態を切り分ける。Windows側のWHEA・異常終了記録も読み取り確認する。観測の再利用を自動retryせず、元receiptは保持する。
