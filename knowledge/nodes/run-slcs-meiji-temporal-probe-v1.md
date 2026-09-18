---
id: run-slcs-meiji-temporal-probe-v1
type: run
title: 'Meiji時間的な人物対応: 4clipで教師改善、5件目は整合性検査で停止'
provider: codex
session: 01a0b314-89b5-7902-b0a1-3141a9323ea9
date: '2026-09-18'
status: failed
config:
  model: selected Meiji PLCS/BLCS 60epoch
  data: meiji_rgb_v7
  people_selection: temporal_continuity_v2; schema4; max_prediction_frames60; max_prediction_distance1
metrics:
  requested_clips: 5
  completed_clips: 4
  failed_clips: 1
  eligible_dataset_clips: 56
  missing_dataset_clips: 52
  clip000_007_p0_frame_fraction_before: 0.8757668711656442
  clip000_007_p0_frame_fraction_after: 1.0
  clip001_000_p0_frame_fraction_before: 0.7697568389057751
  clip001_000_p0_frame_fraction_after: 0.9969604863221885
repro:
  commit: 30a7eeb04714fd548f28d0a58c9813cdfea6ee1b
  branch: codex/slcs-real-rgb
  remote: git@github.com:Motoki0705/tennis-lab.git
  command: 'env TENNIS_RGB_GPU=0 bash /home/kamimura/projects/tennis-lab/.claude/worktrees/slcs-real-rgb/scripts/datasets/build_real_rgb.sh
    --execute meiji clip_ids=\[video_000/clip_007\,video_001/clip_001\,video_000/clip_001\,video_000/clip_009\,video_001/clip_000\] '
artifacts:
  run_dir: knowledge/runs/run-slcs-meiji-temporal-probe-v1
  log: .training_queue/logs/1789732053829847917_689915_slcs-real-rgb-build-meiji.log
  output_dir: outputs/tennis_scene/generate/meiji_rgb_v7/s42-001
  diagnostics: knowledge/runs/run-slcs-meiji-temporal-probe-v1/comparison.json
parents:
- run-slcs-meiji-v6-tracking-qc
relations: []
tags:
- slcs
- meiji
- quality-control
- person-association
- partial
---

## 考察 / Findings

### 要約
時間的な人物対応を適用した5clip probeのうち4clipを生成した。video_000/clip_007とvideo_001/clip_000の通行人への切替が解消し、P0教師の利用率が改善した。video_000/clip_001はDINO checkpoint SHAの検証不一致で失敗したため、run全体はfailedとする。

### アーキテクチャ詳細
初期人物を各半面の最大boxで選び、以降は直前の実観測boxと最大60frame・1box対角までの速度予測から対応付ける。候補なしを未観測として扱い、通行人へ自動再初期化しない。人物観測schema4・temporal_continuity_v2、観測出力s42-004、dataset v7で旧版と分離した。指定Court・outsource ball・選定済みPLCS/BLCSは維持する。

### メトリクスの解釈
000/007のP0正重みframe率は0.875767→1.0、最長欠損81→0frame。001/000は0.769757→0.996960、235→4frame。000/009のP1最終SLCS frame率は0.967742→0.966398と1frame悪化し、最長欠損6frameは不変。001/001は利用率不変。4clipのballは正重み率不変、3clipの座標は完全一致、001/001の最大差は3.8147e-6m。

partial_quality_reportは4/56完成・52未生成・明示除外1を示す。未生成を許容した途中確認であり、全体合格ではない。利用率と再投影は擬似教師の観測整合性で、独立実測3D精度ではない。学習を伴わず収束曲線はない。

### アーキテクチャ⇄メトリクスの因果考察
000/007と001/000では、raw候補と選択IDの比較、および各3frame×3視点の映像確認で実選手を保持し通行人を選ばないことを確認した。001/001 cam2はViTPose receipt SHAも旧版と異なるため、そのclipの変化を人物対応だけに帰属させない。000/009の1frame悪化も保持する。

### 既存実験との比較
旧v6の長欠損maskは誤った人物選択後の教師を除外できたが、人物対応を修正しなかった。今回、選手が検出される区間で教師を回復できた。001/001のcam2画面外区間は未観測を維持し、他視点からの支持と区別した。

### 次に有効な実験
SHA不一致の原因を切り分け、検証を緩めず000/001を再実行する。その後56clipの教師生成・strict品質報告・初期選択を含む画像確認を完了する。全体完成前に追加SLCS学習へ進まない。
