---
id: run-slcs-meiji-observation-review-v2
type: run
title: 'Meiji v8向け人物観測を別収録へ拡張: 2clip・54画像'
provider: codex
date: '2026-09-18'
status: done
config:
  observations: tennis_scene/precompute/meiji_dino_vitpose/s42-004
  clips:
  - video_000/clip_011
  - video_001/clip_002
  device: cpu
  diagnostic_only: true
metrics:
  clips: 2
  camera_streams: 6
  reviewed_frame_camera_images: 54
  min_detection_sample_coverage: 0.8338461538461538
  min_pose_supported_fraction: 1.0
  min_torso_confident_fraction: 0.920585967617579
  max_adjacent_box_center_previous_diagonal: 0.0557592548429966
artifacts:
  run_dir: knowledge/runs/run-slcs-meiji-observation-review-v2
  output_dir: outputs/tennis_scene/analyze/meiji_observation_review/s42-002
  diagnostics: knowledge/runs/run-slcs-meiji-observation-review-v2/results.json
parents:
- run-slcs-meiji-observation-review-v1
relations: []
tags:
- slcs
- meiji
- observation
- cpu-diagnosis
- visual-review
session: 01a0b314-89b5-7902-b0a1-3141a9323ea9
repro:
  commit: 0f500a1202f19ef4dee76900659c4b1baae442a5
  branch: codex/slcs-real-rgb
  command: env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONPATH=.
    .venv/bin/python -B knowledge/runs/run-slcs-meiji-observation-review-v1/probe.py
    --project-root /home/kamimura/projects/tennis-lab --observation-root /home/kamimura/projects/tennis-lab/outputs/tennis_scene/precompute/meiji_dino_vitpose/s42-004
    --clip video_000/clip_011 --clip video_001/clip_002
    --output-dir outputs/tennis_scene/analyze/meiji_observation_review/REPRO_NEW_RUN_ID
---

## 考察 / Findings

### 要約
video_000の最後のclipと、別収録video_001のclip_002について、6カメラ分のcontact sheetに含まれる54画像を親が目視確認した。抽出画像に明白な隣接コートの人物への切替えは見られなかった。前回の3clip・80画像とは重複せず、観測画像の確認範囲を計5clip・2収録へ広げた。全frameや3D教師の採用判定ではない。

### アーキテクチャ詳細
前runのCPU専用probeを変更せず再利用した。各cameraの均等5frameと、各playerの隣接box中心移動量/直前box対角長が最大となる区間の両端を選ぶ。動画・人物NPZ・metadataを前後にdual SHAで確認し、記録済み動画SHAとも照合した。GPU推論・閾値変更・教師生成は行っていない。

### メトリクスの解釈
最小検出sample率83.38%はvideo_000/clip_011のcam0・遠方P1。pose_supported率は全12 player-camera群で100%だが、短い内部欠損のbox補間を含む。腰+肩のconfidence支持率の最小値は同群の92.06%。video_001/clip_002では検出sample率の最小値95.58%、腰+肩支持率の最小値99.33%。単眼の観測支持と、後段の多視点教師利用率は区別する。

### アーキテクチャ⇄メトリクスの因果考察
今回の画像でも対象コートの人物にboxが追従して見える。一方、video_000/clip_011のcam2・frame648では近側P0の脚が画像下端で切れている。遠方P1はcam0で小さく映り、検出欠損や関節誤りが起き得る。疎な画像確認では短時間の誤対応や関節単位の精度を保証できず、腰+肩支持の導入効果もこの画像だけでは判定できない。

### 既存実験との比較
前回はvideo_000先頭3clipのみだった。本runでは同収録の末尾と別収録を確認した。全体生成の進行中に完了済み観測を読む診断であり、v8の3D教師生成済み件数や最終学習データ採用数に加算しない。曲線は学習runではないため対象外。

### 次に有効な実験
全56clipの観測生成とv8教師生成を継続し、最終品質レポートの低支持区間・速度異常・再投影残差を画像と対応させる。未確認のvideo_002を含む全体品質評価が必要であり、この結果だけで閾値の変更やclip採用を行わない。
