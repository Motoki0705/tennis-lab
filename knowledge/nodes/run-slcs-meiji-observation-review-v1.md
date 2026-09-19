---
id: run-slcs-meiji-observation-review-v1
type: run
title: Meiji v8向け人物観測3clip・80画像の確認
provider: codex
date: '2026-09-18'
status: done
config:
  observations: tennis_scene/precompute/meiji_dino_vitpose/s42-004
  clips:
  - video_000/clip_000
  - video_000/clip_001
  - video_000/clip_002
  device: cpu
  diagnostic_only: true
metrics:
  clips: 3
  camera_streams: 9
  reviewed_frame_camera_images: 80
  min_detection_sample_coverage: 0.9147058823529411
  min_pose_supported_fraction: 1.0
  min_torso_confident_fraction: 0.887905604719764
  max_adjacent_box_center_previous_diagonal: 0.04794221743941307
artifacts:
  run_dir: knowledge/runs/run-slcs-meiji-observation-review-v1
  output_dir: outputs/tennis_scene/analyze/meiji_observation_review/s42-001
  diagnostics: knowledge/runs/run-slcs-meiji-observation-review-v1/results.json
parents:
- run-slcs-meiji-temporal-probe-v1
- run-slcs-meiji-inference-repeatability-v1
relations: []
tags:
- slcs
- meiji
- observation
- cpu-diagnosis
- visual-review
session: 01a0b314-89b5-7902-b0a1-3141a9323ea9
repro:
  commit: 28ec0ae0bb0fd456e2a417ba9129c2a6f21696fa
  branch: codex/slcs-real-rgb
  command: env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONPATH=.
    .venv/bin/python -B knowledge/runs/run-slcs-meiji-observation-review-v1/probe.py
    --project-root /home/kamimura/projects/tennis-lab --observation-root /home/kamimura/projects/tennis-lab/outputs/tennis_scene/precompute/meiji_dino_vitpose/s42-004
    --clip video_000/clip_000 --clip video_000/clip_001 --clip video_000/clip_002
    --output-dir outputs/tennis_scene/analyze/meiji_observation_review/REPRO_NEW_RUN_ID
---

## 考察 / Findings

### 要約
Meiji v8用の観測cacheから新たに揃った000/000–002の3clip・9カメラをCPUで画像化し、80枚の抽出frameを親が目視確認した。確認した画像ではP0/P1が対象コートの選手を囲み、隣接コートの人物への明白な切替えは見られなかった。全frame・残りclipの正確性を保証せず、教師生成・採用判定の完了とは数えない。

### アーキテクチャ詳細
GPU推論・重み変更なし。各cameraで均等5frameと、各playerの隣接box中心移動量/直前box対角長が最大のsegment両端を選び、重複を除いた。boxとconfidence≥0.3の関節・骨格を実動画へ重ねた。動画・NPZ・metadataを前後にdual SHA照合し、動画の記録済みSHAとも一致。実行時はproduction base 01ca134bと未追跡probeを使い、同じprobeを28ec0ae0へcommitした。正確なscript SHAと再現commitをanalysis_provenance.jsonに残した。

### メトリクスの解釈
最小検出sample率91.47%、pose_supported率は全18 player-camera群で100%。これは短い検出欠損のbox補間が許可されるという意味で、全関節が正しいという意味ではない。cam0遠方P1は腰+肩のconfidence支持率が88.79–90.89%であり、他camの同選手は99.7%以上。肩支持はこの単眼を部分的に除くため、最終多視点教師の利用率は別途集計する。最大隣接box移動は直前対角長の4.79%。独立3D GT・teacher loss・学習曲線はない。

### アーキテクチャ⇄メトリクスの因果考察
抽出画像には隣のコートの人物も映るが、選択boxは対象コートの同じ選手を維持して見える。画像端の身体切れや遠方の小さな人物は引き続き注意が必要で、短い間の取り違えや関節単位の誤りは今回の疎な目視では検出しきれない。局所観測の信頼性確認であり、3D補正の妥当性は評価していない。

### 既存実験との比較
以前のv7完成4clipのroot診断とは別の3clipを確認した。checkpoint反復試験が推論数値の一致を見たのに対し、本runは実画像上の人物対応を点検した。すべてvideo_000なので収録間の汎化は示さない。

### 次に有効な実験
進行中の全56clip観測生成とv8教師生成を続け、strict品質レポートで問題区間を抽出して画像確認する。画像抽出のscriptはclip指定を変えて再利用し、前回出力を上書きしない。閾値・教師maskの緩和は行わない。
