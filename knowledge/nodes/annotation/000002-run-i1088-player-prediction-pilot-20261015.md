---
id: run-i1088-player-prediction-pilot-20261015
type: run
task: annotation
sequence: 2
recorded_at: '2026-10-15'
title: 新規実clipのplayer予測がCUDA拡張のロードで停止
issue: 1088
provider: codex
session: 01a13898-67ee-7641-86c1-4cf77195d15c
date: '2026-10-15'
status: failed
config:
  campaign: initial-model-preview-20261015
  clip_id: 269ec7212d2f6cfec639ef83b47af4a2d577ff29299d1ca0e1e2ab6bfafb073e
  inference:
    confidence_threshold: 0.3
    coordinates: native_pixel_xyxy_edges
    decode: pyav_bgr24_native
    device: cuda
    max_long_side: 1333
    schema_version: player.annotation-inference.v1
    short_side: 800
metrics:
  stage_elapsed_seconds: 18.749603291999847
  exit_code: 1
repro:
  commit: d734b1308ed9181e7ed2ffb8d3d8c941f747f590
  branch: codex/annotation-budget-roots
  remote: git@github.com:Motoki0705/tennis-lab.git
  command: /home/kamimura/projects/tennis-lab/.claude/worktrees/annotation-budget-roots/.venv/bin/python
    -m src.tasks.base.annotation.execution --launch /home/kamimura/projects/tennis-lab/outputs/chat_annotation/campaigns/initial-model-preview-20261015/jobs/62e31b7a9d38a00c0364387eaa152beb8e054f85f47ee5e0359c949ab4c1ccbe-1/launch.json
    --sha256 8d800c4cf4a4dd634a6add356b75ca0e955bf0eeb349b56177f7cdfcc0182dcb --queued
artifacts:
  run_dir: knowledge/runs/run-i1088-player-prediction-pilot-20261015
  log: /home/kamimura/projects/tennis-lab/.training_queue/logs/1792034618291181895_178064_annotation-8d800c4cf4a4dd63.log
  evidence: knowledge/runs/run-i1088-player-prediction-pilot-20261015/evidence
parents: []
relations:
- to: run-i1088-ball-prediction-pilot-20261015
  rel: same_input
papers: []
tags:
- annotation
- player_detection
- gpu
- pipeline-smoke
- environment-failure
---

## 観測

2026-10-15、ballの実GPU予測と同じ新規clipをplayer候補で推論する工程を共有queueで実行した。
固定したplayer exportと設定を渡したが、DINO初期化の `build_dino` が
`RuntimeError: DINO CUDA extension is not installed` を返して停止した。
工程はexit code 1、status failedであり、予測receiptは得られていない。
経過18.750秒は失敗までの工程時間で、正常推論の速度を表さない。

## 解釈と次の作業

この試行は実環境で必要なCUDA拡張をロードできなかった記録であり、検出精度・
GPU容量不足・重みの不整合を示すものではない。モデルや閾値を変更して成功に見せず、
固定入力・予算scope・失敗記録を保持して、既存拡張の状態と正規ビルド経路を調査する。
環境整備後は理由付きの新しいattemptで同じclipを再実行する必要がある。

未レビュー予測の採用、追加学習、モデル採用、追跡・pose公開は行っていない。
学習曲線と品質指標は存在しない。ballの成功をplayerや全パイプラインの成功へ一般化しない。

## 根拠と再現条件

queueの元repro bundleを改変せず保存し、工程のlaunch・operation・exit・stderr・設定の
bytes/hashを `evidence/files.json` に記録した。大容量の元動画・clip・重みはGitへ複製していない。
再実行には固定root上の入力実体が必要であり、bundleだけでは再現できない。
失敗した工程も同じ累積予算scopeの使用量に含め、baselineを初期化しない。
