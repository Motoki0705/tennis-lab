---
id: run-fasthmr-camerahmr-pt16-review
type: run
task: tennis_scene
sequence: 88
recorded_at: '2026-11-25'
title: 保存済みGVHMRとFastHMR PT16の目視比較用メッシュ
issue: 2491
provider: codex
session: 01a20d45-575e-7a53-90c5-48b09053fa2b
date: '2026-11-25'
status: done
config:
  model: fasthmr-camerahmr
  backend: pt16
  data: meiji_4cam/multiview-player-evaluation-v1
  feature_batch_size: 4
  diffusion_seed: 0
metrics:
  clips: 6
  source_frame_rows: 1711
  person_frame_requests: 6617
  output_archives: 12
summary_impact: none
repro:
  commit: 97cabfef23bdab3e1097af32a2cdd96cf754b9dd
  branch: codex/parent-2492-issue-2491-primary
  remote: git@github.com:Motoki0705/tennis-lab.git
  command: sha256sum -c outputs/fasthmr-setup/review-pt16/source.sha256 && bash outputs/fasthmr-setup/review-pt16/run.sh
artifacts:
  run_dir: knowledge/runs/run-fasthmr-camerahmr-pt16-review
  log: .training_queue/logs/0-8204421172867557512_2026-11-25T03-53-47-132442487Z_31336_i2491-fasthmr-pt16-visual-6clips.log
parents: [run-fasthmr-camerahmr-smoke]
relations: []
papers: []
tags: [fasthmr, body_recovery]
---

## 考察 / Findings

固定manifestのSHA-256 `3dc496513380a39d4511ff7f8711f4192eababa5109f6a0506f88a60baa70580` を使った。同じ選手ID・選択カメラ・crop・タイムラインでFastHMR CameraHMRをPT16で実行した。GVHMRは採用済みのv2パラメータと配置済みsceneを読み、配置前は描画用の頂点を復元した。GVHMRのモデル推論は再実行していない。 [出力SHAと欠測](../../runs/run-fasthmr-camerahmr-pt16-review/review-pt16.json) [保存GVHMRの参照](../../runs/run-fasthmr-camerahmr-pt16-review/review-gvhmr-archive.json)

6 clip、1,711サンプル時刻、6,617人・フレームの復元リクエストを両方式のNPZで照合した。12ファイルのSHA-256と、推定フレーム、選手、選択カメラ、カメラ幾何、有効性を検証した。配置済みの描画可能件数はGVHMR 6,597、FastHMR 6,604。件数は各方式の表示可用性を示す。品質は同期動画の目視確認で判断する。PT32と比較するparity判定は対象外。 [実行記録](../../runs/run-fasthmr-camerahmr-pt16-review/run.json)

元映像・配置前・配置後、COCO17の再投影と共通コート視点を [描画スクリプト](../../runs/run-fasthmr-camerahmr-pt16-review/render_review.py) で生成する。clip_003の2サンプルを使う試作で4種のH.264動画を確認した。clip_004・clip_028にある1フレーム間隔も、保存されたsource frame差を使って元映像の時刻に対応させる。全6 clipの動画出力と目視所見は後続。目視所見による採用判断までは標準pipelineの判断を維持する。
