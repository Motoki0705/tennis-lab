---
id: run-fasthmr-camerahmr-pt16-review
type: run
task: tennis_scene
sequence: 94
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
  comparison_videos: 24
  source_ticks_per_product: 3414
summary_impact: none
repro:
  commit: 97cabfef23bdab3e1097af32a2cdd96cf754b9dd
  branch: codex/parent-2492-issue-2491-primary
  remote: git@github.com:Motoki0705/tennis-lab.git
  command: sha256sum -c outputs/fasthmr-setup/review-pt16/source.sha256 && bash outputs/fasthmr-setup/review-pt16/run.sh
artifacts:
  run_dir: knowledge/runs/run-fasthmr-camerahmr-pt16-review
  render_manifest: knowledge/runs/run-fasthmr-camerahmr-pt16-review/render-manifest.json
  review_index: .worktree-outputs/parent-2492-issue-2491-primary/tennis_scene/evaluate/fasthmr-qualitative/pt16-20261125/index.html
  log: .training_queue/repro/0-8204421172867557512_2026-11-25T03-53-47-132442487Z_31336_i2491-fasthmr-pt16-visual-6clips/queue.log
  log_checksum: knowledge/runs/run-fasthmr-camerahmr-pt16-review/queue-log.sha256.txt
parents: [run-fasthmr-camerahmr-smoke]
relations: []
papers: []
tags: [fasthmr, body_recovery]
---

## 考察 / Findings

固定manifestのSHA-256 `3dc496513380a39d4511ff7f8711f4192eababa5109f6a0506f88a60baa70580` を使った。同じ選手ID・選択カメラ・crop・タイムラインでFastHMR CameraHMRをPT16で実行した。GVHMRは採用済みのv2パラメータと配置済みsceneを読み、配置前は描画用の頂点を復元した。GVHMRのモデル推論は再実行していない。 [出力SHAと欠測](../../runs/run-fasthmr-camerahmr-pt16-review/review-pt16.json) [保存GVHMRの参照](../../runs/run-fasthmr-camerahmr-pt16-review/review-gvhmr-archive.json)

6 clip、1,711サンプル時刻、6,617人・フレームの復元リクエストを両方式のNPZで照合した。12ファイルのSHA-256と、推定フレーム、選手、選択カメラ、カメラ幾何、有効性を検証した。配置済みの描画可能件数はGVHMR 6,597、FastHMR 6,604。件数は各方式の表示可用性を示す。品質は同期動画の目視確認で判断する。PT32と比較するparity判定は対象外。 [実行記録](../../runs/run-fasthmr-camerahmr-pt16-review/run.json)

CPUジョブ `20261125T070233Z_387170703f99_fasthmr-render-review-6clips` が全6 clipの同期動画を生成した（終了値0、2,126.741秒）。各clipで元映像と配置前の選択ビュー、配置後の4カメラへのメッシュ・COCO17再投影、配置前後の3Dを確認できる。推論ビューは `SRC`、欠測は各パネルに表示する。clip_004・clip_028で1フレームになる推論間隔は、60000/1001 fpsの元時刻に合わせて保持した。 [描画スクリプト](../../runs/run-fasthmr-camerahmr-pt16-review/render_review.py)

24本すべてのSHA-256、H.264の寸法、パケットの表示時刻、HTMLのリンクを照合した。各表示系列は合計3,414のsource tickで、1,711の固定推論時刻を含む。メディアは主checkoutの `.worktree-outputs/parent-2492-issue-2491-primary/tennis_scene/evaluate/fasthmr-qualitative/pt16-20261125/` の `index.html` から閲覧できる。目視所見と採否は人間の確認待ち。 [動画と入力のSHA-256](../../runs/run-fasthmr-camerahmr-pt16-review/render-manifest.json) [動画照合](../../runs/run-fasthmr-camerahmr-pt16-review/verify_videos.py)
