---
id: run-i2269-player-court-side-20261119
type: run
task: court_side
sequence: 8
recorded_at: '2026-11-19'
title: 選手の足元だけでmeiji_4camの3clipのhalf-turnを判定
issue: 2269
provider: codex
date: '2026-11-19'
status: done
summary_impact: updated
config:
  dataset: meiji_4cam
  video_id: video_000
  clip_ids: [clip_000, clip_027, clip_037]
  camera_ids: [cam0, cam1, cam2, cam3]
  reference: cam0
  evidence: player
  max_frames: 2048
  match_distance_m: 3.0
  symmetry_distance_m: 0.5
  min_motion_m: 0.15
  bottom_border_px: 4.0
  min_frames: 8
  max_cost: 0.7
  min_support: 0.2
  min_margin: 0.05
metrics:
  clip_matches: 3
  clip_stops: 0
  shared_matches: 3
  shared_stops: 0
  scored_frames: 369
  shared_cost: 0.14564943190466034
  shared_support: 0.46612466124661245
  shared_margin: 0.23719908212715557
  clip_margins: {clip_000: 0.17341812982712787, clip_027: 0.22422588369809465, clip_037: 0.3150889679174889}
  cpu_wall_seconds: 20.458035708987154
repro:
  commit: 0aad14f15b3e16287ec59a3e69f8d8314bbfed06
  branch: codex/court-side-player
  command: bash knowledge/runs/run-i2269-player-court-side-20261119/reproduce.sh
artifacts:
  run_dir: knowledge/runs/run-i2269-player-court-side-20261119
  output_dir: .worktree-outputs/court-side-player/tennis_scene/evaluate/court-side-player-20261119
parents:
- run-court-side-video-shared-20261118
relations: []
papers: []
tags: [player_footpoints, video_shared, cpu_acceptance]
---

meiji_4camのcamera順はcam0、cam1、cam2、cam3。#1752が保存した4cameraの校正・`selected_player_tracks` v2から観測bboxとmaskを読み、選手の足元だけで採点した。clip_000、027、037の単独判定と3clipを統合した動画判定は、Ball実測で確認した `[F,F,T,T]` と一致した。共有成果物を読み直し、単独未決も拒否する `unobservable_policy=fail` で3clipすべてを採用した。停止は0件。

| 範囲 | 元frame | サンプルframe | 採点frame | 最良cost | 最良support | margin |
|---|---:|---:|---:|---:|---:|---:|
| clip_000 | 356 | 178 | 144 | 0.185605 | 0.229167 | 0.173418 |
| clip_027 | 235 | 118 | 107 | 0.107547 | 0.859813 | 0.224226 |
| clip_037 | 292 | 146 | 118 | 0.131441 | 0.398305 | 0.315089 |
| 動画共有 | 883 | 442 | 369 | 0.145649 | 0.466125 | 0.237199 |

全8仮説のcost・support・順位、使用した元frame番号、camera対の件数、距離閾値ごとの停止理由、固定入力の参照を記録した。[計測結果 JSON](../../runs/run-i2269-player-court-side-20261119/report.json)

## 距離閾値

初期の2.5 mではclip_037で正しい向きが最良だったが、同じframeの全camera対が閾値以内に入る割合が0.161で、`min_support=0.2` に達せず `no_consistent_hypothesis` で停止した。同clipで正しい向きを適用した際、frame内の最大対応距離の20パーセンタイルは2.555 m、中央値は3.237 mだった。
3.0 mは試した2.5、3.0、3.5 mのうち3clipがすべて確定する最小値だった。3.5 mではclip_027の別仮説のsupportが0.327、3.0 mでは0.178だったため、3.0 mを採った。採点条件の他の閾値は固定した。

clip_000はcam3を含む対に大きな残差が集中し、cam2–cam3の対応距離の90パーセンタイルは14.261 mだった。最良仮説でもsupportは下限より0.029高い。今回の3clipで閾値を選定しており、別収録や未見clipでの判定性能、全69clipでの選手経路の採否は未計測。

## 保存済み入力の扱い

共有buildは共有コートに登録した全clipを必要とする。検証では69clip版の共有H行列とdrift報告から、対象3clipに範囲を固定したコート成果物を作った。既存のCPU校正を実行し、元storeとカメラ行列・基準camera・コートkeypoint・可視性・ROIが一致することを確認した。検証用storeにはその校正と固定した観測を保存した。
採点、共有build、再ロード、各clipの採用検査を20.46秒で実行した。所要時間には版・配列の照合、校正の再計算、検証用storeへのコピーを含む。[再現コマンド](../../runs/run-i2269-player-court-side-20261119/reproduce.sh)・[評価コード](../../runs/run-i2269-player-court-side-20261119/evaluate.py)
