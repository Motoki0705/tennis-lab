---
id: run-player-camera-stabilization-delivery-20261109
type: run
task: tennis_scene
sequence: 52
recorded_at: '2026-11-09'
title: 広角4強度比較動画の回収と配信監査
issue: 1776
provider: codex
date: '2026-11-09'
status: done
config:
  algorithm: gaussian_markley_quaternion_v1
  sigma_seconds:
  - 0
  - 0.06
  - 0.08
  - 0.12
  translation: unchanged
  horizon_lock: false
  smoothing_context_frames:
  - 0
  - 1010
  source_frames:
  - 0
  - 240
  fps: 59.94006
  lens: equidistant_diagonal_159_approximation
  panel_resolution:
  - 640
  - 360
  pinhole_render_resolution:
  - 2560
  - 1440
  depth_of_field: false
  validation_scope: CPU_delivery_and_release
metrics:
  decoded_video_frames: 3360
  videos: 14
  render_audit_rows_checked: 1920
  camera_missing_rows: 16
  verification_command_seconds: 20.86982533799892
  rendered_images_in_delivery_run: 0
  gpu_allocation_seconds: 10538.681694
  recovered_files: 39
  source_pngs_retained_on_drive: 1920
  raw_pngs_or_zips_recovered: 0
artifacts:
  run_dir: knowledge/runs/run-player-camera-stabilization-delivery-20261109
  verification: knowledge/runs/run-player-camera-stabilization-delivery-20261109/delivery-verification.json
  release: knowledge/runs/run-player-camera-stabilization-delivery-20261109/allocation-release-proof.json
  human_video_links: knowledge/runs/run-player-camera-stabilization-delivery-20261109/human-video-links.json
  sample_review: knowledge/runs/run-player-camera-stabilization-delivery-20261109/delivery-review.json
  raw_drive_outputs: outputs/tennis_scene/visualize/player_view/stabilization-comparison/attempt2/full
parents:
- run-player-camera-stabilization-full-render-20261109
relations:
- rel: evaluates
  to: run-player-camera-stabilization-full-render-20261109
papers: []
tags:
- player-view
- camera-stabilization
- diagnostic
summary_impact: none
repro:
  commit: 1231c2698369f0853368fe90faa3a4439eaa7925
  command: .venv/bin/python knowledge/runs/run-player-camera-stabilization-20261109/verify_delivery.py
    --inputs outputs/tennis_scene/visualize/player_view/stabilization-comparison/inputs
    --run outputs/tennis_scene/visualize/player_view/stabilization-comparison/attempt2/full
    --output outputs/tennis_scene/visualize/player_view/stabilization-comparison/operations/attempt2/delivery-verification.json
  cwd: .
---

このrunのdoneは、比較動画の回収・CPU配信監査・所有VM停止の完了を示す。先行full-render jobのfailed/-15は保持し、人間の強度選択・品質受入はpendingである。

保存済みの全68fileのDrive upload --verify receipt、厳密検証manifestの全source SHA-256/MD5/size、native jobのfailed/-15とdrive_saved_atを照合し、39file（14 MP4・8 auditとmanifest類）を回収した。追加のDrive inventoryはAPI照会で失敗したため、その試行を記録してから、同一の完了済み全tree内容照合receiptを使う明示的な別helperへ切り替えた。回収した各fileのSHA-256/MD5/sizeをsource固定値と比較し、全一致。新しいDriver metadataを合成したり、不明なDrive IDを推定したりしていない。2本の人間用リンクは接続済みDriveで実folderを読み、file名とsizeを回収済みデータと一致させた。共有設定は変更していない。元1920 PNGは8つのlossless ZIPとしてDriveに保持し、PNG/ZIPのローカル全量回収は0。

固定sourceのverify_delivery.pyを実行し、14動画の全3360frameをdecode、各240frame・約59.94006fps・4.004秒・PTS・解像度を検査した。1920行のsource/output/cache index、mask/reason、C2由来のcamera回転/位置、PC2誤差、実OPTIX/L4・camera/color設定、入力・render digestを照合して成功した。コマンドelapsedは20.869825秒。再開ターンで本番コード・pytest caseの追加はなく、既存の受入検証scriptとactual outputの監査を再利用した。

L4は14:18:29 UTCにstop成功。割当要求11:22:50 UTCから10,538.681694秒（2時間55分39秒）で、4時間以内。source差分・untrackedはなく、他のColab assignmentは操作していない。GPU停止後に、両2×2動画からsource0/2/26/60/162/239のcontactと、ID0 source26・ID1 source162の原寸sampleを実際に目視した。左上0・右上0.06・左下0.08・右下0.12秒で同じsource表示が揃い、source0はCAMERA MISSING。source1の欠測保持はaudit/全decodeで確認した。

魚眼周辺のコート線湾曲は強く、細線には段差と軟らかさがある。ID1 source162では大きな首振りが残り、コートが左下へ寄って灰色背景が多い。広角で相手とボールは小さく見える。静止画確認は時間方向のalias、動的なブレ感・酔い・臨場感の受入を代替しない。元4秒区間の最大姿勢偏差はσ0.06でID0/1=7.37/7.39度、0.08で10.59/11.69度、0.12で16.62/20.19度（short-camera-metrics.json）。全1010frameの最大値とは区別する。

人間に見せる2×2は各1280×720。動画リンクとrepo相対pathはhuman-video-links.jsonとreview.mdに記録した。人間が強度を選んだ後に全1010frameを再描画し、本人の定性受入を別途記録する。C5の実NHT GPU、超広角背景support/alpha、メモリ・時間方向aliasは未検証。summaryの標準pipelineと人間確認が独立という判断は変わらずsummary_impactはnone。学習を行わずTensorBoardは対象外。
