---
id: run-i2634-player-selection-ablation
type: run
task: person_tracking
sequence: 15
recorded_at: '2026-11-26'
title: 保存済み検出・追跡から画像2Dとコート座標の選手選別を比較
issue: 2634
provider: codex
date: '2026-11-26'
status: done
summary_impact: updated
config:
  device: cpu
  dataset_3cam: meiji_3cam
  dataset_4cam: meiji_4cam
  conditions: [all_tracks, image_2d, court_raw_dwell, court_standard]
  min_presence_fraction: 0.25
  max_candidates_per_camera: 6
  box_iou: 0.5
metrics:
  dev_image_2d_player_precision: 0.8555270370209249
  dev_court_standard_player_precision: 0.9998473903754197
  dev_image_2d_player_recall: 0.9537342386032978
  dev_court_standard_player_recall: 0.9532007759456838
  reserved_image_2d_player_precision: 0.6324615518290034
  reserved_court_standard_player_precision: 1.0
  reserved_image_2d_group_idf1: 0.7497832433867115
  reserved_court_standard_group_idf1: 0.9971266988962875
  reserved_image_2d_pair_f1: 0.7734828421705077
  reserved_court_standard_pair_f1: 0.9942706648800999
  meiji_4cam_image_2d_triangulated_clips: 0
  meiji_4cam_court_raw_dwell_triangulated_clips: 52
  meiji_4cam_court_standard_triangulated_clips: 55
  meiji_4cam_court_standard_player_frames_3d: 133887
repro:
  commit: dfb03c0785c7a7a41850c5cd6909b1bb383a0b75
  command: bash knowledge/runs/run-i2634-player-selection-ablation/reproduce.sh outputs/i2634/full-replay --jobs 4
artifacts:
  run_dir: knowledge/runs/run-i2634-player-selection-ablation
  output_dir: outputs/i2634/full-replay
parents:
- run-i964-default-merge-r11-20260930
- run-i964-unseen-r16-20261001
- run-i2363-trt-player-video
relations:
- to: run-multiview-pose-link-recall-20261124
  rel: related
  note: 予約3camのvideo_001/clip_003の固定校正と向きを共有する。
tags: [player-selection, ablation]
---

全条件で保存済みDINO・StrongSORT++・ViTPose・CLIPを共有した。3camの開発4本・予約3本では
画像情報だけによる順位付け、同じraw ID単位でのコート滞在、断片を連結する実験時の標準選別と全raw IDを比較した。
コート滞在には実行版の標準であるシングルス幅を用いた。ダブルス幅を採用した比較は[#2440の記録](000014-run-i2440-player-selection.md)にある。
選手・非選手の部分参照、校正、向き、MILP設定を固定した。4camでは同一の初期校正から選手観測による
court sideを含む経路を再生した。 [手順と結果](../../runs/run-i2634-player-selection-ablation/README.md)

**第6章の主張を測定範囲で支持する。** 画像2Dから実験時の標準選別へ替えると、予約3本で選手precisionは
63.25→100%、group IDF1は74.98→99.71%。既知の非選手12,690件の誤採用が0件になり、
選手recallは99.17→99.29%。全3本のpair F1は77.35→99.43%で、両条件で対応できた2本の
pair F1は99.42→99.30%だった。開発4本でもprecisionとIDF1は上がり、recallは0.05ポイント下がった。
[分子・分母とclip別の指標](../../runs/run-i2634-player-selection-ablation/meiji-3cam.json)

4camは画像2Dと選別なしの各69本がcourt sideで停止し、同じ標準選別は55本が三角測量まで進んで
133,887/184,912選手フレームを得た。同じraw ID単位でコート滞在を評価する条件も52本が完了した。
4camは有効3Dの被覆を測定した。ダブルスでの選別・IDの精度と、向きを固定した別プロファイルでの完了差は未測定。
[4camの集計](../../runs/run-i2634-player-selection-ablation/meiji-4cam.json)

[summary](../../summary.md#人物player_detectionperson_trackingplayer_association)には
同一入力での3cam選別と4cam被覆の根拠を追加した。先行run 10・11の13条件を第6章向けの表の元データにした。
[既存2Dアブレーション表](../../runs/run-i2634-player-selection-ablation/historical-2d.csv)
