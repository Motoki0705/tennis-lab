---
id: run-i2443-meiji-4cam-tracker-comparison
type: run
task: person_tracking
sequence: 14
recorded_at: '2026-11-24'
title: meiji_4camでの過去の2D追跡比較と部分注釈での採点
issue: 2443
provider: codex
date: '2026-11-24'
status: failed
summary_impact: none
config:
  dataset: meiji_4cam
  planned_clips: 69
  planned_annotated_frames: 1205
  pilot_reference: run-i2442-meiji-pseudo-association-pilot
metrics:
  initial_queue_failed_before_launch: 4
  full_saved_default_clips: 69
  full_saved_default_frames: 1205
  full_saved_default_stopped_clips: 14
  full_saved_default_pair_f1: 0.7503273378816071
  full_saved_default_valid_player_frames: 3774
  full_saved_default_labelled_player_frames: 4802
  pilot_clips: 3
  pilot_frames: 16
  pilot_boxes: 486
  pilot_stopped_clips: 1
  pilot_raw_idf1: 0.8483516483516483
  pilot_group_idf1: 0.8527472527472527
  pilot_selection_precision: 0.9853658536585366
  pilot_selection_recall: 0.808
  pilot_pair_f1: 0.5333333333333333
  pilot_matched_decided_pair_f1: 0.6997635933806147
  pilot_covered_player_frames: 34
  pilot_labelled_player_frames: 64
artifacts:
  run_dir: knowledge/runs/run-i2443-meiji-4cam-tracker-comparison
parents: [run-i964-tracker-hybrids-r10-20260930]
relations:
  - to: run-i964-tracker-matrix-r8-20260930
    rel: compares
  - to: run-i964-tracker-linking-r9-20260930
    rel: compares
  - to: run-i2442-meiji-pseudo-association-pilot
    rel: related
    note: 採点に使う4camの部分注釈と抽出計画
  - to: run-i2363-trt-player-video
    rel: related
    note: 固定した検出・姿勢・校正と現行基準の保存出力
---
同期4カメラの保存済み現行経路を、3 clip・16 frameのTerra試行ラベルで採点した。
停止したclip_020もpairと3D coverageの分母に含め、実観測に対するraw/group IDF1、選別、対応、3D被覆を測った。
保存CLIPのclip_025では再実行したStrongSORT++＋poseの実観測と元rowの一致、Lab連結の曖昧停止を確認した。
比較条件、除外理由、層別の暫定値と再実行方法は[run記録](../../runs/run-i2443-meiji-4cam-tracker-comparison/README.md)にある。

全1,205 frameを公開後、保存済み現行経路で採点した。pair F1は疑義全体0.7265、確信層の無作為標本0.8957、停止14/69 clip。
全体比較の初回投入は4件ともGPUロックの権限により起動前に失敗した。共有ロックのACL復旧が再投入条件となる。
既定方式の判断には、同じ4camの層別人手誤り率とFP32外観での全件比較が必要。
