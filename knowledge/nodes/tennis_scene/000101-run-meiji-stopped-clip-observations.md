---
id: run-meiji-stopped-clip-observations
type: run
task: tennis_scene
sequence: 101
recorded_at: '2026-11-26'
title: Meiji 4camの停止14 clipで2D観測と停止理由を照合
issue: 2636
provider: codex
date: '2026-11-26'
status: done
summary_impact: updated
config:
  dataset: meiji_4cam/video_000
  cohort: '#2363の停止14 clip'
  source_commit: fef1d626dd0c40ea8ce69705eda777fec4801021
  camera_count: 4
  replay_workers: 4
  replay_appearance: saved_per_detection_clip_reid
metrics:
  stopped_clips: 14
  synchronized_frames_extracted: 154
  camera_frames_extracted: 616
  anchor_player_slots: 672
  anchor_assessable_player_slots: 665
  anchor_detected_player_slots: 665
  anchor_selected_player_slots: 581
  anchor_chair_person_boxes: 155
  anchor_selected_chair_person_boxes: 0
  selected_off_target_boxes: 5
  confirmed_box_merges: 5
  confirmed_box_splits_or_duplicates: 8
  confirmed_detection_gaps: 2
  confirmed_tracking_id_switches: 6
  confirmed_tracking_fragments: 7
  unchanged_stop_codes_reproduced: 14
  dino_row_deletion_clips: 8
  dino_rows_deleted: 17
  dino_row_deletion_clips_recovered: 0
  joined_group_clips: 2
  joined_group_clips_recovered: 0
  runner_up_omission_clips: 9
  runner_up_omission_clips_passed: 7
  runner_up_omission_clips_with_consistent_four_reviewed_ids: 5
  figure_count: 14
  extraction_wall_seconds: 70.09
  replay_wall_seconds: 102.60
repro:
  commit: 618890b83dcb3362171d2b4596ba8a65e59ee5a2
  branch: codex/parent-2627-issue-2636-primary
  command: >-
    env PYTHONPATH="$PWD" CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
    .venv/bin/python -B knowledge/runs/run-meiji-stopped-clip-observations/replay.py "$I2363_RUN" "$OUT/replay" --jobs 4
artifacts:
  run_dir: knowledge/runs/run-meiji-stopped-clip-observations
  output_dir: .worktree-outputs/parent-2627-issue-2636-primary/stopped-clip-replay-cpu
  input_manifest: knowledge/runs/run-meiji-stopped-clip-observations/inputs.json
  review_counts: knowledge/runs/run-meiji-stopped-clip-observations/review-counts.json
  replay_results: knowledge/runs/run-meiji-stopped-clip-observations/replay.json
  figure_index: knowledge/runs/run-meiji-stopped-clip-observations/figures/index.csv
parents:
- run-i2363-trt-player-video
relations: []
papers: []
tags:
- meiji_4cam
- stopped_clips
- player_association
---

## 考察 / Findings

停止14件の共通原因を2D検出に帰属する主張は不支持。42の同期時点で可視性を評価できた665選手・視点枠すべてにDINO boxがあり、選別後は581枠だった。境界で停止した9件すべてで次点groupに実選手が含まれ、別視点の同一選手は異なる対応identityに入っていた。004・028は人物の重なりで独立boxが途切れ、追跡IDも交代した事例で、DINOのboxが追跡と下流の判定に寄与した可能性がある。 [clip別の観測、前後の画像と件数](../../runs/run-meiji-stopped-clip-observations/README.md)

保存外観を使うCPU再生で元の停止コード14/14件を再現した。目視で確認したDINO行17件を除いて再追跡・再選別した8件では、いずれも元の理由で停止した。036・038の同一選手group統合も停止した。次点groupを外す感度比較では7/9件が人物対応を通過し、5件でレビュー時点の4人のIDが一貫して各人2視点以上に残った。付け直すboxのpose・外観を新規計算する比較と全時系列のID採点は未測定。 [CPUの停止と割当](../../runs/run-meiji-stopped-clip-observations/replay.json)

020・058では不足した選手の追跡が4視点にあり、選別で除外されていた。012・020・058は先行するCPU比較で同じ2D観測から選別幅の変更で三角測量まで進んだ。この結果と今回の映像診断を全件評価の判断へ反映した。 [#2440の固定入力比較](https://github.com/Motoki0705/tennis-lab/blob/63e9730fa23a57d46e7df4d9d4fed9e9204e9f94/knowledge/runs/run-i2440-player-selection/comparison.json)
