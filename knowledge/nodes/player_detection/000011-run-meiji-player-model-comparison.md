---
id: run-meiji-player-model-comparison
type: run
task: player_detection
sequence: 11
recorded_at: '2026-11-25'
title: Meiji 4cam 人物検出と統合姿勢の精度・速度比較
issue: 2487
provider: codex
date: '2026-11-25'
status: done
summary_impact: updated
config:
  evaluation_id: meiji_4cam/multiview-player-evaluation-v1
  evaluation_manifest_sha256: 3dc496513380a39d4511ff7f8711f4192eababa5109f6a0506f88a60baa70580
  speed_id: meiji_4cam/person-model-speed-v1
  speed_manifest_sha256: 64c74d75e5f1afe869911f1620b9c8e8a793fc4c4e4ac830f4abd97b21fb1a62
  timing_protocol_sha256: 39e362cb87c7c373f8c1ad25e5e28025aab89b1467be21ca3139c2fb456136a2
  gpu: NVIDIA GeForce RTX 5060 Ti
  measured_backends: [pt16, trt16]
  image_batch_for_frontier: 1
  warmup: 30
  timing_blocks: 3
  repetitions_per_block: 100
metrics:
  evaluation_images: 6844
  eligible_player_boxes: 26276
  fixed_timing_images: 96
  planned_rows: 44
  matched_accuracy_and_speed_conditions: 18
  measured_current_accuracy_conditions: 26
  original_trt_conditions_with_unmatched_timing: 8
  historical_pt32_accuracy_conditions: 5
  trt_diagnostic_conditions_with_matched_timing: 4
  fp32_attention_infeasible_conditions: 2
  fp32_attention_inapplicable_conditions: 2
repro:
  scorer_commit: 0d4bf0990f82ae0b8ebf2afe08d3209e583d2364
  harness_merge_commit: 59faa32a28d2f54e6492abdba2429846fa0397e3
  command: >-
    env PYTHONPATH=. OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
    .venv/bin/python -B knowledge/runs/run-meiji-player-model-comparison/compare_conditions.py
    --artifacts "$PERSON_COMPARISON_ARTIFACTS"
    --evidence knowledge/runs/run-meiji-player-model-comparison/comparison-evidence.json
    --output "$PERSON_COMPARISON_OUTPUT"
artifacts:
  run_dir: knowledge/runs/run-meiji-player-model-comparison
  comparison: knowledge/runs/run-meiji-player-model-comparison/conditions.csv
  evidence: knowledge/runs/run-meiji-player-model-comparison/comparison-evidence.json
  detection_plot: knowledge/runs/run-meiji-player-model-comparison/pareto-detection.png
  joint_plot: knowledge/runs/run-meiji-player-model-comparison/pareto-detection_and_pose.png
  overlay_index: knowledge/runs/run-meiji-player-model-comparison/representative-cam0-gpu-candidates-index.json
  output_dir: .worktree-outputs/parent-2492-issue-2487-primary/player-model-comparison
parents:
- run-meiji-multiview-player-evaluation
- run-person-model-speed-low-precision
- run-meiji-rfdetr-large-speed
relations: []
papers: []
tags:
- meiji-4cam
- player-detection
- pose-estimation
- fp16
- tensorrt
---

## 考察

代表6 clipの6,844画像と26,276選手BBoxを共通採点し、固定96画像の同期込み速度と照合した。44行のうち18条件に現行の精度とホスト設定を揃えた速度がそろい、旧スレッド設定の8条件は精度のみ比較表に掲載した。保存済みのPT32精度は5条件。精度と主比較の速度・VRAMはともに画像batch 1で、batch 2の精度は未計測。選手以外の人物は部分ラベルのためprecisionとAPは算出していない。SMPLの配置に現行DINOとViTPoseの観測を使っているため、数値は疑似正解との一致度を表す。 [全条件表](../../runs/run-meiji-player-model-comparison/conditions.csv) · [照合と再現](../../runs/run-meiji-player-model-comparison/README.md)

検出のParetoはRF-DETR-L PT16/TRT16とECDet-L/X O365 PT16。CPU設定はいずれもPyTorch 4・OpenCV 1スレッド。RF-DETR-L TRT16とDINO TRT16のP50は26.43対94.79 ms/画像、Recall@0.50は96.81%対96.67%、@0.75は42.09%対47.07%だった。ECDet-X O365 PT16は50.75 ms/画像、Recall@0.50/0.75は97.97%/46.62%。Lは52.91 ms/画像、98.19%/44.88%だった。RT-DETRv4-L PT16は53.67 ms/画像で98.12%/45.57%を記録した。 [検出Pareto図](../../runs/run-meiji-player-model-comparison/pareto-detection.png)

検出＋姿勢のParetoはECPose-L/X PT16とDINO＋ViTPose-H PT16/TRT16。ECPose-L/X PT16は56.89/56.06 ms/画像、欠測込み平均OKS 0.2908/0.2588。DINO＋ViTPose-H PT16は266.30 ms/画像と0.7631だった。ECPose-L PT16の小人物Recall@0.50は10.99%。GPU予測で代表6 clipの同じcam0フレームを並べ、検出数と画像のSHA-256を保存した。 [検出＋姿勢Pareto図](../../runs/run-meiji-player-model-comparison/pareto-detection_and_pose.png) · [比較画像の索引](../../runs/run-meiji-player-model-comparison/representative-cam0-gpu-candidates-index.json)

EdgeCrafterとRT-DETRv4のTensorRT変換は原因未特定で調査を終了した。ECDet 4重みのFP32 attention版は精度と matched 速度の両方を図に載せ、Pareto判定から外した。元のTRT16は精度を図の別枠に残した。RT-DETRv4のFP32 attention版は全診断ケースで後処理前にNaNとなり、終端ログで失敗を確認した。ECPoseのFP32 attention版は適用不可。PT32 parityは未計測で、既存の出力差は履歴に残した。 [EdgeCrafterの診断](../../runs/run-meiji-player-model-comparison/EDGECRAFTER.md) · [RT-DETRv4の診断](../../runs/run-meiji-player-model-comparison/rtdetrv4.md#fp32-decoder-attention-probe)

人物検出の置換候補にRF-DETR-LとECDet O365 PT16が残った。候補の実BBoxを使ったViTPose-Hと下流品質、利用条件の最終確認が採用判断に残る。#1739の入口はDINO由来の入力を要求し、ECPose単独recipeの2D成果を受け入れるには別の判断と検証が要る。 [現状判断](../../summary.md#人物player_detectionperson_trackingplayer_association)
