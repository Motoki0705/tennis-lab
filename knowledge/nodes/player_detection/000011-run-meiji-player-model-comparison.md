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
  planned_rows: 36
  current_accuracy_and_speed_conditions: 22
  historical_pt32_accuracy_conditions: 5
  trt_diagnostic_conditions_outside_frontier: 2
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

代表6 clipの6,844画像と26,276選手BBoxを共通採点し、固定96画像の同期込み速度を照合した。36行のうち22条件にPT16/TRT16の精度と速度がそろい、5条件に保存済みPT32精度がある。選手以外の人物は部分ラベルのためprecisionとAPは算出していない。SMPLの配置に現行DINOとViTPoseの観測を使っているため、数値は疑似正解との一致度を表す。 [全条件表](../../runs/run-meiji-player-model-comparison/conditions.csv) · [精度・速度の照合と再現](../../runs/run-meiji-player-model-comparison/README.md)

検出ではRF-DETR-L TRT16が26.43 ms/画像で、同じCPU設定のDINO TRT16の94.79 ms/画像に対してP50が72.1%短い。Recall@0.50は96.81%対96.67%、@0.75は42.09%対47.07%だった。ECDet-L O365 PT16は98.19%と61.46 ms/画像、RT-DETRv4-L PT16は98.12%と52.52 ms/画像を記録した。RF-DETR-LとDINOはPyTorch CPU 4 thread、EdgeCrafterとRT-DETRv4は2 threadで計時した。CPU条件が異なる組み合わせの小差から速度順位は確定できない。 [検出Pareto図](../../runs/run-meiji-player-model-comparison/pareto-detection.png)

検出＋姿勢ではECPose-L PT16が55.11 ms/画像、欠測を含む平均OKSが0.2908。DINO＋ViTPose-H PT16は266.30 ms/画像と0.7631だった。ECPose-L PT16の小人物Recall@0.50は10.99%だった。今回のECPoseは高速で、疑似正解に対する姿勢と小人物の品質は大きく低下した。GPU予測で代表6 clipの同じcam0フレームを並べ、検出数と画像のハッシュを保存した。 [検出＋姿勢Pareto図](../../runs/run-meiji-player-model-comparison/pareto-detection_and_pose.png) · [比較画像の索引](../../runs/run-meiji-player-model-comparison/representative-cam0-gpu-candidates-index.json)

RT-DETRv4 L/XのTRT16はRecallが0.00%と0.03%に落ち、同一入力の診断でも共通後処理の前に差が出た。2条件は測定値を表と図に残し、Pareto frontierの計算から除外した。EdgeCrafterのTRT16計時は採点で使ったplanと一致し、精度差の演算単位での原因は調査中。PT32がスコープ外になった後の計時にparityは適用せず、従来の出力差は履歴に残した。 [RT-DETRv4の診断](../../runs/run-meiji-player-model-comparison/rtdetrv4.md#same-input-tensorrt-diagnosis) · [EdgeCrafterの診断](../../runs/run-meiji-player-model-comparison/EDGECRAFTER.md#diagnosis-of-the-batch-2-traced-plans)

人物検出の置換候補としてRF-DETR-Lを絞り込む根拠をsummaryに追加した。RF-DETR-Lの実際のcropを使ったViTPose-Hと下流の品質は未確認。#1739の入口はDINO由来の入力を要求し、ECPose単独recipeの2D成果を受け入れるには別の判断と検証が要る。 [現状判断](../../summary.md#人物player_detectionperson_trackingplayer_association)
