---
id: run-court-align-synth-eval-aug-r2-weak-mp8-v2
type: run
title: court-align-synth-eval-aug-r2-weak-mp8-v2
provider: codex
session: 01a05a02-ad55-76c0-9fcb-f250929e59a7
date: '2026-09-02'
status: done
config:
  model: CourtAlignmentCNN（weak-structure round 2 checkpoint、σ=2.0）
  loss: inference-only synthetic test
  data: b00_weak_structure_v2
  decoder_threshold: 0.25
  decoder_max_peaks: 8
  decoder_cluster_distance_px: 8.0
  decoder_max_instances: 2
metrics:
  instance_precision: 0.9869621903520208
  instance_recall: 0.9986807387862797
  instance_f1: 0.9927868852459018
  instance_count_accuracy: 0.982421875
  instance_count_mae: 0.017578125
  matched_center_mean_error_px: 0.5056413905879623
  instance_kp_mean_error_px: 1.9290027483734478
  instance_kp_pck_2px: 0.9952135493372607
  instance_kp_pck_4px: 0.9952135493372607
  sim2_translation_error_px: 0.5942258664350317
  sim2_rotation_error_deg: 0.31961303693232485
  sim2_scale_relative_error: 0.0026453123679110314
repro:
  commit: 5de3a7d45e037a799d4dad0ae1ef3ac3cb24897e
  branch: feat/court-alignment-kp14-pipeline
  remote: git@github.com:Motoki0705/tennis-lab.git
  command: PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HYDRA_FULL_ERROR=1 /home/kamimura/projects/tennis-lab/.venv/bin/python
    -m src.tasks.court_alignment.scripts.evaluate paths.checkpoint_root=/home/kamimura/projects/tennis-lab/.claude/worktrees/court-alignment-kp14/outputs
    paths.output_root=/home/kamimura/projects/tennis-lab/.claude/worktrees/court-alignment-kp14/outputs
    data=b00_weak_structure_v2 evaluation.checkpoint_path=court_alignment/augmentation_round2/b00_weak_warm30_lr3e4_s42/logs/version_0/checkpoints/court-alignment-epoch\\=029.ckpt
    decoder.threshold=0.25 decoder.max_peaks=8 decoder.cluster_distance_px=8.0 decoder.max_instances=2
    metrics.threshold=0.25 metrics.max_peaks=8 metrics.match_max_error_px=8.0 run.output_dir=court_alignment/evaluation/b00_weak_warm30_mp8_v2
    run.gpus=1
artifacts:
  run_dir: knowledge/runs/run-court-align-synth-eval-aug-r2-weak-mp8-v2
  predictions: knowledge/runs/run-court-align-synth-eval-aug-r2-weak-mp8-v2/pred_test.npz
  output_dir: outputs/court_alignment/augmentation_round2/b00_weak_warm30_lr3e4_s42/logs/version_0
parents:
- run-court-align-aug-r2-weak-warm-v2
relations: []
tags: [court-alignment, kp14, augmentation, synthetic, weak-structure, inference, max-peaks-8]
---

## 考察 / Findings

### 要約

weak-structure round 2 checkpointを`max_peaks=8`でsynthetic testした。F1=0.9927868852459018を維持し、B00用decoder容量の拡張がsynthetic性能を崩さないことを確認した。

### アーキテクチャ詳細

学習済みのσ=2.0 KP14 + center-vote CNNは固定し、decoderのみthreshold 0.25、NMS 3、channel当たり`max_peaks=8`、cluster distance 8 px、最大2 instanceへ設定した。学習runの2048/256/256 splitのうちtest splitをinference-onlyで評価している。

### メトリクスの解釈

bundle `metrics.json` はprecision=0.9869621903520208、recall=0.9986807387862797、F1=0.9927868852459018、KP誤差=1.9290027483734478 px、center誤差=0.5056413905879623 px、count accuracy=0.982421875である。`diagnostic_metrics.json`ではTP=757、FP=10、FN=1、visible KP coverage=8110/8148=0.9953362788414335だった。`max_peaks=8`は偽陽性を僅かに増やすが高い性能を保つ。

### アーキテクチャ⇄メトリクスの因果考察

channel候補数を4から8へ広げると、複数courtの同一semantic KPを捨てにくくなる一方、noise peakも残るためprecisionがわずかに低下する。観測されたF1=0.9927868852459018は、このtrade-offがB00成功のために許容できることを示す。

### 既存実験との比較

親trainingのdefault decoder test F1=0.9961190168175938に対し0.003332低下したが、依然0.99超である。同じ`max_peaks=8`を使うB00 runはF1=1.0となり、decoder容量拡張の必要性を補強する。

### 次に有効な実験

複数sceneのholdoutで`max_peaks=4/6/8`を比較し、B00以外でもrecall改善がFP増加を上回るか検証する。B00については独立GTを用意する。
