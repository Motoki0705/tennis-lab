---
id: run-court-align-b00-eval-aug-r2-weak-warm-v2
type: run
title: court-align-b00-eval-aug-r2-weak-warm-v2
provider: codex
session: 01a05a02-ad55-76c0-9fcb-f250929e59a7
date: '2026-09-02'
status: done
config:
  model: CourtAlignmentCNN（weak-structure round 2 checkpoint、σ=2.0）
  loss: inference-only
  data: B00 accepted ground-UV mean-probability heatmap（max resize、256x256）
  decoder_threshold: 0.25
  decoder_max_peaks: 4
  decoder_cluster_distance_px: 8.0
  decoder_max_instances: 2
metrics:
  predicted_instance_count: 2.0
  reference_instance_count: 2
  instance_tp: 1.0
  instance_fp: 1.0
  instance_fn: 1.0
  instance_precision: 0.5
  instance_recall: 0.5
  instance_f1: 0.5
  matched_center_mean_error_px: 1.0558865070343018
  matched_center_mean_error_m: 0.14739831480151844
  visible_kp_coverage: 0.39285714285714285
repro:
  commit: 5de3a7d45e037a799d4dad0ae1ef3ac3cb24897e
  branch: feat/court-alignment-kp14-pipeline
  remote: git@github.com:Motoki0705/tennis-lab.git
  command: HYDRA_FULL_ERROR=1 /home/kamimura/projects/tennis-lab/.venv/bin/python
    -m src.tasks.court_alignment.scripts.evaluate_real_heatmap paths.data_root=/home/kamimura/projects/tennis-lab/.tmp/b00-alignment-eval
    paths.checkpoint_root=/home/kamimura/projects/tennis-lab/.claude/worktrees/court-alignment-kp14/outputs
    paths.output_root=/home/kamimura/projects/tennis-lab/.claude/worktrees/court-alignment-kp14/outputs
    real_evaluation.archive_path=synthetic_data_generation/scenes/B00/alignment/line-heatmaps/heatmaps.npz
    real_evaluation.manifest_path=synthetic_data_generation/scenes/B00/alignment/line-heatmaps/manifest.json
    real_evaluation.alignment_path=synthetic_data_generation/scenes/B00/alignment/alignment.json
    real_evaluation.device=cuda real_evaluation.preprocess.method=max decoder.threshold=0.25
    real_evaluation.preprocess.content_fraction=1.0 real_evaluation.checkpoint_path=court_alignment/augmentation_round2/b00_weak_warm30_lr3e4_s42/logs/version_0/checkpoints/court-alignment-epoch\\=029.ckpt
    real_evaluation.output_dir=court_alignment/real_heatmap/aug_r2_weak_warm_v2
artifacts:
  run_dir: knowledge/runs/run-court-align-b00-eval-aug-r2-weak-warm-v2
  predictions: knowledge/runs/run-court-align-b00-eval-aug-r2-weak-warm-v2/pred_test.npz
parents:
- run-court-align-aug-r2-weak-warm-v2
relations:
- to: run-court-align-b00-eval-aug-r2-appearance-warm-v1
  rel: compares
tags: [court-alignment, kp14, augmentation, b00, real-heatmap, weak-structure, inference]
---

## 考察 / Findings

### 要約

round 2 weak-structure checkpointをB00へ適用すると、2面を出力し、両方のraw poseはaccepted referenceに近かった。しかし`max_peaks=4`でsemantic候補が切り捨てられ、既存instance metricではTP=1/FP=1/FN=1、F1=0.5に留まった。

### アーキテクチャ詳細

σ=2.0固定、appearance + 弱いline dropout/ghost lineでwarm-start学習したKP14 + center-vote CNNのinference-only runである。32-view B00 `mean_probability` rasterをmax resizeで256x256へ変換し、threshold 0.25、NMS 3、channel当たり`max_peaks=4`、cluster distance 8 px、最大2 instanceでdecodeした。

### メトリクスの解釈

bundle `metrics.json` はF1=0.5、matched center=1.0558865070343018 px（0.14739831480151844 m）、visible KP coverage=11/28=0.39285714285714285である。`diagnostic_metrics.json`由来の2 posesはscale 7.197665691375732/7.163821697235107 px/m、rotation 83.21643620573992°/83.07568649751347°で、accepted reference 7.1668 px/m、約83°に近い。aggregate 180.43279848247766 px/90.09172944055327°/scale 0.5021585907861315は未マッチ1面のpenaltyを含む。

### アーキテクチャ⇄メトリクスの因果考察

weak structureにより、appearance-onlyの崩壊pose 1件が正しいscale/rotationへ改善した。一方、2面目はsemantic count 6でcoverageに余裕がなく、channel当たり4 peaksの上限がglobalに必要な候補を捨てた。観測上、モデル出力には2面の幾何が存在するため、decoder容量を増やすことが学習追加より直接的である。

### 既存実験との比較

appearance round 2の同じ`max_peaks=4`評価もF1=0.5だが、matched centerは4.61154317855835 px、本runは1.0558865070343018 pxで、両raw posesもreferenceに近い。親trainingはsynthetic F1=0.9961190168175938を保ち、round 1 combinedのsynthetic F1=0.667418より大幅に安定した。

### 次に有効な実験

checkpointを固定したままdecoderを`max_peaks=8`へ拡張し、B00の2面matchを再評価する。同時に同設定のsynthetic testで既存性能を維持することを確認する。
