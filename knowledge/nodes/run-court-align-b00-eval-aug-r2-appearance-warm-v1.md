---
id: run-court-align-b00-eval-aug-r2-appearance-warm-v1
type: run
title: court-align-b00-eval-aug-r2-appearance-warm-v1
provider: codex
session: 01a05a02-ad55-76c0-9fcb-f250929e59a7
date: '2026-09-02'
status: done
config:
  model: CourtAlignmentCNN（appearance round 2 checkpoint、σ=2.0）
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
  matched_center_mean_error_px: 4.61154317855835
  matched_center_mean_error_m: 0.6437563967581381
  visible_kp_coverage: 0.2857142857142857
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
    real_evaluation.preprocess.content_fraction=1.0 real_evaluation.checkpoint_path=court_alignment/augmentation_round2/b00_appearance_warm30_lr3e4_s42/logs/version_0/checkpoints/court-alignment-epoch\\=028.ckpt
    real_evaluation.output_dir=court_alignment/real_heatmap/aug_r2_appearance_warm_v1
artifacts:
  run_dir: knowledge/runs/run-court-align-b00-eval-aug-r2-appearance-warm-v1
  predictions: knowledge/runs/run-court-align-b00-eval-aug-r2-appearance-warm-v1/pred_test.npz
parents:
- run-court-align-aug-r2-appearance-warm-v1
relations:
- to: run-court-align-b00-eval-aug-pilot-appearance-v1
  rel: supersedes
tags: [court-alignment, kp14, augmentation, b00, real-heatmap, appearance, inference]
---

## 考察 / Findings

### 要約

round 2 appearance checkpointをB00実heatmapへ適用したところ、2面を出力したがTP=1/FP=1/FN=1、F1=0.5だった。1面のposeは正しい一方、`max_peaks=4`では複数courtに必要なsemantic候補が切り捨てられ、2面同時成功には至らなかった。

### アーキテクチャ詳細

σ=2.0固定、appearance augmentationでwarm-start学習したKP14 + center-vote CNNのinference-only runである。48 views中32 viewsの`mean_probability` rasterをmax resize/letterboxで256x256へ変換し、threshold 0.25、NMS 3、channel当たり`max_peaks=4`、cluster distance 8 px、最大2 instanceでdecodeした。

### メトリクスの解釈

bundle `metrics.json` は2 predictions/2 referencesに対してF1=0.5、matched center=4.61154317855835 px（0.6437563967581381 m）、visible KP coverage=8/28=0.2857142857142857を示す。`diagnostic_metrics.json`由来では、正しい側のscaleは7.203720569610596 px/m、rotationは83.34676304529293°でreference 7.1668 px/m、約83°に近い。一方のaggregate Sim(2) 180.41010334342718 px/90.15833744533924°/scale 0.5025669766195682は未マッチ1面のpenaltyを含み、連続pose誤差として解釈しない。

### アーキテクチャ⇄メトリクスの因果考察

モデルは2面を数え、1面は正しいposeへ到達したため、appearance domain gapはround 1より安定して縮小した。ただし各KP channelに最大4候補しか残さないdecoderは、2面分の真peakとghost/noise peakが競合するB00で候補不足になる。仮説として学習不足ではなくdecode時のsemantic候補切捨てが第2面のmatchを阻害した。

### 既存実験との比較

親trainingはsynthetic F1=0.9974093264248705を保った。round 1 appearanceもB00 F1=0.5だったがsynthetic F1=0.827103であり、round 2はsynthetic劣化を解消した。ただしappearanceだけでは`max_peaks=4`時の2面成功を達成していない。

### 次に有効な実験

appearance checkpoint単独の追学習より先に、複数court用に`max_peaks=8`を評価する。加えてweak structure checkpointで欠損/ghost line耐性を比較する。
