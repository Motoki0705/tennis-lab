---
id: run-court-align-b00-s200-max-t025-cf08372-v2
type: run
title: court-align-b00-s200-max-t025-cf08372-v2
provider: codex
session: 01a05a02-ad55-76c0-9fcb-f250929e59a7
date: '2026-09-02'
status: done
config:
  model: CourtAlignmentCNN（4-down U-Net、KP14 heatmap + 2ch center vote、sigma=2.0 checkpoint）
  loss: inference-only（decoder threshold 0.25）
  data: B00 accepted ground-UV mean-probability heatmap（max resize、content fraction
    0.8372）
  input_shape: 999x908 -> 256x256
  aggregate_views: 32
  training_scale_range_px_per_m:
  - 3.0
  - 6.0
metrics:
  predicted_instance_count: 2
  reference_instance_count: 2
  instance_tp: 0
  instance_fp: 2
  instance_fn: 2
  instance_f1: 0.0
  matched_instance_count: 0
  coverage_gate_pass_rate: 1.0
  sim2_pair_count: 0
  sim2_translation_error_m: 60.18720805952798
  sim2_rotation_error_deg: 180.0
  sim2_scale_relative_error: 1.0
repro:
  commit: a3861691b1954779fd3dc2ab754cc99313018994
  branch: feat/court-alignment-kp14-pipeline
  remote: git@github.com:Motoki0705/tennis-lab.git
  command: HYDRA_FULL_ERROR=1 /home/kamimura/projects/tennis-lab/.venv/bin/python
    -m src.tasks.court_alignment.scripts.evaluate_real_heatmap paths.data_root=/home/kamimura/projects/tennis-lab/.tmp/b00-alignment-eval
    paths.checkpoint_root=/home/kamimura/projects/tennis-lab/.claude/worktrees/court-alignment-kp14/outputs
    paths.output_root=/home/kamimura/projects/tennis-lab/.claude/worktrees/court-alignment-kp14/outputs
    real_evaluation.archive_path=synthetic_data_generation/scenes/B00/alignment/line-heatmaps/heatmaps.npz
    real_evaluation.manifest_path=synthetic_data_generation/scenes/B00/alignment/line-heatmaps/manifest.json
    real_evaluation.alignment_path=synthetic_data_generation/scenes/B00/alignment/alignment.json
    real_evaluation.device=cuda real_evaluation.checkpoint_path=court_alignment/ablation_sigma_200/logs/version_0/checkpoints/court-alignment-epoch\\=047.ckpt
    real_evaluation.preprocess.method=max decoder.threshold=0.25 real_evaluation.preprocess.content_fraction=0.8372
    real_evaluation.output_dir=court_alignment/real_heatmap/b00_s200_max_t025_cf08372_v2
artifacts:
  run_dir: knowledge/runs/run-court-align-b00-s200-max-t025-cf08372-v2
  predictions: knowledge/runs/run-court-align-b00-s200-max-t025-cf08372-v2/pred_test.npz
  log: .training_queue/logs/1788305344158883902_177404_court-align-b00-s200-max-t025-cf08372-v2.log
parents:
- run-court-align-b00-s200-max-t025-v2
relations:
- to: run-court-align-kp14-ablation-sigma-200
  rel: compares
tags:
- court-alignment
- kp14
- multi-court
- ground-uv
- real-heatmap
- b00
- sigma-200
- scale-normalization
- inference
---

## 考察 / Findings

### 要約

content fractionを0.8372へ縮め、reference scaleを7.1668から5.9968 px/mへ正規化して学習範囲内へ入れたが、TP=0、FP=2、FN=2、F1=0のままだった。scale normalization単独が無効だったことを示すnegative resultである。referenceはaccepted alignmentで、独立 ground truth ではない。

### アーキテクチャ詳細

sigma=2.0の同一checkpointとmax resizeを用い、32/48 views集約の999x908 heatmapを256x256へ配置するときのcontent fractionだけを1.0から0.8372へ変更した。decoder thresholdは0.25、最大2 instances、学習scale範囲は3--6 px/mである。

### メトリクスの解釈

raw semantic countは9/7、predicted scaleは0.01206/0.02304 px/mで、正規化後reference 5.9968 px/mには近づかなかった。coverage gateは全4候補が通ったがmatch pairは0である。60.187 m / 180 deg / relative scale 1.0は未match penaltyであり、連続pose誤差ではない。

### アーキテクチャ⇄メトリクスの因果考察

観測として、apparent scaleを学習上限へ合わせるとcoverageは改善したが、14 semantic peaksの幾何整合とinstance matchは回復しなかった。したがってscale OODだけでは失敗を説明できない。仮説として、線幅/blur、欠落線、false/parallel/ghost lines、確率値分布の差がより支配的である。

### 既存実験との比較

親のcontent fraction 1.0 runはreference 7.1668 px/m、semantic 9/6、coverage 0.5、F1 0だった。本runはreference 5.9968、semantic 9/7、coverage 1.0へ変わったがF1は0のままで、synthetic sigma=2.0のF1 0.99737を再現できなかった。

### 次に有効な実験

scale range自体は余裕を持って拡張しつつ、morphological線幅/blur、line dropout、spurious・parallel・ghost lines、probability amplitude/noise、view dropoutをfactorialに追加する。content fractionによるscale normalizationは補助変換としてのみ扱い、主施策にしない。
