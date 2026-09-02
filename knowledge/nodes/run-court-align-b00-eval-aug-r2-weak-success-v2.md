---
id: run-court-align-b00-eval-aug-r2-weak-success-v2
type: run
title: court-align-b00-eval-aug-r2-weak-success-v2
provider: codex
session: 01a05a02-ad55-76c0-9fcb-f250929e59a7
date: '2026-09-02'
status: done
config:
  model: CourtAlignmentCNN（weak-structure round 2 checkpoint、σ=2.0）
  loss: inference-only
  data: B00 accepted ground-UV mean-probability heatmap（max resize、256x256）
  decoder: b00_v1
  decoder_threshold: 0.25
  decoder_max_peaks: 8
  decoder_cluster_distance_px: 8.0
  decoder_max_instances: 2
metrics:
  predicted_instance_count: 2.0
  reference_instance_count: 2
  instance_tp: 2.0
  instance_fp: 0.0
  instance_fn: 0.0
  instance_precision: 1.0
  instance_recall: 1.0
  instance_f1: 1.0
  matched_center_mean_error_px: 0.9204371273517609
  matched_center_mean_error_m: 0.1284900228846212
  instance_kp_mean_error_px: 129.3010733468192
  visible_kp_coverage: 0.6428571428571429
  pose_raw_kp_count: 18
  pose_raw_kp_coverage: 0.6428571428571429
  pose_raw_kp_error_mean_px: 0.7880935478541586
  pose_raw_kp_error_q95_px: 1.7984622478485104
  pose_reconstructed_kp_count: 28
  pose_reconstructed_kp_error_mean_px: 0.4520937505045107
  pose_reconstructed_kp_error_q95_px: 0.6897681146860122
  pose_reconstructed_kp_error_max_px: 0.7279878258705139
  sim2_translation_error_px: 0.1953481063246727
  sim2_translation_error_m: 0.027269958920868368
  sim2_rotation_error_deg: 0.24728358380572663
  sim2_scale_relative_error: 0.002297769654101167
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
    real_evaluation.device=cuda real_evaluation.checkpoint_path=court_alignment/augmentation_round2/b00_weak_warm30_lr3e4_s42/logs/version_0/checkpoints/court-alignment-epoch\\=029.ckpt
    real_evaluation.preprocess.method=max real_evaluation.preprocess.content_fraction=1.0
    decoder=b00_v1 real_evaluation.output_dir=court_alignment/real_heatmap/aug_r2_weak_success_v2
artifacts:
  run_dir: knowledge/runs/run-court-align-b00-eval-aug-r2-weak-success-v2
  predictions: knowledge/runs/run-court-align-b00-eval-aug-r2-weak-success-v2/pred_test.npz
parents:
- run-court-align-aug-r2-weak-warm-v2
relations:
- to: run-court-align-b00-eval-aug-r2-weak-warm-v2
  rel: supersedes
- to: run-court-align-synth-eval-aug-r2-weak-mp8-v2
  rel: confirms
- to: group-court-align-b00-real-heatmap-eval
  rel: compares
- to: run-court-align-b00-s200-max-t025-v2
  rel: compares
tags: [court-alignment, kp14, augmentation, b00, real-heatmap, weak-structure, inference, max-peaks-8, success]
---

## 考察 / Findings

### 要約

σ=2.0、weak-structure round 2 checkpoint、B00用`max_peaks=8` decoderにより、B00 accepted alignmentの2面へTP=2/FP=0/FN=0、F1=1.0で対応した。再構築14KPの平均誤差は0.4520937505045107 px、回転誤差は0.24728358380572663°で、今回のsystem-relative成功条件を満たす。

### アーキテクチャ詳細

4-down U-NetのKP14 heatmap + 2ch center-vote CNNを、σ=2.0固定、appearance + 弱いline dropout/ghost line、2048/256/256 samples、30 epochs、128 steps/epoch、LR 3e-4でstrict model-only warm-start学習したcheckpointを用いた。実入力はB00の48 views中32 viewsを集約した`mean_probability` rasterである。`decoder=b00_v1`はthreshold 0.25、NMS 3、`max_peaks=8`、cluster distance 8 px、最大2 instancesを明示する。

### メトリクスの解釈

bundle `metrics.json` のinstance結果はTP=2/FP=0/FN=0、precision/recall/F1=1.0である。raw検出KPは18/28、coverage=0.6428571428571429、誤差mean=0.7880935478541586 px、q95=1.7984622478485104 px。検出KPから各courtのSim(2)をfitして14点を再構築すると28/28点でmean=0.4520937505045107 px、q95=0.6897681146860122 px、max=0.7279878258705139 pxだった。centerは0.9204371273517609 px/0.1284900228846212 m、translationは0.1953481063246727 px/0.027269958920868368 m、rotationは0.24728358380572663°、scale相対誤差は0.002297769654101167である。

従来の`instance_kp_mean_error_px=129.3010733468192`は未検出10KPへmissing penaltyを入れた値で、検出済みKPや再構築poseの連続誤差ではない。coverage・raw/reconstructed統計とこのpenaltyはいずれも本runの`metrics.json`が出典である。accepted alignmentは既存systemが採択したreferenceで、独立GTではない。

### アーキテクチャ⇄メトリクスの因果考察

weak appearance/structure augmentationがB00の線幅、blur、欠損、ghost lineに対する2面分の正しいKP候補を生成し、`max_peaks=8`が各semantic channelでそれらを残した。`max_peaks=4`でも2面pose自体はreference近傍だったが、semantic候補切捨てにより1面しかmetric matchしなかった。したがって今回の成功はaugmentationだけでなく、multi-instance出力に十分なdecoder容量との組合せによる。

### 既存実験との比較

同じcheckpointの`max_peaks=4` runはTP=1/FP=1/FN=1、F1=0.5であり、本runがsupersedeする。`max_peaks=8`のsynthetic testはF1=0.9927868852459018を維持し、B00だけへ過剰適合したdecode変更ではないことを確認した。clean σ=2.0 baselineのB00はTP=0/FP=2/FN=2、F1=0だったため、round 2は明確に改善した。ただし比較reference自体がaccepted alignment由来なので、独立した絶対精度の証明ではない。

### 次に有効な実験

B00以外のsceneと手動annotationによる独立GTで再評価する。raw coverageは18/28=0.6428571428571429に留まるため、F1を保ったままsemantic KP coverageを高めるline-dropout/ghost強度とdecoder thresholdの小規模ablationも有効である。
