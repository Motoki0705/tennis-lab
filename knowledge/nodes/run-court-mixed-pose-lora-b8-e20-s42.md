---
id: run-court-mixed-pose-lora-b8-e20-s42
type: run
title: DINOv3 LoRA mixed-source court dense+pose 学習
provider: codex
session: 01a06e82-4115-7af1-b52d-c011c61cfa56
date: '2026-09-10'
status: done
config:
  model: court_hierarchical_dinov3_vitb16_lora_r8
  loss: dense_pose
  data: synthetic_court_4+tennis_court_detector_4
  batch_size: 8
  max_epochs: 20
  precision: bf16-mixed
  input_size: 256
metrics:
  kp_mean_dist: 15.014755
  kp_mean_distance_px: 15.014755
  kp_median_distance_px: 3.130819
  seg_miou: 0.858559
  line_dice: 0.461342
  pose_translation_l2_m: 0.871661
  pose_rotation_geodesic_deg: 2.555641
  pose_focal_relative_error: 0.077253
  pose_log_focal_abs_error: 0.078741
  pose_reprojection_mean_distance_px: 6.513071
  kp_pose_consistency_distance_px: 0.0
  invalid_depth_rate: 0.000454
  visible_point_count: 2203.0
repro:
  commit: 379b5a66dd5ce756a29c5dd3f398b448b9526694
  branch: main
  remote: git@github.com:Motoki0705/tennis-lab.git
  command: CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    .venv/bin/python -m src.tasks.court_detection.scripts.train_mixed training=lora
    data.batch_size=8 mixed.train_batch_counts.synthetic_court=4 mixed.train_batch_counts.tennis_court_detector=4
    run.test_after_fit=true run.output_dir=court_detection/mixed-source/dense-pose-lora-b8-e20-s42
artifacts:
  run_dir: knowledge/runs/run-court-mixed-pose-lora-b8-e20-s42
  predictions: knowledge/runs/run-court-mixed-pose-lora-b8-e20-s42/pred_test.npz
  log: .training_queue/logs/1789052279895515108_1993066_court_mixed_pose_lora_b8_e20_s42.log
  output_dir: outputs/court_detection/mixed-source/dense-pose-lora-b8-e20-s42/logs/version_0
  curves: knowledge/runs/run-court-mixed-pose-lora-b8-e20-s42/curves.png
  tb_logdir: outputs/court_detection/mixed-source/dense-pose-lora-b8-e20-s42/logs/version_0
parents: []
relations:
- to: run-i524-court-seg-baseline
  rel: compares
- to: run-i524-court-seg-ssl
  rel: compares
tags:
- court-detection
- dinov3
- lora
- mixed-source
- keypoint
- segmentation
- line-segmentation
- camera-pose
---

## 考察 / Findings

### 要約
DINOv3 ViT-B/16をLoRAで適応し、synthetic courtとTennisCourtDetectorを1バッチ4:4で混合して、keypoint・court segmentation・line segmentation・camera poseを同時学習した。独立testではkeypoint平均誤差`15.014755 px`、seg mIoU `0.858559`、line Dice `0.461342`、回転誤差`2.555641°`、並進誤差`0.871661 m`を得た。

### アーキテクチャ詳細
入力は`256×256`、encoderはDINOv3 ViT-B/16で、backbone本体を凍結しつつ`qkv`・`proj`・`fc1`・`fc2`へrank 8 / alpha 16のLoRAを挿入した。8層Transformer encoderとlarge DPT decoderに、keypoint・segmentation・line・camera poseの各headを接続した。各dense lossとposeのtranslation・rotation・focal lossをすべてweight 1で最適化し、pose-safe augmentation、bf16、batch size 8、20 epochを使用した。Git管理版の予測bundleは、再評価に必要な出力を保持しながら100MB未満へcompact化しており、詳細はrun directoryのREADMEに記録した。

### メトリクスの解釈
testではkeypoint中央値が`3.130819 px`である一方、平均は`15.014755 px`であり、一部の大きな外れ値が残る。seg mIoUは`0.858559`、line Diceは`0.461342`で、領域推定に比べ細線推定が難しい。camera poseは回転`2.555641°`、並進`0.871661 m`、focal相対誤差`0.077253`、再投影平均`6.513071 px`だった。収束曲線ではseg mIoU、line Dice、pose誤差は概ね改善したが、val lossは最初の`2.12387`から最後の`3.40530`へ単調に悪化しており、総loss基準では過学習が見られる。

### アーキテクチャ⇄メトリクスの因果考察
複数headの共同学習により、court領域とカメラ幾何の双方に有用な特徴を共有できた可能性がある。一方、仮説として、syntheticとrealで利用可能な教師信号や分布が異なるため、weight 1の単純和ではline・pose・keypoint間の勾配競合が起き、train loss低下とval loss悪化が併存したと考えられる。keypoint平均と中央値の乖離も、特定画角または可視性条件への汎化不足を示唆する。

### 既存実験との比較
`run-i524-court-seg-baseline`のbest val mIoU `0.517`および`run-i524-court-seg-ssl`のbest val mIoU `0.800`に対し、本runのtest seg mIoUは`0.858559`だった。ただし、本runはmixed-source、LoRA教師あり適応、multi-head、test splitという異なる契約であり、数値差をLoRAやmixed training単独の効果へ帰属することはできない。現行KP14 deploy runとも入力解像度・出力契約・評価splitが異なるため、置換根拠にはしない。

### 次に有効な実験
同一split・同一multi-head構成で、(1) backbone完全凍結、(2) LoRA、(3) full fine-tuningを比較する。さらにbest checkpointを各指標で再評価し、loss weightの自動平衡またはpose/denseの段階学習を試す。keypoint外れ値についてはsource・scene・可視点数別に分解し、平均誤差を支配する条件を特定する。
