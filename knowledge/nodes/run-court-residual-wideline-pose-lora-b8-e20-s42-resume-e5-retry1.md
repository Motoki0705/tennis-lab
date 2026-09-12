---
id: run-court-residual-wideline-pose-lora-b8-e20-s42-resume-e5-retry1
type: run
title: Residual Dense Head・7.5 cm line・pose監視のCourt LoRA学習
issue: 867
provider: codex
session: 01a08a57-b460-7cd0-8dc1-88c6a3a5a3fb
date: '2026-09-11'
status: done
config:
  model: court_hierarchical_dinov3_vitb16_lora_r8_residual_dense_heads
  loss: dense_pose_pose_monitored_best
  data: synthetic_court_4+tennis_court_detector_4_line_75mm_baseline_150mm
  batch_size: 8
  max_epochs: 20
  precision: bf16-mixed
  input_size: 256
  resume_epoch: 5
metrics:
  kp_mean_dist: 3.436689
  kp_mean_distance_px: 3.436689
  kp_median_distance_px: 1.727726
  seg_miou: 0.802288
  line_dice: 0.541651
  pose_translation_l2_m: 0.904958
  pose_rotation_geodesic_deg: 3.166693
  pose_focal_relative_error: 0.075874
  pose_log_focal_abs_error: 0.077195
  pose_reprojection_mean_distance_px: 15.821514
  kp_pose_consistency_distance_px: 0.0
  invalid_depth_rate: 0.002724
  visible_point_count: 2203.0
repro:
  commit: 0fcdd13237dbb4b14aefc5c6102c4fd6d870ee00
  branch: codex/court-residual-head-target-preview
  remote: git@github.com:Motoki0705/tennis-lab.git
  command: CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    PYTHONPATH=/home/kamimura/projects/tennis-lab-worktrees/court-residual-head-target-preview
    /home/kamimura/projects/tennis-lab/.venv/bin/python -m src.tasks.court_detection.scripts.train_mixed
    paths.data_root=/home/kamimura/projects/tennis-lab/data paths.output_root=/home/kamimura/projects/tennis-lab/outputs
    paths.checkpoint_root=/home/kamimura/projects/tennis-lab/outputs paths.external_asset_root=/home/kamimura/projects/tennis-lab/third_party
    paths.cache_root=/home/kamimura/projects/tennis-lab/.cache training=pose_lora
    data.batch_size=8 data.num_workers=2 mixed.train_batch_counts.synthetic_court=4
    mixed.train_batch_counts.tennis_court_detector=4 run.test_after_fit=true run.resume=court_detection/mixed-source/residual-head-wide-line-lora-b8-e20-s42-retry1/logs/version_0/checkpoints/last.ckpt
    run.output_dir=court_detection/mixed-source/residual-head-wide-line-pose-lora-b8-e20-s42-resume-e5-retry1
artifacts:
  run_dir: knowledge/runs/run-court-residual-wideline-pose-lora-b8-e20-s42-resume-e5-retry1
  predictions: knowledge/runs/run-court-residual-wideline-pose-lora-b8-e20-s42-resume-e5-retry1/pred_test.npz
  output_dir: /home/kamimura/projects/tennis-lab/outputs/court_detection/mixed-source/residual-head-wide-line-pose-lora-b8-e20-s42-resume-e5-retry1/logs/version_0
  log: .training_queue/logs/1789131317875449519_3473120_court_residual_wideline_pose_lora_b8_e20_s42_resume_e5_retry1.log
  curves: knowledge/runs/run-court-residual-wideline-pose-lora-b8-e20-s42-resume-e5-retry1/curves.png
  tb_logdir: outputs/court_detection/mixed-source/residual-head-wide-line-pose-lora-b8-e20-s42-resume-e5-retry1/logs/version_0
  inference_real_kp: assets/court_detection/residual-head-wide-line-pose-best-e17/inference/real-kp.gif
  inference_real_seg: assets/court_detection/residual-head-wide-line-pose-best-e17/inference/real-seg.gif
  inference_real_line: assets/court_detection/residual-head-wide-line-pose-best-e17/inference/real-line.gif
  inference_synthetic_kp: assets/court_detection/residual-head-wide-line-pose-best-e17/inference/synthetic-kp.gif
  inference_synthetic_seg: assets/court_detection/residual-head-wide-line-pose-best-e17/inference/synthetic-seg.gif
  inference_synthetic_line: assets/court_detection/residual-head-wide-line-pose-best-e17/inference/synthetic-line.gif
parents:
- run-court-mixed-pose-lora-b8-e20-s42
relations: []
tags:
- court-detection
- dinov3
- lora
- mixed-source
- residual-dense-head
- keypoint
- segmentation
- line-segmentation
- 75mm-line
- camera-pose
- pose-monitor
---

## 考察 / Findings

### 要約
DINOv3 ViT-B/16 LoRAのmixed-source学習を、タスク別Residual Dense Headと通常線7.5 cm・baseline 15 cmの教師マスクへ変更した。最終epoch 19の独立testではKP平均誤差`3.436689 px`、seg mIoU `0.802288`、line Dice `0.541651`、並進誤差`0.904958 m`、回転誤差`3.166693°`だった。pose監視bestはepoch 17の`val/loss_direct_pose=0.0520`、真の`last.ckpt`はepoch 19 / global step 33160であり、bestとlatestの保存責務が分離されたことも確認した。

### アーキテクチャ詳細
入力は`256×256`、encoderはDINOv3 ViT-B/16で、backbone本体を凍結しつつ`qkv`・`proj`・`fc1`・`fc2`へrank 8 / alpha 16のLoRAを挿入した。共有DPT出力の後段は、KP・SEG・LINEごとに`512→256` projection、depthwise/pointwise residual block 2層、task別出力projectionを持つResidual Dense Headへ拡張した。synthetic courtとTennisCourtDetectorを1バッチ4:4で混合し、各dense lossとpose translation・rotation・focal lossをweight 1で最適化した。line教師は通常線7.5 cm、baseline 15 cmのversioned schemaを使用した。旧runのepoch 5 / global step 9948から再開し、bestは`val/loss_direct_pose`、latestはmonitor非依存の毎epoch保存とした。

### メトリクスの解釈
最終epoch 19のtestではKP中央値`1.727726 px`に対し平均`3.436689 px`で、外れ値は残るものの中央値と平均の乖離はbaselineより小さい。seg mIoUは`0.802288`、line Diceは`0.541651`だった。camera poseは並進`0.904958 m`、回転`3.166693°`、focal相対誤差`0.075874`、再投影平均`15.821514 px`、invalid depth rate `0.002724`である。ここに記録したtest metricsはfit後のin-memory最終epoch 19モデルによるもので、pose監視best epoch 17の値ではない。epoch 17 checkpointを使った実画像・合成画像のKP / SEG / LINE可視化は、Residual Dense Headのcheckpoint復元時にmissing/unexpected keyを出さず完走し、全headで構造化された予測を確認した。

### アーキテクチャ⇄メトリクスの因果考察
KP平均誤差とline Diceが同時に改善したことは、タスク別Residual Dense Headが異なる出力表現へ専用容量を与えた仮説と整合する。line幅拡張もpositive pixelを増やし、line学習を安定させた可能性がある。ただし、Residual Headとline schemaを同時変更しているため寄与は分離できない。seg mIoUとposeの並進・回転・再投影は悪化しており、増えたhead容量だけで共有特徴の勾配競合が解消したとはいえない。特にpose direct lossをbest選択に使っても、幾何評価全体の改善は保証されない。

### 既存実験との比較
親run `run-court-mixed-pose-lora-b8-e20-s42`に対し、KP平均誤差は`15.014755→3.436689 px`、中央値は`3.130819→1.727726 px`、line Diceは`0.461342→0.541651`へ改善した。一方、seg mIoUは`0.858559→0.802288`、並進誤差は`0.871661→0.904958 m`、回転誤差は`2.555641→3.166693°`、再投影平均は`6.513071→15.821514 px`へ悪化した。focal相対誤差は`0.077253→0.075874`と僅かに改善した。教師line幅、head構造、checkpoint選択基準が同時に異なるため、単一要因の効果とは断定しない。

### 次に有効な実験
同一7.5 cm line schemaのままlinear headとResidual Dense Headを比較し、head容量の寄与を分離する。そのうえで、pose-best選択を`loss_direct_pose`だけでなく再投影誤差または複合幾何指標と比較する。SEG低下とpose再投影悪化に対しては、task別勾配normを記録し、loss weightの平衡化またはheadごとの段階学習を検討する。
