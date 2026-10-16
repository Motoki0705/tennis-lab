---
id: run-i983-court-dinov3-splus-s42
type: run
task: court_detection
sequence: 39
recorded_at: '2026-10-12'
title: 凍結DINOv3-S+と共通後段の比較学習 (#983)
issue: 983
provider: codex
session: 01a12538-67e1-7d22-a6ce-2d7add686d59
date: '2026-10-12'
status: done
config:
  backbone: dinov3_vits16plus
  backbone_train_mode: frozen
  native_feature_dim: 384
  projection: four independent learned 1x1 384 -> 1024
  transformer_dim: 1024
  transformer_depth: 8
  heads: 16
  ffn_dim: 2752
  dpt_channels: 512
  long_side: 512
  batch_size: 8
  mixed_batch_counts:
  - 4
  - 4
  seed: 42
  max_epochs: 20
metrics:
  best_val_loss: 1.232303261756897
  selected_epoch: 17
  synthetic_test:
    kp_mean_distance_px: 4.126668930053711
    kp_median_distance_px: 2.270092248916626
    pose_reprojection_mean_distance_px: 14.25533390045166
    pose_translation_l2_m: 0.680153489112854
    pose_rotation_geodesic_deg: 2.3257951736450195
    pose_focal_relative_error: 0.06442432850599289
    line_dice: 0.6443001627922058
    seg_miou: 0.9494113922119141
    semantic_line_miou: 0.45498406887054443
    invalid_depth_rate: 0.0
    visible_point_count: 7928.0
  real_val:
    kp_mean_distance_px: 1.34260094165802
    kp_median_distance_px: 1.1999969482421875
    seg_miou: 0.9695714712142944
    line_dice: 0.7055452466011047
    semantic_line_miou: 0.4845685362815857
repro:
  commit: c4b12f923b243da6b502149aa6f3912c706045e7
  command: .venv/bin/python -m src.tasks.court_detection.scripts.train --config-name
    train_i983_splus run.artifact_store.mode=rclone run.artifact_store.remote=gdrive
    run.artifact_store.remote_root=tennis_lab/outputs/court_detection/train/i983-dino-splus-shared1024/s42
    run.artifact_store.sync_interval_seconds=60
artifacts:
  run_dir: knowledge/runs/run-i983-court-dinov3-splus-s42
  config: knowledge/runs/run-i983-court-dinov3-splus-s42/config.yaml
  environment: knowledge/runs/run-i983-court-dinov3-splus-s42/environment.json
  input_verification: knowledge/runs/run-i983-court-dinov3-splus-s42/input-verification.json
  condition_comparison: knowledge/runs/run-i983-court-dinov3-splus-s42/condition-comparison.json
  running_receipt: knowledge/runs/run-i983-court-dinov3-splus-s42/running-receipt.json
  startup_status: knowledge/runs/run-i983-court-dinov3-splus-s42/startup-status.json
  output_dir: gdrive:tennis_lab/outputs/court_detection/train/i983-dino-splus-shared1024/s42
  log: knowledge/runs/run-i983-court-dinov3-splus-s42/startup.log
  training_completion: knowledge/runs/run-i983-court-dinov3-splus-s42/training-completion.json
  checkpoint_selection: knowledge/runs/run-i983-court-dinov3-splus-s42/checkpoint-selection.json
  evaluation: knowledge/runs/run-i983-court-dinov3-splus-s42/evaluation/evaluation.json
  evaluation_verification: knowledge/runs/run-i983-court-dinov3-splus-s42/evaluation-verification.json
  evaluation_completion: knowledge/runs/run-i983-court-dinov3-splus-s42/evaluation-completion.json
  tb_logdir: knowledge/runs/run-i983-court-dinov3-splus-s42/tensorboard/version_0
  curves: knowledge/runs/run-i983-court-dinov3-splus-s42/curves.png
parents:
- run-i983-court-l512-profile-s42
relations:
- to: run-i983-court-dinov3-s-s42
  rel: compares
- to: run-i983-court-dinov3-l-s42
  rel: compares
papers: []
tags:
- issue-983
- colab-l4
- frozen-backbone
- shared-downstream
---

#983の4条件比較の3条件目として、凍結DINOv3-S+の20epoch学習を新しいColab L4で開始した。backboneは384次元の4段特徴を出し、それぞれ独立した学習可能な1×1射影で共通1024次元のTransformer/DPTへ渡す。共通後段の初期重みをseed42で揃え、backboneと射影の初期化の乱数消費を隔離する。

Sの学習・正式評価・補足診断のDrive保存とsession停止を確認してから、新しいsession04を起動した。親preflightと同じB00〜B03・TennisCourtDetector-v1の5 dataset、およびS+ pretrained weightを取得し、転送検証とmanifest/index/画像shard/weightのSHA-256照合を完了した。入力準備jobのreturncode=0とDrive保存を確認してから本学習を提出した。

実行版はSの成功runと同じc4b12f923である。Python3.11.16/PyTorch2.14.1/Lightning2.6.6、DINOv3 submodule、L4とdriverもSと一致する。実際に解決したconfigをSの成功attemptと比較し、backbone名・weight path・run/Drive保存先だけが変わることを確認した。データ・split・前処理・loss・共通後段・学習条件は維持する。Driveへ保存された設定はVMの設定とbyte一致した。

Notebook kernel経由で実行し、checkpoint保存直後と60秒ごとのArtifactStore同期でDriveへ保存する。世代保持と最良checkpoint選択を分け、全20epoch後に共通validation loss最小の保存checkpointを評価する。合成testを選択に使わず、test_after_fitの終端モデル評価とも区別する。

2026-10-12T02:54:39Zに20epochと終端test・Drive最終保存をreturncode=0で完了した。last checkpointはepoch19/global_step33160で、monitor=val/loss・mode=minのcallbackからepoch17（val/loss=1.232303261756897）を選択した。checkpoint SHA-256は075ecaf156c38174cc0c66856c3ba02043a0ad094f0c38e4643dfb00e28a58d8である。学習後にsession04が消失したため、保存済みlastと選択checkpointをsession05へ検証付きで取得した。再学習は行わず、選択checkpointの正式評価を2026-10-12T08:31:22Zにreturncode=0・Drive保存で完了した。評価JSONと36PNGを取得・検証し、L/Sと比較契約が一致することを確認した。合成test886件のKP平均4.1267px、SEG mIoU0.9494、LINE Dice0.6443、semantic LINE mIoU0.4550、pose再投影平均14.2553pxだった。pose並進L2は0.6802m、回転2.3258度、focal相対誤差6.44%、可視7,928点の無効depthは0だった。実画像val2211件はKP平均1.3426px、SEG0.9696、LINE0.7055、semantic LINE0.4846である。無効化されているconsistency距離のraw値0は精度として報告しない。最終flush後のTensorBoardイベントからtrain/val scalarのみを抽出し、終端モデルの自動testは別JSONに保存して正式評価と区別する。Sでは1画像の深度無効な2点がpose平均再投影誤差を大きく引き上げたため、S+でも平均だけから分布を推定せず、無効depthや固定画像を含む記録を確認する。必要な追加診断は元の指標を変更せず別に記録する。

L/S/S+/Bの4条件が揃うまで一般的なサイズ間の優劣や採用変更は判断しない。実画像valは選択データを兼ね、Lにはcheckpoint再開時のRNG/sampler継続の制約がある。これらを残した上で、同一データの別split表と同一画像の定性比較に統合する。

学習速度については[batch8/16の短時間診断](000042-group-i983-court-throughput-s42.md)に切り分けを記録した。入力はVM内にあり、後段の計算が大きい。追加計測は別runで行い、このS+の学習条件やcheckpointを変更していない。
