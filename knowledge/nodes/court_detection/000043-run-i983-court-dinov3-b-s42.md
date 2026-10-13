---
id: run-i983-court-dinov3-b-s42
type: run
task: court_detection
sequence: 43
recorded_at: '2026-10-12'
title: 凍結DINOv3-Bと共通後段の比較学習 (#983)
issue: 983
provider: codex
session: 01a12538-67e1-7d22-a6ce-2d7add686d59
date: '2026-10-12'
status: done
config:
  backbone: dinov3_vitb16
  backbone_train_mode: frozen
  native_feature_dim: 768
  projection: four independent learned 1x1 768 -> 1024
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
  best_val_loss: 1.1920050382614136
  selected_epoch: 17
  synthetic_test:
    kp_mean_distance_px: 3.986131429672241
    kp_median_distance_px: 2.247342586517334
    pose_reprojection_mean_distance_px: 11.09050464630127
    pose_translation_l2_m: 0.572388768196106
    pose_rotation_geodesic_deg: 1.7561739683151245
    pose_focal_relative_error: 0.062227290123701096
    line_dice: 0.644749104976654
    seg_miou: 0.952130138874054
    semantic_line_miou: 0.45913931727409363
    invalid_depth_rate: 0.0
    visible_point_count: 7928.0
  real_val:
    kp_mean_distance_px: 1.3426450490951538
    kp_median_distance_px: 1.1999969482421875
    seg_miou: 0.9697523713111877
    line_dice: 0.7060995101928711
    semantic_line_miou: 0.48428186774253845
repro:
  commit: c4b12f923b243da6b502149aa6f3912c706045e7
  command: .venv/bin/python -m src.tasks.court_detection.scripts.train --config-name
    train_i983_b run.artifact_store.mode=rclone run.artifact_store.remote=gdrive run.artifact_store.remote_root=tennis_lab/outputs/court_detection/train/i983-dino-b-shared1024/s42
    run.artifact_store.sync_interval_seconds=60
artifacts:
  run_dir: knowledge/runs/run-i983-court-dinov3-b-s42
  config: knowledge/runs/run-i983-court-dinov3-b-s42/config.yaml
  environment: knowledge/runs/run-i983-court-dinov3-b-s42/environment.json
  input_verification: knowledge/runs/run-i983-court-dinov3-b-s42/input-verification.json
  condition_comparison: knowledge/runs/run-i983-court-dinov3-b-s42/condition-comparison.json
  running_receipt: knowledge/runs/run-i983-court-dinov3-b-s42/running-receipt.json
  training_start: knowledge/runs/run-i983-court-dinov3-b-s42/training-start.json
  pipeline_source: knowledge/runs/run-i983-court-dinov3-b-s42/run_pipeline.py
  output_dir: gdrive:tennis_lab/outputs/court_detection/train/i983-dino-b-shared1024/s42
  config_persistence: knowledge/runs/run-i983-court-dinov3-b-s42/config-persistence.json
  environment_comparison: knowledge/runs/run-i983-court-dinov3-b-s42/environment-comparison.json
  startup_status: knowledge/runs/run-i983-court-dinov3-b-s42/startup-status.json
  log: knowledge/runs/run-i983-court-dinov3-b-s42/startup.log
  training_process: knowledge/runs/run-i983-court-dinov3-b-s42/training-process.json
  tb_logdir: knowledge/runs/run-i983-court-dinov3-b-s42/tensorboard/version_0
  curves: knowledge/runs/run-i983-court-dinov3-b-s42/curves.png
  training_completion: knowledge/runs/run-i983-court-dinov3-b-s42/training-completion.json
  checkpoint_selection: knowledge/runs/run-i983-court-dinov3-b-s42/checkpoint-selection.json
  evaluation: knowledge/runs/run-i983-court-dinov3-b-s42/evaluation/evaluation.json
  evaluation_completion: knowledge/runs/run-i983-court-dinov3-b-s42/evaluation-completion.json
  evaluation_verification: knowledge/runs/run-i983-court-dinov3-b-s42/evaluation-verification.json
  evaluation_session_stop: knowledge/runs/run-i983-court-dinov3-b-s42/evaluation-session-stop.json
parents:
- run-i983-court-l512-profile-s42
relations:
- to: run-i983-court-dinov3-l-s42
  rel: compares
- to: run-i983-court-dinov3-s-s42
  rel: compares
- to: run-i983-court-dinov3-splus-s42
  rel: compares
papers: []
tags:
- issue-983
- colab-l4
- frozen-backbone
- shared-downstream
---

#983の4条件比較の最後の条件として、凍結DINOv3-Bの20epoch学習をColab L4で開始した。4段の768次元特徴を独立した学習可能な1×1層で1024次元へ揃え、Transformer/DPT/headsは先行L/S/S+と同じ容量と初期重みに固定する。データ、512長辺の前処理、batch8の4:4混合、seed42、loss、optimizer、学習予算を変えない。

入力はB00〜B03とTennisCourtDetector-v1の5 dataset、およびB pretrained weightである。session07のVM内へ取得し、転送検証とmanifest/index/画像shard/weightのSHA-256照合を完了した。実際に解決したconfigはS+と比較し、backbone名・weight path・run/Drive保存先以外の一致を確認した。科学的commit c4b12f923、Python3.11.16、PyTorch2.14.1、Lightning2.6.6、DINOv3 submodule、L4とdriverも先行S/S+と一致する。

最初の準備VMであるsession06は環境と入力を用意した後、本学習の開始前に消失した。準備・取消・保存の履歴はbundleのprovisioning-attempts/06に残す。VM消失原因は確定していない。session07では1つのNotebook kernel jobが環境構築、入力検証、学習を連続して実行する。job自体の開始は2026-10-12T10:20:13Z、学習childの開始は10:27:51Zであり、両時刻を区別する。準備VMでの検証はsession07自身の入力検証の代用にしていない。

checkpoint保存直後と60秒ごとのArtifactStore同期で学習成果をDriveへ保存する。workerは設定で宣言したrunner出力へ周期書込せず、学習終了後に残りを保存する。pipeline側の小さな準備・開始receiptは別のpersist treeに保存する。旧checkpointを自動間引きせず、20epoch終了後にmonitor=val/loss・mode=minのcallbackから最良保存checkpointを選ぶ。合成testは選択に使わず、終端モデルの自動testとは別に正式評価する。

学習は2026-10-12T20:57:36Zにreturncode 0・Drive最終保存で終了した。学習VM07はその後消失したため、再学習せずに新しいL4のsession08へ同じ入力と保存checkpointを取得した。lastのepoch19/global_step33160と、monitor=val/loss・mode=minのcallbackを確認し、最小値1.1920050382614136のepoch17を選択した。合成testの成績は選択に使っていない。選択checkpoint SHA-256はb8f36ed60b59d11083aaf1001292a7f4f88791efe59bc6ef087693645ec4fae9である。

正式評価は2026-10-12T22:30:12Zにreturncode 0・Drive保存で完了した。合成test886件のKP平均3.9861px、SEG mIoU0.9521、LINE Dice0.6447、semantic LINE mIoU0.4591、pose再投影平均11.0905pxだった。pose並進L2は0.5724m、回転geodesicは1.7562度、focal相対誤差は6.22%、可視7,928点の無効depthは0だった。実画像val2211件はKP平均1.342645px、SEG0.969752、LINE0.706100、semantic LINE0.484282である。無効化したKP-pose consistencyのraw値0を精度として扱わない。

評価JSON・固定36PNGをDriveから取得して検証し、L/S/S+とデータ版、画像ID順序、前処理、共通後段、評価条件の契約が一致することを確認した。学習・評価とも保存完了を確認し、評価session08を停止してsource差分を回収した。最終TensorBoardからtrain/val 24,448 scalarを重複なしで抽出し、終端モデルの自動test 50 scalarは正式best評価と別に保存した。

[4条件の比較](000044-group-i983-court-dinov3-shared1024-s42.md)では、Bが合成KP/SEG、LがLINE/semantic LINE/pose再投影で最良だった。実画像valの差は小さい。単一seedの観測であり、Lの中断再開、実画像valのcheckpoint選択兼用、Sのpose外れ値を残して解釈する。採用checkpointの変更や追加学習は行っていない。
