---
id: run-i983-court-dinov3-b-s42
type: run
task: court_detection
sequence: 43
recorded_at: '2026-10-12'
title: 凍結DINOv3-Bと共通後段の比較学習 (#983)
issue: 983
provider: codex
date: '2026-10-12'
status: running
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
metrics: {}
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
session: 01a12538-67e1-7d22-a6ce-2d7add686d59
repro:
  commit: c4b12f923b243da6b502149aa6f3912c706045e7
  command: .venv/bin/python -m src.tasks.court_detection.scripts.train --config-name
    train_i983_b run.artifact_store.mode=rclone run.artifact_store.remote=gdrive run.artifact_store.remote_root=tennis_lab/outputs/court_detection/train/i983-dino-b-shared1024/s42
    run.artifact_store.sync_interval_seconds=60
---

#983の4条件比較の最後の条件として、凍結DINOv3-Bの20epoch学習をColab L4で開始した。4段の768次元特徴を独立した学習可能な1×1層で1024次元へ揃え、Transformer/DPT/headsは先行L/S/S+と同じ容量と初期重みに固定する。データ、512長辺の前処理、batch8の4:4混合、seed42、loss、optimizer、学習予算を変えない。

入力はB00〜B03とTennisCourtDetector-v1の5 dataset、およびB pretrained weightである。session07のVM内へ取得し、転送検証とmanifest/index/画像shard/weightのSHA-256照合を完了した。実際に解決したconfigはS+と比較し、backbone名・weight path・run/Drive保存先以外の一致を確認した。科学的commit c4b12f923、Python3.11.16、PyTorch2.14.1、Lightning2.6.6、DINOv3 submodule、L4とdriverも先行S/S+と一致する。

最初の準備VMであるsession06は環境と入力を用意した後、本学習の開始前に消失した。準備・取消・保存の履歴はbundleのprovisioning-attempts/06に残す。VM消失原因は確定していない。session07では1つのNotebook kernel jobが環境構築、入力検証、学習を連続して実行する。job自体の開始は2026-10-12T10:20:13Z、学習childの開始は10:27:51Zであり、両時刻を区別する。準備VMでの検証はsession07自身の入力検証の代用にしていない。

checkpoint保存直後と60秒ごとのArtifactStore同期で学習成果をDriveへ保存する。workerは設定で宣言したrunner出力へ周期書込せず、学習終了後に残りを保存する。pipeline側の小さな準備・開始receiptは別のpersist treeに保存する。旧checkpointを自動間引きせず、20epoch終了後にmonitor=val/loss・mode=minのcallbackから最良保存checkpointを選ぶ。合成testは選択に使わず、終端モデルの自動testとは別に正式評価する。

10:35Zの起動確認でsanity validation 2/2を通過し、Epoch0の80/1658 batch、表示train/loss=45.40の有限値まで進行した。設定ファイルはVMとDriveでbyte一致を検証した。現在は学習中であり、正式な精度・最良checkpoint・学習曲線は未確定である。初期lossを精度比較へ用いない。完了後にL/S/S+と同じ順序付き評価IDと比較契約を照合し、合成testの5出力と実画像valの4出力を別表にまとめ、固定した同一画像を並べる。実画像valがcheckpoint選択に使われる点、Lの再開時RNG/sampler継続の制約、Sのpose投影外れ値も比較記録に残す。
