---
id: run-i983-court-dinov3-splus-s42
type: run
task: court_detection
sequence: 39
recorded_at: '2026-10-12'
title: 凍結DINOv3-S+と共通後段の比較学習 (#983)
issue: 983
provider: codex
date: '2026-10-12'
status: running
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
metrics: {}
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
session: 01a12538-67e1-7d22-a6ce-2d7add686d59
repro:
  commit: c4b12f923b243da6b502149aa6f3912c706045e7
  command: .venv/bin/python -m src.tasks.court_detection.scripts.train --config-name
    train_i983_splus run.artifact_store.mode=rclone run.artifact_store.remote=gdrive
    run.artifact_store.remote_root=tennis_lab/outputs/court_detection/train/i983-dino-splus-shared1024/s42
    run.artifact_store.sync_interval_seconds=60
---

#983の4条件比較の3条件目として、凍結DINOv3-S+の20epoch学習を新しいColab L4で開始した。backboneは384次元の4段特徴を出し、それぞれ独立した学習可能な1×1射影で共通1024次元のTransformer/DPTへ渡す。共通後段の初期重みをseed42で揃え、backboneと射影の初期化の乱数消費を隔離する。

Sの学習・正式評価・補足診断のDrive保存とsession停止を確認してから、新しいsession04を起動した。親preflightと同じB00〜B03・TennisCourtDetector-v1の5 dataset、およびS+ pretrained weightを取得し、転送検証とmanifest/index/画像shard/weightのSHA-256照合を完了した。入力準備jobのreturncode=0とDrive保存を確認してから本学習を提出した。

実行版はSの成功runと同じc4b12f923である。Python3.11.16/PyTorch2.14.1/Lightning2.6.6、DINOv3 submodule、L4とdriverもSと一致する。実際に解決したconfigをSの成功attemptと比較し、backbone名・weight path・run/Drive保存先だけが変わることを確認した。データ・split・前処理・loss・共通後段・学習条件は維持する。Driveへ保存された設定はVMの設定とbyte一致した。

Notebook kernel経由で実行し、checkpoint保存直後と60秒ごとのArtifactStore同期でDriveへ保存する。世代保持と最良checkpoint選択を分け、全20epoch後に共通validation loss最小の保存checkpointを評価する。合成testを選択に使わず、test_after_fitの終端モデル評価とも区別する。

L4上でsanity validationの2batchを通過し、Epoch 0の120/1658まで更新と有限のlossを確認した。これは起動の確認である。このノードは実行中の記録で、完了・最良checkpoint・精度は未確定。TensorBoard曲線と正式評価は完了後に保存する。Sでは1画像の深度無効な2点がpose平均再投影誤差を大きく引き上げたため、S+でも平均だけから分布を推定せず、無効depthや固定画像を含む記録を確認する。必要な追加診断は元の指標を変更せず別に記録する。

L/S/S+/Bの4条件が揃うまで一般的なサイズ間の優劣や採用変更は判断しない。実画像valは選択データを兼ね、Lにはcheckpoint再開時のRNG/sampler継続の制約がある。これらを残した上で、同一データの別split表と同一画像の定性比較に統合する。
