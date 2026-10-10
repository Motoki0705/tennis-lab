---
id: run-i983-court-dinov3-l-s42
type: run
task: court_detection
sequence: 37
recorded_at: '2026-10-10'
title: 凍結DINOv3-Lと共通後段の比較学習 (#983)
issue: 983
provider: codex
status: running
config:
  backbone: dinov3_vitl16
  backbone_train_mode: frozen
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
  run_dir: knowledge/runs/run-i983-court-dinov3-l-s42
  config: knowledge/runs/run-i983-court-dinov3-l-s42/config.yaml
  output_dir: gdrive:tennis_lab/outputs/court_detection/train/i983-dino-l-shared1024/s42
parents:
- run-i983-court-l512-profile-s42
relations: []
papers: []
tags:
- issue-983
- colab-l4
- frozen-backbone
- shared-downstream
date: '2026-10-10'
session: 01a12538-67e1-7d22-a6ce-2d7add686d59
repro:
  commit: 98b737829ba8baae8b823c7b7728cf0608f5073f
  command: .venv/bin/python -m src.tasks.court_detection.scripts.train --config-name
    train_i983_l run.artifact_store.mode=rclone run.artifact_store.remote=gdrive run.artifact_store.remote_root=tennis_lab/outputs/court_detection/train/i983-dino-l-shared1024/s42
    run.artifact_store.sync_interval_seconds=60
---

#983の4条件比較のうちViT-Lを、Colab L4で20epochの本学習として開始した。凍結backbone、長辺512、混合4:4、batch8、seed42、共通1024次元・8層・16 heads・FFN2752・DPT512を固定する。初期化時のbackboneと射影層の乱数消費を隔離し、他サイズでも共通後段の初期重みを揃える。

このノードは学習中の記録であり、完了・最良checkpoint・精度をまだ報告しない。TensorBoardはColab上で生成中で、完了後に曲線と評価証跡を収集する。checkpointは各epoch終了後にDriveへ保存し、validation lossによる選択と保存世代の保持を分ける。save_top_k=-1を使い、学習frameworkからDriveへの自動間引きを行わない。

再現commandと実行版はrun.json / request.json、学習設定はconfig.yaml、共通入力snapshotは親preflightのbundleが正本である。モデル更新を伴う3-step preflightから重みを再利用せず、DINO pretrained checkpointとseed42から新規開始した。合成V3 testと実画像valを別集計し、S/S+/Bの同条件runが揃うまでDINOv3サイズ間の結論を出さない。
