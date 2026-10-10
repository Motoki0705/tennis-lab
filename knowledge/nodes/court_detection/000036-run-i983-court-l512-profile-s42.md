---
id: run-i983-court-l512-profile-s42
type: run
task: court_detection
sequence: 36
recorded_at: '2026-10-10'
title: 共通1024次元のViT-LをL4で512入力確認 (#983)
issue: 983
provider: codex
status: done
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
metrics:
  steps: 3
  steady_step_median_seconds: 0.5903714275000311
  peak_allocated_bytes: 10177981952
  peak_reserved_bytes: 10986979328
  downstream_trainable_parameters: 128635436
artifacts:
  run_dir: knowledge/runs/run-i983-court-l512-profile-s42
  config: knowledge/runs/run-i983-court-l512-profile-s42/config.yaml
  output_dir: gdrive:tennis_lab/outputs/court_detection/profile/i983-l512/attempt-01
  profile: knowledge/runs/run-i983-court-l512-profile-s42/profile.json
  log: knowledge/runs/run-i983-court-l512-profile-s42/run.log
parents: []
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
  command: .venv/bin/python -m src.tasks.court_detection.scripts.profile_ablation
    --output outputs/court_detection/profile/i983-l512/attempt-01 --steps 3
---

L4上で実データの混合batch（合成4＋実画像4）を使い、凍結ViT-Lと共通1024次元のTransformer＋DPTについて3回のforward・backward・AdamW更新を完了した。lossとclip前gradient normはすべて有限で、各入力は[8,3,288,512]、pose教師は4件だった。source commit、解決済み設定、24枚のsample ID・K、メモリ値、完了とDrive保存のreceiptをbundleへ保存した。

初回125.19秒はコンパイルを含む。後続2stepの中央値は0.59037秒で、peak allocatedは約10.18 GB、reservedは約10.99 GBだった。これは短い計算区間の診断で、validation・checkpoint転送・全学習時間の見積もりそのものではない。fullgraph=falseの既定設定でgraph breakとcomplex operatorの警告があり、ログに残した。TensorBoard付き学習ではないため、収束曲線は作らない。

この観測から長辺512・batch8を維持して本学習へ進む判断をした。backboneは303,154,176 parameterすべて凍結し、学習する共通後段は128,635,436 parameter。ViT-Lの入力射影はIdentityである。S/S+/Bでは各段の学習可能な射影が加わるため、後段容量と総trainable数は区別する。精度の優劣・収束・production採用は未評価で、過去のLoRA B結果を今回の基準へ流用しない。

全入力のSHA-256をローカル・Drive転送・Colab配置で照合した。共通snapshotとsplit順序はinputs.json / splits.json、512幾何の実データ確認はgeometry512.json、全画像の寸法はimage-shapes512.jsonを参照する。合成V3 test886枚と実画像val2211枚を別表で評価し、同じIDの定性画像を比較することを次工程とする。
