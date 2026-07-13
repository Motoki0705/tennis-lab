---
id: run-i634-slcs-overfit-split-dino-down2
type: run
title: SLCS単一clip過学習（完全分離trunk＋DINO 2×2圧縮）
issue: 634
provider: codex
session: 019f55e6-8819-7e63-8481-72f9effc4079
date: '2026-07-13'
status: done
config:
  model: small
  dataset: tennis_clip/clip_000
  windows: 13
  overfit: true
  require_dino: true
  dino_patch_grid: 8x14
  dino_patch_downsample_factor: 2
  shared_layers: 0
  position_layers: 2
  rotation_layers: 2
  batch_size: 1
  epochs: 100
  learning_rate: 0.0003
metrics:
  player_position_error_m: 0.532336
  player_position_error_median_m: 0.414587
  player_angular_error_deg: 6.452454
  player_angular_error_median_deg: 3.14869
  player_position_accuracy_0.3m: 0.332248
  player_position_accuracy_0.5m: 0.603827
  player_position_accuracy_1.0m: 0.896173
  player_position_accuracy_2.0m: 0.992264
  player_angle_accuracy_10deg: 0.851384
  player_angle_accuracy_15deg: 0.892915
  player_angle_accuracy_30deg: 0.954397
  player_position_pred_b_m: 0.531294
  player_rotation_pred_b_deg: 11.743767
  player_position_conf_error_corr: 0.635849
  player_rotation_conf_error_corr: 0.615148
  ball_position_error_m: 1.69761
  ball_position_error_median_m: 1.228111
  ball_position_accuracy_0.3m: 0.063317
  ball_position_accuracy_0.5m: 0.156917
  ball_position_accuracy_1.0m: 0.394356
  ball_position_accuracy_2.0m: 0.685478
  ball_position_pred_b_m: 1.22972
  ball_position_conf_error_corr: 0.530482
repro:
  commit: 56445ad063a9a87e6453f4d095c084f8efb0b532
  branch: feat/issue-634-dino-compress-split-trunks
  remote: git@github.com:Motoki0705/tennis-lab.git
  command: PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True .venv/bin/python -m src.tasks.slcs.scripts.train
    model=small data.dataset_root=/home/kamimura/projects/tennis-lab/data/tennis_multivew/processed/tennis_clip/dataset
    data.split_file=/home/kamimura/projects/tennis-lab/data/tennis_multivew/processed/tennis_clip/dataset/splits.json
    data.overfit=true data.batch_size=1 data.num_workers=2 training.trainer.max_epochs=100
    training.trainer.check_val_every_n_epoch=10 training.trainer.log_every_n_steps=13
    training.early_stopping.enabled=false training.learning_rate=3e-4 training.warmup_steps=20
    training.checkpoint.save_top_k=1 data.require_dino=true model.num_shared_layers=0
    model.num_position_layers=2 model.num_rotation_layers=2 model.dino_patch_downsample_factor=2
    run.output_dir=outputs/slcs/i634_overfit_split_dino_down2
artifacts:
  run_dir: knowledge/runs/run-i634-slcs-overfit-split-dino-down2
  predictions: knowledge/runs/run-i634-slcs-overfit-split-dino-down2/pred_test.npz
  log: .training_queue/logs/1783905015933516084_1219230_i634_slcs_overfit_split_dino_down2.log
  output_dir: outputs/slcs/i634_overfit_split_dino_down2/logs/version_0
  curves: knowledge/runs/run-i634-slcs-overfit-split-dino-down2/curves.png
  tb_logdir: outputs/slcs/i634_overfit_split_dino_down2/logs/version_0
parents:
- run-i634-slcs-overfit-split-dino
- run-i634-slcs-overfit-dino-down2-shared
relations:
- to: run-i634-slcs-overfit-dino
  rel: compares
- to: run-i634-slcs-overfit-split-dino
  rel: compares
tags:
- slcs
- overfit
- single-clip
- dino
- patch-compression
- split-trunk
---

## 考察 / Findings

### 要約

完全分離trunkとDINO 2×2圧縮を併用した。共有/DINO baseline比ではyaw 7.762→6.452°、ball位置1.954→1.698mを改善したが、player位置は0.470→0.532mへ悪化した。非圧縮の完全分離runには3指標すべてで劣った。

### アーキテクチャ詳細

共有0層、position/rotation各2層とし、16×28 DINO特徴を8×14へbilinear downsampleしてから各branchのcross-attentionへ渡す。各frameのvisual key/valueは448から112 tokenへ1/4圧縮される。train/val/testは同一13 window、100 epochs、seed 42である。

### メトリクスの解釈

DINO baseline比でyawは16.9%、ball位置は13.1%改善し、player位置は13.2%悪化した。非圧縮完全分離比ではplayer位置0.512→0.532m、yaw5.751→6.452°、ball1.616→1.698mとすべて悪化した。ただしball 0.5m以内は13.8%→15.7%へ増え、閾値指標は一様ではない。
収束曲線ではvalidation lossが終盤まで安定して低下した。ball 0.3m accuracyには中盤以降の揺れがあるが、0.5/1.0m accuracyは継続的に上昇している。

### アーキテクチャ⇄メトリクスの因果考察

完全分離では各branchがvisual tokenからtask固有情報を抽出するため、共有trunk時より高解像度patchの価値が高くなった可能性がある（仮説）。圧縮で局所appearanceを落とすとrotation branchのyawとposition branchのballが共に悪化した。一方、共有DINO baselineには勝つため、分離効果自体は圧縮後も残る。

### 既存実験との比較

親 `run-i634-slcs-overfit-split-dino` は今回のyaw/ball最良で、本runは計算量削減との交換で精度を落とした。もう一方の親 `run-i634-slcs-overfit-dino-down2-shared` と比べると、yaw/ballは本runが良く、player位置は共有圧縮runが良い。

### 次に有効な実験

shared 1層 + task 1層と2×2圧縮を組み合わせ、player位置と計算量の折衷点を探す。実際のstep時間・GPU memoryも計測し、112 token化の速度利得を精度差と合わせて評価する。
