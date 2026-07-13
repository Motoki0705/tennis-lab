---
id: run-i634-slcs-overfit-split-no-dino
type: run
title: SLCS単一clip過学習（完全分離trunk・DINOなし）
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
  require_dino: false
  shared_layers: 0
  position_layers: 2
  rotation_layers: 2
  batch_size: 1
  epochs: 100
  learning_rate: 0.0003
metrics:
  player_position_error_m: 0.542678
  player_position_error_median_m: 0.413891
  player_angular_error_deg: 8.31879
  player_angular_error_median_deg: 3.539417
  player_position_accuracy_0.3m: 0.306596
  player_position_accuracy_0.5m: 0.611971
  player_position_accuracy_1.0m: 0.907573
  player_position_accuracy_2.0m: 0.973534
  player_angle_accuracy_10deg: 0.825733
  player_angle_accuracy_15deg: 0.879886
  player_angle_accuracy_30deg: 0.933225
  player_position_pred_b_m: 0.520738
  player_rotation_pred_b_deg: 12.581306
  player_position_conf_error_corr: 0.712602
  player_rotation_conf_error_corr: 0.44036
  ball_position_error_m: 1.826645
  ball_position_error_median_m: 1.326379
  ball_position_accuracy_0.3m: 0.062629
  ball_position_accuracy_0.5m: 0.139023
  ball_position_accuracy_1.0m: 0.354439
  ball_position_accuracy_2.0m: 0.660014
  ball_position_pred_b_m: 1.3189
  ball_position_conf_error_corr: 0.560462
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
    training.checkpoint.save_top_k=1 data.require_dino=false model.num_shared_layers=0
    model.num_position_layers=2 model.num_rotation_layers=2 run.output_dir=outputs/slcs/i634_overfit_split_no_dino
artifacts:
  run_dir: knowledge/runs/run-i634-slcs-overfit-split-no-dino
  predictions: knowledge/runs/run-i634-slcs-overfit-split-no-dino/pred_test.npz
  log: .training_queue/logs/1783905015815052209_1219186_i634_slcs_overfit_split_no_dino.log
  output_dir: outputs/slcs/i634_overfit_split_no_dino/logs/version_0
  curves: knowledge/runs/run-i634-slcs-overfit-split-no-dino/curves.png
  tb_logdir: outputs/slcs/i634_overfit_split_no_dino/logs/version_0
parents:
- run-i634-slcs-overfit-no-dino
relations:
- to: run-i634-slcs-overfit-no-dino
  rel: compares
tags:
- slcs
- overfit
- single-clip
- no-dino
- split-trunk
---

## 考察 / Findings

### 要約

共有trunkを0層、position/rotation trunkを各2層にして勾配経路を分離した。共有/DINOなしbaseline比でyawは9.849→8.319°、ball位置は2.105→1.827mへ改善したが、player位置は0.484→0.543mへ悪化した。

### アーキテクチャ詳細

観測embedding後にtoken gridを複製し、position branchがplayer/ball位置、rotation branchがplayer yawを担当する。各branchの実効深さはbaselineと同じ2層で、rotation lossはposition axial trunkへ流れない。共通embeddingは共有される。DINOは無効、train/val/testは同一13 window、100 epochs、seed 42である。

### メトリクスの解釈

baseline比でyawは15.5%、ball位置は13.2%改善した。一方player位置は12.1%悪化し、0.5m以内も61.9%→61.2%へ微減した。yaw 10°以内は73.5%→82.6%へ増加した。
収束曲線ではvalidation lossが100 epochsを通して低下し、ball accuracyも終盤まで上昇しているため、分離trunkの最適化崩壊ではない。

### アーキテクチャ⇄メトリクスの因果考察

rotation lossがposition trunkを直接更新しなくなったことでyaw自身の最適化も安定した可能性があり、負の転移仮説と整合する。一方、player位置はrotationとの共有表現から得ていた正の転移も失った可能性がある（仮説）。ball改善はposition branchがrotation勾配から隔離された結果と整合するが、parameter数増加も含むため因果は分離だけに限定できない。

### 既存実験との比較

親run `run-i634-slcs-overfit-no-dino` に対し、yawとballは改善、player位置は悪化した。PLCS知見の「分離すれば位置が改善」をSLCSへそのまま移した結果にはならず、SLCSではplayerとballを同じposition branchに置く構造差がある。

### 次に有効な実験

shared 1層 + task 1層の中間配分を試し、正の転移を残しつつrotation勾配支配を弱める。parameter数を揃えた幅縮小対照も必要である。
