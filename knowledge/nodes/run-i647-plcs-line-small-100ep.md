---
id: run-i647-plcs-line-small-100ep
type: run
title: i647_plcs_line_small_100ep
issue: 647
provider: codex
session: 019f6013-0a29-7183-87ae-e4c221387139
date: '2026-07-14'
status: done
config:
  model: multiview_axial_line_small
  loss: canonical_rot
  data: chunked_multiview_sequence_line_bs8
metrics:
  position_error_m: 7.235765
  position_error_std_m: 3.567188
  position_error_median_m: 7.327524
  angular_error_deg: 99.070404
  angular_error_std_deg: 49.896454
  angular_error_median_deg: 103.048508
  x_error_m: 2.576758
  y_error_m: 6.265839
  z_error_m: 0.223974
  position_accuracy: 0.003449
  angle_accuracy: 0.033912
  position_accuracy_0.5m: 0.003449
  position_accuracy_1m: 0.019543
  position_accuracy_2m: 0.074531
  angle_accuracy_10deg: 0.020501
  angle_accuracy_15deg: 0.033912
  angle_accuracy_30deg: 0.108315
repro:
  commit: 6fe0a82cf1c67dc9b0c6e5b7599e92f75ad39292
  branch: feat/issue-647-court-line-token
  remote: git@github.com:Motoki0705/tennis-lab.git
  command: PYTHONPATH=. PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True .venv/bin/python
    -m src.tasks.plcs.scripts.train model=multiview_axial_line_small data=chunked_multiview_sequence_line_bs8
    data.scene_dir=data/plcs_broadcast data.chunk.chunks_dir=data/plcs_broadcast/chunks
    camera=broadcast data.batch_size=8 data.num_workers=4 data.num_views_range=[1,1]
    data.seq_len_range=[64,256] data.chunk.generation_workers=6 data.chunk.epochs_per_chunk=30
    loss=canonical_rot loss.position_weight=8.0 loss.canonical_pose_weight=0.0 loss.joint_angle_weight=0.0
    loss.torsion_angle_weight=0.0 loss.torso_twist_weight=0.0 loss.bone_length_weight=0.0
    training.trainer.max_epochs=100 training.trainer.check_val_every_n_epoch=5 training.qualitative_logging.enabled=false
    training.early_stopping.enabled=false run.output_dir=outputs/plcs/issue647_line_small_100ep
    run.gpus=1
artifacts:
  run_dir: knowledge/runs/run-i647-plcs-line-small-100ep
  predictions: knowledge/runs/run-i647-plcs-line-small-100ep/pred_test.npz
  log: .training_queue/logs/1784026204479700992_1930930_i647_plcs_line_small_100ep.log
  output_dir: outputs/plcs/issue647_line_small_100ep/logs/version_0
  curves: knowledge/runs/run-i647-plcs-line-small-100ep/curves.png
  tb_logdir: outputs/plcs/issue647_line_small_100ep/logs/version_0
parents: []
relations:
- to: run-i647-plcs-line-moderate-v2-100ep
  rel: compares
- to: run-i647-plcs-kp-small-100ep
  rel: compares
tags:
- plcs
- court-line
- ransac
- augmentation
- negative-result
---

## 考察 / Findings

### 要約

RANSAC court-line tokenと強いline-map増強を用いたPLCSの100 epoch baseline。train/validation/testは完走したが、test位置誤差 `7.236m`、角度誤差 `99.07°` で、世界座標位置・回転を学習できなかった。

### アーキテクチャと増強

`multiview_axial_line_small`（13.1M params）、broadcast single-view、`[court token, player token]`の2-token camera-axis入力。部分欠損0.8、遮蔽0.7、false-positive 0.5、blur 0.3、morphology 0.4、far dropout 0.3、near-only 0.15を独立に適用し、各windowでRANSAC線分を再抽出した。

### メトリクスの解釈

best validationでも位置 `6.692m`（epoch 9付近）、角度 `77.11°`（epoch 54付近）に留まり、後半で改善しなかった。testの中央値も位置 `7.328m`、角度 `103.05°` で、少数の外れ値だけではなく分布全体が未学習である。

### 因果考察

forward依存性テストによりcourt linesを変えると予測が変わるため、court tokenの未接続ではない。独立な劣化が1 mapに平均3種類超重なること、欠損・false-positiveでdeterministic sort後のflatten位置が大きく入れ替わることが、MVPのflatten embeddingには過剰だった可能性が高い。

### 次の実験

全劣化タイプを残しつつ重畳確率と最大個数を約半分にした [[run-i647-plcs-line-moderate-v2-100ep]] と比較する。同一split/modelのKP対照は [[run-i647-plcs-kp-small-100ep]]。moderateでも未学習なら、line-wise set encoderまたは段階的augmentationを次のablation候補とする。
