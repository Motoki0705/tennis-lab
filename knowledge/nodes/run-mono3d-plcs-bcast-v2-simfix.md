---
id: run-mono3d-plcs-bcast-v2-simfix
type: run
title: mono3d_plcs_bcast_v2_simfix
issue: 593
provider: claude
session: dd9d56e0-5f2d-4d9d-a8c7-c80704a806b3
date: '2026-07-05'
status: done
config:
  model: multiview_axial_split
  loss: canonical_rot
  data: chunked_singleview_sequence
metrics:
  position_error_m: 0.344998
  position_error_std_m: 0.361734
  position_error_median_m: 0.262619
  angular_error_deg: 21.085461
  angular_error_std_deg: 25.587015
  angular_error_median_deg: 13.365777
  x_error_m: 0.129537
  y_error_m: 0.285046
  z_error_m: 0.043938
  position_accuracy: 0.85553
  angle_accuracy: 0.543948
  position_accuracy_0.5m: 0.85553
  position_accuracy_1m: 0.958723
  position_accuracy_2m: 0.986403
  angle_accuracy_10deg: 0.385213
  angle_accuracy_15deg: 0.543948
  angle_accuracy_30deg: 0.810853
repro:
  commit: 0d18de3aca52485905b364a12647e9066c9b95ac
  branch: main
  remote: git@github.com:Motoki0705/tennis-lab.git
  command: MPLBACKEND=Agg PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True .venv/bin/python
    -m src.tasks.plcs.scripts.train model=multiview_axial_split model.num_layers=0
    model.num_task_layers=6 data=chunked_singleview_sequence data.scene_dir=data/plcs_broadcast
    data.chunk.chunks_dir=data/plcs_broadcast/chunks data.batch_size=8 data.seq_len_range=[64,256]
    data.num_court_kp=14 data.num_workers=2 data.chunk.generation_workers=6 data.chunk.epochs_per_chunk=30
    loss=canonical_rot loss.position_weight=8.0 loss.canonical_pose_weight=0.0 loss.joint_angle_weight=0.0
    loss.torsion_angle_weight=0.0 loss.torso_twist_weight=0.0 loss.bone_length_weight=0.0
    training.trainer.accumulate_grad_batches=1 training.trainer.max_epochs=200 training.trainer.check_val_every_n_epoch=5
    training.qualitative_logging.enabled=false training.early_stopping.enabled=false
    camera=broadcast run.gpus=1
artifacts:
  run_dir: knowledge/runs/run-mono3d-plcs-bcast-v2-simfix
  predictions: knowledge/runs/run-mono3d-plcs-bcast-v2-simfix/pred_test.npz
  log: .training_queue/logs/1783252014501234340_346610_mono3d_plcs_bcast_v2_simfix.log
  output_dir: outputs/plcs/plcs_multiview_axial_split/logs/version_22
  checkpoint: ckpt/plcs/run-mono3d-plcs-bcast-v2-simfix-epoch199.ckpt
  curves: knowledge/runs/run-mono3d-plcs-bcast-v2-simfix/curves.png
  tb_logdir: outputs/plcs/plcs_multiview_axial_split/logs/version_22
parents:
- run-mono3d-plcs-bcast
relations:
- to: run-mono3d-plcs-bcast
  rel: supersedes
tags:
- monocular
- broadcast
- sim-to-real
---

## 考察 / Findings

### 要約
射影鏡像バグ修正 + 広域カメラランダム化 + 実測校正augmentation後の単眼PLCS再学習 (PR #603)。実映像で崩壊した [[run-mono3d-plcs-bcast]] を置き換える。カメラ分布が大幅に広がった(setback 15-40m / height 5-15m / 幅比 0.5-0.9)にもかかわらず position 0.345m / 回転 median 13.4° と、旧run(狭い固定プリセット、0.313m / median 16.2°)から回転は改善・位置は同水準。

### アーキテクチャ詳細
モデル/損失/データ構成は [[run-mono3d-plcs-bcast]] と同一 (multiview_axial_split H=0/S=6, canonical_rot pos_weight=8 aux=0, chunked_singleview_sequence C=1, court_kp=14, seq_len [64,256])。差分は環境側: (1) `make_look_at_camera` の左右鏡像バグ修正 (OpenCV規約化)、(2) camera=broadcast プリセットのレンジ化 (HFOVはコート見かけ幅比から閉形式で解決)、(3) augmentation実測校正 (human noise σ 0.0001→0.003, burst 4→12 等、tennis_clip ViTPose実測準拠)、(4) 高速化: epochs_per_chunk 10→30, val 5epochごと, qualitative off (200ep が約75分で完走、旧runの数時間から短縮)。

### メトリクスの解釈
test/pos_error 0.345m (median 0.263m)、angular 21.1° (median 13.4°、30deg内 81%)。z_error 0.044m。x_error 0.130m < y_error 0.285m は単眼の奥行き(Y)不確実性で想定どおり。旧runよりタスクは難化(カメラジオメトリのばらつき増)しているため、同水準の位置誤差は実質改善。角度median 16.2°→13.4° はノイズ強化による正則化効果の可能性(仮説)。

### アーキテクチャ⇄メトリクスの因果考察
崩壊の主因はモデルではなく学習環境の鏡像バグ (実court KPが合成分布から15-30σ外) + カメラOOD (4-12σ)。修正後、実クリップの court KP は合成分布内 max|z|≈0.7 に収まっており、in-dist metrics が実映像性能に転移する前提条件が初めて成立した。角度誤差std 25.6°はflip由来の裾が残存(単眼の既知課題)。

### 既存実験との比較
[[run-mono3d-plcs-bcast]] (0.313m/29.7°) 比: 平均角度 29.7°→21.1°、median 16.2°→13.4°。位置 0.313→0.345m はカメラ分布拡大分のコスト。実映像評価は本runが初の有効測定になる(旧runは鏡像世界での学習のため実映像では無効)。

### 次に有効な実験
(1) 本ckptで tennis_clip パイプライン再実行 (窓分割推論 window=256/overlap=64 + vis閾値0.35込み) — 実映像での定性検証が最優先。(2) 回転flip対策 (#593 のidea list: heading-consistency loss / multi-hypothesis)。(3) seq_len_range を [64,384] に拡大して窓境界依存を低減。
