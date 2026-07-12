---
id: run-mono3d-blcs-bcast-v3-simfix
type: run
title: mono3d_blcs_bcast_v3_simfix
issue: 593
provider: claude
session: dd9d56e0-5f2d-4d9d-a8c7-c80704a806b3
date: '2026-07-05'
status: done
config:
  model: multiview_axial_base
  loss: trajectory_position + reprojection(0.1) + axis_weights[1,4,1]
  data: chunked_multiview_sequence C=1
metrics:
  mean_position_error_m: 1.845229
  mean_x_error_m: 0.510848
  mean_y_error_m: 1.615349
  mean_z_error_m: 0.369423
  mean_endpoint_error_m: 3.407946
  position_accuracy_0_3m: 0.155612
  position_accuracy_0_6m: 0.392604
  position_accuracy_1_2m: 0.64788
  endpoint_accuracy_0_5m: 0.13
  endpoint_accuracy_1m: 0.27
repro:
  commit: 16a21f752374d48ea2c1e6d0ae6d5556b3d87593
  branch: fix/issue-593-sim-to-real-projection
  remote: git@github.com:Motoki0705/tennis-lab.git
  command: MPLBACKEND=Agg PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True .venv/bin/python
    -m src.tasks.blcs.scripts.train --config-name train_chunked data.scene_dir=data/blcs_broadcast
    data.chunk.chunks_dir=data/blcs_broadcast/chunks data.num_views_range=[1,1] data.camera_mode=random
    data.num_court_kp=14 data.num_workers=2 data.chunk.generation_workers=8 data.chunk.epochs_per_chunk=20
    camera=broadcast training.position_axis_weights=[1.0,4.0,1.0] training.trainer.max_epochs=200
    training.trainer.check_val_every_n_epoch=5 training.qualitative_logging.enabled=false
    run.gpus=1
artifacts:
  run_dir: knowledge/runs/run-mono3d-blcs-bcast-v3-simfix
  predictions: knowledge/runs/run-mono3d-blcs-bcast-v3-simfix/pred_test.npz
  log: .training_queue/logs/1783252014517687220_346625_mono3d_blcs_bcast_v3_simfix.log
  output_dir: outputs/blcs/blcs_multiview_axial/logs/version_2
  checkpoint: ckpt/blcs/run-mono3d-blcs-bcast-v3-simfix-epoch189.ckpt
  curves: knowledge/runs/run-mono3d-blcs-bcast-v3-simfix/curves.png
  tb_logdir: outputs/blcs/blcs_multiview_axial/logs/version_2
parents:
- run-mono3d-blcs-bcast-v2-yw4
relations:
- to: run-mono3d-blcs-bcast-v2-yw4
  rel: supersedes
tags:
- monocular
- broadcast
- sim-to-real
---

## 考察 / Findings

### 要約
射影鏡像バグ修正 + 広域カメラランダム化 + burst延長後の単眼BLCS再学習、200ep (PR #603)。実映像で崩壊した [[run-mono3d-blcs-bcast-v2-yw4]] を置き換える。in-dist pos_error は 1.46m→1.85m と増加したが、これはカメラ分布の大幅拡大(setback 15-40m / height 5-15m / 幅比 0.5-0.9)による難化で、実映像転移の有効性は本runが初の測定になる。

### アーキテクチャ詳細
モデル/損失は [[run-mono3d-blcs-bcast-v2-yw4]] と同一 (multiview_axial_base, C=1, court_kp=14, position_axis_weights [1,4,1], reprojection 0.1)。差分は環境側: (1) 射影のOpenCV規約化(鏡像解消、DifferentiableProjection含む)、(2) broadcastプリセットレンジ化、(3) burst_dropout 4→16フレーム×3回 (実測ボール欠損 run 最長21フレーム準拠)、(4) max_epochs 100→200 + 高速化 (epochs_per_chunk 5→20, val 5epごと, qualitative off; 200epが約79分)。

### メトリクスの解釈
test/pos_error 1.81m。軸別: x 0.51m / y 1.62m / z 0.37m — 単眼奥行き(Y)が支配的なのは想定どおり。endpoint 3.41m は旧run (3.64m) から微改善。position_accuracy_1_2m 0.65。

### アーキテクチャ⇄メトリクスの因果考察
in-dist誤差増 (1.46→1.85m) はカメラジオメトリの多様化 + burst延長(欠損中の補間はY方向に不確実)によるタスク難化が原因(仮説)。旧runは鏡像世界での自己一貫学習のため実映像では無効であり、in-dist数値の直接比較よりも実クリップでの挙動が評価軸。

### 既存実験との比較
[[run-mono3d-blcs-bcast-v2-yw4]] (1.46m/endpoint 3.64m) 比: in-dist位置は悪化、endpointは同等。y_error比率 (1.62/1.85=0.87) は旧run同様に奥行きがボトルネック。

### 次に有効な実験
(1) 本ckptで tennis_clip 再推論 (窓分割込み) — 実映像でのY発散 (+22.9m外挿) の解消確認が最優先。(2) 物理prior (z̈≈-g) + バウンドアンカー (#593 idea list) — endpoint 3.4m の構造的改善はこちら。(3) カメラ条件付け (court KPからのPnP特徴を明示入力)でY不確実性の低減。
