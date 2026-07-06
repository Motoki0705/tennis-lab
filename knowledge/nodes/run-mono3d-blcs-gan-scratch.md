---
id: run-mono3d-blcs-gan-scratch
type: run
title: mono3d_blcs_gan_scratch
issue: 593
provider: claude
session: 59c50b41-eb5d-40e1-aecf-09958a0bf02e
date: '2026-07-06'
status: done
config:
  model: multiview_axial_base
  loss: trajectory_position + reprojection(0.1) + axis_weights[1,4,1] + LSGAN adversarial
    (target_weight 0.05, warmup 10ep, disc=trajectory_transformer hidden256); deterministic
    GAN transition start_epoch=100 (100ep 教師あり warmup → 100ep GAN)
  data: chunked_multiview_sequence C=1 (broadcast, court_kp=14)
metrics:
  mean_position_error_m: 2.06025
  mean_x_error_m: 0.598959
  mean_y_error_m: 1.747141
  mean_z_error_m: 0.365075
  mean_endpoint_error_m: 3.553755
  position_accuracy_0_3m: 0.058349
  position_accuracy_0_6m: 0.220291
  position_accuracy_1_2m: 0.481895
  endpoint_accuracy_0_5m: 0.02
  endpoint_accuracy_1m: 0.19
repro:
  commit: 692b4f6c94bb4bd7036101112a50de61e90983ab
  branch: main
  remote: git@github.com:Motoki0705/tennis-lab.git
  command: MPLBACKEND=Agg PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True .venv/bin/python
    -m src.tasks.blcs.scripts.train --config-name train_chunked_gan data.scene_dir=/home/kamimura/projects/tennis-lab/data/blcs_broadcast
    data.chunk.chunks_dir=/home/kamimura/projects/tennis-lab/data/blcs_broadcast/chunks
    data.num_views_range=[1,1] data.camera_mode=random data.num_court_kp=14 data.num_workers=2
    data.chunk.generation_workers=8 data.chunk.epochs_per_chunk=20 camera=broadcast
    training.position_axis_weights=[1.0,4.0,1.0] training.trainer.check_val_every_n_epoch=5
    training.qualitative_logging.enabled=false training.gan.target_weight=0.05 training.gan.warmup_epochs=10
    training.trainer.max_epochs=200 training.gan.transition.start_epoch=100 run.output_dir=/home/kamimura/projects/tennis-lab/outputs/blcs/blcs_multiview_axial
    run.gpus=1
artifacts:
  run_dir: knowledge/runs/run-mono3d-blcs-gan-scratch
  predictions: knowledge/runs/run-mono3d-blcs-gan-scratch/pred_test.npz
  log: .training_queue/logs/1783307565131814992_687133_mono3d_blcs_gan_scratch.log
  output_dir: outputs/blcs/blcs_multiview_axial/logs/version_8
  curves: knowledge/runs/run-mono3d-blcs-gan-scratch/curves.png
  tb_logdir: outputs/blcs/blcs_multiview_axial/logs/version_8
parents: []
relations:
- to: run-mono3d-blcs-bcast-v4-physics
  rel: compares
- to: run-mono3d-blcs-gan-ft-v3
  rel: compares
- to: run-mono3d-blcs-bcast-v3-simfix
  rel: compares
tags:
- monocular
- broadcast
- gan
- negative-result
- real-clip
---

## 考察 / Findings

### 要約
GAN を **from-scratch** で入れた対照 run。決定論的 transition (start_epoch=100) で ep0-99 純教師あり (振幅確立) → ep100 から GAN 起動。物理 from-scratch ([[run-mono3d-blcs-bcast-v4-physics]]) が振幅崩壊で 2.438m に悪化したのに対し、GAN from-scratch は **2.060m** と物理より軽症だが、fine-tune ([[run-mono3d-blcs-gan-ft-v3]] 1.548m) にも純教師あり200ep ([[run-mono3d-blcs-bcast-v3-simfix]] 1.845m) にも劣る。real-clip ジッターはむしろ悪化 (jerk 0.280→0.384)。**「fine-tune > from-scratch」の教訓は GAN でも成立**。

### アーキテクチャ詳細
init_weights なし、LR 1e-4 (default)、200ep。`training.gan.transition.start_epoch=100` で前半100ep は純教師あり、後半100ep で GAN (warmup 10ep で weight→0.05)。generator/discriminator は [[run-mono3d-blcs-gan-ft-v3]] と同一。狙いは「教師ありで振幅を確立してから GAN を入れれば from-scratch の崩壊を避けられるか」の検証。

### メトリクスの解釈
in-dist test: pos_error **2.060m** / x 0.599 / y 1.747 / z 0.365 / endpoint 3.554。val は教師あり相 (ep0-99) で 4.75→~2.5m まで低下したが、GAN 起動後 (ep100-) は 2.5-3.3m で不安定に振動し、最終 2.06m。real-clip (可視区間): ball jerk **0.384** (v3 0.280 より悪化)、jerk_Y 0.333、gravity a_z∈[-15,-4] 0.080 (v3 0.106 より悪化)、height_max 3.44m、|Y|>15m 10。

### アーキテクチャ⇄メトリクスの因果考察
100ep の教師あり warmup では v3 の 200ep 純教師あり収束 (1.845m) に届かず、ep100 時点で ~2.5m と未収束。その未収束モデルに GAN 圧をかけたため、val が広い帯で振動し (adversarial 不安定)、収束を阻害したと考える (仮説)。fine-tune 版が滑らかに改善したのと対照的で、GAN は「十分収束したモデルの後段 refine」でこそ機能し、未収束からの同時最適化には向かない。物理 from-scratch より軽症なのは、GAN が明示 prior ほど強く振幅を潰さないため。

### 既存実験との比較
[[run-mono3d-blcs-bcast-v4-physics]] (物理 from-scratch 2.438m): GAN の方が in-dist は良い (2.060m) が、real-clip jerk は GAN の方が悪い (0.384 vs 物理は改善方向)。[[run-mono3d-blcs-gan-ft-v3]] (GAN fine-tune 1.548m): fine-tune が明確に優位。[[run-mono3d-blcs-bcast-v3-simfix]] (純教師あり 1.845m): それにも未達。

### 次に有効な実験
from-scratch 路線は非推奨。GAN を使うなら収束済み ckpt からの fine-tune 一択 ([[run-mono3d-blcs-gan-ft-v3]] 参照)。どうしても single-run にしたいなら教師あり warmup を大幅に延ばす (例 max_epochs 400, start_epoch 300) 必要があるが費用対効果は低い。
