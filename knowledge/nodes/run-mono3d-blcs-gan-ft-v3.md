---
id: run-mono3d-blcs-gan-ft-v3
type: run
title: mono3d_blcs_gan_ft_v3
issue: 593
provider: claude
session: 59c50b41-eb5d-40e1-aecf-09958a0bf02e
date: '2026-07-06'
status: done
config:
  model: multiview_axial_base
  loss: trajectory_position + reprojection(0.1) + axis_weights[1,4,1] + LSGAN adversarial
    (target_weight 0.05, warmup 10ep, disc=trajectory_transformer hidden256); deterministic
    GAN transition start_epoch=10
  data: chunked_multiview_sequence C=1 (broadcast, court_kp=14)
metrics:
  mean_position_error_m: 1.547709
  mean_x_error_m: 0.404254
  mean_y_error_m: 1.346746
  mean_z_error_m: 0.293058
  mean_endpoint_error_m: 3.288679
  position_accuracy_0_3m: 0.163556
  position_accuracy_0_6m: 0.384316
  position_accuracy_1_2m: 0.638579
  endpoint_accuracy_0_5m: 0.05
  endpoint_accuracy_1m: 0.17
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
    training.learning_rate=3e-5 training.trainer.max_epochs=60 training.gan.transition.start_epoch=10
    run.init_weights=/home/kamimura/projects/tennis-lab/outputs/blcs/blcs_multiview_axial/logs/version_2/checkpoints/last.ckpt
    run.output_dir=/home/kamimura/projects/tennis-lab/outputs/blcs/blcs_multiview_axial
    run.gpus=1
artifacts:
  run_dir: knowledge/runs/run-mono3d-blcs-gan-ft-v3
  predictions: knowledge/runs/run-mono3d-blcs-gan-ft-v3/pred_test.npz
  log: .training_queue/logs/1783307565114299774_687118_mono3d_blcs_gan_ft_v3.log
  output_dir: outputs/blcs/blcs_multiview_axial/logs/version_7
  curves: knowledge/runs/run-mono3d-blcs-gan-ft-v3/curves.png
  tb_logdir: outputs/blcs/blcs_multiview_axial/logs/version_7
parents:
- run-mono3d-blcs-bcast-v3-simfix
relations:
- to: run-mono3d-blcs-ftc-axis-s04g03
  rel: compares
- to: run-mono3d-blcs-bcast-v3-simfix
  rel: compares
tags:
- monocular
- broadcast
- gan
- finetune
- real-clip
- physics-prior
---

## 考察 / Findings

### 要約
物理prior ([[run-mono3d-blcs-ftc-axis-s04g03]]) と**完全同一設定** (v3 init_weights / 60ep / LR 3e-5) で、手作りの jerk+gravity loss を **LSGAN discriminator に差し替えた** BLCS fine-tune。決定論的GAN transition (start_epoch=10) を新規実装して使用。in-dist は全軸で改善し **pos_error 1.845→1.548m** (物理ftC 1.947m を明確に下回る=全variant最良)。しかし real-clip の非物理ジッターは **ほぼ不変** (ball jerk 0.280→0.278)。GANは「当てはまりの良いモデル」を作るが物理priorのような平滑化はしない=**物理priorの代替にはならない**。

### アーキテクチャ詳細
[[run-mono3d-blcs-bcast-v3-simfix]] checkpoint (version_2/last.ckpt) から weight-only fine-tune。generator=既存 multiview_axial_base、discriminator=trajectory_transformer (hidden256, num_layers4, 系列に1 logit の real/fake 判定)。manual optimization で `hybrid_loss = supervised_loss + w·LSGAN_generator_loss`。GAN transition は本PRで **損失監視→エポック監視 (決定論的)** に変更: `training.gan.transition.start_epoch=10` で ep0-9 純教師あり、ep10 から GAN 起動し warmup 10ep で weight 0→0.05。`target_weight=0.05` は投入前に v3収束点の教師あり loss≈0.05 と LSGAN generator loss≈O(0.1-0.5) を実測較正して決定 (blcs既定 2.0 は GAN項が教師ありの約20倍で振幅崩壊するため不採用)。

### メトリクスの解釈
in-dist test: pos_error **1.548m** / x 0.404 / y 1.347 / z 0.293 / endpoint 3.289。全軸で v3 (0.511/1.615/0.369) と ftC (0.690/1.584/0.409) を下回る。単眼depthのボトルネックである y も 1.615→1.347 と改善。val 軌跡は ep10 起動→ep16 に 2.34m の過渡bump→ep20 (weight full) で 1.71m へ回復→ep59 1.51m と健全 (崩壊なし)。real-clip (tennis_clip, 可視区間限定; マスクは ball検出共有で4条件同一): ball jerk **0.278** (v3 0.280, ftC 0.106)、jerk_Y 0.226 (v3 0.231)、jerk_Z 0.064 (v3 0.065)、gravity a_z∈[-15,-4] 0.096 (v3 0.106, ftC 0.303)、height_max 4.54m (v3 3.22, 圧縮せず)、|Y|>15m 9 (v3 10)。

### アーキテクチャ⇄メトリクスの因果考察
in-dist が改善したのは、discriminator が GT 軌道分布との整合を促し、位置教師と競合せず「もっともらしく当てはまる」方向へ緩く正則化したため (仮説)。振幅も潰れない (height 4.54m)。一方 real-clip ジッターが不変なのは、discriminator が**系列1本に1 logit の大域判定**で、per-frame の高周波ジッターに局所的な勾配を与えないため (仮説)。明示的な jerk 罰則 (ftC) は各フレームの3階差分を直接抑えるので、同じ「物理的妥当性」でも GAN と最適化対象が異なる。留保: in-dist 改善の一部は 60ep 追加学習の寄与もありうるが、完全同一設定の ftC が 1.947m に悪化した事実から、物理loss→GAN の差替え効果が支配的。

### 既存実験との比較
[[run-mono3d-blcs-ftc-axis-s04g03]] (物理ftC, 同一設定): in-dist は GAN が全軸で優位 (1.548 vs 1.947m)。real-clip は逆に物理が圧勝 (jerk 0.106 vs 0.278、gravity 0.303 vs 0.096)。→ **両者はトレードオフが真逆**。[[run-mono3d-blcs-bcast-v3-simfix]] (baseline): in-dist 改善、real-clip ジッターは同等。

### 次に有効な実験
(1) discriminator を **速度/加速度系列** 入力 or **per-frame/patch 判定** にして高周波ジッターを狙い撃ちさせる (系列レベル大域判定の弱点を補う)。(2) 純教師あり60ep fine-tune の対照を取り、in-dist 改善のうち GAN 寄与を厳密に分離。(3) GAN と軽い jerk prior の併用 (精度は GAN、平滑化は明示prior)。(4) best-checkpoint 選択 (現状 last-epoch)。
