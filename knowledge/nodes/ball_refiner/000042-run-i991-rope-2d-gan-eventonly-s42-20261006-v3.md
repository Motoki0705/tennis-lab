---
id: run-i991-rope-2d-gan-eventonly-s42-20261006-v3
type: run
task: ball_refiner
sequence: 42
recorded_at: '2026-10-06'
title: 2D RoPE Transformer＋軌道GAN：ノイズなしイベント連続欠損（seed42）
issue: 991
provider: codex
session: 01a10697-63da-75b2-ae84-f2980da51c23
date: '2026-10-06'
status: done
config:
  model:
    dimensions: 2
    architecture: regression
    width: 256
    layers: 8
    heads: 4
    dropout: 0.05
    window_length: 128
    flow_steps: 16
    ffn_dim: 704
    rope_dim: 64
    rope_theta: 10000.0
  corruption:
    event_probability: 0.5
    isolated_probability: 0.0
    gap_min: 3
    gap_max: 10
    noise_p95_px: 0.0
    jitter_sigma_px: 0.0
    outlier_probability: 0.0
    triangulation_steps: 3
  training:
    steps: 4000
    batch_size: 32
    learning_rate: 0.0003
    weight_decay: 0.01
    gradient_clip: 1.0
    evaluate_every: 500
    log_every: 100
    cpu_threads: 2
    gan:
      enabled: true
      target_weight: 2.0
      transition:
        start_step: 500
      warmup_steps: 1000
      discriminator:
        name: trajectory_transformer
        hidden_dim: 256
        num_layers: 4
        num_heads: 4
        ffn_dim: 704
        ffn_type: swiglu
        dropout: 0.1
        rope_dim: 64
        rope_theta: 10000.0
        max_seq_len: 128
        invalid_init_std: 0.02
        cls_init_std: 0.02
  run:
    output_dir: ball_refiner/train/rope-2d-gan-eventonly/20261006-v3-s42
    seed: 42
    device: cuda
metrics:
  best_step: 4000.0
  inference_ms_per_frame: 0.015961
  test_event_rmse_px: 14.639831
  test_frame_missing_rate: 0.113212
  test_missing_rmse_px: 19.226862
  test_rmse_px: 6.980916
  validation_rmse_px: 7.23055362701416
  test_observed_rmse_px: 2.7857491970062256
  training_seconds: 369.53814448899357
  peak_gpu_memory_bytes: 1889165824
  velocity_rmse_per_s: 120.04348172074236
  acceleration_rmse_per_s2: 7661.825681896389
  event_acceleration_magnitude_ratio: 1.9594446269943335
repro:
  commit: c5cee39b26e8f876c23df177d627df2f6ea8fece
  branch: codex/coordinate-refiner-rope-gan
  remote: git@github.com:Motoki0705/tennis-lab.git
  command: OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 .venv/bin/python -m src.tasks.ball_refiner.scripts.train_coordinates
    model.dimensions=2 run.output_dir=ball_refiner/train/rope-2d-gan-eventonly/20261006-v3-s42
artifacts:
  run_dir: knowledge/runs/run-i991-rope-2d-gan-eventonly-s42-20261006-v3
  predictions: knowledge/runs/run-i991-rope-2d-gan-eventonly-s42-20261006-v3/pred_test.npz
  log: /home/kamimura/projects/tennis-lab/.training_queue/logs/1791248190791221542_3435495_i991-rope-2d-gan-eventonly-s42-20261006-v3.log
  output_dir: /home/kamimura/projects/tennis-lab/outputs/ball_refiner/train/rope-2d-gan-eventonly/20261006-v3-s42/logs/version_0
  config: knowledge/runs/run-i991-rope-2d-gan-eventonly-s42-20261006-v3/config.yaml
  checkpoint: /home/kamimura/projects/tennis-lab/outputs/ball_refiner/train/rope-2d-gan-eventonly/20261006-v3-s42/logs/version_0/checkpoints/best.ckpt
  last_checkpoint: /home/kamimura/projects/tennis-lab/outputs/ball_refiner/train/rope-2d-gan-eventonly/20261006-v3-s42/logs/version_0/checkpoints/last.ckpt
  curves: knowledge/runs/run-i991-rope-2d-gan-eventonly-s42-20261006-v3/curves.png
  motion_metrics: knowledge/runs/run-i991-rope-2d-gan-eventonly-s42-20261006-v3/motion_metrics.json
  tb_logdir: /home/kamimura/projects/tennis-lab/outputs/ball_refiner/train/rope-2d-gan-eventonly/20261006-v3-s42/logs/version_0
  gan_curve: knowledge/runs/run-i991-rope-2d-gan-eventonly-s42-20261006-v3/gan_curve.png
parents:
- group-i991-i1014-coordinate-refiners-s42
relations: []
papers: []
tags:
- coordinate-refiner
- rope
- trajectory-gan
- event-only
- seed42
---

## 観測

4,000更新を完了し、固定validationの全frame RMSEでstep 4000を選択した。選択重みはGAN係数2の期間に属する。testの全frame RMSEは6.9809px、欠損19.2269px、観測2.7857px、イベント近傍14.6398px。共通データの同一splitと評価seed20991を使い、入力jitter・外れ値・離散欠損は0、イベント選択率50%に対する実testフレーム欠損率は11.3212%だった。

Generatorは共通RoPE/SwiGLU Transformer、DiscriminatorはBLCS/PLCS共通のCLS Transformerで座標軌道だけを判別する。座標と欠損maskから全frameを直接回帰し、観測区間もlossの対象とした。GANは最初の500更新が0、次の1000更新で2に到達して維持する。重み・全frame予測・実際のGAN係数とlossを含むログを保存した。

## 解釈と制約

既存15条件（親group）は入力にP95約200pxのノイズと4%の離散欠損を含み、モデルも異なる。この差を単独のアーキテクチャ効果やGAN係数の効果として比較しない。今回のGANなし対照と追加seedは実施していない。座標誤差が有限で学習したことと、物理的な曲線の連続性・イベント時の速度変化を保証することは別である。合成・全軌道画面内の条件であり、実検出への一般化は未確認。

次は同一のノイズなし条件でGANなしを比較し、イベント近傍の位置・速度・加速度を合わせて判断する。testの結果を使って選択checkpointやハイパーパラメータを再調整していない。

TensorBoardは存在するが、`kg_curves.py`が対象とするtrain/val同名系列がないためスキップされた。保存済みJSONログから既存の`tests/benchmarks/ball_refiner_coordinates_report.py::plot_curves`で学習・validation曲線を作成し、GAN係数とlossも別図にした。速度・加速度は同ファイルの`motion_metrics`で、全test軌道の有限差分を60fpsに換算した。
