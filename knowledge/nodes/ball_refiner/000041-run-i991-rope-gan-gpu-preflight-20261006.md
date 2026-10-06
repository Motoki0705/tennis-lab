---
id: run-i991-rope-gan-gpu-preflight-20261006
type: run
task: ball_refiner
sequence: 41
recorded_at: '2026-10-06'
title: RoPE座標GANの実サイズGPU動作確認（4更新、精度評価対象外）
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
    steps: 4
    batch_size: 32
    learning_rate: 0.0003
    weight_decay: 0.01
    gradient_clip: 1.0
    evaluate_every: 4
    log_every: 1
    cpu_threads: 2
    gan:
      enabled: true
      target_weight: 2.0
      transition:
        start_step: 1
      warmup_steps: 2
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
    output_dir: ball_refiner/train/rope-2d-gan-gpu-preflight/20261006-s42
    seed: 42
    device: cuda
metrics:
  best_step: 4.0
  inference_ms_per_frame: 0.015586
  test_event_rmse_px: 2929.113281
  test_frame_missing_rate: 0.113212
  test_missing_rmse_px: 2866.937256
  test_rmse_px: 2935.939453
  velocity_rmse_per_s: 1224.6870154263988
  acceleration_rmse_per_s2: 102205.81866822844
  event_acceleration_magnitude_ratio: 11.619981236792873
repro:
  commit: 9588b33833c54423c361d150fbac442f9631497f
  branch: codex/coordinate-refiner-rope-gan
  remote: git@github.com:Motoki0705/tennis-lab.git
  command: OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 .venv/bin/python -m src.tasks.ball_refiner.scripts.train_coordinates
    training.steps=4 training.evaluate_every=4 training.log_every=1 training.gan.transition.start_step=1
    training.gan.warmup_steps=2 run.output_dir=ball_refiner/train/rope-2d-gan-gpu-preflight/20261006-s42
artifacts:
  run_dir: knowledge/runs/run-i991-rope-gan-gpu-preflight-20261006
  predictions: knowledge/runs/run-i991-rope-gan-gpu-preflight-20261006/pred_test.npz
  log: /home/kamimura/projects/tennis-lab/.training_queue/logs/1791247437343443332_3420896_i991-rope-gan-gpu-preflight-20261006.log
  output_dir: /home/kamimura/projects/tennis-lab/outputs/ball_refiner/train/rope-2d-gan-gpu-preflight/20261006-s42/logs/version_0
  config: knowledge/runs/run-i991-rope-gan-gpu-preflight-20261006/config.yaml
  checkpoint: /home/kamimura/projects/tennis-lab/outputs/ball_refiner/train/rope-2d-gan-gpu-preflight/20261006-s42/logs/version_0/checkpoints/best.ckpt
  last_checkpoint: /home/kamimura/projects/tennis-lab/outputs/ball_refiner/train/rope-2d-gan-gpu-preflight/20261006-s42/logs/version_0/checkpoints/last.ckpt
  curves: knowledge/runs/run-i991-rope-gan-gpu-preflight-20261006/curves.png
  motion_metrics: knowledge/runs/run-i991-rope-gan-gpu-preflight-20261006/motion_metrics.json
  tb_logdir: /home/kamimura/projects/tennis-lab/outputs/ball_refiner/train/rope-2d-gan-gpu-preflight/20261006-s42/logs/version_0
  gan_curve: knowledge/runs/run-i991-rope-gan-gpu-preflight-20261006/gan_curve.png
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

共通single_objectの実データ、Generator幅256・8層／Discriminator幅256・4層、batch32でGPU上の4更新・validation・test・checkpoint保存を完了した。GPU allocated peakは1,888,051,200 bytes。新しい軌道だけを判別するLSGANとRoPE共通部品の数値・保存経路を確認するsmokeである。

動作確認では待機1更新・増加2更新を明示overrideし、係数0→1→2→2を観測した。本学習の待機500・増加1000とは違う。test RMSE 2935.94pxという値を品質達成と解釈せず、4更新のみの重みを正式比較へ含めない。データ・コード・予測は保存した。通常テストと独立評価の後、本学習へ進む判断の根拠は正常終了と有限値、メモリ余裕であり、この位置精度ではない。

TensorBoardは存在するが、`kg_curves.py`が対象とするtrain/val同名系列がないためスキップされた。保存済みJSONログから既存の`tests/benchmarks/ball_refiner_coordinates_report.py::plot_curves`で学習・validation曲線を作成し、GAN係数とlossも別図にした。速度・加速度は同ファイルの`motion_metrics`で、全test軌道の有限差分を60fpsに換算した。
