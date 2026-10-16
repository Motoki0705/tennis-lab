---
id: run-i1014-rope-3d-gan-eventonly-s42-20261006-v3
type: run
task: ball_refiner
sequence: 43
recorded_at: '2026-10-06'
title: 3D RoPE Transformer＋軌道GAN：ノイズなしイベント連続欠損（seed42）
issue: 1014
provider: codex
session: 01a10697-63da-75b2-ae84-f2980da51c23
date: '2026-10-06'
status: done
config:
  model:
    dimensions: 3
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
    output_dir: ball_refiner/train/rope-3d-gan-eventonly/20261006-v3-s42
    seed: 42
    device: cuda
metrics:
  best_step: 4000.0
  inference_ms_per_frame: 0.018082
  test_event_rmse_m: 0.296786
  test_frame_missing_rate: 0.113212
  test_missing_rmse_m: 0.401636
  test_rmse_m: 0.147149
  validation_rmse_m: 0.14348241686820984
  test_observed_rmse_m: 0.06183363497257233
  training_seconds: 372.2396025880007
  peak_gpu_memory_bytes: 1889256960
  velocity_rmse_per_s: 3.2509710741498634
  acceleration_rmse_per_s2: 241.40067124054613
  event_acceleration_magnitude_ratio: 2.71123562720467
repro:
  commit: c5cee39b26e8f876c23df177d627df2f6ea8fece
  branch: codex/coordinate-refiner-rope-gan
  remote: git@github.com:Motoki0705/tennis-lab.git
  command: OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 .venv/bin/python -m src.tasks.ball_refiner.scripts.train_coordinates
    model.dimensions=3 run.output_dir=ball_refiner/train/rope-3d-gan-eventonly/20261006-v3-s42
artifacts:
  run_dir: knowledge/runs/run-i1014-rope-3d-gan-eventonly-s42-20261006-v3
  predictions: knowledge/runs/run-i1014-rope-3d-gan-eventonly-s42-20261006-v3/pred_test.npz
  log: /home/kamimura/projects/tennis-lab/.training_queue/logs/1791248190831634287_3435512_i1014-rope-3d-gan-eventonly-s42-20261006-v3.log
  output_dir: /home/kamimura/projects/tennis-lab/outputs/ball_refiner/train/rope-3d-gan-eventonly/20261006-v3-s42/logs/version_0
  config: knowledge/runs/run-i1014-rope-3d-gan-eventonly-s42-20261006-v3/config.yaml
  checkpoint: /home/kamimura/projects/tennis-lab/outputs/ball_refiner/train/rope-3d-gan-eventonly/20261006-v3-s42/logs/version_0/checkpoints/best.ckpt
  last_checkpoint: /home/kamimura/projects/tennis-lab/outputs/ball_refiner/train/rope-3d-gan-eventonly/20261006-v3-s42/logs/version_0/checkpoints/last.ckpt
  curves: knowledge/runs/run-i1014-rope-3d-gan-eventonly-s42-20261006-v3/curves.png
  motion_metrics: knowledge/runs/run-i1014-rope-3d-gan-eventonly-s42-20261006-v3/motion_metrics.json
  tb_logdir: /home/kamimura/projects/tennis-lab/outputs/ball_refiner/train/rope-3d-gan-eventonly/20261006-v3-s42/logs/version_0
  gan_curve: knowledge/runs/run-i1014-rope-3d-gan-eventonly-s42-20261006-v3/gan_curve.png
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

4,000更新を完了し、固定validationの全frame RMSEでstep 4000を選択した。選択重みはGAN係数2の期間に属する。testの全frame RMSEは0.14715m、欠損0.40164m、観測0.06183m、イベント近傍0.29679m。欠損率11.3212%、イベント選択率49.2047%だった。入力はノイズなし2D観測の三角測量結果と欠損maskであり、GTの3D座標を入力へ転記していない。

GeneratorとDiscriminatorの構成・学習スケジュールは2Dと共通。3Dでは座標チャンネルを3へ変更した。同じ共有データのラリーsplit・評価seed20991・イベント欠損を再利用した。学習はseed42、batch32、128frame、SmoothL1 beta0.02＋LSGAN、最初の500更新はGANなし、次の1000更新で係数2に到達する。旧3D Flow方式と重みを保持したが、この実験ではFlowを再学習していない。

## 解釈と制約

同じtest入力への線形補間は全体0.30156m・欠損0.89624mであり、このrunは欠損の位置誤差を低減した。ノイズなし観測にも再推論する指定のため、ほぼ正確だった観測区間には約6.18cmのRMSEが発生した。観測座標を固定した手法とのトレードオフは未比較。

既存15条件は2Dノイズ・離散欠損・モデル構造が異なるため、この差をGANやRoPE単独の効果と解釈しない。同条件のGANなし対照、追加seed、実検出への一般化は未確認。連続性とイベント付近の力学的自然さは位置RMSEだけでは保証できない。testを用いた再選択・再調整は行っていない。

TensorBoardは存在するが、`kg_curves.py`が対象とするtrain/val同名系列がないためスキップされた。保存済みJSONログから既存の`tests/benchmarks/ball_refiner_coordinates_report.py::plot_curves`で学習・validation曲線を作成し、GAN係数とlossも別図にした。速度・加速度は同ファイルの`motion_metrics`で、全test軌道の有限差分を60fpsに換算した。
