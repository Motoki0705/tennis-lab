---
id: run-i991-rope-2d-gan-only-eventonly-s42-20261006-v4
type: run
task: ball_refiner
sequence: 45
recorded_at: '2026-10-06'
title: 2D Refiner：位置lossを0へ減衰したGAN-only学習（seed42）
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
    ffn_type: swiglu
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
    reconstruction:
      initial_weight: 1.0
      final_weight: 0.0
      start_step: 2000
      decay_steps: 1000
    gan:
      enabled: true
      target_weight: 1.0
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
    output_dir: ball_refiner/train/rope-2d-gan-only-eventonly/20261006-v4-s42
    seed: 42
    device: cuda
metrics:
  best_step: 3000
  last_step: 4000
  training_seconds: 367.8468093830161
  peak_gpu_memory_bytes: 1889165824
  best_all_rmse_px: 9.476381301879883
  best_missing_rmse_px: 23.475875854492188
  best_observed_rmse_px: 5.559487342834473
  best_event_rmse_px: 17.9011173248291
  best_validation_rmse_px: 9.86307430267334
  best_velocity_rmse_per_s: 142.0474882835111
  best_acceleration_rmse_per_s2: 9684.918706174249
  best_event_acceleration_magnitude_ratio: 2.502993452918707
  last_all_rmse_px: 289.4901428222656
  last_missing_rmse_px: 267.4595947265625
  last_observed_rmse_px: 292.1831359863281
  last_event_rmse_px: 262.9962463378906
  last_validation_rmse_px: 288.4842529296875
  last_velocity_rmse_per_s: 297.7136036277659
  last_acceleration_rmse_per_s2: 21272.133115595316
  last_event_acceleration_magnitude_ratio: 2.7492108692113226
repro:
  commit: b415d791c7275ba871bb2c2cdc541090cfacec12
  branch: codex/coordinate-refiner-gan-only
  remote: git@github.com:Motoki0705/tennis-lab.git
  command: OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 .venv/bin/python -m src.tasks.ball_refiner.scripts.train_coordinates
    training=coordinate_gan_only model.dimensions=2 run.output_dir=ball_refiner/train/rope-2d-gan-only-eventonly/20261006-v4-s42
artifacts:
  run_dir: knowledge/runs/run-i991-rope-2d-gan-only-eventonly-s42-20261006-v4
  predictions: knowledge/runs/run-i991-rope-2d-gan-only-eventonly-s42-20261006-v4/pred_test.npz
  log: /home/kamimura/projects/tennis-lab/.training_queue/logs/1791259707735987111_3712392_i991-rope-2d-gan-only-eventonly-s42-20261006-v4.log
  output_dir: /home/kamimura/projects/tennis-lab/outputs/ball_refiner/train/rope-2d-gan-only-eventonly/20261006-v4-s42/logs/version_0
  config: knowledge/runs/run-i991-rope-2d-gan-only-eventonly-s42-20261006-v4/config.yaml
  curves: knowledge/runs/run-i991-rope-2d-gan-only-eventonly-s42-20261006-v4/curves.png
  loss_schedule: knowledge/runs/run-i991-rope-2d-gan-only-eventonly-s42-20261006-v4/loss-schedule.png
  last_predictions: knowledge/runs/run-i991-rope-2d-gan-only-eventonly-s42-20261006-v4/predictions_last/pred_test.npz
  last_diagnostics: knowledge/runs/run-i991-rope-2d-gan-only-eventonly-s42-20261006-v4/predictions_last/diagnostic_metrics.json
  checkpoint: /home/kamimura/projects/tennis-lab/outputs/ball_refiner/train/rope-2d-gan-only-eventonly/20261006-v4-s42/logs/version_0/checkpoints/best.ckpt
  last_checkpoint: /home/kamimura/projects/tennis-lab/outputs/ball_refiner/train/rope-2d-gan-only-eventonly/20261006-v4-s42/logs/version_0/checkpoints/last.ckpt
  tb_logdir: /home/kamimura/projects/tennis-lab/outputs/ball_refiner/train/rope-2d-gan-only-eventonly/20261006-v4-s42/logs/version_0
parents:
- group-i991-i1014-rope-gan-eventonly-s42
relations:
- to: run-i991-rope-2d-gan-eventonly-s42-20261006-v3
  rel: compares
papers: []
tags:
- coordinate-refiner
- gan-only
- position-loss-decay
- event-only
- seed42
- negative-result
---

## 結果

共通single_objectを使い、batch32・seed42で4,000更新を完了した。GAN係数は500更新待機後の1,000更新で0→1、位置SmoothL1係数は2,000更新まで1、次の1,000更新で1→0、3,001〜4,000更新はGANのみとした。3,000更新も係数0に到達している。Generator/Discriminator・入力劣化・AdamW・cosine LR・split・評価seedは前回条件を維持した。

最終step4000のtest全体RMSEは**289.4901px**、欠損267.4596px、観測292.1831px、イベント近傍262.9962pxだった。validationで選ばれたbestはstep3000で、test全体9.4764px。bestの選択はtestを開く前に確定し、最終GAN-only区間の結果にbestの値を代用していない。

前回（位置係数1を維持・GAN最大2）の全体RMSEは6.9809pxだった。今回のlastは前回より位置精度が悪化した。同一の評価入力hash、全frameの入力・GT・mask・IDが一致し、保存NPZから指標を再計算して診断JSONと照合した。lastのG更新回数4000・D更新回数3500、位置係数0・GAN係数1、checkpoint hashも検証した。

## 解釈と次の比較

GANのみへの移行後に入力位置との一致が崩れた。軌道だけを判別するDiscriminatorには、出力と入力観測の対応を直接評価する経路がない。この欠如が位置のずれを許したという仮説はあるが、今回だけで機序を確定しない。前回からGAN最大係数と位置lossの両方を変更しており、個々の効果は分離できない。1 seed・合成軌道であり、実動画への一般化も評価していない。

最終モデルの速度RMSEは297.7136px/s、イベント近傍の平均加速度ノルムはGTの2.749倍。位置のずれに加えて、これらを軌道の自然さの診断として残したが、知覚的自然さの保証とは扱わない。次に比較するならGAN最大1を固定し、位置係数を1に維持する対照と、小さい正値に留める条件を同一seed群で比較する。今回は追加学習・本番重みの切替は行っていない。

## 再現性と検証

学習commitはb415d791c7275ba871bb2c2cdc541090cfacec12、共有queueのresource=allで順次実行した。best予測はbundle直下、最終予測はpredictions_last/に保持した。学習時のmetrics.jsonにはcheckpoint_kindの文字列が含まれていたため、数値専用の登録器は自動抽出を省略した。履歴bundleを保持したまま、このnodeには診断JSONに基づく明示的なbest/last指標を記録した。後続f9d61e5a6では文字列を診断JSONだけに置き、数値metrics互換性を修正した。学習計算・重みへの変更はない。

TensorBoardは存在する。kg_curves.pyの対応tagではない系列は、追跡済みtests/benchmarks/ball_refiner_coordinates_report.pyのplot_curvesとmotion_metrics、および保存した解析scriptで描画・集計した。全更新の係数、重み付きloss、validation曲線を残した。関連テスト44件が成功し、最終の保存形式修正後に統合7件を再確認した。今回の新規validator指定はなく、validatorは起動していない。
