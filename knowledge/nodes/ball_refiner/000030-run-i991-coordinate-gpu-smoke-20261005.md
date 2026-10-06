---
id: run-i991-coordinate-gpu-smoke-20261005
type: run
task: ball_refiner
sequence: 30
recorded_at: '2026-10-05'
title: 座標2D RefinerのGPU試運転：共有BLCS・GAN・保存再読込
issue: 991
provider: codex
session: 01a10697-63da-75b2-ae84-f2980da51c23
date: '2026-10-05'
status: done
config:
  dimensions: 2
  architecture: regression
  gan_weight: 0.002
  steps: 100
  batch_size: 32
  train_rallies: 3
  val_rallies: 1
  test_rallies: 1
  dataset_scope: CPU smoke fixture; not production dataset
metrics:
  best_step: 100.0
  inference_ms_per_frame: 0.023001
  test_event_rmse_px: 202.351974
  test_frame_missing_rate: 0.12711
  test_missing_rmse_px: 299.927277
  test_rmse_px: 131.139374
repro:
  commit: ea96f1ffaa378f5b36d5347f29dabbd3edaee038
  branch: codex/coordinate-ball-refiners-991-1014
  remote: git@github.com:Motoki0705/tennis-lab.git
  command: OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 .venv/bin/python -m src.tasks.ball_refiner.scripts.train_coordinates
    paths.data_root=/tmp/ball-refiner-coordinate-smoke run.output_dir=ball_refiner/train/coordinates-smoke/gpu-20261005
    training.steps=100 training.batch_size=32 training.evaluate_every=100 training.log_every=25
    training.gan_warmup_steps=25
artifacts:
  run_dir: knowledge/runs/run-i991-coordinate-gpu-smoke-20261005
  predictions: knowledge/runs/run-i991-coordinate-gpu-smoke-20261005/pred_test.npz
  log: /home/kamimura/projects/tennis-lab/.training_queue/logs/1791165244385535955_1534041_i991-coordinate-gpu-smoke-20261005.log
  output_dir: /home/kamimura/projects/tennis-lab/outputs/ball_refiner/train/coordinates-smoke/gpu-20261005/logs/version_0
parents: []
relations: []
papers: []
tags:
- coordinate-refiner
- gpu-smoke
- gan
---

共有BLCSデータ5ラリー（train 3 / val 1 / test 1）を使い、座標＋欠損maskから全frameを予測する2D回帰＋GANをGPUで100更新した。学習・validation選択・test保存・best checkpoint再読込・図出力が完了した。学習中peak allocatedは211,538,432 bytes（約202 MiB）であり、本比較の2ジョブ同時実行に十分な余裕がある。GPU学習は共有training queue経由で実行した。

これは処理系の試運転であり、性能比較や採用の根拠にはしない。testの全frame RMSEは131.14 px、欠損frameは299.93 pxで、線形補間の78.90 / 75.32 pxより悪い。trainが3ラリー・100更新に限られ、視点と軌道の一般化を判断できない。次は同じ生成仕様の本データ1,280ラリーでイベント選択率・GAN・3D Flowを共通予算で比較する。

本データではなく `/tmp/ball-refiner-coordinate-smoke/ball_refiner/single_object` のfixtureを使用した。再生成は `src.tasks.ball_refiner.scripts.generate_coordinates` に `paths.data_root=/tmp/ball-refiner-coordinate-smoke generation.train_rallies=3 generation.val_rallies=1 generation.test_rallies=1 generation.workers=2` を明示する。既存の出力先への上書きは拒否される。

TensorBoardはrunの `logs/version_0/` に保存した。少数更新によるloss減少は学習経路の確認に限る。過去のGMM実験とは入力、教師、split、目的関数が異なるため、数値を直接比較しない。

kg_curvesを実行したが、train/reconstruction・val/rmseのtagは共通描画の対象外でskipされた。この試運転／中断runの曲線は添付せず、元のTensorBoardとログを保持する。本比較15条件には専用scriptで描いた曲線を添付した。
