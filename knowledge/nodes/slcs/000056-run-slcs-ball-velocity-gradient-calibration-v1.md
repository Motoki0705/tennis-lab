---
task: slcs
sequence: 56
recorded_at: 2026-09-19
date_source: experiment_date
papers: []
id: run-slcs-ball-velocity-gradient-calibration-v1
type: run
title: 'SLCS速度整合: train-only初期勾配10%からweightを固定'
provider: codex
session: 01a0b71f-c516-75b2-ac58-0c92e8a4f098
date: '2026-09-19'
status: done
config:
  model: fresh SLCSFusionModel hidden128/shared4/DINO-downsample2
  loss: supervised ball velocity, Smooth L1 beta1
  data: slcs/real_rgb_v1 train only, saved augmentation
  batch_size: 16
  seed: 42
  velocity_scale_mps: 11.259468485469933
metrics:
  supervised_loss: 0.28767937421798706
  velocity_loss: 13.338584899902344
  supervised_gradient_norm: 0.017141269257452938
  velocity_gradient_norm: 1.5418442357252726
  target_gradient_ratio: 0.1
  calibrated_ball_velocity_weight: 0.0011117380640846516
  train_windows: 466
  selected_windows: 16
  valid_positive_weight_pairs: 1579
  pair_confidence_sum: 1318.949951171875
repro:
  commit: 42b1429268c3e4502071ac09535d466163e9e8a2
  branch: codex/slcs-real-rgb
  remote: git@github.com:Motoki0705/tennis-lab.git
  command: env PYTHONUNBUFFERED=1 PYTHONFAULTHANDLER=1 CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4
    MKL_NUM_THREADS=4 .venv/bin/python -m scripts.analysis.calibrate_slcs_ball_velocity
    --output-root /home/kamimura/projects/tennis-lab/outputs --training-run slcs/train/real_rgb_no_ball_smooth/s42-takeover-003
    --output slcs/analyze/ball_velocity_calibration/s42-001 --device cuda --batch-size
    16 --seed 42 --velocity-scale-mps 11.259468485469933 --gradient-ratio 0.1
artifacts:
  run_dir: knowledge/runs/run-slcs-ball-velocity-gradient-calibration-v1
  output_dir: /home/kamimura/projects/tennis-lab/outputs/slcs/analyze/ball_velocity_calibration/s42-001
  log: knowledge/runs/run-slcs-ball-velocity-gradient-calibration-v1/queue.log
parents:
- run-slcs-ball-velocity-train-scale-v1
relations:
- to: run-slcs-full-real-rgb-no-ball-smooth-e60-v3
  rel: compares
tags:
- slcs
- real-rgb
- train-only
- velocity
- calibration
---

## 考察 / Findings

### 要約

学習16窓の初期化時ball出力勾配を使い、追加速度項が既存ball supervised項の10%になるweight=0.0011117380640846516を得た。
以後の60epoch比較ではこの値とtrain-only scale=11.259468485469933m/sを固定する。
validation/testもcheckpoint重みも使用していない。係数決定であり、モデル改善の実証ではない。

### アーキテクチャ詳細

保存control configから同じmodelをseed42でfresh初期化し、train466窓だけを構築した。
独立CPU Generator(seed42)のrandpermで16窓を一様非復元抽出する。
augmentation前とdropout前にそれぞれ再seedし、保存済みaugmentation・DINO・training modeの単一forwardを使用した。
既存supervised項はball_position_weight×position lossとball_position_nll_weight×NLLの和。
両項のpred_ball_positionに対するL2勾配normを測り、weight=0.1×既存norm/速度normとした。
選択window IDs、RNG順序、元config全体とSHA、torch/numpy版をcalibration.jsonへ保存した。

### メトリクスの解釈

既存loss0.287679、速度loss13.338585に対し、出力勾配normは0.0171413と1.5418442。
weighted速度normは0.00171413で、宣言した比率0.1を満たす。有効pair1579、confidence総和1318.95。
学習曲線はない。GPU共有queueのall予約による1回のprobeであり、学習epoch数・test metricではない。

### アーキテクチャ⇄メトリクスの因果考察

実FPSで差分を速度へ換算する項は位置項より出力勾配が大きかったため、単純に同じweightにはしない。
10%は初期のバランスを定める工学的仮説で、学習中のparameter勾配比や最適係数を保証しない。
単一batchでの較正の不確実性を残し、結果を見てvalidationに合わせて係数を再調整しない。

### 既存実験との比較

親runは教師統計からscaleだけを決めた。本runはfresh modelに同じ教師版と通常のtrain augmentationを与えweightを決める。
gap48のbroadcast退行を踏まえ、次の比較には既存burst24 controlを使う。
旧model/checkpointを較正に読み込んだ結果ではなく、学習開始時の出力勾配に基づく設定である。

### 次に有効な実験

`train_real_rgb_velocity`でseed42・60epoch、同じmodel/data/optimizer/augmentation、ball jerk=0を維持して学習する。
自動testを無効にし、validation最良checkpointを同じ5入力条件で比較する。
位置平均/p95、visibility境界、教師の速い区間、速度ベクトル誤差、playerを確認し、最大速度低下だけで採用しない。
