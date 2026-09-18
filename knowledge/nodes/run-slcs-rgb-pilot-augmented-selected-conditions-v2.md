---
id: run-slcs-rgb-pilot-augmented-selected-conditions-v2
type: run
title: 'SLCS augmentation: validation選定重みのCPU四条件評価'
provider: codex
date: '2026-09-18'
status: done
config:
  model: validation-selected augmented, epoch=55 (0-based)
  loss: evaluation only, masked unweighted headline metrics
  data: slcs/real_rgb_pilot_v2, min_window_label_ratio=0.5, train window/token settings
  device: cpu float32
  batch_size: 4
metrics:
  val/full/player_position_error_m: 2.669083833694458
  val/full/ball_position_error_m: 7.030832767486572
  val/full/player_angular_error_deg: 45.82729721069336
  val/no_rgb/player_position_error_m: 2.823474168777466
  val/no_rgb/ball_position_error_m: 7.0195231437683105
  val/no_rgb/player_angular_error_deg: 47.43182373046875
  val/detector_gap/player_position_error_m: 2.880582571029663
  val/detector_gap/ball_position_error_m: 7.026538848876953
  val/detector_gap/player_angular_error_deg: 46.047821044921875
  val/rgb_only/player_position_error_m: 3.363882303237915
  val/rgb_only/ball_position_error_m: 7.018841743469238
  val/rgb_only/player_angular_error_deg: 45.55490493774414
  test/full/player_position_error_m: 2.863278388977051
  test/full/ball_position_error_m: 5.5149245262146
  test/full/player_angular_error_deg: 47.48822784423828
  test/no_rgb/player_position_error_m: 2.9235401153564453
  test/no_rgb/ball_position_error_m: 5.56441068649292
  test/no_rgb/player_angular_error_deg: 50.6858024597168
  test/detector_gap/player_position_error_m: 3.190944194793701
  test/detector_gap/ball_position_error_m: 5.515984058380127
  test/detector_gap/player_angular_error_deg: 50.309234619140625
  test/rgb_only/player_position_error_m: 4.028789043426514
  test/rgb_only/ball_position_error_m: 5.528722286224365
  test/rgb_only/player_angular_error_deg: 56.186466217041016
artifacts:
  run_dir: knowledge/runs/run-slcs-rgb-pilot-augmented-selected-conditions-v2
  log: knowledge/runs/run-slcs-rgb-pilot-augmented-selected-conditions-v2/evaluation.log
  output_dir: /home/kamimura/projects/tennis-lab/outputs/slcs/analyze/real_rgb_pilot_augmented/s42-002
  predictions: knowledge/runs/run-slcs-rgb-pilot-augmented-selected-conditions-v2/test_full/eval_arrays.npz
parents:
- run-slcs-rgb-pilot-augmented-e60-v2
relations:
- to: run-slcs-rgb-pilot-baseline-selected-conditions-v2
  rel: compares
tags:
- slcs
- real-rgb
- pilot
- ablation
- pseudo-teacher
repro:
  command: CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONPATH=.
    .venv/bin/python knowledge/runs/run-slcs-rgb-pilot-augmented-selected-conditions-v2/reproduce.py
    augmented
  commit: 44e529721933b08f5703526f554a155247b3df6c
  checkpoint_sha256: b925fc2a8cc2eb3dc80aacb7e2e6f8389d40ab48f8afb3c74408b97cbd17d30c
---

## 考察 / Findings

### 要約
validation選定の同一checkpointをval/test×4入力条件でCPU FP32評価した。選手はRGB・検出入力を利用しているが、ballは大誤差で予測分散が極小。独立実測3D精度ではなく擬似教師との一致度である。

### アーキテクチャ詳細
選定epoch55（0-based）、SHA256 `b925fc2a8cc2eb3dc80aacb7e2e6f8389d40ab48f8afb3c74408b97cbd17d30c`。selection.jsonにretained top3全候補とvalidation値を記録し、test値は選定に使用しない。trainのdata/window/token設定をそのまま使い、min_window_label_ratio=0.5を明示、batch4・strict checkpoint load・CPU FP32。full/no_rgb/detector_gap/rgb_onlyは入力だけを変更。動画prefix video_=Meiji、それ以外broadcastという明示規則でdomain集計した。

### メトリクスの解釈
以下は選定モデルのtestで、training runに登録したlastモデルのtestとは別。testはeastbourneの2 windowのみ、Meiji testは未収録。valは43 window。mask付き非加重平均で、教師weightは合計・平均を別記しheadlineには乗じない。

| 条件 | player m | ball m | yaw deg |
|---|---:|---:|---:|
| full | 2.8633 | 5.5149 | 47.488 |
| no_rgb | 2.9235 | 5.5644 | 50.686 |
| detector_gap | 3.1909 | 5.5160 | 50.309 |
| rgb_only | 4.0288 | 5.5287 | 56.186 |

各val/testの全体・video・domain値、valid counts/weights、fullとの差はcomparison JSON/CSVに保存。CPU FP32 valとBF16保存monitorの小差を精度改善とは解釈しない。

### アーキテクチャ⇄メトリクスの因果考察
test ball予測std XYZ=[0.011515014804899693, 0.008649986237287521, 0.007399260066449642]m、教師std=[2.462275743484497, 4.57837438583374, 0.4464415907859802]m、std norm比=0.00310（val比=0.01831）。full対no_rgb予測差RMS3D=0.1556m、対gap=0.0167m。ほぼ定数に近い低分散を定量化したが、厳密な定数とは断言しない。終端train ball=6.586mにも誤差が残るため、未学習・縮退の仮説を優先検証する。ball_diagnostics.jsonにclip別誤差、軸mean/std、入力差RMS、TB終盤5点を保存した。

### 既存実験との比較
モデル内はSHAと全IDs/window/frame/target/mask/weightの完全一致を要求。baseline/augmented間は異なるSHAを明示的に許容する別scriptで同一配列対応を完全照合した。augmentationはtest full playerを2.8213→2.8633mへ微悪化させる一方、no_rgbを3.6113→2.9235m、gapを3.5483→3.1909m、rgb_onlyを5.1445→4.0288mへ改善した。通常精度・入力依存・欠損耐性は分けて解釈する。単一seed・小規模pilotで因果効果や広い汎化を断定しない。

### 次に有効な実験
ballを小規模trainで過学習可能か確かめ、観測と教師の対応、ball lossの勾配・重み、単純定数予測との差を調べる。full Meijiの学習でもball予測分散とtrain誤差を早期監視し、欠損耐性の比較は同じteacher maskで継続する。Meiji testと独立実測ラベルなしに最終性能と主張しない。
