---
task: slcs
sequence: 51
recorded_at: 2026-09-18
date_source: experiment_date
papers: []
id: run-slcs-rgb-pilot-baseline-selected-conditions-v2
type: run
title: 'SLCS baseline: validation選定重みのCPU四条件評価'
provider: codex
date: '2026-09-18'
status: done
config:
  model: validation-selected baseline, epoch=47 (0-based)
  loss: evaluation only, masked unweighted headline metrics
  data: slcs/real_rgb_pilot_v2, min_window_label_ratio=0.5, train window/token settings
  device: cpu float32
  batch_size: 4
metrics:
  val/full/player_position_error_m: 2.571880578994751
  val/full/ball_position_error_m: 7.02938175201416
  val/full/player_angular_error_deg: 43.872947692871094
  val/no_rgb/player_position_error_m: 2.873530387878418
  val/no_rgb/ball_position_error_m: 7.013044357299805
  val/no_rgb/player_angular_error_deg: 56.573673248291016
  val/detector_gap/player_position_error_m: 3.0093183517456055
  val/detector_gap/ball_position_error_m: 7.022319793701172
  val/detector_gap/player_angular_error_deg: 49.24721908569336
  val/rgb_only/player_position_error_m: 4.152010917663574
  val/rgb_only/ball_position_error_m: 7.021568298339844
  val/rgb_only/player_angular_error_deg: 50.54702377319336
  test/full/player_position_error_m: 2.821348190307617
  test/full/ball_position_error_m: 5.586698055267334
  test/full/player_angular_error_deg: 38.81786346435547
  test/no_rgb/player_position_error_m: 3.611276388168335
  test/no_rgb/ball_position_error_m: 5.62885856628418
  test/no_rgb/player_angular_error_deg: 84.11768341064453
  test/detector_gap/player_position_error_m: 3.548280715942383
  test/detector_gap/ball_position_error_m: 5.580329895019531
  test/detector_gap/player_angular_error_deg: 61.319889068603516
  test/rgb_only/player_position_error_m: 5.144540786743164
  test/rgb_only/ball_position_error_m: 5.609474182128906
  test/rgb_only/player_angular_error_deg: 69.09800720214844
artifacts:
  run_dir: knowledge/runs/run-slcs-rgb-pilot-baseline-selected-conditions-v2
  log: knowledge/runs/run-slcs-rgb-pilot-baseline-selected-conditions-v2/evaluation.log
  output_dir: /home/kamimura/projects/tennis-lab/outputs/slcs/analyze/real_rgb_pilot_baseline/s42-002
  predictions: knowledge/runs/run-slcs-rgb-pilot-baseline-selected-conditions-v2/test_full/eval_arrays.npz
parents:
- run-slcs-rgb-pilot-baseline-e60-v2
relations: []
tags:
- slcs
- real-rgb
- pilot
- ablation
- pseudo-teacher
repro:
  command: CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONPATH=.
    .venv/bin/python knowledge/runs/run-slcs-rgb-pilot-baseline-selected-conditions-v2/reproduce.py
    baseline
  commit: 44e529721933b08f5703526f554a155247b3df6c
  checkpoint_sha256: 318eef5423c4183d894c8c032e7a85b3d8f83ca3cf2b0909aabbadd344fe1c73
---

## 考察 / Findings

### 要約
validation選定の同一checkpointをval/test×4入力条件でCPU FP32評価した。選手はRGB・検出入力を利用しているが、ballは大誤差で予測分散が極小。独立実測3D精度ではなく擬似教師との一致度である。

### アーキテクチャ詳細
選定epoch47（0-based）、SHA256 `318eef5423c4183d894c8c032e7a85b3d8f83ca3cf2b0909aabbadd344fe1c73`。selection.jsonにretained top3全候補とvalidation値を記録し、test値は選定に使用しない。trainのdata/window/token設定をそのまま使い、min_window_label_ratio=0.5を明示、batch4・strict checkpoint load・CPU FP32。full/no_rgb/detector_gap/rgb_onlyは入力だけを変更。動画prefix video_=Meiji、それ以外broadcastという明示規則でdomain集計した。

### メトリクスの解釈
以下は選定モデルのtestで、training runに登録したlastモデルのtestとは別。testはeastbourneの2 windowのみ、Meiji testは未収録。valは43 window。mask付き非加重平均で、教師weightは合計・平均を別記しheadlineには乗じない。

| 条件 | player m | ball m | yaw deg |
|---|---:|---:|---:|
| full | 2.8213 | 5.5867 | 38.818 |
| no_rgb | 3.6113 | 5.6289 | 84.118 |
| detector_gap | 3.5483 | 5.5803 | 61.320 |
| rgb_only | 5.1445 | 5.6095 | 69.098 |

各val/testの全体・video・domain値、valid counts/weights、fullとの差はcomparison JSON/CSVに保存。CPU FP32 valとBF16保存monitorの小差を精度改善とは解釈しない。

### アーキテクチャ⇄メトリクスの因果考察
test ball予測std XYZ=[0.027336621657013893, 0.02193128690123558, 0.019509369507431984]m、教師std=[2.462275743484497, 4.57837438583374, 0.4464415907859802]m、std norm比=0.00769（val比=0.01779）。full対no_rgb予測差RMS3D=0.1435m、対gap=0.0488m。ほぼ定数に近い低分散を定量化したが、厳密な定数とは断言しない。終端train ball=6.572mにも誤差が残るため、未学習・縮退の仮説を優先検証する。ball_diagnostics.jsonにclip別誤差、軸mean/std、入力差RMS、TB終盤5点を保存した。

### 既存実験との比較
モデル内はSHAと全IDs/window/frame/target/mask/weightの完全一致を要求。baseline/augmented間は異なるSHAを明示的に許容する別scriptで同一配列対応を完全照合した。augmentationはtest full playerを2.8213→2.8633mへ微悪化させる一方、no_rgbを3.6113→2.9235m、gapを3.5483→3.1909m、rgb_onlyを5.1445→4.0288mへ改善した。通常精度・入力依存・欠損耐性は分けて解釈する。単一seed・小規模pilotで因果効果や広い汎化を断定しない。

### 次に有効な実験
ballを小規模trainで過学習可能か確かめ、観測と教師の対応、ball lossの勾配・重み、単純定数予測との差を調べる。full Meijiの学習でもball予測分散とtrain誤差を早期監視し、欠損耐性の比較は同じteacher maskで継続する。Meiji testと独立実測ラベルなしに最終性能と主張しない。
