---
id: run-slcs-full-real-rgb-no-ball-smooth-val-v3
type: run
title: 'SLCS全体版validation 4条件: ball定数崩壊を脱するが欠損・裾誤差は残る'
provider: codex
session: 01a0b71f-c516-75b2-ac58-0c92e8a4f098
date: '2026-09-19'
status: done
config:
  model: SLCSFusionModel hidden128/shared4/DINO-downsample2
  loss: inference only
  data: slcs/real_rgb_v1
  selected_epoch_zero_based: 56
  split: val
  device: cuda
  precision: float32
metrics:
  val_windows: 343
  ball_valid_window_occurrences: 34238
  full_player_position_error_m: 1.3826085329055786
  full_ball_position_error_m: 2.521042585372925
  no_rgb_ball_position_error_m: 2.6812899112701416
  detector_gap_ball_position_error_m: 3.0972740650177
  rgb_only_ball_position_error_m: 7.753563404083252
  train_mean_ball_position_error_m: 7.692206743130705
  full_meiji_ball_position_error_m: 2.52386474609375
  full_broadcast_ball_position_error_m: 2.4293246269226074
  full_ball_position_error_p95_m: 7.693350160280706
  full_ball_predicted_mean_speed_mps: 11.675494300947951
  full_ball_target_mean_speed_mps: 12.78773650879646
  full_ball_velocity_error_mps: 8.75650502498033
  full_ball_std_norm_ratio: 0.8958946332164204
repro:
  commit: 61d94edb5c394de2cb7e905fce362da299bf73d9
  branch: codex/slcs-real-rgb
  remote: git@github.com:Motoki0705/tennis-lab.git
  command: env PYTHONUNBUFFERED=1 PYTHONFAULTHANDLER=1 CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4
    MKL_NUM_THREADS=4 .venv/bin/python -m scripts.analysis.evaluate_slcs_run --output-root
    /home/kamimura/projects/tennis-lab/outputs --training-run slcs/train/real_rgb_no_ball_smooth/s42-takeover-003
    --output slcs/evaluate/real_rgb_no_ball_smooth/s42-takeover-003 --domain-prefix
    video_=meiji --default-domain broadcast --device cuda --batch-size 4 --ball-train-mean
artifacts:
  run_dir: knowledge/runs/run-slcs-full-real-rgb-no-ball-smooth-val-v3
  output_dir: outputs/slcs/evaluate/real_rgb_no_ball_smooth/s42-takeover-003
  log: knowledge/runs/run-slcs-full-real-rgb-no-ball-smooth-val-v3/queue.log
parents: [run-slcs-full-real-rgb-no-ball-smooth-e60-v3]
relations:
- {to: run-slcs-pilot-no-ball-smooth-eval-v1, rel: compares}
tags: [slcs, real-rgb, validation, input-conditions, motion, train-mean]
---

## 考察 / Findings

### 要約

validation最良epoch56を固定して343窓の4入力条件を評価した。full ball誤差2.5210mは
train-only平均位置定数7.6922mを明確に下回り、旧pilotの定数に近い出力から前進した。
ただし検出欠損で3.0973m、RGB-onlyで7.7536mとなり、裾誤差と時間的な跳びも残る。

### アーキテクチャ詳細

公開CLIの`--ball-train-mean`を使用し、保存training configのdata/quality/windowを維持してaugmentationだけ無効化。
GPU float32、batch4、同じcheckpoint・教師・mask・weight・frameの4条件を厳密に照合した。
checkpoint SHAは`870407e89ca0ff31d07a33395acf3a3538f49146274f3566bd36ff969b6733a2`。
選定monitorはval scene位置誤差のみ。testは本runで評価していない。
平均位置定数はtrain466窓の重複を除いた25913正weight camera-frameからfitした。

### メトリクスの解釈

| 条件 | player位置誤差m | ball位置誤差m |
|---|---:|---:|
| full | 1.3826 | 2.5210 |
| no_rgb | 1.5170 | 2.6813 |
| detector_gap | 1.4631 | 3.0973 |
| rgb_only | 1.8890 | 7.7536 |

Meiji333窓のball full/no_rgbは2.5239/2.6415m、broadcast10窓は2.4293/3.9750m。
broadcastの教師weightはball平均0.15、player平均0.3585であり、同じ実測3D品質の評価とは扱わない。
full ball予測速度平均11.6755m/s（教師12.7877）、分散比0.8959だが、速度誤差平均8.7565m/s、
位置p95=7.6934m、最大予測速度481.77m/s（教師最大63.94）と大きな失敗が残る。
headlinesは有効window occurrencesの非加重平均で、定数fitのconfidence重みと区別する。
本runは学習ではなく、収束曲線は対象外。全配列・設定・motion・比較をevaluation/へ保存した。

### アーキテクチャ⇄メトリクスの因果考察

RGB除去時に位置精度が低下し、特にbroadcastでは融合の有用性を示す観測が得られた。
ただし単一checkpointへの入力除去であり、RGBなし再学習との差や因果効果の推定ではない。
RGB-onlyでは位置変動があっても定数誤差を下回れず、低解像度・疎な特徴と学習配分のどちらが原因か未確定。
時間的な外れ値の残存は、平均位置誤差・分散比だけでは頑健性を判断できないことを示す。

### 既存実験との比較

旧pilotのvalとはclip構成・教師版が異なるため、その7.0030mとの比率を同条件の改善率と呼ばない。
今回の定数baselineは同じ全体版val・同じ有効maskなので直接比較できる。
fullは定数より5.1712m良いが、RGB-onlyは約0.0614m悪い。

### 次に有効な実験

Meiji/broadcastの固定val clipを既存predict_clipで可視化し、欠損区間と時間的スパイクを確認する。
現行augmentationと評価の欠損長・消去対象の違いを調べ、論文に基づく単一変更の60epoch比較を優先する。
full・gapのplayer/ball誤差、裾、motion、RGB有無を同時に確認し、RGB-onlyの誤差だけで採用しない。
