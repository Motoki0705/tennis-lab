---
id: run-slcs-pilot-augmented-portable-eval-v1
type: run
title: 'SLCS pilotの汎用run評価: 実FPSでほぼ静止したballを確認'
provider: codex
date: '2026-09-19'
status: done
session: 01a0b71f-c516-75b2-ac58-0c92e8a4f098
config:
  data: slcs/real_rgb_pilot_v2
  training_run: slcs/train/real_rgb_pilot_augmented/s42-002
  selected_epoch_zero_based: 55
  split: val
  device: cpu
  precision: float32
metrics:
  val_windows: 43
  full_player_position_error_m: 2.669083833694458
  full_ball_position_error_m: 7.030832767486572
  no_rgb_ball_position_error_m: 7.0195231437683105
  detector_gap_ball_position_error_m: 7.026538848876953
  rgb_only_ball_position_error_m: 7.018841743469238
  full_ball_predicted_mean_speed_mps: 0.05456269424680978
  full_ball_target_mean_speed_mps: 13.70674498997783
  full_ball_std_norm_ratio: 0.018313855601182986
repro:
  commit: fccebe1e
  branch: codex/slcs-real-rgb
  command: >-
    env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
    .venv/bin/python -m scripts.analysis.evaluate_slcs_run
    --output-root /home/kamimura/projects/tennis-lab/outputs
    --training-run slcs/train/real_rgb_pilot_augmented/s42-002
    --output slcs/evaluate/real_rgb_pilot_augmented_selected/s42-takeover-001
    --domain-prefix video_=meiji --default-domain broadcast --device cpu --batch-size 4
artifacts:
  run_dir: knowledge/runs/run-slcs-pilot-augmented-portable-eval-v1
  output_dir: outputs/slcs/evaluate/real_rgb_pilot_augmented_selected/s42-takeover-001
parents: [run-slcs-rgb-pilot-augmented-e60-v2]
relations:
- {to: run-slcs-rgb-pilot-augmented-selected-conditions-v2, rel: confirms}
tags: [slcs, pilot, evaluation, motion, ball-collapse, cpu]
---

## 考察 / Findings

### 要約

汎用CLIで実checkpointを選定し、val 43window・4条件の評価が成功した。
ballは誤差だけでなく実FPSでの平均速度でもほぼ静止している。学習済みモデルの採用根拠にはならない。

### アーキテクチャ詳細

保存training configのdata/window/token/qualityを保持し、augmentationだけ無効にした。
last checkpoint内のretained validation scoreからepoch55を選び、testもmtimeも選定には使わない。
Meiji 59.94006FPS、broadcast 30FPSを各clip manifestから取得し、有効な連続frameのみで微分した。
これは旧7clip pilotの評価であり、生成中のMeiji v9全体教師の学習ではない。

### メトリクスの解釈

full ball誤差7.0308m、予測平均速度0.0546m/sに対し教師13.7067m/s。
全windowの標準偏差ノルム比0.0183はclip間の中心差を含み、video別ではMeiji 0.00290、
broadcast 0.00348とさらに小さい。4条件でball誤差は7.0188〜7.0308mに留まる。
新規学習がないためこのrun独自の収束曲線は対象外。教師一致度であり実測3D精度ではない。

### アーキテクチャ⇄メトリクスの因果考察

入力欠損への誤差変化の小ささは頑健性ではなく、入力に追従しない予測でも説明できる。
位置分散と実速度はこの解釈を支持するが、正則化・勾配競合など個別原因は未確定。

### 既存実験との比較

以前のselected条件評価とfull val誤差が一致し、新しいCLIの実データ動作を確認した。
train-mean baseline（別runでval約7.0333m）との差が小さいという既存結論を維持する。

### 次に有効な実験

Meiji全体教師の監査後、予定済みのball smoothnessだけを無効にする60epoch比較を行う。
同じval maskの平均値baseline・軌跡分散・速度・player/yawを併用して判断し、testは選定に使わない。
