---
id: run-slcs-full-real-rgb-velocity-e60-v1
type: run
title: 'SLCS教師速度整合: 固定係数で60epoch・1800更新を完走'
provider: codex
session: 01a0b71f-c516-75b2-ac58-0c92e8a4f098
date: '2026-09-19'
status: done
config:
  model: SLCSFusionModel hidden128/shared4/DINO-downsample2
  loss: ball jerk=0, supervised ball velocity
  data: slcs/real_rgb_v1, burst24
  ball_velocity_weight: 0.0011117380640846516
  ball_velocity_scale_mps: 11.259468485469933
  max_epochs: 60
  seed: 42
  test_after_fit: false
metrics:
  completed_epochs: 60
  global_step: 1800
  terminal_train_ball_position_error_m: 2.433317184448242
  terminal_train_player_position_error_m: 1.3221697807312012
  terminal_val_ball_position_error_m: 2.5645251274108887
  terminal_val_player_position_error_m: 1.4232771396636963
  terminal_val_scene_position_error_m: 1.9939011335372925
  best_val_scene_position_error_m: 1.9769423007965088
  selected_epoch_zero_based: 49
  terminal_train_ball_velocity_loss: 3.1056883335113525
  terminal_val_ball_velocity_loss: 0.1452355831861496
repro:
  commit: 34023014f38b3ba4d3c175f7b7e3917d57a00a93
  branch: codex/slcs-real-rgb
  remote: git@github.com:Motoki0705/tennis-lab.git
  command: env PYTHONUNBUFFERED=1 PYTHONFAULTHANDLER=1 CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4
    MKL_NUM_THREADS=4 .venv/bin/python -m src.tasks.slcs.scripts.train --config-name
    train_real_rgb_velocity paths.checkpoint_root=/home/kamimura/projects/tennis-lab/outputs
    run.output_dir=slcs/train/real_rgb_velocity/s42-001
artifacts:
  run_dir: knowledge/runs/run-slcs-full-real-rgb-velocity-e60-v1
  output_dir: /home/kamimura/projects/tennis-lab/outputs/slcs/train/real_rgb_velocity/s42-001/logs/version_0
  log: knowledge/runs/run-slcs-full-real-rgb-velocity-e60-v1/queue.log
  curves: knowledge/runs/run-slcs-full-real-rgb-velocity-e60-v1/curves.png
  tb_logdir: outputs/slcs/train/real_rgb_velocity/s42-001/logs/version_0
parents:
- run-slcs-ball-velocity-gradient-calibration-v1
- run-slcs-full-real-rgb-no-ball-smooth-e60-v3
relations:
- to: run-slcs-full-real-rgb-gap48-val-v1
  rel: compares
tags:
- slcs
- real-rgb
- velocity
- gpu
- train-only-calibration
---

## 考察 / Findings

### 要約

train-onlyで固定したscale/weightの速度整合項を追加し、ローカル共有queueのall予約で60epoch・1800更新を中断なく完走した。
終端checkpointのepoch59/global_step1800と、validation scene最良epoch49を確認した。
基準置換の可否は別runの同一val5条件・motion診断で判断する。本runではtestを実行していない。

### アーキテクチャ詳細

`train_real_rgb_velocity`はno-ball-smooth・burst24のcontrolへ教師速度とのSmooth L1整合だけを追加する。
連続frame・両端valid・非padding・実timestamp・両端confidenceの最小値を使い、入力欠損側の教師も保持する。
model、教師版、split、augmentation、optimizer、seed42、60epoch/1800更新はcontrolと同じ。
保存configの差分も読み取り専用で照合した。速度設定・出力先のほか、今回は自動testを無効化し、
入力checkpoint rootをoutputsへ明示したが、resume/init_weightsを使わないfresh学習なので入力重みの差はない。
controlの既存自動test記録を今回の係数やcheckpointの選択には使用していない。

### メトリクスの解釈

train ballは9.2049→2.4333m、player12.1163→1.3222m。terminal valはball2.5645m/player1.4233m。
追加した速度lossはtrain12.0877→3.1057、val0.2859→0.1452。これはscaleで無次元化したSmooth L1値で、速度誤差m/sそのものではない。
trainには入力欠損augmentationがあり、valとのloss差をそのまま過学習の証拠とは扱わない。
曲線は全60epochを対象とし、終盤のvalidationは頭打ちで、最良epochは終端ではなく49だった。

### アーキテクチャ⇄メトリクスの因果考察

追加lossを含む最適化は成立したが、初期ball出力勾配比10%は学習中のparameter勾配比を保証しない。
velocity lossの低下だけで検出境界のspike改善や、高速な教師運動の保存を断定しない。
教師そのものの3D精度・時間整合性も独立実測GTでは保証されていない。

### 既存実験との比較

controlのbest val scene1.956797mに対し、本runは1.976942mで小幅に悪い。
選定monitorだけで採否を決めず、full/gapのball平均・p95・velocity vector error・domain・playerを固定条件で比較する。
gap48はbroadcastの退行から採用せず、本runでは元のburst24へ戻している。

### 次に有効な実験

公開評価CLIでval5条件を比較し、保存配列のvisibility遷移とtrain-only p95閾値以上の高速教師区間を診断する。
追加lossの係数をこのvalに合わせて再調整せず、改善・退行の両方を記録して基準置換の可否を判断する。

後続の[初回val評価](run-slcs-full-real-rgb-velocity-val-interrupted-v1.md)は再起動後に空出力が見つかり、採用不可となった。
本学習のTensorBoard各epoch系列60点・末尾step1799は再確認でき、0バイトになっていた曲線画像はその記録から再生成した。
当時はホスト診断を優先してGPU実験を保留したが、後続のgoal優先指示で再開した。
採否は別runの[新しいval5条件評価](run-slcs-full-real-rgb-velocity-val-v2.md)に記録する。
