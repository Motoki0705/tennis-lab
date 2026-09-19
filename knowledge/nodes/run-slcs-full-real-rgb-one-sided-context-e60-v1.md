---
id: run-slcs-full-real-rgb-one-sided-context-e60-v1
type: run
title: 'SLCS片側観測context: 60epoch完走・validation選定epoch56'
provider: codex
session: 01a0b71f-c516-75b2-ac58-0c92e8a4f098
date: '2026-09-19'
status: done
config:
  model: SLCSFusionModel hidden128/shared4, temporal=true, one_sided=true
  loss: direct temporal control unchanged, ball jerk=0, velocity weight=0
  data: slcs/real_rgb_v1, seed42, burst24, unbalanced train shuffle
  epochs: 60
  test_after_fit: false
metrics:
  completed_epochs: 60
  global_step: 1800
  selected_epoch_zero_based: 56
  selected_val_scene_monitor_m: 1.963842749595642
  selected_training_val_ball_position_error_m: 2.5395989418029785
  selected_training_val_player_position_error_m: 1.3880865573883057
  terminal_train_ball_position_error_m: 2.409202814102173
  terminal_train_player_position_error_m: 1.3112317323684692
  terminal_val_ball_position_error_m: 2.5437731742858887
  terminal_val_player_position_error_m: 1.4007446765899658
repro:
  commit: f8f83a6c457c153640dcba04ed86722cbdb794f9
  branch: codex/slcs-real-rgb
  remote: git@github.com:Motoki0705/tennis-lab.git
  command: env PYTHONUNBUFFERED=1 PYTHONFAULTHANDLER=1 CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4
    MKL_NUM_THREADS=4 .venv/bin/python -m src.tasks.slcs.scripts.train --config-name
    train_real_rgb_one_sided_context paths.checkpoint_root=/home/kamimura/projects/tennis-lab/outputs
    paths.data_root=/home/kamimura/projects/tennis-lab/data paths.output_root=/home/kamimura/projects/tennis-lab/outputs
    run.output_dir=slcs/train/real_rgb_one_sided_context/s42-001
artifacts:
  run_dir: knowledge/runs/run-slcs-full-real-rgb-one-sided-context-e60-v1
  output_dir: outputs/slcs/train/real_rgb_one_sided_context/s42-001
  summary: knowledge/runs/run-slcs-full-real-rgb-one-sided-context-e60-v1/summary.json
  preflight: knowledge/runs/run-slcs-full-real-rgb-one-sided-context-e60-v1/preflight.json
  log: knowledge/runs/run-slcs-full-real-rgb-one-sided-context-e60-v1/queue.log
  tb_logdir: outputs/slcs/train/real_rgb_one_sided_context/s42-001/logs/version_0
  curves: knowledge/runs/run-slcs-full-real-rgb-one-sided-context-e60-v1/curves.png
parents:
- run-slcs-full-real-rgb-ball-temporal-context-e60-v1
- run-slcs-full-real-rgb-temporal-domain-balanced-val-v1
relations:
- to: run-slcs-full-real-rgb-ball-temporal-context-val-v1
  rel: compares
tags:
- slcs
- real-rgb
- one-sided-ball-context
- training-complete
---

## 考察 / Findings

### 要約

片側だけ観測anchorがあるball欠損へfeature contextを追加したfresh学習を、共有queueのall予約で60epoch・1800更新まで完走した。
validation scene monitor最小のepoch56を選び、SHA `55b8795d3d2003acd04c9b09faed99453454f8d64298d95d3b42f90a0afd4120` に固定した。
学習中の選定monitor1.96384mは直接対照TemporalContextの1.94919mより悪く、ここから改善・採用とは判断しない。独立した5条件validation診断を次に行う。

### アーキテクチャ詳細

直接対照は非domain-balancedのTemporalContext。旧保存設定の明示的な無効値移行を適用すると、保存config差は
`model.missing_ball_one_sided_context` と出力先だけだった。loss・教師・augmentation・train shuffle・60epoch・seed42は維持する。
元の観測court+ball embedding、左/右flag、最近傍観測までのframe距離のlog1pをbiasなし射影し、片側欠損実frameへだけ直接加算する。
両側補間は変更せず、観測frame・両側anchorあり・全欠損・paddingへの直接残差は0。未来側も許すoffline処理で、出力補間や速度clampではない。
GRU-Dのmask/経過時間とBRITSの双方向観測contextからの転用仮説であり、論文再現ではない。設計と原典はSLCS READMEを参照。

### メトリクスの解釈

lastはepoch59/global_step1800、主要7 TensorBoard系列は各60点・最終step1799で有限だった。
last/選定checkpointのmodel・optimizerを含む浮動値を監査し、非有限値は非監視ModelCheckpointの未使用kth_value=+infだけだった。
選定時の学習内val ball2.53960m/player1.38809mと、終端val ball2.54377m/player1.40074mは異なる時点の値である。
これらはbf16学習内の指標で、後続float32条件別評価値とは区別する。負の総lossはlearned-scale Laplace NLLを含むため、位置誤差や学習破綻と同一視しない。
追加射影128×131の16,768要素はすべて有限・非ゼロで、L2 normは2.34950。既存temporal射影も有限・非ゼロだった。
集計・保存config差・checkpoint選定・SHA・監査commandはbundleへ保存する。testは実行していない。

### アーキテクチャ⇄メトリクスの因果考察

学習前の実データCPU検証で、共有parameter・RNG・6出力tensorは対照と初期状態で完全一致し、追加射影への有限な勾配L1=1.48013を確認した。
自然欠損のtrain 55,920 frame occurrence中、片側欠損は2,953（Meiji2,936、broadcast17）で114windowに存在した。
観測までの距離は全体中央値22/p95 86/max119frame。長距離でanchorが古くなるリスクがあり、片側経路を追加しただけで連続性を保証しない。
事前検証はaugmentation/cacheを明示的に無効化した適用範囲・初期同値確認であり、本学習のaugmentation有効・cache有効とは区別する。
単一seedの指標差から機構の因果や一般的な頑健性を断定しない。

### 既存実験との比較

TemporalContextと同じ466 train窓・30batch/epoch・1800更新を維持した。自然の片側欠損が多いMeijiと少ないbroadcastを全体平均だけでまとめない。
DomainBalancedの巨大な境界不連続を契機にした仮説だが、このrunはsamplingを継承していない。
元のno-smooth基準とは両側・片側contextの2変更があるため、片側追加の直接因果比較はTemporalContextとの比較に限定する。
source `f8f83a6c` はcleanで開始した。後から統合したanchor分類診断はこの学習sourceには含まれない。

### 次に有効な実験

epoch56のSHAを固定し、同じ343 validation窓のfull/no_rgb/rgb_only/detector_gap/detector_gap_no_rgbを評価する。
TemporalContextおよび元の基準とteacher/mask/window/FPS/観測maskを照合し、位置平均・p95・domain別・高速区間・欠損境界の速度誤差を比較する。
追加CPU診断では左だけ/右だけ/両側/観測なしと、事前固定距離bucket all/1/2–8/9–24/25以上を使う。testを選定に使わず、外れ値を除去しない。
