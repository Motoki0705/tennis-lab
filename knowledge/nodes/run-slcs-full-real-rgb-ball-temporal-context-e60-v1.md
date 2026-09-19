---
id: run-slcs-full-real-rgb-ball-temporal-context-e60-v1
type: run
title: 'SLCS観測ballの時間的feature context: 60epoch完走・validation選定epoch49'
provider: codex
session: 01a0b71f-c516-75b2-ac58-0c92e8a4f098
date: '2026-09-19'
status: done
config:
  model: SLCSFusionModel hidden128/shared4, missing_ball_temporal_context=true
  loss: baseline ball jerk=0, velocity weight=0
  data: slcs/real_rgb_v1, seed42, burst24, fixed recording-disjoint split
  epochs: 60
  test_after_fit: false
metrics:
  completed_epochs: 60
  global_step: 1800
  selected_epoch_zero_based: 49
  selected_val_scene_monitor_m: 1.9491939544677734
  terminal_train_ball_position_error_m: 2.4886646270751953
  terminal_train_player_position_error_m: 1.323377251625061
  terminal_val_ball_position_error_m: 2.4843881130218506
  terminal_val_player_position_error_m: 1.4387983083724976
  terminal_val_scene_position_error_m: 1.9615931510925293
  temporal_projection_nonzero_weights: 16384
  temporal_projection_norm: 2.3306963443756104
repro:
  commit: 5ab9afc7da2e9cebcd461b806e0676037d97622c
  branch: codex/slcs-real-rgb
  remote: git@github.com:Motoki0705/tennis-lab.git
  command: env PYTHONUNBUFFERED=1 PYTHONFAULTHANDLER=1 CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4
    MKL_NUM_THREADS=4 .venv/bin/python -m src.tasks.slcs.scripts.train --config-name
    train_real_rgb_ball_temporal_context paths.checkpoint_root=/home/kamimura/projects/tennis-lab/outputs
    paths.data_root=/home/kamimura/projects/tennis-lab/data paths.output_root=/home/kamimura/projects/tennis-lab/outputs
    run.output_dir=slcs/train/real_rgb_ball_temporal_context/s42-001
artifacts:
  run_dir: knowledge/runs/run-slcs-full-real-rgb-ball-temporal-context-e60-v1
  output_dir: outputs/slcs/train/real_rgb_ball_temporal_context/s42-001
  summary: knowledge/runs/run-slcs-full-real-rgb-ball-temporal-context-e60-v1/summary.json
  log: knowledge/runs/run-slcs-full-real-rgb-ball-temporal-context-e60-v1/queue.log
  curves: knowledge/runs/run-slcs-full-real-rgb-ball-temporal-context-e60-v1/curves.png
  tb_logdir: outputs/slcs/train/real_rgb_ball_temporal_context/s42-001/logs/version_0
parents:
- run-slcs-full-real-rgb-no-ball-smooth-e60-v3
- run-slcs-full-real-rgb-missing-ball-court-val-v1
relations:
- to: run-slcs-full-real-rgb-missing-ball-court-e60-v1
  rel: compares
tags:
- slcs
- real-rgb
- missing-ball
- temporal-feature-context
- training-complete
---

## 考察 / Findings

### 要約

共有GPU queueのall予約で60epoch・1800更新を完走した。last checkpointはepoch59/global_step1800、
全浮動stateが有限、主要7種類のepoch系列が60点・最終step1799であることを確認した。
追加した128×128射影の全16384重みが更新された。5条件validationと欠損境界の評価前に採用とは判断しない。

### アーキテクチャ詳細

基準no-ball-smoothからの単独入力施策。元のcourt+ball embeddingの観測tokenを、同じoffline windowの最近の左右anchor間で線形合成する。
ゼロ初期化・biasなし射影を通し、両側anchorを持つ欠損実frameのinvisible tokenにだけ加える。
元のvisibility・UV・教師・loss weightを変えず、片側anchor・全欠損・paddingの追加contextは0。
court残差とvelocity lossは無効、hidden128/shared4・DINO downsample2・seed42・burst24・batch16・60epochを基準と固定した。
resume/initなしのfresh学習で、test_after_fit=false。
[TrackNetV3](https://people.cs.nycu.edu.tw/~yushuen/data/TrackNetV3.pdf) §3.3を参考にした入力側の転用仮説で、原論文の学習補完器の再現ではない。
§4.5の単純な出力線形補間の限界は残り、特徴補間は物理UV/3D補間ではない。

### メトリクスの解釈

終端train ball2.4887m/player1.3234m、val ball2.4844m/player1.4388m。
epoch49のval scene monitor1.9491939545mが保存候補epoch52/56より小さく選定された。
選定checkpoint SHA256は`94793e761d22a1f163d8089ae0210b6cb45f7c37144deb498c31b4d85fd6f961`。
終端epoch59の値と選定epoch49の固定条件評価を混同しない。負のlossはNLLを含む目的関数による。
curves.pngは保存TensorBoardのvalidation lossとball accuracyから生成する。
学習済み射影norm2.3307は入力経路が更新された証拠で、性能改善の証明ではない。

### アーキテクチャ⇄メトリクスの因果考察

一定の不可視tokenだけよりも、欠損前後の観測済み幾何特徴を使いやすくする狙いである。
打撃・バウンド・長い欠損を線形特徴で十分表現できるとは限らず、複雑な軌道では誤誘導し得る。
validation入力maskの事前集計では、教師有効な自然欠損2269 occurrence中1778に両側anchorがある。
自然の観測→欠損329ペア中312、欠損→観測338ペア中317が追加contextの対象で、それ以外の境界はこの経路だけでは解決しない。

### 既存実験との比較

基準のbest validation scene monitor1.9568mに対して今回1.9492mと小幅に低下したが、単一monitorだけを採否根拠にしない。
前のcourt-only contextは位置と境界の退行により不採用だった。今回それを重ねず、元の基準から比較する。
基準保存configとの差は時間的context、有効時にしか使わないvelocity既定値の明示、保存root/出力先、自動終端test無効化である。
学習開始sourceは5ab9afc7。途中にCIのcomputation-only規約へ合わせてforwardの重複形状検証だけを除去した6dbb1c19を統合したが、実行中processは変更していない。
constructorとforward計算部分のASTは同一と確認し、有効入力の計算・gradient・mask・パラメータ構造は変えていない。

### 次に有効な実験

選定epoch49を固定し、full/no_rgb/detector_gap/rgb_only/detector_gap_no_rgbの343 validation窓を比較する。
既存train-only高速閾値26.4250385982m/s、両visibility境界、位置平均・p95、domain別・playerを確認する。
曲線や最大速度低下だけで頑健性を主張せず、validationの選定を閉じてからheld-out testへ進む。
