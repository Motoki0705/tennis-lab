---
id: run-slcs-full-real-rgb-gap48-interrupted-v1
type: run
title: 'SLCS全体版gap48: 12epoch・360更新で外部signal停止、比較未完了'
provider: codex
session: 01a0b71f-c516-75b2-ac58-0c92e8a4f098
date: '2026-09-19'
status: failed
config:
  model: SLCSFusionModel hidden128/shared4/DINO-downsample2
  loss: ball_position_smoothness_weight=0
  data: slcs/real_rgb_v1
  burst_max_frames: 48
  max_epochs: 60
  test_after_fit: false
metrics:
  exit_code: 143
  completed_epochs: 12
  checkpoint_global_step: 360
  last_val_player_position_error_m: 2.0682528018951416
  last_val_ball_position_error_m: 7.64935827255249
repro:
  commit: 0e1e32c19841f3a594f9e1c0d7b812ab54e8df5f
  branch: codex/slcs-real-rgb
  remote: git@github.com:Motoki0705/tennis-lab.git
  command: env PYTHONUNBUFFERED=1 PYTHONFAULTHANDLER=1 CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4
    MKL_NUM_THREADS=4 .venv/bin/python -m src.tasks.slcs.scripts.train --config-name
    train_real_rgb_gap48 run.test_after_fit=false run.output_dir=slcs/train/real_rgb_gap48/s42-takeover-001
artifacts:
  run_dir: knowledge/runs/run-slcs-full-real-rgb-gap48-interrupted-v1
  output_dir: /home/kamimura/projects/tennis-lab/outputs/slcs/train/real_rgb_gap48/s42-takeover-001/logs/version_0
  log: knowledge/runs/run-slcs-full-real-rgb-gap48-interrupted-v1/queue.log
  curves: knowledge/runs/run-slcs-full-real-rgb-gap48-interrupted-v1/curves.png
  tb_logdir: outputs/slcs/train/real_rgb_gap48/s42-takeover-001/logs/version_0
parents:
- run-slcs-full-real-rgb-no-ball-smooth-val-v3
relations: []
tags:
- slcs
- real-rgb
- detector-gap
- interrupted
- gpu
---

## 考察 / Findings

### 要約

連続欠損の最大長だけ24→48にした60epoch比較は、12epoch・360更新でsignal停止した。
workerは2026-09-19 13:02:20 JST開始、13:06:24にFAILED(signal)、logはexit_code=143を記録した。
完了した60epoch baselineとの採否比較はまだ行えない。

### アーキテクチャ詳細

`train_real_rgb_gap48`。seed42・全体版61clip・model・loss・学習予算はno-smooth baselineと同じ。
追加探索のため自動終端testだけを無効化した。ModDropの欠損モダリティ学習を参考にしたrepo固有の仮説であり、
48frameは論文の推奨値ではない。根拠: https://arxiv.org/abs/1501.00102 。
保存configとqueue bundleに実行設定・source commitを保持した。

### メトリクスの解釈

TensorBoardにはtrain/val各12epoch、最終step359。last.ckptはepoch11/global_step360で
optimizer stateとLR scheduler stateを各1個保持している。読み取り時SHA256は
`016cf87b2181ade355b75aa82faf68e499f004f27c2400f80622fcc3dc11a14f`。
最終val ball7.6494m/player2.0683mは学習途中の値で、60epoch施策の最終性能ではない。
曲線は中断時点までの推移だけを示す。

### アーキテクチャ⇄メトリクスの因果考察

143はSIGTERM相当の終了を示すが、signal送信者・原因は未確定。
確認時の環境起動時刻は13:18:19 JSTで、worker停止の約12分後だった。
したがって後の環境再起動を13:06の学習停止原因と断定しない。OOMやモデル不具合の証拠もない。

### 既存実験との比較

no-smooth baselineは60epoch完走済みで、今回の12epoch数値をその最終値へ直接比較しない。
最大長以外の学習設定を保持するprofile同値テストは通過している。

### 次に有効な実験

環境方針についてユーザーへローカル継続/Colab切替を確認した。
ローカル継続なら保存checkpointの全状態から目標60epochまで再開する。中断前の証拠は保全する。
複数logger versionのlast.ckptがある場合は、評価CLIへ明示的なlast指定を追加し、mtimeによる推測を避ける。
