---
task: slcs
sequence: 67
recorded_at: 2026-09-19
date_source: experiment_date
papers: []
id: run-slcs-full-real-rgb-gap48-e60-resume-v2
type: run
title: 'SLCS全体版gap48: 全状態再開で60epoch・1800更新を完走'
provider: codex
session: 01a0b71f-c516-75b2-ac58-0c92e8a4f098
date: '2026-09-19'
status: done
config:
  model: SLCSFusionModel hidden128/shared4/DINO-downsample2
  loss: ball_position_smoothness_weight=0
  data: slcs/real_rgb_v1
  burst_max_frames: 48
  max_epochs: 60
  resumed_from_epoch_zero_based: 11
  test_after_fit: false
metrics:
  completed_epochs: 60
  resumed_epochs: 48
  global_step: 1800
  terminal_train_ball_position_error_m: 2.416276454925537
  terminal_train_player_position_error_m: 1.2647202014923096
  terminal_val_ball_position_error_m: 2.4533321857452393
  terminal_val_player_position_error_m: 1.362619400024414
  best_val_scene_position_error_m: 1.9005093574523926
  selected_epoch_zero_based: 55
repro:
  commit: 1758eda44debb618395fc7fe86e9ecb30f32a4fa
  branch: codex/slcs-real-rgb
  remote: git@github.com:Motoki0705/tennis-lab.git
  command: env PYTHONUNBUFFERED=1 PYTHONFAULTHANDLER=1 CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4
    MKL_NUM_THREADS=4 .venv/bin/python -m src.tasks.slcs.scripts.train --config-name
    train_real_rgb_gap48 paths.checkpoint_root=/home/kamimura/projects/tennis-lab/outputs
    run.resume=slcs/train/real_rgb_gap48/s42-takeover-001/logs/version_0/checkpoints/last.ckpt
    run.test_after_fit=false run.output_dir=slcs/train/real_rgb_gap48/s42-takeover-001
artifacts:
  run_dir: knowledge/runs/run-slcs-full-real-rgb-gap48-e60-resume-v2
  output_dir: /home/kamimura/projects/tennis-lab/outputs/slcs/train/real_rgb_gap48/s42-takeover-001/logs/version_1
  log: knowledge/runs/run-slcs-full-real-rgb-gap48-e60-resume-v2/queue.log
  curves: knowledge/runs/run-slcs-full-real-rgb-gap48-e60-resume-v2/curves.png
  tb_logdir: outputs/slcs/train/real_rgb_gap48/s42-takeover-001/logs/version_1
parents:
- run-slcs-full-real-rgb-gap48-interrupted-v1
relations:
- to: run-slcs-full-real-rgb-no-ball-smooth-e60-v3
  rel: compares
tags:
- slcs
- real-rgb
- detector-gap
- resumed
- gpu
---

## 考察 / Findings

### 要約

12epochで中断したgap48をoptimizer/schedulerを含む全状態から再開し、60epoch・global_step1800まで完走した。
再開後は48epoch。ユーザーの環境方針はローカル継続であり、共有GPU queueのall予約で実行した。
validation選定はepoch55、scene誤差1.900509m。条件別の採否評価は別runで行う。

### アーキテクチャ詳細

no-smooth controlからの学習施策差分はburst最大長24→48のみ。seed42、model、loss、教師版、splitを維持。
同じ出力runにrun.resumeを明示し、元のversion_0/last.ckptからversion_1へ継続した。
元checkpointのSHAは016cf87b2181ade355b75aa82faf68e499f004f27c2400f80622fcc3dc11a14f。
再開前configと途中曲線は親nodeへ保存済み。自動終端testは無効であり、testの数値は本runにない。

### メトリクスの解釈

再開後train ballは7.5569→2.4163m、player2.2740→1.2647m。terminal valはball2.4533m/player1.3626m。
曲線はversion_1のepoch12〜59だけを示すため、中断前12epochとは親nodeの曲線を併読する。
validation lossは低下後に頭打ちとなり、位置閾値内の割合も終盤は概ね横ばいだった。trainとvalidationにはaugmentation条件差があるため、両者の差だけで過学習の有無を断定しない。
終端last.ckptをCPUで読み、epoch59/global_step1800を確認した。選定monitorはval scene位置誤差最小。

### アーキテクチャ⇄メトリクスの因果考察

保存状態から再開できたことは確認したが、中断なし実行とのaugmentation RNG完全一致は保証しない。
logger directory変更によりModelCheckpointの候補辞書が再初期化された旨の警告があるため、
評価はversion_1/last.ckptを明示し、mtimeやversion番号による自動推測を避ける。
元version_0のlast.ckpt内のbest scoreもCPUで確認し、4.8588056564m（epoch11）だった。再開前候補が今回の最良1.9005093575mを上回らないことを確認した。
今回の完走は過去のsignal・CRC原因が解決したことの証明ではない。

### 既存実験との比較

controlのbest val scene1.956797mに対して今回は1.900509mだが、これだけで頑健性改善とは判定しない。
同じ公開CLIで5条件・固定val・train-only平均位置・motion・domain・裾を評価して比較する。
新profileの採用によって既存baseline成果物を置換する操作はまだ行っていない。

### 次に有効な実験

`--last-checkpoint logs/version_1/checkpoints/last.ckpt`を指定したval5条件評価を行う。
gap48の採否を判断後、採用する欠損設定を固定し、準備中の教師速度lossだけを追加する60epoch比較へ進む。
