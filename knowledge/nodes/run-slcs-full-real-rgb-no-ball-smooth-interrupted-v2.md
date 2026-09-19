---
id: run-slcs-full-real-rgb-no-ball-smooth-interrupted-v2
type: run
title: 'SLCS全体版再実行: 環境再起動後に学習/worker消失、checkpoint未保存'
provider: codex
session: 01a0b71f-c516-75b2-ac58-0c92e8a4f098
date: '2026-09-19'
status: failed
config:
  model: SLCSFusionModel hidden128/shared4/DINO-downsample2
  loss: ball_position_smoothness_weight=0
  data: slcs/real_rgb_v1
  planned_epochs: 60
metrics:
  retained_checkpoints: 0
  tensorboard_scalar_tags: 0
repro:
  commit: 48f75c7777cb7cf4d597589bed612750da975d49
  branch: codex/slcs-real-rgb
  remote: git@github.com:Motoki0705/tennis-lab.git
  command: env CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 .venv/bin/python
    -m src.tasks.slcs.scripts.train --config-name train_real_rgb loss.ball_position_smoothness_weight=0.0
    run.output_dir=slcs/train/real_rgb_no_ball_smooth/s42-takeover-002
artifacts:
  run_dir: knowledge/runs/run-slcs-full-real-rgb-no-ball-smooth-interrupted-v2
  output_dir: /home/kamimura/projects/tennis-lab/outputs/slcs/train/real_rgb_no_ball_smooth/s42-takeover-002/logs/version_0
  log: knowledge/runs/run-slcs-full-real-rgb-no-ball-smooth-interrupted-v2/queue.log
parents: [run-slcs-real-rgb-dino-crc-v1, run-slcs-full-real-rgb-no-ball-smooth-start-failure-v1]
tags: [slcs, real-rgb, interrupted, environment]
---

## 考察 / Findings

### 要約

path付きCRC例外を追加後、同じ全体版60epoch条件を新出力先で再実行した。
後続観測では学習PID434335・worker PID434154がともに存在せず、環境起動時刻はjob開始より後だった。
checkpointとTensorBoard scalarは未保存。正常完了やlive waitとは扱わない。

### アーキテクチャ詳細

前runと同じデータ・loss・batch16・workers2・seed42を使用。queue開始は11:57:13 JST。
読込例外の文脈追加以外のmodel/data数値処理は変更していない。

### メトリクスの解釈

03:31 UTCの確認で`uptime -s`は2026-09-19 12:24:44 JSTを示し、記録済み開始時刻より後だった。
`ps -p 434335,434154`はheaderのみ・終了コード1。TensorBoardの全tag種別が空で、checkpointは0。
queueのrunning fileは残存しているが、実processの存在を表していない。exit codeは記録されておらず捏造しない。
更新がどこまで進んだかは保存記録から確定できない。収束曲線・精度は得られていない。

### アーキテクチャ⇄メトリクスの因果考察

実行環境が再起動したこととjob消失は確認できたが、再起動理由は未確定。
OOM・GPU故障・user操作のいずれとも断定しない。hardware原因の追跡は本タスクの条件にしない。

### 既存実験との比較

直前のCRCエラーとは停止形態が異なり、このrunのlogにはCRC例外を認めない。
CPUスモークと7clip pilotは完了しているが、61clip GPU学習の完了証拠はまだない。

### 次に有効な実験

再開可能checkpointがないため、新runで60epochを開始する必要がある。
先にstep数計数とLightning setupによる同一train/val datasetの重複構築を確認し、
意味を変えず不要な再読込を省けるならtests付きで修正する。
