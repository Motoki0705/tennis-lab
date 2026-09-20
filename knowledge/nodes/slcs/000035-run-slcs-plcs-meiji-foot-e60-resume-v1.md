---
task: slcs
sequence: 35
recorded_at: 2026-09-18
date_source: experiment_date
papers: []
id: run-slcs-plcs-meiji-foot-e60-resume-v1
type: run
title: Meiji PLCS再開学習のネイティブ異常終了
provider: codex
session: 01a0b314-89b5-7902-b0a1-3141a9323ea9
date: '2026-09-18'
status: failed
config:
  model: multiview_axial_foot_residual
  loss: all_outputs_beta01_reprojection
  data: camera_view_real_rgb_ft_v1
metrics:
  saved_epoch_index: 37
  exit_code: 139
repro:
  commit: 4348077d8f7a0b35ab58350232cf4ccb607990f1
  branch: codex/slcs-real-rgb
  remote: git@github.com:Motoki0705/tennis-lab.git
  command: CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 .venv/bin/python
    -m src.tasks.plcs.scripts.train --config-name train_meiji_foot_real_rgb run.output_dir=plcs/train/meiji_foot_real_rgb/s42-001
    run.init_weights=null run.resume=plcs/train/meiji_foot_real_rgb/s42-001/logs/version_0/checkpoints/last.ckpt
artifacts:
  run_dir: knowledge/runs/run-slcs-plcs-meiji-foot-e60-resume-v1
  output_dir: /home/kamimura/projects/tennis-lab/outputs/plcs/train/meiji_foot_real_rgb/s42-001/logs/version_1
  curves: knowledge/runs/run-slcs-plcs-meiji-foot-e60-resume-v1/curves.png
  tb_logdir: outputs/plcs/train/meiji_foot_real_rgb/s42-001/logs/version_1
parents: [run-slcs-plcs-meiji-foot-e60-v1]
relations: []
tags:
- slcs
- plcs
- meiji
- interrupted
- runtime
---

## 考察 / Findings

### 要約
WSL再起動後、epoch29の全学習状態から再開し、epoch37のlast checkpoint保存まで進んだがexit139で失敗。60epoch実験は未完了。保存重みのCPU読込は成功。

### アーキテクチャ詳細
78.4Mパラメータのfoot-residual PLCS、3視点、32–128 frame、batch4、lr5e-5、bf16。元のoptimizer/schedulerを復元し、総学習上限60epochを維持した。

### メトリクスの解釈
saved_epoch_indexは0始まり。validation最良はepoch35の約0.224mで、合成source-motion分離データに対する値。test評価も実RGBの独立3D精度確認も完了していない。

### アーキテクチャ⇄メトリクスの因果考察
queue logとdmesgでPython本体およびDataLoader workerのネイティブsegfaultを確認。OOM killerの記録はなく、原因は未特定。モデル設計が原因とは結論しない。

### 既存実験との比較
元のversion_0はWSL再起動で中断し、epoch29まで保存。今回version_1はstep7500から9500へ進み、epoch27の約0.240mを下回るvalidation checkpointを保存した。再開前後のlogger版を横断してvalidation最良を選ぶ必要がある。

### 次に有効な実験
version_1/last.ckptのoptimizerを含む状態から、GPUを単独予約して継続する。PYTHONFAULTHANDLER=1を設定し、再発した場合のスタックを記録する。学習済み状態を捨てて初期化しない。
