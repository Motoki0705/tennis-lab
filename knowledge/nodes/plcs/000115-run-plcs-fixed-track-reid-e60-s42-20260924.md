---
id: run-plcs-fixed-track-reid-e60-s42-20260924
type: run
task: plcs
sequence: 115
recorded_at: '2026-09-24'
title: 固定track Re-ID初回本学習をvalidation NaNで停止
issue: 915
provider: codex
session: 01a0d0f9-6057-7961-ad6b-adc46744d619
date: '2026-09-24'
status: failed
config:
  model: plcs_player_reid
  hidden_dim: 256
  num_stages: 4
  batch_size: 4
  accumulate_grad_batches: 4
  planned_epochs: 60
  last_completed_epoch: 4
  seed: 42
  compile: true
  precision: bf16-mixed
  data: plcs/tracked_person_reid_v1
metrics:
  train_loss_epoch4: 0.20049546658992767
repro:
  commit: 53db1868054e8330126d34a458edb1424646d61e
  branch: codex/plcs-track-reid
  remote: git@github.com:Motoki0705/tennis-lab.git
  command: CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    .venv/bin/python -m src.tasks.plcs.scripts.train --config-name train_reid paths.project_root=/home/kamimura/projects/tennis-lab/.claude/worktrees/plcs-track-reid
    paths.data_root=/home/kamimura/projects/tennis-lab/data paths.output_root=/home/kamimura/projects/tennis-lab/outputs
    paths.artifact_root=/home/kamimura/projects/tennis-lab/outputs paths.checkpoint_root=/home/kamimura/projects/tennis-lab/outputs
    data.scene_dir=plcs/tracked_person_reid_v1 run.output_dir=plcs/train/fixed_track_reid_v1_s42
    training.trainer.enable_progress_bar=false
artifacts:
  run_dir: knowledge/runs/run-plcs-fixed-track-reid-e60-s42-20260924
  log: /home/kamimura/projects/tennis-lab/.training_queue/logs/1790218935954276826_740451_plcs-fixed-track-reid-e60-s42-20260924.log
  output_dir: /home/kamimura/projects/tennis-lab/outputs/plcs/train/fixed_track_reid_v1_s42
  tb_logdir: outputs/plcs/train/fixed_track_reid_v1_s42/logs/version_0
  curves: knowledge/runs/run-plcs-fixed-track-reid-e60-s42-20260924/curves.png
parents:
- run-plcs-track-reid-gpu-smoke-20260924
relations: []
papers: []
tags:
- reid
- fixed_tracks
- failed
- nonfinite
- compile
---

## 観測と停止理由

60epochを予定したが、epoch0〜4でtrain lossは1.4473から0.2005へ低下する一方、全validation lossがNaNとなったため停止した。queue上は取消による終了で、研究上は失敗として記録する。保存checkpoint epoch4の全parameterは有限値であり、validation精度として利用できる測定値はない。NaNを0や失敗率へ置き換えず、未測定とする。

同じcheckpoint・validation入力によるCPU/CUDA/eager/compiled probeではfresh evalは有限だった。compiledモデルでは2回目以降の評価、または評価→学習→評価でNaNを再現し、eagerでは再現しなかった。dropout=0では解消せず、全無効attention行だけ自己参照を許して直後に出力をゼロ化するmaskでは反復評価・モード切替が有限になった。内部kernelの完全な原因確定ではなく、この環境での反証実験である。

固定slot・metric learningそのものの失敗とは切り分ける。修正では有効queryが見られるkeyを変えず、無効rowのsoftmaxを未定義にしない。CPUで有効出力と勾配の同値性を確認し、非有限lossの即時停止とsanity validationを含む2epoch GPUスモークを追加する。次の本学習はこのrunをresumeせず、seed42で初期化し直す。sideは学習しない。

TensorBoard曲線はtrain損失と非有限validationの発生確認用であり、収束や精度比較には用いない。
