---
id: run-i1031-court-ckpt-qualitative-20261008a
type: run
task: court_detection
sequence: 34
recorded_at: '2026-10-08'
title: Court実checkpoint定性保存：固定batch選択の契約不一致
issue: 1031
provider: codex
session: 01a113c9-1114-71a2-9261-80815f662d33
date: '2026-10-08'
status: failed
config:
  checkpoint: ckpt/court_detection/multiscale_depth3/b863df1f01f0.ckpt
  checkpoint_sha256: b863df1f01f00d2ff00a21d56a461879e3c5af193a9f32cec47a88d2842f8383
  checkpoint_epoch: 17
  validation_subset_indices:
  - 0
  - 935
  mode: validation_only
  qualitative_interval: 1
metrics: {}
repro:
  commit: c751331f589a40432920a4f9fc2bbf34eb3c96db
  branch: fix/issue-1031-qualitative-schedule
  remote: git@github.com:Motoki0705/tennis-lab.git
  command: MPLBACKEND=Agg OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True .venv/bin/python -m scripts.verify_qualitative_checkpoints
    --task court_detection --asset-root /home/kamimura/projects/tennis-lab --checkpoint
    /home/kamimura/projects/tennis-lab/ckpt/court_detection/multiscale_depth3/b863df1f01f0.ckpt
    --output-dir /home/kamimura/projects/tennis-lab/outputs/court_detection/evaluate/issue-1031-qualitative/20261008a
    --sample-indices 0 935
artifacts:
  run_dir: knowledge/runs/run-i1031-court-ckpt-qualitative-20261008a
  log: /home/kamimura/projects/tennis-lab/.training_queue/logs/1791417113966994965_70627_i1031-court-ckpt-qualitative-20261008a.log
  output_dir: /home/kamimura/projects/tennis-lab/outputs/court_detection/evaluate/issue-1031-qualitative/20261008a
  verification: knowledge/runs/run-i1031-court-ckpt-qualitative-20261008a/verification.json
  config: knowledge/runs/run-i1031-court-ckpt-qualitative-20261008a/config.yaml
parents: []
relations: []
papers: []
tags:
- qualitative-logging
- real-checkpoint
- cuda
- pr-1041
---

既存checkpointを用いたPR #1041の定性保存検証。Court側parserが古いfirst/random/indicesを許可し、共通callbackのfixed_indicesを拒否した。モデルロード・描画の前の設定検証で停止した。

エラーは `SemanticConfigurationError: training.qualitative_logging.selection_mode is invalid.`。出力画像・GIFおよび精度の測定値はなく、metricsは空とする。追加学習はしていない。失敗時のコマンド・config・reportをbundleに保存した。

修正後は `run-i1031-court-ckpt-qualitative-20261008b` で同じcheckpoint・validation subsetを再検証する。TensorBoardイベントを作成する段階より前に停止したため学習曲線はない。
