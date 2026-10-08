---
id: run-i1031-player-ckpt-qualitative-20261008a
type: run
task: player_detection
sequence: 7
recorded_at: '2026-10-08'
title: Player実checkpoint定性保存：worktreeのCUDA拡張未配置
issue: 1031
provider: codex
session: 01a113c9-1114-71a2-9261-80815f662d33
date: '2026-10-08'
status: failed
config:
  checkpoint: ckpt/player_detection/chat-player-v1-e8-best-epoch03.ckpt
  checkpoint_sha256: f14401da5214c3f32f44171c466b5775e7daebf6bd5236c95508b3c6d68d7e05
  checkpoint_epoch: 3
  validation_subset_indices:
  - 0
  - 347
  mode: validation_only
  qualitative_interval: 1
metrics: {}
repro:
  commit: c751331f589a40432920a4f9fc2bbf34eb3c96db
  branch: fix/issue-1031-qualitative-schedule
  remote: git@github.com:Motoki0705/tennis-lab.git
  command: MPLBACKEND=Agg OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True .venv/bin/python -m scripts.verify_qualitative_checkpoints
    --task player_detection --asset-root /home/kamimura/projects/tennis-lab --checkpoint
    /home/kamimura/projects/tennis-lab/ckpt/player_detection/chat-player-v1-e8-best-epoch03.ckpt
    --output-dir /home/kamimura/projects/tennis-lab/outputs/player_detection/evaluate/issue-1031-qualitative/20261008a
    --sample-indices 0 347 --player-max-frames 60 --player-display-width 640
artifacts:
  run_dir: knowledge/runs/run-i1031-player-ckpt-qualitative-20261008a
  log: /home/kamimura/projects/tennis-lab/.training_queue/logs/1791417113827962195_70604_i1031-player-ckpt-qualitative-20261008a.log
  output_dir: /home/kamimura/projects/tennis-lab/outputs/player_detection/evaluate/issue-1031-qualitative/20261008a
  verification: knowledge/runs/run-i1031-player-ckpt-qualitative-20261008a/verification.json
  config: knowledge/runs/run-i1031-player-ckpt-qualitative-20261008a/config.yaml
parents: []
relations: []
papers: []
tags:
- qualitative-logging
- real-checkpoint
- cuda
- pr-1041
---

既存checkpointを用いたPR #1041の定性保存検証。DINOのMultiScaleDeformableAttentionが当該worktreeに配置されておらず、モデル構築で停止した。trained state_dictのstrict loadと推論は未実行である。

エラーは `RuntimeError: DINO CUDA extension is not installed. Run: TENNIS_LAB_BUILD_CUDA_OPS=1 .venv/bin/python setup.py build_ext --inplace`。出力画像・GIFおよび精度の測定値はなく、metricsは空とする。追加学習はしていない。失敗時のコマンド・config・reportをbundleに保存した。

修正後は `run-i1031-player-ckpt-qualitative-20261008b` で同じcheckpoint・validation subsetを再検証する。TensorBoardイベントを作成する段階より前に停止したため学習曲線はない。
