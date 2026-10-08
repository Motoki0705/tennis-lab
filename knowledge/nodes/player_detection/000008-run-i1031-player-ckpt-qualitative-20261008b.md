---
id: run-i1031-player-ckpt-qualitative-20261008b
type: run
task: player_detection
sequence: 8
recorded_at: '2026-10-08'
title: Player学習済みcheckpointからvalidation元動画のbbox GIFを保存
issue: 1031
provider: codex
session: 01a113c9-1114-71a2-9261-80815f662d33
date: '2026-10-08'
status: done
config:
  checkpoint: ckpt/player_detection/chat-player-v1-e8-best-epoch03.ckpt
  checkpoint_sha256: f14401da5214c3f32f44171c466b5775e7daebf6bd5236c95508b3c6d68d7e05
  checkpoint_epoch: 3
  validation_subset_indices:
  - 0
  - 347
  mode: validation_only
  qualitative_interval: 1
  frame_stride: 2
  max_frames: 60
  display_width: 640
  score_threshold: 0.3
metrics:
  validation_samples: 2
  gif_files: 2
  gif_frames: 120
  tensorboard_image_tags: 2
  state_tensors_strict_loaded: 716
  peak_cuda_allocated_mib: 2644.515625
repro:
  commit: 1dcf9407897cefc38842fe02e57e6895c008e6f6
  branch: fix/issue-1031-qualitative-schedule
  remote: git@github.com:Motoki0705/tennis-lab.git
  command: 'env TENNIS_LAB_BUILD_CUDA_OPS=1 ''TENNIS_LAB_DINO_OPS_BUILD_CONFIG={"paths":
    {"project_root": "/home/kamimura/projects/tennis-lab/.claude/worktrees/issue-1031-qualitative-schedule",
    "data_root": "/home/kamimura/projects/tennis-lab/data", "checkpoint_root": "/home/kamimura/projects/tennis-lab/ckpt",
    "artifact_root": "/home/kamimura/projects/tennis-lab/.claude/worktrees/issue-1031-qualitative-schedule/outputs",
    "output_root": "/home/kamimura/projects/tennis-lab/.claude/worktrees/issue-1031-qualitative-schedule/outputs",
    "cache_root": "/home/kamimura/projects/tennis-lab/.claude/worktrees/issue-1031-qualitative-schedule/.cache",
    "external_asset_root": "/home/kamimura/projects/tennis-lab/third_party"}, "source_role":
    "external_asset", "source": "DINO/models/dino/ops/src", "destination_role": "cache",
    "destination": "dino_ops/src", "compressed_time_local_bindings": "src/utils/models/components/ops/compressed_time_local/bindings.cpp",
    "compressed_time_local_kernels": "src/utils/models/components/ops/compressed_time_local/kernels.cu"}''
    TORCH_CUDA_ARCH_LIST=12.0 MAX_JOBS=2 .venv/bin/python setup.py build_ext --inplace
    && env MPLBACKEND=Agg OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True .venv/bin/python -m scripts.verify_qualitative_checkpoints
    --task player_detection --asset-root /home/kamimura/projects/tennis-lab --checkpoint
    /home/kamimura/projects/tennis-lab/ckpt/player_detection/chat-player-v1-e8-best-epoch03.ckpt
    --output-dir /home/kamimura/projects/tennis-lab/outputs/player_detection/evaluate/issue-1031-qualitative/20261008b
    --sample-indices 0 347 --player-max-frames 60 --player-display-width 640'
artifacts:
  run_dir: knowledge/runs/run-i1031-player-ckpt-qualitative-20261008b
  log: /home/kamimura/projects/tennis-lab/.training_queue/logs/1791417281637748388_74360_i1031-player-ckpt-qualitative-20261008b.log
  output_dir: /home/kamimura/projects/tennis-lab/outputs/player_detection/evaluate/issue-1031-qualitative/20261008b
  verification: knowledge/runs/run-i1031-player-ckpt-qualitative-20261008b/verification.json
  config: knowledge/runs/run-i1031-player-ckpt-qualitative-20261008b/config.yaml
  qualitative: assets/player_detection/issue-1031-checkpoint-validation
  tb_logdir: /home/kamimura/projects/tennis-lab/outputs/player_detection/evaluate/issue-1031-qualitative/20261008b/logs/version_0
parents:
- run-pr937-player-inference-real-frame-20260928
relations:
- to: run-i1031-player-ckpt-qualitative-20261008a
  rel: resolves
papers: []
tags:
- qualitative-logging
- real-checkpoint
- cuda
- pr-1041
---

epoch 3・global step 8000の既存Lightning checkpointを716 tensorすべてstrict loadし、学習中保存と同じmodule・共通callbackでTrainer.validateを実行した。元動画sourceが異なるvalidation clip 29（2Fa16bdg8pI）と64（5T2anHdqhXU）を選び、元動画のmanifest・hash・寸法・frame数・PTS照合を通してDINO Swin-Lを実推論した。mockや追加学習は行っていない。

worktree内で正規setup.pyからCUDA拡張をbuildした後、各clipの先頭120 source frameをstride 2で読み、60 frameずつのbbox・confidence付きGIFを640×360、12.5 FPSで保存した。計2 GIF・120 frameの全decodeとTensorBoardの画像2 tag、callback状態更新を確認した。先頭・中間・末尾を目視し、選手の動きに対応してbboxが変わることを確認した。GIFは元clip全体ではなく先頭4.8秒ずつ。checkpoint SHAと全state_dict tensorの実行前後一致も確認し、重みは不変だった。

近方・遠方の検出が見える一方、画面外に出る選手や小さい遠方人物の全frameでのrecallを保証する検査ではない。学習時に選別したframeだけでなく元動画の連続区間を読む保存経路が成立した証拠とし、report内の2 frameのvalidation metricを全体精度へ一般化しない。既存の採用・export方針は変更しない。

画像を含むTensorBoard証拠はあるが、1回のvalidationのみで学習・収束曲線の作成は該当しない。再現bundleにはCUDA buildを含む正規コマンドを保持した。次は異なる会場・検出欠落を含むvalidation clipを固定してbboxの時間安定性を評価する。全動画精度の主張には別の定量評価が必要である。
