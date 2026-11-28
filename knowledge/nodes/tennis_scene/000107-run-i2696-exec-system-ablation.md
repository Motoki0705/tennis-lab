---
id: run-i2696-exec-system-ablation
type: run
task: tennis_scene
sequence: 107
recorded_at: '2026-11-28'
title: 固定3クリップでモデル共有・横断バッチ・TensorRT FP16を統制比較
issue: 2696
provider: codex
session: 01a21d37-57e6-7ff2-9ab1-33d5bcf8b8e9
date: '2026-11-27'
status: done
summary_impact: updated
config:
  dataset: meiji_4cam
  video: video_000
  clips: [clip_000, clip_037, clip_027]
  source_frames_per_camera: 883
  cameras: 4
  players_per_side: 2
  court: video_shared, clip-level player side evidence
  ball_enabled: false
  capture_sample_interval_seconds: 0.5
  capture_cuda_synchronize: false
  batch_sizes: {dino: 1, vitpose_crops: 8, clip_reid: 16, hmr2: 16, gvhmr: 4}
metrics:
  c1_cli_wall_seconds: 2186.313835958019
  c2_cli_wall_seconds: 1686.1479646930238
  c3_cli_wall_seconds: 1706.8127263109782
  c4_cli_wall_seconds: 1319.3085901879822
  c4_speedup_vs_c1: 1.6571663765537228
  c3_peak_vram_gib: 4.490329742431641
  c4_peak_vram_gib: 1.7618141174316406
  c3_peak_rss_gib: 8.157665252685547
  c4_peak_rss_gib: 3.7743797302246094
  generated_clips_per_condition: 3
  matched_player_ids_per_condition: 12
  c4_joint_l2_median_m: 0.001812496599231846
  c4_joint_l2_p95_m: 0.009062803599121639
  c4_joint_l2_max_m: 0.6978441422130778
repro:
  commit: 2dbd8c355faa2d530b8b0e17164f691e60a57d07
  queue_job: 0-8204212722975544204_2026-11-27T13-47-57-024455795Z_973716_i2696-ablation-v1.job
  branch: codex/parent-2627-issue-2696-primary
  remote: git@github.com:Motoki0705/tennis-lab.git
  command: 'env CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/home/kamimura/projects/tennis-lab/.claude/worktrees/parent-2627-issue-2696-primary
    OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 .venv/bin/python -B
    -u knowledge/runs/run-i2696-exec-system-ablation/run.py ablation-v1 --expect-head
    2dbd8c355faa2d530b8b0e17164f691e60a57d07 --dataset data/tennis_multiview/processed/meiji_4cam/dataset
    --prior /mnt/wsl/tennis-data/home/kamimura/projects/tennis-lab/.worktree-outputs/parent-2278-issue-2363-primary
    --dino-extension /home/kamimura/projects/tennis-lab/.worktree-outputs/video-batch-cli/tennis_scene/evaluate/video-batch-runtime-20261117/dino-extension-a2/lib/MultiScaleDeformableAttention.so '
artifacts:
  run_dir: knowledge/runs/run-i2696-exec-system-ablation
  log: .training_queue/logs/0-8204212722975544204_2026-11-27T13-47-57-024455795Z_973716_i2696-ablation-v1.log
  output_dir: outputs/i2696/ablation-v1
  queue_receipt: knowledge/runs/run-i2696-exec-system-ablation/queue-receipt.json
  timings: knowledge/runs/run-i2696-exec-system-ablation/metrics.json
  parity: knowledge/runs/run-i2696-exec-system-ablation/parity.json
  figure: knowledge/runs/run-i2696-exec-system-ablation/comparison.pdf
parents:
- run-i2363-trt-player-video
- run-i1752-shared-fp16-20261118
relations: []
papers: []
tags: [tensorrt, fp16, video-batch, profiling, ablation]
---

## 考察

同一HEAD・固定3クリップ・空の独立storeでプレーヤー3Dを4条件各1回計測し、全条件で3件を生成した。逐次PyTorch FP16から3clip一括のモデル共有へ切り替えると、管理対象6モデルのロード・解放が各18→6回になり、CLIは2,186.31→1,686.15秒。横断バッチではViTPose／CLIP-ReID／HMR2の充填率が99.77%以上に達したが、CLIは1,706.81秒と共有のみより20.66秒長かった。横断バッチのままTensorRT FP16にすると、CLIは1,319.31秒、Peak VRAMは4.490→1.762 GiB、Peak RSSは8.158→3.774 GiB。 [計測値・バッチ内訳](../../runs/run-i2696-exec-system-ablation/README.md#計測結果)

C1に対して全条件の12選手IDと3D関節の有効フレームが一致した。C2の3D関節座標は完全一致し、C3の最大差は0.079 mm。C4はSMPL有効フレームが3,496→3,495で、共通有効59,533関節点の3D差は中央値1.81 mm、P95 9.06 mm、最大0.698 mだった。計測は単一収録の固定条件・固定順に限る。 [出力整合](../../runs/run-i2696-exec-system-ablation/README.md#出力整合)
