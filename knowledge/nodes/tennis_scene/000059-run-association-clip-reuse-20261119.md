---
id: run-association-clip-reuse-20261119
type: run
task: tennis_scene
sequence: 59
recorded_at: '2026-11-19'
title: Meiji4camの3clipでの人物対応CLIP再利用と照合費用
issue: 1756
provider: codex
session: 01a1f1fc-a0b2-7cf3-ad55-092e6e3022cc
date: '2026-11-19'
status: done
summary_impact: none
config:
  video: video_000
  clips: [clip_037, clip_000, clip_027]
  cameras_per_clip: 4
  players_per_side: 2
  precision: float32
  batch_size: 64
  order: [recompute, reuse, reuse, recompute]
  cosine_atol: 1.0e-06
  diagnostic_atol: 0.0001
  replay_commit: a65a729fc58d60f52c07a29778fdecbf72bae692
metrics:
  eligible_samples: 816
  reused_samples: 816
  recomputed_samples_when_reuse_enabled: 0
  player_ids_equal: true
  association_decisions_and_appearance_mask_equal: true
  sampled_frames_and_rejections_equal: true
  max_diagnostic_delta: 3.1047079573909286e-08
  seconds_saved_association_sum_of_clip_medians: 0.57876359
  seconds_saved_clip_calls_sum_of_clip_medians: 2.3096738555
  proof_hashed_frames: 3532
  proof_hash_cpu_seconds: 15.638074869
  proof_hash_wall_seconds: 15.823703692
  gpu_model_loads: 1
  gpu_model_unloads: 1
  gpu_allocator_peak_bytes: 860329984
  cpu_saved_replay_exact: true
repro:
  commit: ba460cbe53073bc0a0d4320ad16bab171c2a9ec2
  branch: codex/clip-embedding-reuse
  remote: git@github.com:Motoki0705/tennis-lab.git
  command: timeout --signal=TERM 900s env PYTHONPATH=. OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2
    .venv/bin/python knowledge/runs/run-association-clip-reuse-20261119/compare.py
    --dataset ../video-batch-cli/outputs/tennis_scene/evaluate/video-batch-shared-fp16-20261118/datasets/batch
    --output outputs/clip-embedding-reuse/gpu-doubles
artifacts:
  run_dir: knowledge/runs/run-association-clip-reuse-20261119
  log: .training_queue/logs/0-8204937428478711273_2026-11-19T04-29-31-521288726Z_630111_i1756-association-clip-reuse-doubles.log
  output_dir: outputs/clip-embedding-reuse/gpu-doubles
parents: [run-person-feature-artifacts-20261117]
relations: []
papers: []
tags: []
---

## 比較条件

#1756（#1734のC9）の比較。#1752の4カメラ・ダブルス3clipから検出、選別済みtrack、コート校正、sideを固定した。
保存成果物の依存・モデル重み・動画を照合したうえで、現行の外観特徴ノードでCLIP v4特徴を生成し、保存後に読み直した。
[参照・設定・入力ハッシュ](../../runs/run-association-clip-reuse-20261119/inputs.json)

RTX 5060 Tiを共有training queueのall枠で使用した。モデルが常駐した状態で、clipごとに全再推論→再利用→再利用→全再推論の順で比較した。
画像照合・decode・対応solverを含む人物対応処理の時間と、CLIP呼出しの時間をC1 recorderで記録した。
全試行でCLIPの重み、精度、sampling、batch上限をそろえ、連続値は事前の許容差で比較した。
[計測スクリプト](../../runs/run-association-clip-reuse-20261119/compare.py)

## 結果

| clip | 再利用数 / 対象数 | 人物対応の短縮（秒） | CLIP呼出しの短縮（秒） |
|---|---:|---:|---:|
| `clip_037` | 256 / 256 | 0.195 | 0.696 |
| `clip_000` | 272 / 272 | 0.230 | 0.792 |
| `clip_027` | 288 / 288 | 0.154 | 0.822 |

人物ID、group化、候補の採否とappearanceの有無、sample frame、重複・遮蔽による除外診断が一致した。
固定元の人物IDとも一致した。association診断の連続値の最大差は `3.10e-8` で、事前に固定した許容差内だった。
短縮値は各条件2回の中央値の差。3clipで人物対応は計0.579秒、CLIP呼出しは計2.310秒短縮した。
[比較JSON](../../runs/run-association-clip-reuse-20261119/comparison.json)
[appearance診断](../../runs/run-association-clip-reuse-20261119/appearance-diagnostics.json)
[recorder](../../runs/run-association-clip-reuse-20261119/recorder.json)

保存済み特徴の画像SHA-256を独立したCPU再読込で再現した。3,532フレームのSHA-256の計算だけで15.82秒を要した。
この時間にはdecode、推論、保存を含めていない。再利用を有効にする初回はこの証拠を生成するため、C9の設定は既定で `false` とした。
新規生成を含む3clip全体の短縮は未計測。今回の内訳では初回の照合証拠生成にかかる時間が、人物対応で短縮できた時間を上回る。
標準pipelineの採用判断を更新する証拠は得ていない。
[SHA-256の計測結果](../../runs/run-association-clip-reuse-20261119/frame-hash-cost.json)

初回試行では検証スクリプトが固定入力をシングルスとして読み、全再推論側が `player_association_ambiguous_players` で終了した。
固定成果物の試合形式を照合するよう修正し、ダブルス設定で完走した。
[初回のログ](../../runs/run-association-clip-reuse-20261119/attempt-1/run.log)
[初回のrecorder](../../runs/run-association-clip-reuse-20261119/attempt-1/recorder.json)

画像証拠の生成をC9有効時に限定した最終コードでは、GPUで保存した3clipをCPUで再生し、CLIP起動0回でGPUの再利用結果と完全一致した。
最終コードでの追加GPU計測は、別ジョブの終了待ちで共有枠を取得できなかったため開始前に取り下げた。
[最終コードの再生結果](../../runs/run-association-clip-reuse-20261119/saved-replay.json)
