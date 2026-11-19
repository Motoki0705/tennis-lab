---
id: run-i1757-serial-fb6255a-first-1
type: run
task: tennis_scene
sequence: 80
recorded_at: '2026-11-20'
title: 同一HEADの逐次対照でボール差と横断batchの人物3D差を分離
issue: 1757
provider: codex
date: '2026-11-20'
status: done
summary_impact: updated
parents: [run-i1757-video-batch-meiji, run-i1757-batched-heartbeat-1]
repro:
  commit: fb6255a2853a6992e3d96287d7b989b681c2e336
  branch: codex/video-batch-benchmark
metrics:
  first_wall_including_startup_seconds: 2128.194058364985
  allocated_peak_bytes: 4181805056
  reserved_peak_bytes: 4460642304
  rss_peak_bytes: 9486848000
  scenes_reloaded: 3
  recorder_errors: 0
  ball_validity_changed_vs_old_serial_frames: 9
  ball_validity_changed_vs_same_commit_batched_frames: 6
  player_joint_validity_changed_vs_same_commit_batched_entries: 6
  quality_acceptance_vs_model_reuse: true
  quality_acceptance_vs_same_commit_batched: false
artifacts:
  run_dir: knowledge/runs/run-i1757-serial-fb6255a-first-1
  output_dir: outputs/tennis_scene/evaluate/video-batch-benchmark-20261119/serial-fb6255a-first-1
  queue_job: 0-8204898262595246886_2026-11-19T15-22-17-404753113Z_3013697_i1757-serial-fb6255a-first-1
  metrics: knowledge/runs/run-i1757-serial-fb6255a-first-1/serial-fb6255a-first-1.json
---

3クリップの逐次対照を条件3と同じHEADで初回1回実行し、3件を公開後に再読込した。旧逐次から実行commitを変えると、先行の条件2と同じ9 frameでボール3Dの有効性が変わった。旧逐次から人物のscene配列は完全一致。新しい逐次対照と条件2の初回は47 node×3クリップと最終scene配列がすべて完全一致した。

条件3は同一HEADの逐次より初回6.76%短いが、球の有効性が6 frameで変わり、人物3Dの固定許容差も超えた。保存済みの実GPU出力を使い、ボールなしで幾何、三角測量、身体view選択をCPU再計算した。カメラと身体入力が各測定の保存時と一致し、再組立てした人物配列にも同じ差が残った。人物3Dの品質受入は未達。 [結果と比較](../../runs/run-i1757-serial-fb6255a-first-1/result.md)
