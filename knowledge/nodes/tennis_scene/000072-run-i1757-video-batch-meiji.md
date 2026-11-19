---
id: run-i1757-video-batch-meiji
type: run
task: tennis_scene
sequence: 72
recorded_at: '2026-11-19'
title: meiji_4camの3クリップ計測はGPU probeの監視タイムアウトで停止
issue: 1757
provider: codex
date: '2026-11-19'
status: failed
summary_impact: none
parents: [run-i1752-shared-fp16-20261118]
metrics:
  probe_elapsed_seconds: 201.485
  probe_sampled_device_peak_mib: 3355
  completed_serial_repetitions: 0
artifacts:
  run_dir: knowledge/runs/run-i1757-video-batch-meiji
---

meiji_4camのclip_000・clip_027・clip_037を固定した。共有court、player証拠のcourt side、ダブルス、DINO・ViTPose fp16、X3D/CSN＋DPT T32の全ball段を、逐次と一括の両方で使う。採用済みのballによるhalf-turnをplayer判定の参照とする。入力・計測条件・GPU probe・集計は[再現手順](../../runs/run-i1757-video-batch-meiji/reproduce.md)にあり、[許容誤差](../../runs/run-i1757-video-batch-meiji/tolerances.json)は実行前に固定した。

13:30 JSTに共有training queueのall枠へ登録した。13:32 JSTのprobe開始後、clip_027/cam0のX3D/CSNによるball_detectionを保存し、fp16 DINOの初回forwardが完了した。13:35 JSTに `nvidia-smi` の応答が10秒以内に返らず、監視がprobeを停止した。13:36 JSTのprobe終了記録は `nvidia_smi_unresponsive` で、逐次本計測は未開始。資源サンプル・停止時刻・ログの参照は [probe記録](../../runs/run-i1757-video-batch-meiji/probe-summary.json) にある。

全段wall time、playerの向き、sceneの再読込、各条件の速度と品質、中断後の再開、cache混在は未確認。今回得られたのは停止までの観測であり、summaryの現在の判断を維持する。
