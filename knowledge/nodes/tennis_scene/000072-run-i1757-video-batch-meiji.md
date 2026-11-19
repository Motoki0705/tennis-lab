---
id: run-i1757-video-batch-meiji
type: run
task: tennis_scene
sequence: 72
recorded_at: '2026-11-19'
title: meiji_4camの3クリップ逐次計測と保存出力の一致
issue: 1757
provider: codex
date: '2026-11-19'
status: done
summary_impact: updated
parents: [run-i1752-shared-fp16-20261118]
metrics:
  probe_elapsed_seconds: 201.485
  probe_sampled_device_peak_mib: 3355
  completed_serial_repetitions: 3
  first_worker_wall_seconds: 2118.0081680389994
  first_including_startup_seconds: 2131.753765767993
  warm_wall_seconds: [2031.498292328004, 2005.0597923399764]
  warm_sample_stdev_seconds: 18.694842625934783
  allocator_peak_bytes: {allocated: 4184557568, reserved: 4460642304}
  warm_process_lifetime_rss_bytes: 9646534656
  scenes_reloaded: 9
  comparisons_exact: 3
repro:
  commit: 8c91bf074816b3d809ac2548f8d8050fbcda3ac9
  branch: codex/video-batch-benchmark
artifacts:
  run_dir: knowledge/runs/run-i1757-video-batch-meiji
  output_dir: outputs/tennis_scene/evaluate/video-batch-benchmark-20261119/serial-heartbeat-2
  queue_job: 0-8204925093921193151_2026-11-19T07-55-06-078806848Z_1424107_i1757-serial-heartbeat-2
  metrics: knowledge/runs/run-i1757-video-batch-meiji/serial-heartbeat-2.json
---

meiji_4camのclip_000・clip_027・clip_037を固定した。共有court、player証拠のcourt side、ダブルス、DINO・ViTPose fp16、X3D/CSN＋DPT T32の全ball段を、逐次と一括の両方で使う。採用済みのballによるhalf-turnをplayer判定の参照とする。入力・計測条件・GPU probe・集計は[再現手順](../../runs/run-i1757-video-batch-meiji/reproduce.md)にあり、[許容誤差](../../runs/run-i1757-video-batch-meiji/tolerances.json)は実行前に固定した。

13:30 JSTに共有training queueのall枠へ登録した。13:32 JSTのprobe開始後、clip_027/cam0のX3D/CSNによるball_detectionを保存し、fp16 DINOの初回forwardが完了した。13:35 JSTに `nvidia-smi` の応答が10秒以内に返らず、監視がprobeを停止した。13:36 JSTのprobe終了記録は `nvidia_smi_unresponsive` で、逐次本計測は未開始。資源サンプル・停止時刻・ログの参照は [probe記録](../../runs/run-i1757-video-batch-meiji/probe-summary.json) にある。

heartbeat監視に切り替えた `serial-heartbeat-2` では3クリップを全3回で生成し、初回はworker起動込み2131.75秒、warmは2031.50秒と2005.06秒だった。全3組で、各組47 node × 3クリップの保存payloadとscene配列を固定許容誤差で比較し、すべてexactだった。player証拠のcourt_sideは9件とも `[false, false, true, true]`。全sceneを再読込して検証した。 [逐次計測の内訳と品質確認](../../runs/run-i1757-video-batch-meiji/serial-heartbeat-2.md)

この逐次基準をsummaryへ反映した。条件間の速度と品質、中断後の再開、cache混在は未確認。初期probeの停止記録は当時の観測として保持する。
