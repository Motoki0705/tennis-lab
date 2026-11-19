---
id: run-i1757-embedding-reuse-heartbeat-1
type: run
task: tennis_scene
sequence: 78
recorded_at: '2026-11-19'
title: Meiji3クリップのCLIP再利用は全命中し外観特徴の照合費用が増加
issue: 1757
provider: codex
date: '2026-11-19'
status: done
summary_impact: updated
parents: [run-i1757-model-reuse-heartbeat-1, run-association-clip-reuse-20261119]
repro:
  commit: 795d63df1682d0d86f769d584dfbfa263419ad0d
  branch: codex/video-batch-benchmark
metrics:
  first_wall_including_startup_seconds: 1921.968139019009
  warm_wall_seconds: [1823.3925161799998, 1815.32166731701]
  warm_sample_stdev_seconds: 5.706951960951828
  allocated_peak_bytes: 4184557568
  reserved_peak_bytes: 4460642304
  rss_peak_bytes: 9483579392
  clip_hits_per_repeat: 816
  clip_recomputed_per_repeat: 0
  scenes_reloaded: 9
  same_scene_arrays_vs_model_reuse: true
  quality_acceptance_vs_serial: false
artifacts:
  run_dir: knowledge/runs/run-i1757-embedding-reuse-heartbeat-1
  output_dir: outputs/tennis_scene/evaluate/video-batch-benchmark-20261119/embedding-reuse-heartbeat-1
  queue_job: 0-8204917572671425689_2026-11-19T10-00-27-328574310Z_1772178_i1757-embedding-reuse-heartbeat-1
  metrics: knowledge/runs/run-i1757-embedding-reuse-heartbeat-1/embedding-reuse-heartbeat-1.json
---

3回とも816件すべてに命中し、人物対応のCLIP forwardが16回から0回になった。外観特徴段は毎回14.84〜16.02秒延び、同じHEADのモデル再利用よりwarm中央値が16.60秒長かった。frame SHAの準備と追加artifactの読込・照合があり、forward削減が全体の時間短縮につながらなかった。warmは各2回なので増分の推定にはばらつきがある。

9 sceneを再読込してcourt_sideとball参照の一致を確認した。同一HEADのモデル再利用に対して全scene配列と人物IDが一致した。証拠・命中診断に加え、clip_000の人物対応診断の連続値1項目は固定許容差を超えた。旧逐次とはモデル再利用と同じボール差が残った。summaryには再利用率、費用と精度の範囲を反映する。 [時間内訳と差分](../../runs/run-i1757-video-batch-meiji/model-reuse-comparison.md)
