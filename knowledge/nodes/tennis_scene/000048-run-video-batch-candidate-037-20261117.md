---
id: run-video-batch-candidate-037-20261117
type: run
task: tennis_scene
sequence: 48
recorded_at: '2026-11-17'
title: 現行mainの短い4camera clip_037の全段到達確認
issue: 1748
provider: codex
date: '2026-11-17'
status: planned
config:
  main_commit: 486aa1ab09be857909d0fcf8c2b616c75a53f5b0
  clips:
  - video_000/clip_037
  cameras:
  - cam0
  - cam1
  - cam2
  - cam3
  num_frames: 292
  condition: single cold qualification
  repetitions: 1
  allocator_limit_gib: 10
  ball_model: i935
  purpose: Confirm where current main stops; not the final multi-clip baseline
metrics: {}
artifacts:
  run_dir: knowledge/runs/run-video-batch-candidate-037-20261117
  output_dir: outputs/tennis_scene/evaluate/video-batch-candidate-037-20261117
  input_manifest_sha256: 497f75bf338ec5a918f9011cea789b7375bd184ce9407a7b6583126c24194002
parents:
- run-video-batch-baseline-035037-20261116
relations: []
papers: []
tags: []
summary_impact: none
---

## 実行前の固定

#1748の続行指示で、長さ上限内の実clipを現行mainで全段確認する。
C12後のmainはlegacy既定を維持し、C13のPR #2222はまだ未mergeだった。
旧全276画像のCPU replayでは既定が完全一致し、旧全camera採用の035/037のうち035だけが長さ上限を超える。
まず037（292frame・4camera）を1回cold実行して、Courtの先から何が止まるかを記録する。
この単一clip・単一条件は最終の複数clip cold/warm/cache基準値にはしない。summaryの未達判断は変えない。

入力は登録情報・clip.json・4動画のSHAと全frame数で固定し、outputs内へコピーした。元datasetへ書かない。
実行commitとcommandはqueue reproで固定する。実行は共有training queueのall枠、provider=codex、issue=1748。
既定i935・model寿命・dtype・batch・判定条件を維持し、最初の例外で停止する。

GPUは16GB。前回035のpre-body peakはallocated3.07GiB/reserved4GiBだった。
短い037のHMR2/GVHMRを含む逐次loadと既定batchから10GiB以内を事前見積もりとし、
従来C1と同じallocator10GiB上限で超過を明示OOMにする。見積もりを実測値として扱わない。
DINO拡張は前回検証済みの同じbinaryを明示し、data/ckptとvenvを共有する。
