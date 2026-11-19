---
id: run-i1752-association-stop-20261118
type: run
task: tennis_scene
sequence: 69
recorded_at: '2026-11-18'
title: 共有コート接続後のC4受入は人物対応で停止
issue: 1752
provider: codex
session: 01a1e957-c745-72f3-b662-6dcb032b4374
date: '2026-11-18'
status: failed
config:
  pipeline: pipeline_x3d_dpt
  clips:
  - video_000/clip_000
  - video_000/clip_001
  cameras: 4
  court_mode: video_shared
  continue_on_error: false
  sampling_max_frames: 1024
  seed: 42
metrics:
  completed_clips: 0
  single_cold_prefix_seconds: 1200.933115269
  dino_frames: 1424
  full69_prefix_linear_hours: 43.31830216343269
repro:
  commit: fa40a8dc38e9e3cec78e10b7d8cad2d351b0cdb3
  branch: codex/video-batch-cli
  remote: git@github.com:Motoki0705/tennis-lab.git
  command: env TENNIS_LAB_DINO_EXTENSION=/home/kamimura/projects/tennis-lab/.claude/worktrees/video-batch-cli/outputs/tennis_scene/evaluate/video-batch-runtime-20261117/dino-extension-a2/lib/MultiScaleDeformableAttention.so
    PYTHONPATH=/home/kamimura/projects/tennis-lab/.claude/worktrees/video-batch-cli/outputs/tennis_scene/evaluate/video-batch-runtime-20261117/dino-extension-a2/lib
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python knowledge/runs/run-video-batch-runtime-20261117/acceptance.py
    run --root outputs/tennis_scene/evaluate/video-batch-runtime-20261118-a2 --court-settings
    knowledge/runs/run-video-batch-runtime-20261117/court-settings.json
artifacts:
  run_dir: knowledge/runs/run-i1752-association-stop-20261118
  log: .training_queue/logs/0-8205066608261606027_2026-11-17T16-36-31-738393972Z_188023_video-batch-runtime-parity-reboot-a2.log
  output_dir: .worktree-outputs/video-batch-cli/tennis_scene/evaluate/video-batch-runtime-20261118-a2
parents:
- run-i1752-missing-dino-20261118
- run-court-video-drift-lines-20261117
relations:
- to: run-i1751-runtime-gpu-a4
  rel: related
  note: 同じGPUで直前にハング疑い。C4は開始後にこの記録を把握し、終了後の追加投入を停止。
papers: []
tags:
- video_batch
- negative_result
- player_association
summary_impact: updated
---

固定したmeiji_4camのclip_000（356frame）とclip_001（561frame）で、X3D/CSN＋DPT T32、承認済みvideo_sharedコート・drift、seed42を使い、単一入口／一括入口のcold・cacheを比較するqueue jobを実行した。入力はC1保存済み15fileとSHA-256が一致する。元datasetへは書き込まず、独立したstoreを使った。

**2026-11-18 02:03:16 JST（17:03:16 UTC）**、単一入口のcold条件でclip_000のplayer_associationが`player_association_ambiguous_players`を返して停止した。例外は`Identity 2 (354 frames) is too close to player identity 0 (354 frames)`。検出結果から得たidentity候補を区別できないという再構成不能で、今回のjobにはCUDA例外を観測していない。閾値や人物選択を変更して通過させていない。

[例外全文・時刻](../../runs/run-i1752-association-stop-20261118/failure.json)、[段階・モデル・資源の集計](../../runs/run-i1752-association-stop-20261118/prefix-summary.json)、[展開済み設定](../../runs/run-i1752-association-stop-20261118/pipeline.yaml.json)、[固定入力](../../runs/run-i1752-association-stop-20261118/input-manifest.json)を保存した。raw recorder・component配列はoutputsに保持し、raw recorderのbyte数・SHA-256を集計へ残す。

停止までのrecorder wallは**1200.933秒**。DINOは4camera・計1,424frameを実行して解放され、Court共有・ball・人物検出・追跡・pose・court_sideを通過した。完成markerは0。clip_001、単一入口cache、一括入口のcold・cache、GVHMR／body placement／最終scene比較は未到達。したがって、モデル再利用のGPUでの効果・clip数に依存しないload回数・単一入口との最終出力一致は未確認である。

部分計測のframe単価を全46,228frameへ単純に掛けると**43.3時間**になる。これは単一入口の失敗prefixの換算で、batch全工程の見積り・上限／下限・高速化率ではない。人物数・clip長・cache・import／compileの初回分・モデル再利用による削減・未到達の身体復元を含む差は推定していない。全69clipの本番は未実行。信頼できる所要時間には、再構成できる小subsetの完走が必要である。

測定sourceはqueueが採取した`fa40a8dc3`。実行中に依存branchの記録と設定上限2048を統合したが、Python sourceとdriverのhashは不変で、runの設定は開始時に1024で固定されている。[照合結果](../../runs/run-i1752-association-stop-20261118/source-integration-proof.json)を残す。対象2clipはいずれも旧上限未満。最終CLIの既定上限はmainの#1748に従う2048で、clip_035の旧長さ制限は解消している。

直前のC3 jobでDINO heartbeat停止によるハング疑いが記録されていたことを、当job開始後に統合した記録から確認した。C4のDINO通過は原因を特定する証拠ではなく、入力、code版、native binary、監視方法等が異なる。C3の異常時刻・全文はrun-i1751-runtime-gpu-a4を参照する。追加GPU投入はせず、GPU再開と比較用subset／人物対応の扱いを親へ返す。

再現入口はoriginal-repro.txt、run.json、元training command、patchを保持し、repro.shの準備段へnative binary inventoryと再build scriptを追加した。native binaryはGitへ保存しない。原コマンドを別のsource版へ書き換えてはいない。

summary.mdの動画一括推論について、旧Court／長さ上限だけでなく、共有コート接続後の人物対応停止と完成比較0を現行の未解決点へ反映したため、summary_impactはupdatedとする。GPUの再開判断前に再現入口を実行しない。

| prefixの資源・読込 | 観測値 |
|---|---:|
| sampled RSS peak | 8,210.89 MiB |
| CUDA allocated peak | 3,124.86 MiB |
| CUDA reserved peak | 4,008.00 MiB |
| ball / DINO / ViTPose / appearance load・unload | 各1 / 各1 |
| AFLink（CPU）のload・unload記録 | 4 / 0 |

資源は失敗prefixだけの観測で、CUDAはPyTorch allocatorのpeak、RAMは100ms sampleによるprocess RSS。AFLinkはcameraごとに構築されるCPU部品で、このrecorderではloadが4回、unloadは記録されていない。メモリ漏れとは断定しない。AFLinkは#1751が明示するruntime対象（Court、DINO、Ball、ViTPose、CLIP、HMR2、GVHMR、body-model）に含まれないCPU部品として別記し、全モデル合計が1回の読込とは主張しない。1clipしか処理していないため、他モデルもclip数によらないというGPUでの受入証拠にはしない。
