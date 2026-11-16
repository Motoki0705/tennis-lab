---
id: run-video-batch-baseline-035037-20261116
type: run
task: tennis_scene
sequence: 46
recorded_at: '2026-11-16'
title: 採用済み4camera・2clipでの逐次推論の計測
issue: 1748
provider: codex
date: '2026-11-16'
status: planned
config:
  main_commit: 2736a24a3
  clip_order:
  - video_000/clip_035
  - video_000/clip_037
  repetitions: 3
  cache_order:
  - cold_component_cache
  - completed_court_prefix
  - all_node_cache_hit
  allocator_limit_gib: 10
  ball_checkpoint: ball_detection/i935-mixed-ft-s42-epoch09.ckpt
  stop_on_error: true
metrics: {}
artifacts:
  run_dir: knowledge/runs/run-video-batch-baseline-035037-20261116
  output_dir: outputs/tennis_scene/evaluate/video-batch-baseline-035037-20261116
  input_manifest_sha256: 3375770dc3bda6f7f4aea73d0bdb46d1a5055a225ee4ec877e17b42587777a11
parents:
- run-video-batch-baseline-20261116
relations: []
papers: []
tags: []
summary_impact: updated
session: 01a1dfbd-594e-7f72-9f3a-53d3f7f3a340
---

## 計測前に固定した手順

親判断 [#2150](https://github.com/Motoki0705/tennis-lab/issues/2150#issuecomment-6909180940) に従い、
video_000/clip_035（2,108frame）→clip_037（292frame）、cam0→cam1→cam2→cam3、全4cameraを使う。
どちらも59.94005994005994fps・1920×1080。全cameraのCourt採用は
[court-input-qualification.json](../../runs/run-video-batch-baseline-035037-20261116/court-input-qualification.json) に残した。
元datasetの登録情報と10ファイル（clip manifest 2＋動画8）のSHA-256を
[input-manifest.json](../../runs/run-video-batch-baseline-035037-20261116/input-manifest.json) に固定した。
動画はffprobeの全frame走査で完成とframe数を確認した。元datasetへは書かない。

[benchmark.py](../../runs/run-video-batch-baseline-035037-20261116/benchmark.py) のfreezeでoutputs配下へコピーし、
同じscriptのrunを共有training queueへresource=all・provider=codex・issue=1748で登録する。
実行commit・command・queue jobは実行時のrepro bundleに固定する。比較の3条件を同じprocessで次の順に実行し、
3回繰り返す。repeatごとに空のcomponent storeを持つ新しいdatasetコピーを用意し、clipはdataset生成APIから逐次処理する。

1. cold_component_cache: component artifactが存在しない状態から全段を通常設定で実行する。
2. completed_court_prefix: cold完了後のstoreを退避し、検証済みCourt検出・校正の8artifactだけを同じsource pathの新storeへコピーして採用する。下流artifact・export・完成markerは存在しない状態から再開する。準備・退避は計測の外。
3. all_node_cache_hit: 前条件で完成した全artifactを保持し、datasetの完成markerはoverwrite=Trueで再公開する。runnerの全nodeがcachedでモデルloadが0であることを必須とする。dataset markerだけのskipを全node hitの証拠にはしない。

repeat 1のcoldは初回import・CUDA初期化・資産hashを含む。以後は同一processのwarm状態だが、モデル寿命や解放時点を変えない。
OS page cacheはdropせず、入力・資産の事前hash検証も行うため、coldはcomponent cacheについての呼称である。
初回とwarm（repeat 2/3）の平均・標本SDを分け、段階別wall割合とモデルload/unload回数・時間、
PyTorch allocated/reserved peak、sampled RSSとprocess-lifetime RSS、一時artifact/export disk peakを記録する。
raw spanは大きくなるためoutputsへ保持し、knowledgeへ小さい集計とファイルhashを保存する。
途中段が失敗したら最初の例外で停止し、後続clip・条件・repeatを実行しない。閾値・camera・入力は変えない。

## 資源と環境の事前確認

GPUは16,311MiBのRTX 5060 Ti。モデルを逐次load/unloadする既存処理で、最大モデル（DINO/ViTPose/HMR2）は
FP32重み約2.4～2.5GiB、Courtは約1.2GiB。既存の単モデル実行と既定batchからPyTorch peakを10GiB以内と見積もる。
この見積もりは実測値ではない。従来のC1と同じ10GiBのPyTorch allocator上限を設定して
超過を明示的なOOMとして止め、driver/native分にはGPU全量との差を残す。batch・dtypeは既定のまま。
共有queueで全slotを確保してから実行する。上限は費用・所要時間の予算ではない。

配置切替後のdata/ckptはD:のmain実体を参照し、出力はこのworktree専用のpersistent outputsに置く。
データは既にD:にあるためhydrate不要。旧E:からDINO拡張だけを今回のoutputsへコピーし、SHA一致と
validate_dino_extensionのCPU dispatchを確認した。記録はrestored-artifacts.json。以前の失敗計測はGitのrun bundleを維持する。
third_party/DINO・dinov3は前回初期化済みの固定gitlinkを使い、共有venv・submodule sourceは変更しない。

## 実行前の停止リスク

既定frame_sampling.max_framesは1,024。clip_035の30fps格子は計算上1,055となり、court_sideの既定上限を超える。
設定は変更せず最初のcoldで実際の到達段と停止理由を記録する。これより前に停止した場合もそこで打ち切る。

## 計画時点の判断

結果はまだない。旧3clip＋追加8clipのCourt停止時間を基準値として流用しない。
summaryを新入力の計測待ちに改めるためsummary_impactはupdatedとする。
