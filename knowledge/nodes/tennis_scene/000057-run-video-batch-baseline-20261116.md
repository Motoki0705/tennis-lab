---
id: run-video-batch-baseline-20261116
type: run
task: tennis_scene
sequence: 57
recorded_at: '2026-11-16'
date: '2026-11-16'
title: 旧入力3clipのCourt停止例と逐次計測基盤
provider: codex
session: 01a1dfbd-594e-7f72-9f3a-53d3f7f3a340
issue: 1748
status: failed
summary_impact: updated
config:
  main_commit: e561377ce
  repetitions: 3
  clip_order:
  - video_000/clip_000
  - video_000/clip_001
  - video_000/clip_002
  cache_order:
  - cold_component_cache
  - cache_hit
  allocator_limit_gib: 10
  ball_checkpoint: ball_detection/i935-mixed-ft-s42-epoch09.ckpt
metrics:
  scope: Court failure prefix only
  cold_initial_seconds: 337.037786362
  cold_process_warm_mean_seconds: 254.2736666415
  cold_process_warm_sample_stdev_seconds: 2.0378536139647725
  prefix_cache_mean_seconds: 103.70480205
  prefix_cache_sample_stdev_seconds: 3.390068559437144
  completed_reconstructions: 0
  full_cache_hit_established: false
repro:
  commit: 572e36a035eca82a875277f29b38d421859bb53b
  branch: codex/video-batch-baseline
  command: TENNIS_LAB_DINO_EXTENSION=$PWD/outputs/player_detection/evaluate/coco_fullframe/i964-r5-20260929/dino_extension/lib/MultiScaleDeformableAttention.so
    .venv/bin/python knowledge/runs/run-video-batch-baseline-20261116/benchmark.py
    run --root outputs/tennis_scene/video_batch_baseline/20261116 --expect-commit
    572e36a035eca82a875277f29b38d421859bb53b --allocator-limit-gib 10
artifacts:
  run_dir: knowledge/runs/run-video-batch-baseline-20261116
  output_dir: outputs/tennis_scene/video_batch_baseline/20261116
  queue_job: 0-8205243842583373001_2026-11-15T15-22-37-416626998Z_259713_video-batch-baseline-c1
  metrics: knowledge/runs/run-video-batch-baseline-20261116/metrics.json
  recordings: knowledge/runs/run-video-batch-baseline-20261116/recordings
  input_manifest: knowledge/runs/run-video-batch-baseline-20261116/input-manifest.json
  queue_bundle: knowledge/runs/run-video-batch-baseline-20261116/queue
parents: []
relations: []
papers: []
tags:
- inference
- measurement
- sequential
---


計測対象はmain e561377ceへ計測のみを加えた572e36a03である。モデル寿命・clip/cameraの順序・閾値・精度・batch設定は変更していない。6条件のrecorder保存は完了したが、全3clipがCourtで停止し、C1の全モデル基準値は未達である。queueの終了コード0・recorderのcompleteは記録処理の完了であり、シーン再構成やC1の成功を意味しない。meiji_4camでの全モデル計測がCourt条件に阻まれることをsummaryへ反映したため、summary_impactはupdatedとする。

比較入力はmeiji_4camのvideo_000/clip_000（356frame）、clip_001（561）、clip_002（1516）の各4camera（cam0〜3）、59.94005994fps、1920×1080で固定した。dataset.jsonへの登録と12動画のffprobe全frame数を照合し、clip.jsonを含む15fileのコピー前後のSHA-256一致を確認した。選択登録は[dataset.json](../../runs/run-video-batch-baseline-20261116/dataset.json)、内容hashは[input-manifest.json](../../runs/run-video-batch-baseline-20261116/input-manifest.json)に保存した。manifest自体のSHA-256は4d0013ff12b47f3a274d9501e5bb65476ae12755f1f176fb49dcb16f43668009。元dataset・clip_studio・同期processには書き込まない。

再現スクリプトは[benchmark.py](../../runs/run-video-batch-baseline-20261116/benchmark.py)。repo rootからfreeze --root outputs/tennis_scene/video_batch_baseline/20261116で選択入力をコピーし、上記repro.commandを共有training queueのresource=allで実行する。既にfreezeしたrootは上書きせず、再実行は新しいoutputs配下のrootを使い、15fileのhashを固定manifestと照合する。checkpoint・sourceが当時と一致することを別途確認する。

1process内で3回、clip_000→001→002の順にdataset生成API generate_pseudo_annotationsを呼ぶ。各repeatは新しいstoreによるcold component cache、その直後に同じstoreを使うcache hitの順。dataset完成markerのskipを測らないようpublicationはoverwrite=True、component runnerのoverwriteは既定falseとする。repeat1は初回import・CUDA初期化込み、repeat2/3はprocess-warmだが現行寿命のモデル再loadを含む。OS page cacheはdropしないのでdisk coldを意味しない。再構成不能例では完成prefixだけがcache hitし、失敗nodeは再検査される。この条件を全node hitと呼ばない。入力の空検出・再構成不能は実際の結果から記録し、仮定しない。

RTX 5060 Ti（16,311MiB）、事前GPU使用1,581MiB、利用可能RAM28GiB、disk空き3.4TiBを確認した。同GPUでの過去の標準pipeline記録000030のGPU全体sample最大4,928MiBを見積りの根拠とし、ホスト余裕を保つためPyTorch allocatorを10GiBに制限した。これは人間が指定した費用・時間予算ではなく、本計測の資源制約である。上限を超えた失敗をbatch変更で回避しない。

DINOとdinov3のsubmoduleはrepoが固定する版でworktree内へinitした。既存のDINO CUDA binaryをCPUのvalidate_dino_extensionで検証し、TENNIS_LAB_DINO_EXTENSIONで明示選択した。PyTorchは2.13.0+cu130、CUDA toolkit13.0。GVHMRはsrcへ移植済みのコードを使うのでthird_party/GVHMRのinitは不要。

記録の時間・資源の意味は[recorder API](../../../src/utils/inference_recording.md)を正本とする。GPU同期を入れた同一recorder条件で比較し、inclusive累積と重複を除いたwall内訳を混同しない。raw JSON、receipt、outcomeはoutputs配下に保持する。


## 実測した結果と到達範囲

単位は秒。以下は3clipの**Court停止までだけ**を測った表であり、人物・ボール・身体復元を含むpipeline全体の基準値ではない。各値は重複を除いたwall attributionで、他のstageも含む全内訳・割合は[metrics.json](../../runs/run-video-batch-baseline-20261116/metrics.json)にある。

| repeat / 条件 | 総wall | model load | decode | 推論 | 後処理 | save | 公開 | unload | load/unload回数 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 / cold（初回込み） | 337.038 | 49.306 | 1.417 | 4.413 | 185.749 | 0.006 | 0.546 | 2.814 | 7/7 |
| 1 / 完成prefix再利用 | 102.241 | 16.820 | 0.396 | 1.591 | 70.407 | 0.000 | 0.000 | 1.110 | 3/3 |
| 2 / cold（process-warm） | 252.833 | 43.260 | 1.028 | 4.052 | 186.330 | 0.007 | 0.489 | 2.734 | 7/7 |
| 2 / 完成prefix再利用 | 107.581 | 18.470 | 0.439 | 1.634 | 73.550 | 0.000 | 0.000 | 1.060 | 3/3 |
| 3 / cold（process-warm） | 255.715 | 45.268 | 1.318 | 3.873 | 188.380 | 0.006 | 0.566 | 2.732 | 7/7 |
| 3 / 完成prefix再利用 | 101.293 | 16.539 | 0.480 | 1.654 | 70.077 | 0.000 | 0.000 | 1.091 | 3/3 |

初回coldは337.038秒。process-warmでcold component cacheを作り直した2/3回目は平均254.274秒、標本標準偏差2.038秒（n=2）。完成prefix再利用は平均103.705秒、標本標準偏差3.390秒（n=3）。初回とprocess-warmを混ぜて速度改善とは扱わない。全node cache hitは成立していない。

| 条件 | load割合 | decode割合 | 推論割合 | 後処理割合 | save＋公開割合 |
|---|---:|---:|---:|---:|---:|
| repeat 1 cold | 14.63% | 0.42% | 1.31% | 55.11% | 0.16% |
| repeat 1 prefix再利用 | 16.45% | 0.39% | 1.56% | 68.86% | 0.00% |
| repeat 2 cold | 17.11% | 0.41% | 1.60% | 73.70% | 0.20% |
| repeat 2 prefix再利用 | 17.17% | 0.41% | 1.52% | 68.37% | 0.00% |
| repeat 3 cold | 17.70% | 0.52% | 1.51% | 73.67% | 0.22% |
| repeat 3 prefix再利用 | 16.33% | 0.47% | 1.63% | 69.18% | 0.00% |

残りはモデルunload・前処理・転送・artifact load・計測用hash/CUDA初期化・未分類時間である。`measurement`は初回81.107秒、以降7.334〜7.856秒で、初回のimport/CUDAとpublication identity/weight hashを含む。重みloadの時間へ合算しない。

実際にload/unloadされたモデルはCourtのみで、coldは各7/7回、prefix再利用は各3/3回、6条件合計30/30回。checkpointのSHA-256はb863df1f01f00d2ff00a21d56a461879e3c5af193a9f32cec47a88d2842f8383、device=cuda、dtype=float32。DINO・Ball・ViTPose・CLIP・HMR2・GVHMR・body-modelは未到達であり、その速度・VRAMを測定済みとしない。Court入力の実batch shapeは各raw recordingの`batch_shapes`にある。

| repeat / 条件 | PyTorch peak allocated MiB | peak reserved MiB | sampled RAM peak MiB | temporary disk bytes |
|---|---:|---:|---:|---:|
| 1 / cold | 1028.15 | 1176.00 | 3562.11 | 11940 |
| 1 / prefix再利用 | 1028.90 | 1176.00 | 3560.43 | 0 |
| 2 / cold | 1028.90 | 1176.00 | 3561.78 | 11940 |
| 2 / prefix再利用 | 1028.90 | 1176.00 | 3560.36 | 0 |
| 3 / cold | 1028.90 | 1176.00 | 3561.56 | 11940 |
| 3 / prefix再利用 | 1028.90 | 1176.00 | 3561.28 | 0 |

RAMは100ms sampling、VRAMはPyTorch allocatorのpeakであり、他processやdriverを含むGPU全体の使用量ではない。失敗時もCourt unloadが30回すべて記録された。CPUのfake component検証では計測有無でartifact bytes・artifact ID・execution_keyが一致し、cache/load時のモデル未実行を確認した。追加17ケースはcollect-onlyで収集し、各関数の最大展開数5、最長call0.50秒、setupは0.005秒未満。関連382テストは27.39秒で全pass、pre-commitのruff/mypyもpassした。

停止は毎回同じで、clip_000はcourt_detection/cam0、clip_001はcam1、clip_002はcam3。停止cameraの候補はいずれも最大7/7/6 inliersで、既定の8点支持条件を満たさない。成功済みCourt成果物はclip_001/cam0とclip_002/cam0〜2の4件だけで、cache条件ではこの4件を再利用し、失敗した3nodeを再実行した。空検出の実データ例・全モデル実行・全cache hitは未確認である。

[追加8clipの入力選定](000066-run-video-batch-input-qualification-20261116.md)でもcam3が全件で不成立だった。この8件のtimingは本表へ混ぜない。clip_011以降やdataset全69clipの失敗は主張しない。閾値・精度・モデル・処理順の変更、新clip作成、source datasetへの書込みは行っていない。C1を完了するには、Court側の対処を別途先行させるか、比較入力を見直す親判断が必要である。

raw recorder JSON・receipt・outcomeは本run bundleへコピーし、[内容hash一覧](../../runs/run-video-batch-baseline-20261116/evidence-files.json)と再集計用`summarize_evidence.py`を保存した。sourceの厳密な再現は`repro.commit`へcheckoutして実行する。記録後に非有限intervalの拒否・resource失敗JSON保存・ゼロ長span集計・BallのMDD変換後のbatch shape記録を補強した。実測spanは全て正の長さでBallへ未到達のため、これらは保存済みの計測値を変えない。モデル処理・寿命・設定を変更していない。独立validatorは未指定のため指定0回・試行0回・完了0回である。
