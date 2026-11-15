---
id: run-video-batch-baseline-20261116
type: run
task: tennis_scene
sequence: 42
recorded_at: '2026-11-16'
date: '2026-11-16'
title: 動画逐次推論の固定入力・計測基盤と基準値
provider: codex
session: 01a1dfbd-594e-7f72-9f3a-53d3f7f3a340
issue: 1748
status: running
summary_impact: none
config:
  main_commit: e561377ce
  repetitions: 3
  clip_order: [video_000/clip_000, video_000/clip_001, video_000/clip_002]
  cache_order: [cold_component_cache, cache_hit]
  allocator_limit_gib: 10
  ball_checkpoint: ball_detection/i935-mixed-ft-s42-epoch09.ckpt
metrics: {}
repro:
  commit: 572e36a035eca82a875277f29b38d421859bb53b
  branch: codex/video-batch-baseline
  command: TENNIS_LAB_DINO_EXTENSION=$PWD/outputs/player_detection/evaluate/coco_fullframe/i964-r5-20260929/dino_extension/lib/MultiScaleDeformableAttention.so .venv/bin/python knowledge/runs/run-video-batch-baseline-20261116/benchmark.py run --root outputs/tennis_scene/video_batch_baseline/20261116 --expect-commit 572e36a035eca82a875277f29b38d421859bb53b --allocator-limit-gib 10
artifacts:
  run_dir: knowledge/runs/run-video-batch-baseline-20261116
  output_dir: outputs/tennis_scene/video_batch_baseline/20261116
  queue_job: 0-8205243842583373001_2026-11-15T15-22-37-416626998Z_259713_video-batch-baseline-c1
parents: []
relations: []
papers: []
tags: [inference, measurement, sequential]
---

計測対象はmain e561377ceへ計測のみを加えた572e36a03である。モデル寿命・clip/cameraの順序・閾値・精度・batch設定は変更していない。結果は計測中であり、高速化率やモデル読込の割合はまだ確定しない。summaryの標準pipelineについての判断を変える証拠は未取得なのでsummary_impactはnoneとする。

比較入力はmeiji_4camのvideo_000/clip_000（356frame）、clip_001（561）、clip_002（1516）の各4camera（cam0〜3）、59.94005994fps、1920×1080で固定した。dataset.jsonへの登録と12動画のffprobe全frame数を照合し、clip.jsonを含む15fileのコピー前後のSHA-256一致を確認した。選択登録は[dataset.json](../../runs/run-video-batch-baseline-20261116/dataset.json)、内容hashは[input-manifest.json](../../runs/run-video-batch-baseline-20261116/input-manifest.json)に保存した。manifest自体のSHA-256は4d0013ff12b47f3a274d9501e5bb65476ae12755f1f176fb49dcb16f43668009。元dataset・clip_studio・同期processには書き込まない。

再現スクリプトは[benchmark.py](../../runs/run-video-batch-baseline-20261116/benchmark.py)。repo rootからfreeze --root outputs/tennis_scene/video_batch_baseline/20261116で選択入力をコピーし、上記repro.commandを共有training queueのresource=allで実行する。既にfreezeしたrootは上書きせず、再実行は新しいoutputs配下のrootを使い、15fileのhashを固定manifestと照合する。checkpoint・sourceが当時と一致することを別途確認する。

1process内で3回、clip_000→001→002の順にdataset生成API generate_pseudo_annotationsを呼ぶ。各repeatは新しいstoreによるcold component cache、その直後に同じstoreを使うcache hitの順。dataset完成markerのskipを測らないようpublicationはoverwrite=True、component runnerのoverwriteは既定falseとする。repeat1は初回import・CUDA初期化込み、repeat2/3はprocess-warmだが現行寿命のモデル再loadを含む。OS page cacheはdropしないのでdisk coldを意味しない。再構成不能例では完成prefixだけがcache hitし、失敗nodeは再検査される。この条件を全node hitと呼ばない。入力の空検出・再構成不能は実際の結果から記録し、仮定しない。

RTX 5060 Ti（16,311MiB）、事前GPU使用1,581MiB、利用可能RAM28GiB、disk空き3.4TiBを確認した。同GPUでの過去の標準pipeline記録000030のGPU全体sample最大4,928MiBを見積りの根拠とし、ホスト余裕を保つためPyTorch allocatorを10GiBに制限した。これは人間が指定した費用・時間予算ではなく、本計測の資源制約である。上限を超えた失敗をbatch変更で回避しない。

DINOとdinov3のsubmoduleはrepoが固定する版でworktree内へinitした。既存のDINO CUDA binaryをCPUのvalidate_dino_extensionで検証し、TENNIS_LAB_DINO_EXTENSIONで明示選択した。PyTorchは2.13.0+cu130、CUDA toolkit13.0。GVHMRはsrcへ移植済みのコードを使うのでthird_party/GVHMRのinitは不要。

記録の時間・資源の意味は[recorder API](../../../src/utils/inference_recording.md)を正本とする。GPU同期を入れた同一recorder条件で比較し、inclusive累積と重複を除いたwall内訳を混同しない。raw JSON、receipt、outcomeはoutputs配下に保持する。
