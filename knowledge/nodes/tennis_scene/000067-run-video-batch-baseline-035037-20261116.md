---
id: run-video-batch-baseline-035037-20261116
type: run
task: tennis_scene
sequence: 67
recorded_at: '2026-11-16'
title: 固定2clipの逐次計測：clip_035がcourt_sideの長さ上限で停止
issue: 1748
provider: codex
date: '2026-11-16'
status: failed
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
metrics:
  scope: clip_035 failure prefix only
  cold_initial_seconds: 6289.219690991
  model_load_seconds: 52.978398999
  model_load_wall_percent: 0.8423683954766117
  inference_seconds: 5638.880673035
  completed_reconstructions: 0
  measured_conditions: 1
  warm_repeats_measured: 0
  prefix_cache_measured: false
  all_node_cache_hit_established: false
artifacts:
  run_dir: knowledge/runs/run-video-batch-baseline-035037-20261116
  output_dir: outputs/tennis_scene/evaluate/video-batch-baseline-035037-20261116
  input_manifest_sha256: 3375770dc3bda6f7f4aea73d0bdb46d1a5055a225ee4ec877e17b42587777a11
  queue_job: 0-8205188447301195999_2026-11-16T06-45-52-698804000Z_2775652_video-batch-baseline-035037
  persistent_output_dir: .worktree-outputs/video-batch-baseline/tennis_scene/evaluate/video-batch-baseline-035037-20261116
parents:
- run-video-batch-baseline-20261116
relations: []
papers: []
tags: []
summary_impact: updated
session: 01a1dfbd-594e-7f72-9f3a-53d3f7f3a340
repro:
  commit: cf4a1b0459588cf77f5ed81af9b69751c5b65ec7
  branch: codex/video-batch-baseline
  command: env TENNIS_LAB_DINO_EXTENSION=/home/kamimura/projects/tennis-lab/.worktree-outputs/video-batch-baseline/tennis_scene/evaluate/video-batch-baseline-035037-20261116/toolchain/MultiScaleDeformableAttention.so
    .venv/bin/python knowledge/runs/run-video-batch-baseline-035037-20261116/benchmark.py
    run --root outputs/tennis_scene/evaluate/video-batch-baseline-035037-20261116
    --expect-commit cf4a1b0459588cf77f5ed81af9b69751c5b65ec7 --allocator-limit-gib
    10
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
2. completed_court_prefix: cold完了後のstoreを退避し、検証済みCourt検出4＋全cameraまとめた校正1の5artifactだけを同じsource pathの新storeへコピーして採用する。下流artifact・export・完成markerは存在しない状態から再開する。準備・退避は計測の外。
3. all_node_cache_hit: 前条件で完成した全artifactを保持し、datasetの完成markerはoverwrite=Trueで再公開する。runnerの全nodeがcachedでモデルloadが0であることを必須とする。dataset markerだけのskipを全node hitの証拠にはしない。

repeat 1のcoldは初回モデルload・CUDA初期化・資産hashを含む。入力検証がTorch/OpenCVを先にimportするため、これらのimportは計測外。Hydra/orchestratorはその時点では未importで、CUDAも未初期化（measurement-window-imports.jsonの独立CPU確認）。以後は同一processのwarm状態だが、モデル寿命や解放時点を変えない。
OS page cacheはdropせず、入力・資産の事前hash検証も行うため、coldはcomponent cacheについての呼称である。
初回とwarm（repeat 2/3）の平均・標本SDを分け、段階別wall割合とモデルload/unload回数・時間、
PyTorch allocated/reserved peak、sampled RSSとprocess-lifetime RSS、一時artifact/export disk peakを記録する。
raw spanは大きくなるためoutputsへ保持し、knowledgeへ小さい集計とファイルhashを保存する。
途中段が失敗したら最初の例外で停止し、後続clip・条件・repeatを実行しない。閾値・camera・入力は変えない。

再実行は既存runを置換せず、新しいoutputs内の`--root`へfreezeしてから、同じrootでrunする。
元の測定を再現する場合はrepro.commitの版を使い、freeze後に保存済み10ファイルのSHAを照合する。
全datasetの登録情報だけが変わった場合は、旧runのfrozen inputとinput-manifest.jsonを新rootへコピーしてverifyする。
DINO拡張はrestored-artifacts.jsonのSHAと同じbinaryを明示し、runの正確な引数はqueue/run.jsonを参照する。
`--expect-commit`は測る版のfull SHAを指定し、共有queueへ`--resource all --provider codex --issue 1748`と自身のsessionで登録する。

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

## 観測結果

最初のcold条件でclip_035の29nodeが実行・保存された後、court_sideが
`ReconstructionUnavailable(reason=clip_too_long)` で停止した。
エラーは `Full clip has 1055 sampled frames; maximum is 1024`。
clip_037・prefix再開・全node hit・repeat 2/3は実行していない。
共有queueはexit 1 / failedで終わり、process groupの終了後に予約が解放された。
条件・閾値・camera・dtype・batchを変更して回避していない。



総wallは **6289.220秒（104.820分）**。以下は重複しないwall配分で、
並列decodeとGPU処理はrecorderの規約で配分する。inclusive値とraw spanも保持する。
measurementにはrecorderの資産hash・CUDA初期化と、通常のpublication identity準備も含む。全量をrecorderだけの追加コストとは解釈しない。

| stage | wall秒 | 総wall比 |
|---|---:|---:|
| artifact_load | 10.486 | 0.1667% |
| decode | 19.762 | 0.3142% |
| inference | 5638.881 | 89.6595% |
| measurement | 85.736 | 1.3632% |
| model_load | 52.978 | 0.8424% |
| model_unload | 6.817 | 0.1084% |
| postprocess | 256.094 | 4.0720% |
| preprocess | 173.390 | 2.7569% |
| publication | 0.361 | 0.0057% |
| save | 10.659 | 0.1695% |
| transfer | 27.637 | 0.4394% |
| unattributed | 6.418 | 0.1021% |

| model（全てFP32） | device | load / unload境界のevent回数 | load秒 | unload秒 |
|---|---|---:|---:|---:|
| Court | cuda | 4 / 4 | 17.611143 | 1.281592 |
| Ball | cuda | 4 / 4 | 1.077374 | 1.139187 |
| DinoPersonDetector | cuda | 4 / 4 | 10.018469 | 1.425057 |
| AFLink | cpu | 4 / 0 | 0.159524 | 0.000000 |
| CLIP-ReID | cuda | 4 / 4 | 2.122259 | 1.315723 |
| ViTPosePose2D | cuda | 4 / 4 | 21.989630 | 1.655124 |

AFLinkは既存実装に明示unloadがなく、終了時点の解放時間を捏造していない。
CLIPのunloadは既存のdel/cleanup境界の時間であり、FeatureExtractorが参照を保持し得るため、VRAM実解放完了の時間ではない。
player_associationのCLIP・HMR2・GVHMR・body-modelは未到達。
各資産のSHA・実batch shape分布・件数はmetrics.json、load/unloadとnodeのraw eventはlifecycle/にある。
資産16entryは実行前後でSHAとctimeが一致した。

PyTorch allocator peak allocated **3148.43MiB** / reserved **4096.00MiB**。
sampled RSSとprocess-lifetime peakはともに **9417.94MiB**、
登録したartifact/export一時directoryのdisk peakは **464.04MiB**。
GPU全体・driver/native allocation・未登録の一時file・input準備コピーはこのpeakの対象外。

## 証拠と限界

[metrics.json](../../runs/run-video-batch-baseline-035037-20261116/metrics.json) と
[receipt](../../runs/run-video-batch-baseline-035037-20261116/receipts/repeat-1-cold_component_cache.json) が集計と停止位置の正本。
raw recorder JSON（約75MB）はpersistent outputsのrecordings/へ保持し、evidence-files.jsonにSHAとbyte数を固定する。
小さいlifecycle/node event、全資産・入力hash、queueのrun/repro/terminal記録はGitへ保存した。
再集計はpromote.pyで行う。原測定のdriverはmeasured-benchmark.pyへそのまま保存し、元commandはqueue/repro.shに残す。
測定直後の照合でsrcとdriverが測定commitに一致した（measured-source-after-run.json）。

実測後の修正は、未知nodeの例外型を維持すること、完了node再呼出しの二重計測を防ぐこと、
未実行のprefix準備で校正を1artifactとして数えること。
前二者は3caseで修正前に失敗し、修正後に成功した。
raw測定の各node実行は1回ずつで、未知node・完了node再呼出し・prefix準備を通っていないため、保存値には影響しない。
修正後のGPU再実行は行っていない。

prefix準備はcheck_prefix.pyで実store/codecと2clipのfake CPU成果物を使って確認し、
5artifactのbytes/ref/identityの維持、下流artifactとexportの不在を確認した。
これは実GPUのprefix再開・全cache hit測定の代用ではない。

関連 **580case / 55.68秒**、focused **50case / 18.73秒**が成功。
追加はPR全体22case（この継続で5case）、最大展開5、最長call 0.64秒、setup 0.015秒未満。
collect-only・pre-commit・knowledge validatorの証拠はcpu-validation.jsonとログに残す。独立validatorは指定/試行/完了0/0/0。

モデル読込の総wall比はこの停止prefixで0.84%、推論は約89.66%だった。
全段・2clip完走の基準値、warmの分散、全node hit、実データの空検出例は未確認。
C4/C8の高速化率へ外挿しない。長さ上限と入力の扱いを親が判断する必要があり、C1完了とはしない。
旧Court停止例も失敗記録として保持する。
summaryを現在の停止点へ更新したためsummary_impactはupdatedとする。
