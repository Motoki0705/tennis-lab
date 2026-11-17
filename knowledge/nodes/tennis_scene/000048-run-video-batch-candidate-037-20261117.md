---
id: run-video-batch-candidate-037-20261117
type: run
task: tennis_scene
sequence: 48
recorded_at: '2026-11-17'
title: C12後のCourt確認とclip_037の人物対応での停止実測
issue: 1748
provider: codex
date: '2026-11-17'
status: failed
config:
  main_commit: 486aa1ab09be857909d0fcf8c2b616c75a53f5b0
  instrumented_commit: 701165bd2deeddc20d4773b3dbacb6890f8c15ae
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
metrics:
  total_wall_seconds: 1450.508127319
  model_load_seconds: 269.473720212
  model_load_percent: 18.57788420048863
  inference_seconds: 838.04937187
  inference_percent: 57.776261717263274
  completed_nodes: 30
  failed_nodes: 1
  load_events: 25
  unload_events: 21
  ram_rss_peak_bytes: 8637988864
  cuda_allocated_peak_bytes: 3325809152
  cuda_reserved_peak_bytes: 4389339136
  tracked_temporary_disk_peak_bytes: 23518833
artifacts:
  run_dir: knowledge/runs/run-video-batch-candidate-037-20261117
  output_dir: outputs/tennis_scene/evaluate/video-batch-candidate-037-20261117
  input_manifest_sha256: 497f75bf338ec5a918f9011cea789b7375bd184ce9407a7b6583126c24194002
parents:
- run-video-batch-baseline-035037-20261116
relations: []
papers: []
tags: []
summary_impact: updated
---

## 実行前の固定

#1748の続行指示で、長さ上限内の実clipを現行mainで全段確認する。
C12後のmainはlegacy既定を維持し、C13のPR #2222はまだ未mergeだった。
旧全276画像のCPU replayでは既定が完全一致し、旧全camera採用の035/037のうち035だけが長さ上限を超える。
まず037（292frame・4camera）を1回cold実行して、Courtの先から何が止まるかを記録する。
この単一clip・単一条件は最終の複数clip cold/warm/cache基準値にはしない。
全段基準値は未達のまま、新たな人物対応での停止と次に必要な判断をsummaryへ反映する。

入力は登録情報・clip.json・4動画のSHAと全frame数で固定し、outputs内へコピーした。元datasetへ書かない。
実行commitとcommandはqueue reproで固定する。実行は共有training queueのall枠、provider=codex、issue=1748。
既定i935・model寿命・dtype・batch・判定条件を維持し、最初の例外で停止する。

GPUは16GB。前回035のpre-body peakはallocated3.07GiB/reserved4GiBだった。
短い037のHMR2/GVHMRを含む逐次loadと既定batchから10GiB以内を事前見積もりとし、
従来C1と同じallocator10GiB上限で超過を明示OOMにする。見積もりを実測値として扱わない。
DINO拡張は前回検証済みの同じbinaryを明示し、data/ckptとvenvを共有する。

## C12後の既定Courtと元3clip

GPU確認と独立に、#2190の保存済みneural raw・LINE距離場・RGBへ、同じmainの
既定legacy後処理をCPUで再適用した。入力12camera・144regionは
旧処理とのH・投影・支持mask・残差・領域採否・選択が全件一致した。
source HEADは`701165bd2deeddc20d4773b3dbacb6890f8c15ae`、
main基点は上記486aa1ab0。後続mainの7955a300fはworker実行記録だけの変更である。

| clip | 既定で不採用のcamera | 最大支持候補の診断 |
|---|---|---|
| 000 | cam0 | inliers 7 < 8 |
| 001 | cam1 / cam3 | 7 < 8 / 9点候補もcontent内包条件がfalse |
| 002 | cam3 | inliers 6 < 8 |

従って#2190のmergeで元の3clipの停止が解消されたとは扱えない。
#2190はrevision 4で旧処理一致を受入条件としており、今回の観測もその契約と一致する。
#2191の動画共有（PR #2222）は既定clip推論と別の明示モードで、このCPU確認には使わない。
新たなneural forward・camera除外・閾値変更・既存cacheのidentityを無視したloadはない。

406.195秒は旧/新solver両方・入力hash再確認を含む診断runのwallで、基準値ではない。
GPU未初期化、入力/sourceの内容hashは実行前後で不変。
[CPU集計](../../runs/run-video-batch-candidate-037-20261117/court-default-replay/summary.json)、
[入力/source manifest](../../runs/run-video-batch-candidate-037-20261117/court-default-replay/inputs-and-source.json)、
[再現script](../../runs/run-video-batch-candidate-037-20261117/court-default-replay/replay_original_inputs.py)
と全12件の比較診断を保存した。

clip_035の1055点上限は、独立した[PR #2229](https://github.com/Motoki0705/tennis-lab/pull/2229)で
CPU検証と既定2048への変更を扱う。ここでは037の入力・sampling条件を変更しない。

## clip_037の実行結果

共有queue job
`0-8205081638961772343_2026-11-17T12-26-01-038227656Z_1037232_video-batch-candidate-037`
はexit 1、terminal stateはfailed。Court/Ball/DINO/追跡/poseを4cameraとcourt_sideの
計30nodeが完了し、31番目のplayer_associationが
`player_association_ambiguous_players`で停止した。
diagnosticsには各sideに2 identity、計4 identity（presenceは全て290/292 frame）がある。
既定の`players_per_side=1`ではside -1のplayer候補0とrunner-up 1を絞れなかった。
これは検出対象の実人数を人間が確認したという記録ではない。
人数・閾値・camera・入力を変えず、再試行せずに停止した。
[receiptと全診断](../../runs/run-video-batch-candidate-037-20261117/receipts/repeat-1-cold_component_cache.json)、
[queue terminal record](../../runs/run-video-batch-candidate-037-20261117/queue/completion.json)。

| 測定区間 | 秒 | wall比率 % |
|---|---:|---:|
| total | 1450.5081 | 100 |
| decode | 5.1900 | 0.3578 |
| preprocess | 25.6218 | 1.7664 |
| transfer | 3.9449 | 0.2720 |
| inference | 838.0494 | 57.7763 |
| model load | 269.4737 | 18.5779 |
| model unload | 67.2488 | 4.6362 |
| postprocess | 111.5826 | 7.6927 |
| save | 0.8451 | 0.0583 |
| publication処理 | 15.0677 | 1.0388 |
| artifact load | 0.9558 | 0.0659 |
| measurement | 106.8211 | 7.3644 |
| unattributed | 5.7071 | 0.3935 |

重複しないwall分割の合計はtotalと一致する。publicationには失敗時処理を含み、
SceneResult/完成datasetを公開できたという値ではない。
measurementにはnativeなpublication identity生成とrecorderのhash/CUDA処理が入り、
全てを追加observer overheadとは扱わない。Torch/OpenCVのCPU importと入力検証・コピーは窓外、
初回CUDA/model load・pipeline importは窓内。原3clipのCPU replayを別processで並行した
到達確認であり、比較用の最終baselineとはしない。

| モデル（全てfloat32） | device | load/unload回数 | load秒 | unload秒 |
|---|---|---:|---:|---:|
| Court | cuda | 4/4 | 17.7241 | 1.1878 |
| Ball（i935） | cuda | 4/4 | 2.9340 | 1.2209 |
| DinoPersonDetector | cuda | 4/4 | 12.1689 | 62.0987 |
| AFLink | cpu | 4/0 | 0.1937 | 0 |
| CLIP-ReID | cuda | 5/5 | 167.6618 | 1.3797 |
| ViTPosePose2D | cuda | 4/4 | 68.7911 | 1.3617 |

CLIPは追跡4回＋人物対応1回。人物対応の例外時もunload eventを記録した。
CLIPのunloadは元の削除/cleanup境界で、FeatureExtractorが保持する参照の実解放完了を
保証する時刻ではない。AFLinkは元のコードに明示unloadがなく、記録上も0回。
HMR2/GVHMR/body-model資産の段には到達しておらず、その実測を捏造しない。
重みSHA-256とdtype/deviceは[metrics](../../runs/run-video-batch-candidate-037-20261117/metrics.json)に保持した。

batch/call実数はCourt 52、Ball 72（288 items）、DINO 1168、ViTPose 1168
（9356 crops、flip後18712）、CLIP 1172（9595 crops）。AFLinkは2回。
モデル別inclusive時間と実tensor形状別集計はmetricsに保存した。
これらはstageのwall partitionと異なる集計で、相互に足してtotalとはしない。

peakはPyTorch CUDA allocated 3,325,809,152 bytes／reserved 4,389,339,136 bytes、
process RSS 8,637,988,864 bytes、追跡した一時artifact/export disk 23,518,833 bytes。
VRAMはPyTorchのpeakで、driver/desktop全体使用量ではない。RAMはこのprocessの値。
diskは登録した一時領域で、窓外に作った動画入力コピー全体を含まない。

raw recorderは40,897 events・11,072,957 bytes、SHA-256
`6446c4c948a467101d0c71bf43be7bf735466a12e22ddb8d63119d016e7dcae9`。
outputsへ保持し、小さいmetrics・31 node eventsを含むlifecycle・queue repro・全receiptをGitへ保存した。
source/driverは実行前後でcommitと内容が一致し、入力SHAも一致した
（[照合](../../runs/run-video-batch-candidate-037-20261117/measured-source-after-run.json)）。
原035/037の測定commitは`evidence/i1748/clip-035-20261116`と
`evidence/i1748/clip-037-20261117`のannotated tagでもoriginへ保存した。
[再現手順](../../runs/run-video-batch-candidate-037-20261117/reproduce.md)を参照する。

## 最新mainとの関係と残件

測定後に#2191がmergeされたため、報告する実装はmain
`1ccbede7654552590b09474e6a6daf8d5f39875f`へrebaseした。
そのmainも既定は`court_kp.mode=clip`、`players_per_side=1`、sampling cap1024。
人物対応の全Pythonとcomponentは測定時とbyte一致する
（[parity](../../runs/run-video-batch-candidate-037-20261117/current-main-association-parity.json)）。
GPU測定をC13後のcommitで実行したとは扱わない。
動画共有は明示モードなので、元3clipの既定Court停止が自動で解消されたとも扱わない。

rebase後のrecorder/runner/Court/configurationは90件・14.69秒、動画共有のintegration/policyは
45件・21.30秒で成功。source更新前に測った44件の成功ログも保持する。
上記135件のcollect-onlyは4.42秒、pre-commitは変更Python32ファイルでruff/mypyとも成功。
実装の既存CPU契約と新mainの接続検証で、全段GPU成功や最終baselineの代替ではない。

C1の全段・複数clip・cold/完成prefix/全cache hit・各3反復は未成立。
人物数/選別条件と動画共有モードをC1にどう適用するかは親へ返す。
勝手に片側2人へ変えたり、通る別camera/clipへ替えたりしない。
PR #2154はdraftを維持し、merge/issue closeは行わない。
