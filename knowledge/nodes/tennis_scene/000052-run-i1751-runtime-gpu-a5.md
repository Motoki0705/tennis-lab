---
id: run-i1751-runtime-gpu-a5
type: run
task: tennis_scene
sequence: 52
recorded_at: '2026-11-18'
title: runtime両版がGPU完走し読込は各1回、BF16 Ball差分で厳密比較は不合格
issue: 1751
provider: codex
session: 01a1e969-a58c-7d10-be3e-4e0795fd9e18
date: '2026-11-18'
status: failed
config:
  profile: pipeline_x3d_dpt
  clip: meiji_3cam/video_000/clip_000
  cameras: 3
  source_frames: 1010
  sampled_frames: 506
  seed: 42
  ball_batch_size: 1
  ball_precision: bf16
  compile_mode: default
  sampling_max_frames: 1024
  main_commit: 18a2fea05291eeaccb6d1dbfc81160e26c4080dd
  runtime_commit: 1bef628a9c2c5761d241ddc5c1b98c0f89625b35
  torch: 2.13.0+cu130
  driver: '621.24'
metrics:
  main_frames: 1010
  runtime_frames: 1010
  completed_variants: 2
  gpu_anomalies: 0
  gpu_attempts: 1
  gpu_retries: 0
  queue_exit_code: 1
  numerical_acceptance: false
  component_arrays: 266
  scene_array_fields: 29
  all_discrete_exact: true
  runtime_resource_loads:
    court: 1
    ball: 1
    dino: 1
    vitpose: 1
    appearance: 1
    hmr2: 1
    gvhmr: 1
    body_geometry: 1
  runtime_resource_unloads:
    court: 1
    ball: 1
    dino: 1
    vitpose: 1
    appearance: 1
    hmr2: 1
    gvhmr: 1
    body_geometry: 1
  runtime_max_owned_models: 2
  runtime_unclosed_instances: 0
  main_nvml_peak_mib: 4533
  runtime_nvml_peak_mib: 4568
  main_peak_allocated_bytes: 3159937536
  runtime_peak_allocated_bytes: 3159313920
  main_wall_seconds: 2222.512450213
  runtime_wall_seconds: 2263.276122110001
  ball_3d_max_abs_m: 0.0002738386392593384
  ball_3d_max_l2_m: 0.00036189522009106684
  raw_detection_max_axis_px: 0.1287841796875
  raw_confidence_max_abs: 0.0008233785629272461
repro:
  commit: 1bef628a9c2c5761d241ddc5c1b98c0f89625b35
  branch: codex/scene-inference-runtime
  remote: git@github.com:Motoki0705/tennis-lab.git
  command: PYTHONHASHSEED=42 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1
    TENNIS_LAB_DINO_EXTENSION=/home/kamimura/projects/tennis-lab/.worktree-outputs/video-batch-baseline/tennis_scene/evaluate/video-batch-baseline-035037-20261116/toolchain/MultiScaleDeformableAttention.so
    .venv/bin/python -u /home/kamimura/projects/tennis-lab/.worktree-outputs/scene-inference-runtime/tennis_scene/evaluate/runtime-i1751-20261118-a5/run_pair.py
artifacts:
  run_dir: knowledge/runs/run-i1751-runtime-gpu-a5
  log: .training_queue/logs/0-8205064145120710330_2026-11-17T17-17-34-879289669Z_313929_i1751-runtime-main-pair-20261118-a5.log
  output_dir: .worktree-outputs/scene-inference-runtime/tennis_scene/evaluate/runtime-i1751-20261118-a5
parents:
- run-i1751-runtime-gpu-a4
relations: []
papers: []
tags:
- inference-runtime
- bf16
- numerical-mismatch
summary_impact: none
---

## 結論と判定

親の指示によるa5を共有training queue（resource=all）から1回だけ実行した。
main・runtimeの両版が全1010frameを完走し、31component成果物とSceneResult `ok` を保存した。
CUDAエラー・heartbeat停止・NVML無応答は検出せず、GPU再投入は0回だった。
モデル共有・最終需要後の解放は実GPUのload/unload記録で確認できたが、
**事前のfloat atol=1e-6 / rtol=1e-5をBallの差分が超えたため、厳密な成果物比較は不合格**。
比較処理exit1によりqueueもfailedとなった。GPU推論の終了コードは両方0であり、区別して扱う。
PR #2182はDraftを維持し、差分の受入判断を親へ返す。数値条件を事後に広げて合格とはしていない。

[集計と全モデルの計測値](../../runs/run-i1751-runtime-gpu-a5/acceptance-summary.json)、
[全配列・JSONの比較](../../runs/run-i1751-runtime-gpu-a5/comparison-completed.json)、
[元の比較結果](../../runs/run-i1751-runtime-gpu-a5/comparison.json)、
[queue終端](../../runs/run-i1751-runtime-gpu-a5/terminal.json)を保存した。

## 固定条件と実行版

基準mainは `18a2fea05`、runtimeは `1bef628a9`。Meiji 3cam `video_000/clip_000` の
全3camera・1010frame、約30fps格子506frameを使い、切り詰めていない。
`pipeline_x3d_dpt`、人物・GVHMR・body placement有効、seed42、Ball batch1・BF16・compile(default)。
入力・全重み・DINO拡張のSHAと有効設定はa4のpinをそのまま使い、両worktreeで実行前にCPU照合した。
最新mainの上限変更の影響を除くため、frame_sampling.max_frames=1024を両版に明示した。
Torch 2.13.0+cu130 / CUDA 13.0 / Python 3.11.15 / RTX 5060 Ti / driver621.24が両版で一致する。
lockのTorch版と今回の共有venvを同一とは扱わない。
[inputs](../../runs/run-i1751-runtime-gpu-a5/inputs.json)と[固定7ファイル](../../runs/run-i1751-runtime-gpu-a5/frozen-files.json)は
終了後も不変だった。推論中にsrc・固定scriptを編集していない。

同じ起動担当sessionを共通runnerから再開し、登録は1件。
02:20:00 JSTにmain Court cam0の公開とchecksum一致を確認して起動担当は終了し、その後はcallerが
異常・variant完了・job leader終了をfile event / pidfdで監視した。
mainは02:54:47 JST、runtimeは03:32:40 JSTに終了した。
stop flagは登録前から無く、停止していたworkerをlauncher外の一時user serviceで起動した。
他担当のjobやretainedな旧queue記録は操作していない。

## 読込回数と寿命

| 対象 | mainのbackend load | runtimeのbackend load |
|---|---:|---:|
| Court | 3 | 1 |
| Ball | 3 | 1 |
| DINO | 3 | 1 |
| ViTPose | 3 | 1 |
| CLIP-ReID（追跡＋人物対応） | 4 | 1 |
| HMR2 | 1 | 1 |
| GVHMR | 1 | 1 |
| body vertices / COCO17 / root資産 | 各1 | 各1 |

runtimeの8resource（最後のbody資産群は `body_geometry` としてまとめる）はload/unload各1回。
同時所有は最大2、終了後の未解放instanceは0、failed hookは0。
[main計測](../../runs/run-i1751-runtime-gpu-a5/main-measurement.json)と
[runtime計測](../../runs/run-i1751-runtime-gpu-a5/runtime-measurement.json)にbackend記録・runtime hook16件を保持する。
C1 PR #2154は未mergeのため、この実測は同じAPIのrecorder非依存hookへ送った記録である。
clip/camera数の増加に対するCPU fake計画の証拠と合わせ、3cameraで共有することを確認した。

## VRAMと時間

| 指標 | main | runtime |
|---|---:|---:|
| Torch allocated peak（bytes） | 3159937536 | 3159313920 |
| Torch reserved peak（bytes） | 3776970752 | 3802136576 |
| NVML device観測最大（MiB、約1秒間隔） | 4533 | 4568 |
| GC後Torch allocated（bytes） | 33554432 | 33554432 |
| 計測wall（秒） | 2222.512450213 | 2263.276122110001 |

Torch counterのpeakと、desktop等も含むNVMLのsample最大を区別する。
この1組はPython loader観測・入力hash・compileを含み、ばらつきの反復はない。
読込回数は減ったが、wallの高速化を示した結果ではない。

## 数値差分の位置と影響

266component配列・29scene配列を比較した。shape/dtypeは一致し、全ての離散値・mask・IDは完全一致。
Court、DINO、ViTPose、CLIP、HMR2、GVHMR、body placementのモデル出力も完全一致した。
差分の起点はBallのdense heatmapで、cameraごとの最大絶対差は
0.001535416 / 0.001592189 / 0.001630783。
cam0 frame424のraw検出だけが変わり、座標成分最大差0.128784px（L2約0.135px）、score差0.000823379だった。
cam1/2の検出位置・scoreは完全一致。候補cell・valid・selected windowなど離散evidenceも一致した。

平滑化を経たcam0の2D位置差は最大0.022766px。最終ball 3Dでは14frame・42座標成分に差があり、
成分最大0.000273839m（0.274mm）、位置L2最大0.000361895m（0.362mm）。
正規化ball_uvの最大差は1.1861324e-5。
JSON payloadと最終metadataも別途比較し、side検証costの差などは同じfloat条件内だった。
両archiveを公開APIで再読込してschema検証した。
ball 3D有効772frame、2人のSMPL有効1000/993frameなどのvalidity集計も両版で一致する。

## 原因について分かった範囲と親への判断

全入力pin・dtype・batch・モデル重みは一致し、`src/tasks/ball_detection` は両commitでbyte変更なし。
従ってBallの移植コード差を見つけた結果ではない。一方、この1組には同じmainの再実行対照がなく、
BF16/compile backendの数値変動と、モデル寿命・実行順の副作用を切り分けてはいない。
原因は未確定であり、GPUが正常終了したことだけを数値受入成功とはしない。

[既存X3D接続の記録](000043-run-scene-x3d-dpt-20261116.md)にも、同一device/入力のclass比較で
BF16 dense heatmap最大0.0004399512の差が残った記録がある。
同記録の履歴validation用BF16境界1px / score0.005には今回の検出差は収まるが、
その境界はC3の事前比較条件ではないので、自動で転用していない。
[PyTorchの公式再現性文書](https://docs.pytorch.org/docs/2.15/notes/randomness.html#cuda-convolution-determinism)は
CUDA畳込みのalgorithm選択と演算自体の決定性を分けて説明する。参照文書は2.15、実行環境は2.13であり、
これも今回の原因を証明する資料とはしない。

親には、今回のBall差分をC3受入として許容するかの判断を返す。
追加のGPU対照実験は今回の「a5を1回だけ」の指示に含めず、未実施のまま残す。

## CPU検証・保存範囲

診断用guard smokeは既存9＋追加5の14項目成功（全体8.11秒、CUDA未初期化）。
全thread stackとGPU詳細診断をSIGTERM前に保存する順序、診断失敗時の終了、main失敗後のruntime未起動を検査した。
a5では異常が無かったため、停止診断用GPU詳細queryは発動していない。
CPU postprocessingは数値配列5・JSON境界7の12チェック成功。
JSON sidecarとpayloadも事前と同じNumPy float条件で照合する追加scriptを、保存出力に対して実行した。
元のGPU計測script・元比較レポートは保持している。

推論source `1bef628a9` の[Python CI](https://github.com/Motoki0705/tennis-lab/actions/runs/44131464826)と
[Knowledge CI](https://github.com/Motoki0705/tennis-lab/actions/runs/44131465350)は全成功。
通常のmodel unit/integrationはsource変更がないため今回再実行していない（前ターン354件・設定回帰65件の成功を継承）。

queueが作った原本は `queue-repro.sh` にbyte単位で保存し、run.json・原command・patchを変更していない。
`repro.sh` は保存scriptを新しいoutputsへコピーして使う入口で、queue外の起動を拒否する。
再生には記載した両worktreeの固定HEADと、新たな親のGPU実行指示が必要。
CPU再集計は新しい出力先で `compare_completed.py` → `analyze_completed.py` の順に行える。
予測NPY/NPZ・NVMLの生sample・profileは元outputsへ保持し、knowledgeには移していない。

`summary_impact: none`。実行基盤の受入比較であり、標準pipelineのcheckpoint・X3D採否・3D精度の現行判断を更新しない。
学習ではないためTensorBoard曲線は対象外。
