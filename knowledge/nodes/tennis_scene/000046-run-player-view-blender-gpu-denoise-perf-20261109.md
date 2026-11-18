---
id: run-player-view-blender-gpu-denoise-perf-20261109
type: run
task: tennis_scene
sequence: 46
recorded_at: '2026-11-09'
title: OpenImageDenoiseのGPU使用で選手視点描画を約0.21秒へ短縮
issue: 1727
provider: codex
date: '2026-11-09'
status: done
config:
  clip: clip000_20261103
  player_ids:
  - 0
  - 1
  source_frames:
  - 16
  - 17
  - 19
  - 731
  - 735
  width: 1280
  height: 720
  fps: 59.94006
  samples: 16
  persistent_data: true
  denoiser: OPENIMAGEDENOISE
  denoising_use_gpu: true
  human_acceptance: pending
  throughput_gate: passed
metrics:
  rendered_frames: 10.0
  mean_interval_id0_s: 0.211273
  mean_interval_id1_s: 0.217024
  worst_valid_interval_id0_s: 0.250028
  worst_valid_interval_id1_s: 0.26703
  max_evaluated_pc2_error_m: 0.0
  metadata_crc_passed: 1.0
  throughput_gate_passed: 1.0
repro:
  commit: 993c7e2310a76ebc7145ca33b9698be5b4996ddc
  command: bash outputs/tennis_scene/visualize/player_view/blender-acceptance/run_perf_gpu_denoise_queue.sh
  cwd: .
artifacts:
  run_dir: knowledge/runs/run-player-view-blender-gpu-denoise-perf-20261109
  drive_record: operations/colab/player-view-c3-20261109/c3-perf-gpu-denoise
  drive_outputs:
  - outputs/tennis_scene/visualize/player_view/blender-acceptance/perf-gpu-denoise/artifacts
parents: []
relations:
- to: run-player-view-blender-persistent-perf-20261109
  rel: related
  note: 同じstill loopと固定入力でGPU denoisingを明示した性能比較。
- to: run-player-view-blender-batched-perf-20261109
  rel: related
  note: 改善しなかったanimation dispatchを撤去し、直接のframe更新方式へ戻した。
papers: []
tags: []
summary_impact: none
---

# 実測と判断

commit `993c7e231` でOpenImageDenoise方式とGPU使用を明示し、
保存済みC1/C2入力のID0/ID1それぞれ元frame16・17・19・731・735を実描画した。
効果のなかったanimation dispatchは撤去し、元のstill loopへ戻した。
Blender5.2.1 LTS・NVIDIA L4/OPTIX、1280×720、16 samples、元FPS59.94006。
camera/球/PC2/mask・材質・motion blurの設定を変えず、各rowへrender_seconds、
描画記録へenabled/denoiser/use_gpuを追加した。球半径0.05m・水平画角90度・本人mesh全体非表示は仮設定。

初期化後のPNG mtime間隔はID0が平均0.21127秒・有効cameraで最大0.25003秒、
ID1は平均0.21702秒・最大0.26703秒だった。
still loopのpersistent data単独で測った4.20046/4.21246秒に対し、約19倍の速度になった。
source/frame選択・解像度・sample数は共通だが、cold cache状態や計測時刻が違うため厳密な統計実験ではない。
これは描画wall timeであり、生成動画の59.94006fpsを変更する値ではない。

`--debug-cycles`の実ログで、ray tracingとdenoisingの両方が
`NVIDIA L4 (OptiX)`、denoiser typeがOpenImageDenoiseであることを確認した。
設定flagだけからGPU使用と判断せず、同梱`device_evidence.txt`へ実ログの該当行を保持した。
[Blender5.2公式addon](https://github.com/blender/blender/blob/blender-v5.2-release/intern/cycles/blender/addon/properties.py)
のGPU denoisingが既定Falseであることと合わせ、先行runの大きい固定費はCPU denoisingによるものと考えられる。
先行runはdenoising deviceを直接記録していないので、その部分は推論である。

全10frameについてcamera姿勢・人物ID・元frame/cache index・全mask・本人非表示・alpha1/emission0とPNG CRCを照合した。
PC2全頂点の最大誤差は0m。これは保存meshの忠実な読出しで、元の3D再構成精度の測定ではない。
既存previewとの画像差は平均約0.28〜0.46LSBで、比較元のdither差も含む。
球16→17→19、camera欠測731/735と相手meshだけの欠測の表示更新を確認した。
元FPS・入力区間を切り詰めず、全2020frameを同じ最終設定で新規描画する保守的見積りが
約887秒となり、保存・停止用の余裕を残したgateを通過した。
**このrunでは全区間動画をまだ生成していない。**

# 保存と再現

Colab job `c3-perf-gpu-denoise` はcompleted・returncode0・Drive保存済み。
Driveの運用記録をdownload --verifyして登録した。出力tarと原本記録はfrontmatterのDrive参照に保持する。
入力packageはDrive root相対
`outputs/tennis_scene/visualize/player_view/blender-acceptance-20261109/package`。

同梱のqueue/wrapperは実行時の原文で、`colab-original-repro.sh`は記録から生成したcommandを保持する。
外側`repro.sh`が同梱scriptを元pathへ配置する。run bundleは`TENNIS_REPO`の新規checkout外へ置く。
歴史的perfは先行runの監査済みprefix388frameも検証するため、同runのDrive checkpoint tarから
元PNG/rowを復元する必要がある。新しい本番では旧prefixを採用せず、全frameを最終設定に統一する。
過去の絶対deadlineや履歴を上書きせず、再実行は新しい予算・runとして記録する。

学習・採用checkpoint・標準pipeline・3D精度に関する現在の判断は変えず、summary_impactはnone。
TensorBoardは該当しない。doneはこの小規模検証の終了だけを意味する。
人間の定性受入はpendingで、C3受入完了やC5開始の承認ではない。
