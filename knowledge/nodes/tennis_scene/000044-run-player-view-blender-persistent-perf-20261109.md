---
id: run-player-view-blender-persistent-perf-20261109
type: run
task: tennis_scene
sequence: 44
recorded_at: '2026-11-09'
title: 選手視点のpersistent data単独改善は残frameの時間枠に不足
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
  human_acceptance: pending
  throughput_gate: failed
metrics:
  rendered_frames: 10.0
  mean_frame_interval_id0_s: 4.200463
  mean_frame_interval_id1_s: 4.212464
  max_evaluated_pc2_error_m: 0.0
  remaining_frames: 1632.0
  green_mask_iou_all_frames: 1.0
  pixels_different_by_more_than_8_lsb: 0.0
  throughput_gate_passed: 0.0
repro:
  commit: d8fa68cb2894d6e5036f246a73a83df4103465e1
  command: bash outputs/tennis_scene/visualize/player_view/blender-acceptance/run_perf_queue.sh
  cwd: .
artifacts:
  run_dir: knowledge/runs/run-player-view-blender-persistent-perf-20261109
  drive_record: operations/colab/player-view-c3-20261109/c3-perf-persistent
  drive_outputs:
  - outputs/tennis_scene/visualize/player_view/blender-acceptance/perf-preview/artifacts
parents: []
relations:
- to: run-player-view-blender-full-attempt1-20261109
  rel: related
  note: 監査済み388frameを残し、残1632frame再開の速度条件を検証した。
- to: run-player-view-blender-preview-20261109
  rel: related
  note: 共通sampleの画像とmaskを比較。
papers: []
tags: []
summary_impact: none
---

# 観察と判断

保存済みC1/C2入力を変えず、Blenderの`scene.render.use_persistent_data=True`を有効にし、
ID0/ID1の元frame16・17・19・731・735を各5枚描いた。refは`d8fa68cb2`、
Blender 5.2.1 LTS・NVIDIA L4/OPTIX、1280×720、16 samples、元FPS59.94006。
水平画角90度・球半径0.05m・本人mesh全体非表示は引き続き仮設定。
材質alpha=1、emission=0、motion blurなし、dither=0を維持した。

完成PNGのmtimeの初期化後間隔はID0が平均4.20046秒、ID1が4.21246秒。
先行previewでの平均5.114秒より短いが、sample集合とwarm状態が異なるため、
単独の厳密な性能比較ではない。残1632frameの単純外挿は6867.3秒で、
保存・セッション停止に15分以上を残す枠には約819秒不足した。
**描画job自体は成功したが、再開のthroughput gateは失敗し、新本番は起動しなかった。**
最初の全区間jobの監査済みprefix ID0 `[0,388)` は保持している。

全10frameでcamera姿勢・人物ID・元frame/cache index・camera/人物/球mask・本人非表示を照合し、
評価済みPC2全頂点の最大誤差は0mだった。元3D復元精度の評価ではない。
既存previewとの共通sample画像では、8LSBを超える画素差は0、緑の球maskのIoUは全枚1。
球16→17→19、ID0 camera731、ID1 camera735、相手meshだけの欠測の画面更新を確認した。
微小差の平均は約0.27〜0.46LSBで、比較元previewの標準ditherと異なる。
この検証はcache導入時の古い画像の取り残しを調べるもので、画質向上や人間の受入の証拠ではない。

暗いcamera欠測frameでもframe間隔は約4.2秒であり、場面の複雑さ以外の固定費が大きいと推測した。
次は連続区間を一度のanimation renderへまとめ、frameごとの直接設定をhandlerへ移す。
保存camera/PC2/球・maskの監査と画像更新を再検証してから再開を判断する。
このrunではそのdispatch変更の効果は測っていない。

# 保存と再現

Colab job `c3-perf-persistent` はreturncode=0で終了し、Drive保存は2026-11-08T18:22:56Z。
Driveの運用記録をdownload --verifyして登録した。全区間動画はこのrunでは生成していない。
運用Codexは親による次の性能修正の指示へ切り替えるため明示停止した。
remoteの終了・保存を独立確認しており、runtime停止とGPU job失敗を混同しない。

入力packageはDrive root相対
`outputs/tennis_scene/visualize/player_view/blender-acceptance-20261109/package`。
`perf-preview.tar`と運用記録の参照はfrontmatterにある。GPU実行済みwrapperは
同梱`resume_full.py`（SHA-256 `4d896c4aa087943b83f1924347b89a81d761109c1cc4e4fb2956c32fa1588ac9`）。
実行開始後に親が別用途のatomic staging修正を作ったため、実行時の元ファイルをbackupから保持した。
既にload済みのperf処理にはその修正を適用していない。

`colab-original-repro.sh`は記録の原文で、外側の`repro.sh`が同梱queue/wrapperを元の場所へ配置する。
新規checkoutを`TENNIS_REPO`に指定し、run bundleはそのcheckoutの外へ置く。
この歴史的perfは先行jobの旧prefixを前提にするため、同jobのDrive checkpoint tarを
元frame順に展開し、各tarのcheckpoint rowを連結して`full/work/renders/player_0`を復元する必要がある。
ID1は未描画。異なる既存出力は上書きしない。再開modeには過去の絶対deadlineがあり、
新しい予算を与えず履歴のまま実行すれば拒否する。

学習・checkpoint採用・標準pipeline・3D精度に関する判断は変えず、summary_impactはnone。
TensorBoardは該当しない。doneはこの小規模検証jobの終了だけを意味する。
人間の定性受入はpendingであり、C5開始の承認ではない。
