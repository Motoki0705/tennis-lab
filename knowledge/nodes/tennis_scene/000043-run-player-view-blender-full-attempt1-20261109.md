---
id: run-player-view-blender-full-attempt1-20261109
type: run
task: tennis_scene
sequence: 43
recorded_at: '2026-11-09'
title: 選手視点全区間の低速描画を監査済み388frameで停止
issue: 1727
provider: codex
date: '2026-11-09'
status: failed
config:
  clip: clip000_20261103
  player_ids:
  - 0
  - 1
  source_interval:
  - 0
  - 1010
  fps: 59.94006
  width: 1280
  height: 720
  persistent_data: false
  human_acceptance: pending
  completion: partial_controlled_cancellation
metrics:
  audited_frames: 388.0
  fully_rendered_views: 0.0
  target_frames: 2020.0
  max_reported_pc2_error_m: 0.0
  preview_mean_frame_interval_s: 5.114
  full_command_timeout_s: 9345.0
repro:
  commit: 2791ae2fc3fe660c01b79d337ef46b0752aa0fa3
  command: bash outputs/tennis_scene/visualize/player_view/blender-acceptance/run_full_queue.sh
  cwd: .
artifacts:
  run_dir: knowledge/runs/run-player-view-blender-full-attempt1-20261109
  drive_record: operations/colab/player-view-c3-20261109/c3-full-2791
  drive_outputs:
  - outputs/tennis_scene/visualize/player_view/blender-acceptance/full/checkpoints
  - outputs/tennis_scene/visualize/player_view/blender-acceptance/full/artifacts
parents: []
relations:
- to: run-player-view-blender-preview-20261109
  rel: related
  note: 同じ固定入力の少数frame検証を経て全区間へ進んだ。
papers: []
tags: []
summary_impact: none
---

# 結果と制約

C3の固定scene/camera packageから両人1010frameを描く最初の全区間jobを起動したが、
描画速度がGPU上限内の完了に足りないことが判明したため、queue経由で安全に停止した。
**失敗した部分実行であり、全区間動画の完成や人間の受入を意味しない。**

実行はcommit `2791ae2fc`、Blender 5.2.1 LTS、NVIDIA L4/OPTIX。
1280×720、水平画角90度、元FPS59.94006、球半径0.05mは仮設定。
本人mesh全体を非表示、相手は灰色・球は緑色、alpha=1・emission=0、motion blurなし、dither=0。
入力package manifest SHA-256は
`708affe5b630ba50014bf9f3f7a5ff4a730440906c1b1a4f99e920bfe5247280`。

停止時点でID0の元frame `[0,388)` に監査rowと完成PNGがあり、ID1は0frame。
全388rowについて元frame/cache index、camera/人物/球mask、本人非表示、保存camera姿勢と球位置を確認した。
評価済みPC2全頂点の最大誤差は0m。これは保存入力の読出し検証であり、3D復元精度の評価ではない。
rowがないPNGは採用しない。完成動画は0本で、入力全体の切り詰めによる代替納品はしていない。
完成済みprefixは閉じたtarとして旧Drive checkpoint namespaceへ残す。

少数frame previewのPNG mtime間隔は平均5.114秒だった。
この値の2020frameへの単純外挿は約10330秒で、最初の本番command timeout9345秒を超える。
この予測を本番開始前に確認できておらず、約35分の部分実行を停止する必要が生じた。
同じ形状・camera・材質を保ってBlenderのpersistent dataを有効にする性能修正を別commitへ置き、
少数frameで速度と球/欠測の画面更新を確認してから、監査済みprefixの残りだけを再開する方針とした。
性能改善の効果と全区間の成功はこのrunでは測定していない。

# 終了・再現

Colab `c3-full-2791` は2026-11-08T17:27:51Z開始。
queue cancel後のColab状態はfailed・returncode=1、Drive保存は2026-11-08T18:02:51Z。
Blender子孫の停止とGPU queue予約0を確認した。前の運用Codexのruntimeが
`descendants_reaped` で終わった時点ではremote jobは継続していたため、
親workerがremote状態を独立に照合した。runtimeの終了をGPU処理の成功に読み替えない。

Driveの入力はroot相対
`outputs/tennis_scene/visualize/player_view/blender-acceptance-20261109/package`。
成果物と原本の運用記録はfrontmatterのDrive参照に保持し、予測配列・PC2・動画はGitへ複製しない。
同梱のqueue scriptとrender wrapperは実行時の内容を保持する。
`colab-original-repro.sh` は記録から生成されたcommandの原文で、外側の`repro.sh`が
同梱scriptを元のoutputs場所へ配置して呼ぶ。run bundleを`TENNIS_REPO`の新規checkout外へ置く。
歴史的queue scriptの絶対deadlineは2026-11-08T20:19:37Zのままであり、現在は安全に拒否する。
新しい再実行には別の実行予算と新しいrun記録を用意し、この履歴を上書きしない。

このrunはrendererの性能・保存運用の負例で、学習/推論精度の比較ではない。
TensorBoardは該当しない。標準pipeline・採用checkpoint・3D精度に関する既存判断を変えないため、
summary_impactはnone。人間の定性確認は引き続きpendingで、C5開始の承認ではない。
