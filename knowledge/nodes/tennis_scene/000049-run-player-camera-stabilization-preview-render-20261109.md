---
id: run-player-camera-stabilization-preview-render-20261109
type: run
task: tennis_scene
sequence: 49
recorded_at: '2026-11-09'
title: 回転補正比較previewの描画とwrapper判定失敗
issue: 1776
provider: codex
date: '2026-11-09'
status: failed
config:
  session: player-view-stabilization-20261109-a2
  sigma_seconds:
  - 0
  - 0.06
  - 0.08
  - 0.12
  preview_source_frames:
  - 0
  - 26
  - 162
  lens: equidistant_diagonal_159_approximation
  panel_resolution:
  - 640
  - 360
  pinhole_render_resolution:
  - 2560
  - 1440
  depth_of_field: false
  new_render: true
metrics:
  rendered_source_pngs: 24
  images_per_player: 12
  renderer_elapsed_seconds: 66.517762
  missing_camera_rows: 8
  colab_returncode: 1
repro:
  commit: d133993421c206af28340f4ffe082c65b70fe7a7
  command: timeout --signal=TERM --kill-after=20s 12045s .venv/bin/python outputs/tennis_scene/visualize/player_view/stabilization-comparison/operations/attempt2/worker_runtime.py
    preview
  cwd: .
artifacts:
  run_dir: knowledge/runs/run-player-camera-stabilization-preview-render-20261109
  drive_record: operations/colab/player-view-stabilization-20261109-a2/preview-a2
  drive_outputs:
  - outputs/tennis_scene/visualize/player_view/stabilization-comparison/operations/attempt2
  - outputs/tennis_scene/visualize/player_view/stabilization-comparison/attempt2/preview
  capsule_manifest: knowledge/runs/run-player-camera-stabilization-preview-render-20261109/capsule-files.json
parents:
- run-player-camera-stabilization-20261109
relations:
- rel: retries
  to: run-player-camera-stabilization-20261109
papers:
- paper-2007-markley-quaternion-averaging
tags:
- player-view
- camera-stabilization
- diagnostic
summary_impact: none
---

第1回のL4割当拒否（run-player-camera-stabilization-20261109）を、新しい明示的なsession `player-view-stabilization-20261109-a2` で再試行した。2026-11-09 11:22:50.337862 UTCの割当要求から4時間を上限とし、コマンド期限15:06:50.337862 UTC・解放期限15:22:50.337862 UTCを記録した。ほかの学習sessionは操作していない。

描画自体は元frame 0/26/162 ×4強度×2人の24 PNG（各人12枚）と2 contactを生成し、queueはdone・serve returncode=0、rendererはCOMPARISON_COMPLETEを出した。しかし、使い捨てwrapperが成功logにも `exit_code=0` 行を要求し、実際のqueueの成功log形式に合わずreturncode=1となった。**このColab jobのstatusはfailedのまま保存する。** 保存済みColab request/status・stdoutと、厳密な元wrapper snapshotをこのbundleへ残す。snapshotのSHA-256は259efec420a2393b62b052b7e3da90081c47c4a46b7d1dba4fa115efef088063で、upload receiptとの一致を確認した。

preview実描画の合計elapsedは66.517762秒、24row中16有効・8欠測、報告render時間合計28.482959秒だった。cameraの欠測は元frame0であり、黙った補間ではなくCAMERA MISSINGを描いた。GPUはNVIDIA L4/OPTIX、Blender5.2.1 LTS、CPU torch、対角159度を設計値とする等距離魚眼、パネル640×360・元描画2560×1440・DOF無効である。σと入力・設定は元runの固定manifestに従う。

画像の正しさ・回収は後続のpreview-audit runで評価した。この失敗は映像方式の棄却ではなくwrapperの終了判定の誤りであり、レンダーを再実行しない監査へ進んだ。学習を行わないのでTensorBoardは対象外。summaryの標準pipelineとC3の人間確認が独立に必要という判断は変わらず、summary_impactはnone。
