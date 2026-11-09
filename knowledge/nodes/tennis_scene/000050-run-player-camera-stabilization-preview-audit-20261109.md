---
id: run-player-camera-stabilization-preview-audit-20261109
type: run
task: tennis_scene
sequence: 50
recorded_at: '2026-11-09'
title: 既存24画像のpreview監査とハッシュ照合回収
issue: 1776
provider: codex
date: '2026-11-09'
status: done
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
  new_render: false
metrics:
  validated_source_pngs: 24.0
  new_pngs_rendered_by_validation_job: 0.0
  render_audits_checked: 8.0
  images_per_player: 12.0
  reference_renderer_pipeline_elapsed_seconds: 66.517762
  reference_total_render_seconds: 28.482959
  missing_camera_rows: 8.0
repro:
  commit: d133993421c206af28340f4ffe082c65b70fe7a7
  command: timeout --signal=TERM --kill-after=20s 10085s .venv/bin/python outputs/tennis_scene/visualize/player_view/stabilization-comparison/operations/attempt2/validate_existing_preview.py
  cwd: .
artifacts:
  run_dir: knowledge/runs/run-player-camera-stabilization-preview-audit-20261109
  drive_record: operations/colab/player-view-stabilization-20261109-a2/preview-a2-validate
  drive_outputs:
  - outputs/tennis_scene/visualize/player_view/stabilization-comparison/operations/attempt2
  - outputs/tennis_scene/visualize/player_view/stabilization-comparison/attempt2/preview
  capsule_manifest: knowledge/runs/run-player-camera-stabilization-preview-audit-20261109/capsule-files.json
parents:
- run-player-camera-stabilization-preview-render-20261109
relations:
- rel: evaluates
  to: run-player-camera-stabilization-preview-render-20261109
papers: []
tags:
- player-view
- camera-stabilization
- diagnostic
summary_impact: none
---

失敗したpreview-render runの**既存画像だけ**を再監査した。queueのdone状態・completion receipt・resource=all・serve returncode=0・rendererのCOMPARISON_COMPLETEを照合し、誤った成功log行の条件を置き換えた。Colab jobはcompleted/returncode=0、Drive保存は2026-11-09 12:29:46.461498 UTCに確認した。新しい描画は0枚である。

元frame0/26/162×4強度×2人の24画像（各人12枚）、8 render audit、2 contactを確認した。監査前後の全26 PNGのSHA-256が同一である。source frame、欠測理由、package digest、実Blender camera/colour設定、NVIDIA L4/OPTIXを照合し、localへはcontactと小さいauditのみを回収した。Driveのsize/MD5とVM/localのSHA-256が一致し、24元画像はDriveへ保持した。元frame0の8欠測rowは明示表示であり、補間していない。

技術previewとして両contactを実際に目視した。コート線の強い湾曲は等距離魚眼として想定内で、細いコート/ネット線には小さな段差と軟らかさが見える。静止画だけで時間方向のaliasの程度は判断できない。ID1の元frame162は大きな首振りが残り、コートが左下へ寄り灰色の背景が大半を占める。相手・ボールも広角で小さく見える。元frame0は4案とも正しい人物ID・σ・レンズ名・source付きのCAMERA MISSINGである。水平固定やrecenterを追加せず、同じ設定で4秒比較へ進む技術判断をした。**人間の強度選択・品質受入はpending。**

参考renderer elapsed66.517762秒・render合計28.482959秒は先行描画の値であり、この監査jobの新規描画実測とは扱わない。先行runのwrapper失敗を成功へ書き換えていない。GPU起動担当runtimeもdescendants_reaped/exit1で終了したが、cleanup_complete・全group停止を別に確認し、成果はDriveハッシュと実監査で独立に検証した。runtime終了を成功証拠としていない。

学習を行わないのでTensorBoardは対象外。summaryの標準pipelineとC3の人間確認が独立に必要という判断は変わらず、summary_impactはnone。次は同じ4時間枠・同じsession内で240frame×4強度×2人を描き、全decode/FPS/PTS/source対応を確認する。全1010frame版は人間の強度選択後に行う。
