---
id: run-player-camera-stabilization-full-render-20261109
type: run
task: tennis_scene
sequence: 51
recorded_at: '2026-11-09'
title: 広角4強度の4秒比較描画と保存失敗の復旧
issue: 1776
provider: codex
date: '2026-11-09'
status: failed
config:
  algorithm: gaussian_markley_quaternion_v1
  sigma_seconds:
  - 0
  - 0.06
  - 0.08
  - 0.12
  translation: unchanged
  horizon_lock: false
  smoothing_context_frames:
  - 0
  - 1010
  source_frames:
  - 0
  - 240
  fps: 59.94006
  lens: equidistant_diagonal_159_approximation
  panel_resolution:
  - 640
  - 360
  pinhole_render_resolution:
  - 2560
  - 1440
  depth_of_field: false
  validation_scope: render_outputs
metrics:
  source_pngs: 1920.0
  valid_camera_audit_rows: 1904.0
  missing_camera_audit_rows: 16.0
  views: 8.0
  source_frames_per_view: 240.0
  mp4s: 14.0
  archives_on_drive: 8.0
  drive_verified_files: 68.0
  drive_verified_bytes: 1065261659.0
  original_colab_returncode: -15.0
  rerendered_views_after_failure: 0.0
  driver_elapsed_seconds: 2706.457732
  reported_frame_render_seconds_total: 1695.624283
repro:
  commit: d133993421c206af28340f4ffe082c65b70fe7a7
  command: timeout --signal=TERM --kill-after=20s 8640s .venv/bin/python outputs/tennis_scene/visualize/player_view/stabilization-comparison/operations/attempt2/run_full_queue.py
  cwd: .
artifacts:
  run_dir: knowledge/runs/run-player-camera-stabilization-full-render-20261109
  drive_record: operations/colab/player-view-stabilization-20261109-a2/full-a2
  drive_outputs:
  - outputs/tennis_scene/visualize/player_view/stabilization-comparison/attempt2/full
  - outputs/tennis_scene/visualize/player_view/stabilization-comparison/operations/attempt2
  strict_validation: knowledge/runs/run-player-camera-stabilization-full-render-20261109/validated-full.json
  recovery_handoff: knowledge/runs/run-player-camera-stabilization-full-render-20261109/recovery-handoff.json
  drive_copy_receipt: knowledge/runs/run-player-camera-stabilization-full-render-20261109/static-precopy.json
parents:
- run-player-camera-stabilization-preview-audit-20261109
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

各人12枚のpreviewを確認した後、固定ref d133993421c206af28340f4ffe082c65b70fe7a7、同じL4で元frame [0,240) の4強度比較を実行した。入力・camera・lens・sigma・描画qualityを変更せず、全1010frameを補正した後に先頭240frameを選ぶ。全区間の再描画は行っていない。

管理job full-a2は13:14:48 UTCのperiodic_saveで `rclone command could not complete: copy` を記録し、returncode=-15で終了した。queueもfailedである。rendererは別PGIDに残り、重複起動せず追跡した結果、13:32:02 UTCにprocessが終了し、全8 view・1920 PNG・8 ZIP・8 audits・14 MP4が揃った。元のrequestとdriver/archiverは未変更で、request SHA-256はf33d58d52e9ee69854c7d15d56596facf2b7766ef77cbee66ea1cc5241cc3fa2。個別orphan processのexit statusは直接取得しておらず、完成manifest・process消滅・全stageの厳密照合で成果を検証した。元の管理失敗を成功へ読み替えない。

既存1920 PNGのSHA-256、ZIPの全uncompressed member bytes、8 audits、package/source/lens/sigmaを再照合した。追加render job・再描画・画像変更は0。1904rowは有効camera、16rowは両IDのsource0/1を各4強度に保持した欠測表示。driver elapsedは2706.457732秒、各rowの報告render_seconds合計は1695.624283秒で、純粋なGPU kernel時間とは扱わない。

復旧中のDrive listingで403 RATE_LIMIT_EXCEEDED、project単位のQueries/minute制限を実測した。元copyのgeneric exceptionはOSError/TimeoutExpiredを区別して保存しておらず、その正確な原因は未確定。成功済みlistingを使った既存immutable fileの衝突確認を明示し、静的な全成果物をDrive CLI upload --verifyで保存した。13:54:30 UTC、715.577964秒で68file・1,065,261,659 bytesすべての内容照合に成功。元requestやcore publisher、credentialを変更していない。公式saveも成功し、full-a2はfailed/-15のままdrive_saved_at=2026-11-09 14:02:44.079477 UTCを保持した。最後のstatus atomic公開の完了はその後に確認した。

Markley平均は採用方式の背景であり、論文全体を再現した実験ではない。学習/TensorBoardは対象外。summaryの標準pipelineとC3/C5の人間確認が独立に必要という判断は変わらず、summary_impactはnone。Colab管理停止の別PGID子processへの伝播とDrive API制限は運用課題として残る。完成動画の独立した全decode・回収・目視・L4停止は後続delivery runを参照。
