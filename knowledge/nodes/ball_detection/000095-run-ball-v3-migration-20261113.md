---
id: run-ball-v3-migration-20261113
type: run
task: ball_detection
sequence: 95
recorded_at: '2026-11-13'
title: 既存v2注釈のプレー球v3移行と再利用検証
issue: 1714
provider: codex
date: '2026-11-13'
status: running
summary_impact: none
config:
  source_schema: video_ball_annotation.v2
  target_schema: video_ball_annotation.v3
  target_semantics: active_ball
  selection: knowledge/runs/run-ball-v3-migration-20261113/selection.json
metrics:
  inventoried_cameras: 171
  inventoried_frames: 88872
  selected_cameras: 132
  selected_frames: 46212
  held_cameras: 39
  dry_run_cameras: 3
  dry_run_frames: 3030
artifacts:
  run_dir: knowledge/runs/run-ball-v3-migration-20261113
  output_dir: outputs/ball_detection/migrate/video-v3/20261113
parents: []
relations: []
papers: []
tags: [annotation, migration, active-ball]
---

元v2のMeiji 57clip・171camera入力と旧store 3版を棚卸しし、132本の初回移行対象を
[固定一覧](../../runs/run-ball-v3-migration-20261113/selection.json)へ保存した。
対象定義が手持ち球を含む3本と、処理完了に関する別reportの解釈が必要な39本（前者を含む）は
初回集合から保留した。falseの目視網羅性をunprocessedへ読み替える判断はしていない。

少数dry-runではvideo_000/clip_000の全3camera・3,030frameを実動画のbytes・復号PTSと
照合してconverterを通した。observed 2,767、interpolated 105、occlusion_estimated 10、
最終unknown 148を保持した。新しい不在・未処理・座標なし遮蔽は生成していない。
この段階では本実行・snapshot公開の完了を主張しない。

[実行記録と検証設計](../../runs/run-ball-v3-migration-20261113/README.md)に
対応表・旧版の保持・再現入口を残す。実GPT、モデル認識品質の比較、追加学習は対象外。
この移行は採用モデル・評価指標の現在の判断を変えないためsummary_impactはnone。
TensorBoardとcheckpointは生成しない。
