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
  selection: knowledge/runs/run-ball-v3-migration-20261113/selection-reviewed.json
metrics:
  inventoried_cameras: 171
  inventoried_frames: 88872
  selected_cameras: 123
  selected_frames: 63540
  held_cameras: 48
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

元v2のMeiji 57clip・171camera入力と旧store 3版を棚卸しした。初回一覧を保持し、
注記の対象定義と処理完了の根拠を読んだ[本実行集合](../../runs/run-ball-v3-migration-20261113/selection-reviewed.json)は
123本・63,540frameへ固定した。手持ち・プレー前後の対象相違39本と、処理完了の確認に必要な
reportが欠ける15本（6本重複）、計48本を保留した。目視網羅性falseだけを処理未完了と
読み替えず、全frame処理の明記がある入力は採用する。

少数dry-runではvideo_000/clip_000の全3camera・3,030frameを実動画bytes・復号PTSと
照合した。observed 2,767、interpolated 105、occlusion_estimated 10、最終unknown 148を保持。
同じ集合を実publisher、学習reader、scene importで読み、同一要求の再送が同じreceiptへ
戻ることを確認した。目視非網羅だが処理完了の根拠がある別clipの3,051frameもdry-runした。
この段階では本集合の公開は未実施であり、runはrunning。

Driveの旧archiveを検証付きで取得し、99注釈すべてがローカル現行版と同一内容であることを
確認した。欠けるreview/reportはarchiveにも含まれない。Drive配布storeの108 Meiji参照は
ローカルの元版・媒体・group/splitと一致し、別の移行対象版は見つからなかった。

[実行記録と検証設計](../../runs/run-ball-v3-migration-20261113/README.md)に
対応表・旧版の保持・再現入口を残す。実GPT、モデル認識品質の比較、追加学習は対象外。
この移行は採用モデル・評価指標の現在の判断を変えないためsummary_impactはnone。
TensorBoardとcheckpointは生成しない。


WSL停止とEドライブへの移設後、89本の保存済み準備を全件照合し、不整合0本・残り34本を
確認した。復旧後の少数3本では再びreader・scene import・同じreceiptへの再送を検証した。
実行commitは `64f61a41879713169151530c09c8df9fbd2aa4f7`。移設前の旧画像内容hashはなく、
旧画像の全内容照合の基準は復旧後に固定した。注釈・metadata/indexは中断前のhashと一致する。
本集合の実行中のため、公開件数の実測と対応表は未確定である。
