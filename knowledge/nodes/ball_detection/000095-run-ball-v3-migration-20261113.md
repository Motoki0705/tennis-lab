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
status: done
summary_impact: none
config:
  source_schema: video_ball_annotation.v2
  target_schema: video_ball_annotation.v3
  target_semantics: active_ball
  selection: knowledge/runs/run-ball-v3-migration-20261113/selection-reviewed.json
  execution_commit: 64f61a41879713169151530c09c8df9fbd2aa4f7
  dataset_version: meiji-active-ball-v3-20261113
metrics:
  inventoried_cameras: 171
  inventoried_frames: 88872
  selected_cameras: 123
  selected_frames: 63540
  held_cameras: 48
  dry_run_cameras: 3
  dry_run_frames: 3030
  migrated_cameras: 123
  migrated_frames: 63540
  observed: 53141
  interpolated: 840
  occlusion_estimated: 353
  final_unknown: 9206
  position_usable: 54334
  negative_candidates: 0
  unprocessed: 0
  scene_clips: 41
  scene_camera_artifacts: 123
  native_jpeg_bytes: 20690065319
  protected_prepared_archives: 123
artifacts:
  run_dir: knowledge/runs/run-ball-v3-migration-20261113
  output_dir: outputs/ball_detection/migrate/video-v3/20261113
  results_summary: knowledge/runs/run-ball-v3-migration-20261113/results-summary.json
  correspondence: knowledge/runs/run-ball-v3-migration-20261113/correspondence.csv
  held_inputs: knowledge/runs/run-ball-v3-migration-20261113/held-inputs.csv
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
少数検証後に本集合の公開と再利用検証を実施した。

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
本集合123本・63,540frameの公開、全41clip・123cameraのscene import、同一receiptへの
再送を完了した。観測53,141、補間840、遮蔽推定353、最終unknown 9,206を保持した。
位置利用可54,334、新規の負例候補0、未処理0である。元収録groupに対応するval 26,961、
test 18,453、train 18,126frameを保持した。native JPEGは20,690,065,319 bytesで、
123archiveすべてがsnapshotへhardlinkされ、公開保護にも登録されている。
元媒体171本、元注釈171本、snapshotのmetadata/index 6件を固定した内容SHAへ照合した。
旧画像513件は復旧時に固定した内容SHAへ一致する。

実データを扱うrunは終了コード0、1時間43分19秒、最大RSS 3,012,596 KiBで完了した。
48本の保留は元版と理由を[一覧](../../runs/run-ball-v3-migration-20261113/held-inputs.csv)に残し、
今回の対象除外の扱いを親へ返す。再利用はE上の物理rootを指定した実行scriptで検証した。
既定factoryによる移設後outputs symlinkの対応は親へ共有する。
初回PR HEADのPythonとKnowledge libraryは成功し、実行commitとsrc/testsのtreeも一致した。
CIとlocal検証の実測・中断は[検証記録](../../runs/run-ball-v3-migration-20261113/validation-summary.json)に保持する。
