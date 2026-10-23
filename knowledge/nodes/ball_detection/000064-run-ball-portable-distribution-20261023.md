---
id: run-ball-portable-distribution-20261023
type: run
task: ball_detection
sequence: 64
recorded_at: '2026-10-23'
title: 長系列用配布storeの可搬化とDrive・Colab照合
issue: 1374
provider: codex
date: '2026-10-23'
status: running
config:
  train_validation_version: ball-mix1800-720p-all-frame-train-val-v1
  test_version: ball-mix1800-720p-all-frame-test-v1
  source_snapshot: outputs/ball_detection/precompute/mdd_hybrid_1231/20261018-mix1800-720p-v1/ball_snapshot
  frame_step: 1
  common_interval_min_frames: 128
  sample_lengths: [32, 64, 128]
  upload_workers: 4
  colab_sha256_workers: 4
  colab_gpu: L4
metrics:
  train_validation_clips: 1537
  train_validation_files: 2926
  train_validation_bytes: 157029428456
  train_validation_common_frames: 793017
  test_clips: 225
  test_bytes: 28359923470
  local_generated_bytes: 295439829
  local_source_files_stat_checked: 3336
  local_train_validation_sample_windows: 18
  local_test_sample_windows: 9
  drive_verified_files: 2926
  drive_verified_bytes: 157029428456
  resume_retained_payloads: 963
  resume_uploaded_payloads: 574
repro:
  commit: 52f91b6ec0b5e728efe61ec085d91eeafcad87a4
  branch: codex/ball-temporal-distribution
  command: PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$PWD" .venv/bin/python knowledge/runs/run-ball-portable-distribution-20261023/prepare_data.py --source-repo-root /home/kamimura/projects/tennis-lab --output-root "$PWD/tmp/ball-portable-distribution"
artifacts:
  run_dir: knowledge/runs/run-ball-portable-distribution-20261023
  output_dir: tmp/ball-portable-distribution
parents:
- run-ball-all-frame-protocol-20261022
relations: []
papers: []
tags: [portable-store, drive, colab, temporal]
---

## 考察 / Findings

[全フレーム教師の合意済みprotocol](000063-run-ball-all-frame-protocol-20261022.md)に従い、
train/validationと固定testを独立した配布store版にした。clip内の全native frameを保持し、
学習対象は同梱した共通区間の入力から定まる。配布storeのframe数と共通区間のframe数は異なる。
必要clipだけをmetadataに宣言するため、読込み側が不在shardを黙認する分岐は追加していない。

[ローカル準備receipt](../../runs/run-ball-portable-distribution-20261023/local-preparation.json)では、
train/validationはtrain 1,342 clip・validation 195 clip、固定testは225 clipである。
新規生成はindex・metadata・原注釈・protocol入力・内容manifestの計約295 MBで、
RGB shardは既存snapshotを読み取っただけである。全3,336入力ファイルのinode・size・mtime・ctimeが
準備前後で一致した。shardの複製やhardlinkは作らなかった。空きディスクの前後差は他のhost処理も含むため、
この作業単独の使用量とはしない。

内容manifestの全件hash/size検証と、source・splitごとのT=32/64/128サンプル（train/validation 18窓、
test 9窓）で教師入力・原Chat注釈・JPEG・RGB・native PTSが元snapshotと一致した。
原注釈の内容を照合することで、空ラベルのnotesも入力として保持した。
不在教師を生成するpolicyや最終窓manifestの実装は#1372の範囲である。
[旧凍結RGB hashとの照合](../../runs/run-ball-portable-distribution-20261023/prior-payloads.json)も別途通過した。

配布版の原注釈はstore相対pathである。消費側は`store.annotation_path(clip)`を使う必要があり、
既存窓helperの絶対path直接参照は並行する#1372で接続する。
同梱catalog・interval・packのindexは元snapshotの番号のままなので、clip IDと配布manifestの対応表でjoinする。

Driveへの初回全量uploadは[配置receipt](../../runs/run-ball-portable-distribution-20261023/drive-upload.json)で確定した。
`data/ball_detection/ball-mix1800-720p-all-frame-train-val-v1`の全2,926ファイル・157.03 GBが
最終inventoryのsize/MD5と一致し、重複pathはない。内容のSHA-256は配布manifestで固定した。
元snapshotのshardを直接uploadし、metadata転送用の追加stageは約258 MBでRGBを含まない。
全receiptとColab実行用Git bundleは同receiptが示すDriveのoperations archiveへ保存・検証した。
2台のColab VMでの取得・SHA-256・サンプル意味照合は未実施で、これから行う。
最初の全量転送は962 shard・約104.03 GBの照合後、親の報告によるモデルcapacityエラーで
sessionとともに終了した。全量成功のterminal receiptはなく、成功完了とは扱わない。
[中断と再開の記録](../../runs/run-ball-portable-distribution-20261023/session-interruption.json)に従い、
新しいOS sessionへ転送を切り離して再開した。元ファイルの再hashと新しいDrive inventoryで
963ファイルを保持し、残り574 payloadだけを追加した。前回のログに残らなかった1ファイルも
内容一致で保持したため、ログの962件とDriveの963件は異なる。
[監督処理](../../runs/run-ball-portable-distribution-20261023/drive-supervisor.json)は終了コード0を記録した。
初回の実転送開始から最終完了までのUTC差は約7.12時間で、途中の再hash・中断を含む。
個別の所要時間は単調時計で計測し、hostのUTC差との不一致もreceiptへ残した。並列のファイル所要時間を合算しない。
CPU runtimeを指定できない事前調査に対し、親からL4の例外許可を得た。
これは今回の照合に限定した判断で、skill変更や学習実行の許可ではない。
固定testのuploadは最終評価時へ回し、今回はtrain/validationの転送・再取得・検証を優先する。
再現コマンドとreceiptの位置は[run手順](../../runs/run-ball-portable-distribution-20261023/README.md)に記録する。
