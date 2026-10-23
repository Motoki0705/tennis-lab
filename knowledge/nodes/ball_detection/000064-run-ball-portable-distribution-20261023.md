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
status: done
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
  colab_a_sha256_seconds: 459.4853404579999
  colab_a_samples_equal: true
  colab_a_free_after_verification_bytes: 42658000896
  colab_b_download_seconds: 2795.175334792
  colab_b_sha256_seconds: 441.4833025070002
  colab_b_samples_equal: true
  colab_b_free_after_verification_bytes: 42662158336
  stopped_colab_sessions: 2
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

train/validationの可搬な配布storeをDriveへ配置し、異なる2台のL4 VMで全2,926ファイル・157.03 GBの
SHA-256/sizeと18窓の教師入力・JPEG・RGB・native PTSの一致を確認した。両sessionの停止も確認済みである。
配布・取得・照合は成立したが、長系列の学習速度・VRAM・精度は測っていない。
後続のColab試走はこの入力版と再取得費用を使って予算を決められる。

[全フレーム教師のprotocol](000063-run-ball-all-frame-protocol-20261022.md)から必要clipを決め、
train 1,342・validation 195 clipを独立した版にした。clip内の全native frame 903,807を保持し、
学習に使う共通区間は同梱protocol入力の793,017 frameである。この2つのframe数は異なる。
index・metadata・split・原Chat注釈・共通区間と窓生成の入力・内容manifestを含めた。
必要clipだけをmetadataに宣言する新versionなので、不在shardを黙認する読込み分岐はない。
固定testは別版の225 clip・28.36 GBとして作成し、同じ手順で配布できる。

[ローカル準備](../../runs/run-ball-portable-distribution-20261023/local-preparation.json)では、
元snapshotとtrain/validation 18窓・test 9窓（source・split別、T=32/64/128）の教師入力・原注釈・画像・PTSが一致した。
全3,336入力ファイルのinode・size・mtime・ctimeが準備前後で一致した。
既存RGBの複製やhardlinkは作らず、小さいファイルだけを計約295 MB生成した。
後のmetadata upload stageは追加約258 MBで、ここにもRGBを含めていない。
空きディスクの前後差は他のhost処理も含むため、この作業単独の使用量とはしない。
[旧凍結RGB hash](../../runs/run-ball-portable-distribution-20261023/prior-payloads.json)との照合も通過した。

Driveの確定資産は`data/ball_detection/ball-mix1800-720p-all-frame-train-val-v1`である。
[配置receipt](../../runs/run-ball-portable-distribution-20261023/drive-upload.json)にDrive ID、manifestのidentity、
全件inventoryと運用archiveの参照を固定した。全ファイルのsize/MD5が一致し、重複pathはない。
shardを元snapshotから直接uploadし、metadataと最終manifestを順に公開した。
初回実転送開始から完了までのUTC差は約7.12時間で、以後の再hash・中断を含み、初回source hashは含めない。
metadata uploadだけの単調時計計測は約15.03分である。

uploadでは意図したtimeout中断と3ファイル後の停止に加え、モデルcapacityエラーによるsession終了も経験した。
[中断記録](../../runs/run-ball-portable-distribution-20261023/session-interruption.json)の962 shard・約104.03 GBは
ログ上の件数で、全量成功のterminal receiptはなかった。
新しいOS sessionへ転送を切り離し、freshな元ファイルhashとDrive inventoryで963ファイルを保持した。
ログに残らなかった1ファイルも内容一致で保持し、未完了574 payloadだけを追加して完了した。
[監督処理](../../runs/run-ball-portable-distribution-20261023/drive-supervisor.json)は終了コード0を記録した。
再開の単位は未完了ファイル全体であり、HTTP転送のbyte offset継続を保証するものではない。
個別の所要時間は単調時計、UTCの差は別の観測として保存した。hostで両者の差があるため補正値は推定せず、並列時間を合算しない。

[Colab実行記録](../../runs/run-ball-portable-distribution-20261023/colab-execution.json)では、
各VMを新しく割り当て、kernel boot IDの相違、Python 3.11 locked環境、OpenCV版の一致を確認した。
2台目の取得前ファイル一覧は0件で、中断を入れずに同じ配布版をローカルディスクへ取得した。

| 実測 | 環境構築（秒） | 取得＋CLI照合（分） | 全件SHA-256（分） | 18窓照合（秒） | 照合後の空き（GB） |
| --- | ---: | ---: | ---: | ---: | ---: |
| VM A：意図した中断・再開あり | 25.79 | 48.63 | 7.66 | 18.12 | 42.658 |
| VM B：新VMで中断なし | 29.63 | 46.59 | 7.36 | 18.23 | 42.662 |

取得時間にはmanifest取得とDrive CLIの内容照合を含む。VM Aは意図した中断分も含み、
その後の再開コピー単体は約47.18分だった。venvの論理サイズは両VMとも約6.55 GB、
構築後の空きは約199.80 GB、アプリをimportした取得前の空きは約199.70 GBだった。
VM Bの取得・検証プログラム全体は約54.25分、bootstrap・Git反映・保存・操作待ちを含むsessionのUTC経過は約63.37分である。
帯域等を固定した統計的な速度比較ではなく、この2台の実測として後続の予算に使う。

VM Aの取得はCLI照合まで成功したが、その後の検証APIが相対pathで失敗した。
[修復記録](../../runs/run-ball-portable-distribution-20261023/colab-a-recovery.json)のとおり、
検査後のpath正規化と、JSON往復でのtrack IDsの型を修正して回帰テストを通した。
Notebook接続断でpendingになった修正版の提出はcancel・保存確認・既存VMへのreconnectを経て新jobへ置き換えた。
取得済みデータを再転送せず全件照合を完了したため、verification-onlyの所要時間を新規取得とは数えていない。
最終実行commitと検証済みのGit bundleは[code版の記録](../../runs/run-ball-portable-distribution-20261023/colab-code-revisions.json)で固定した。

[VM Aの停止](../../runs/run-ball-portable-distribution-20261023/colab-a-stop.json)と
[VM Bの停止](../../runs/run-ball-portable-distribution-20261023/colab-b-stop.json)は、CLI成功・停止後status・割当0件で確認した。
全jobのterminal状態とDrive保存を確認して停止し、tracked差分と未回収sourceはない。
大きい運用receiptは各sessionの検証済みDrive archiveへ保存し、credential・registry・鍵は記録へ含めていない。
CPU runtimeを指定できない事前調査に対し、親が今回のL4照合を例外許可した。skill変更や学習は行っていない。

消費側は原Chat注釈を`store.annotation_path(clip)`で解決する必要がある。
同梱catalog・interval・packのindexは元snapshotの番号のままなので、clip IDと配布manifestの対応表でjoinする。
原注釈の内容hashで空ラベルのnotesも保持・照合したが、不在教師policyと最終窓manifestは#1372の範囲である。
固定testのuploadは最終評価時へ回した。今回はtrain/validation全量と2台のVM照合を優先し、比較条件を縮めていない。
再現コマンドと56件の回帰テストの詳細は[run手順](../../runs/run-ball-portable-distribution-20261023/README.md)と
[validation](../../runs/run-ball-portable-distribution-20261023/validation.json)を参照する。
