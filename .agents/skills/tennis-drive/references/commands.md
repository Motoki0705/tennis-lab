# CLIの利用と復旧

入口はrepo rootの `.venv/bin/python .agents/skills/tennis-drive/scripts/drive.py`。
`--remote-root`、`--rclone-config`、`--rclone-bin`、`--timeout-seconds` はsubcommandより前に指定する。

## 接続設定

既存の設定ファイルを `--rclone-config /private/path/rclone.conf` または `RCLONE_CONFIG` で明示する。
両方あれば `--rclone-config` が優先される。設定ファイルがない場合や指定がない場合は操作を失敗させ、
専用設定・rcloneの既定設定へ自動的に切り替えない。各操作の結果JSONの `rclone_config` は
選択した設定の絶対pathを返す。設定未指定のエラーでは `null` になる。
内容は表示せず、認証手順は [authentication.md](authentication.md) を参照する。

```bash
export RCLONE_CONFIG="$HOME/.config/tennis-lab/rclone-drive.conf"
```

## 操作例

以下のpathは例。実際の対象をlist/inspectで確定して使う。

```bash
.venv/bin/python .agents/skills/tennis-drive/scripts/drive.py list --path ckpt --max-depth 2
.venv/bin/python .agents/skills/tennis-drive/scripts/drive.py search --path outputs --name '*.ckpt' --limit 50
.venv/bin/python .agents/skills/tennis-drive/scripts/drive.py inspect data/example-v1
.venv/bin/python .agents/skills/tennis-drive/scripts/drive.py inspect . --metadata-only
.venv/bin/python .agents/skills/tennis-drive/scripts/drive.py quota

.venv/bin/python .agents/skills/tennis-drive/scripts/drive.py upload ./data/example-v1 data/example-v1 --dry-run
.venv/bin/python .agents/skills/tennis-drive/scripts/drive.py upload ./data/example-v1 data/example-v1 --verify
.venv/bin/python .agents/skills/tennis-drive/scripts/drive.py download outputs/task/train/example/run-id ./recovered/run-id --verify
.venv/bin/python .agents/skills/tennis-drive/scripts/drive.py verify ./data/example-v1 data/example-v1

.venv/bin/python .agents/skills/tennis-drive/scripts/drive.py mkdir staging/candidate --dry-run
.venv/bin/python .agents/skills/tennis-drive/scripts/drive.py mkdir staging/candidate
.venv/bin/python .agents/skills/tennis-drive/scripts/drive.py inventory --path outputs/task --output /tmp/task-inventory-unique.json
.venv/bin/python .agents/skills/tennis-drive/scripts/drive.py copy staging/candidate data/example-v2 --dry-run
.venv/bin/python .agents/skills/tennis-drive/scripts/drive.py move staging/draft staging/renamed --expected-id VERIFIED_ID
.venv/bin/python .agents/skills/tennis-drive/scripts/drive.py trash staging/obsolete --recursive --expected-id VERIFIED_ID --dry-run
```

ディレクトリ転送はその中身を指定先へコピーする。既定stdoutはJSON、転送ログはstderr。
list/searchの `truncated: true` は全件を返していない。範囲を狭めるかlimitを変える。
完全な観測をファイルへ残す場合はinventoryを使う。同名項目もID付きで記録する。
mkdirは既存ディレクトリへの再実行も許容するが、既存ファイルとproject rootは対象にできない。

既存転送先へのupload/downloadは `--overwrite` が必要。これは既存ファイルの更新であり、
コピー元にない宛先ファイルを消す同期ではない。入力版の更新には新しい版へのコピーを優先する。
copy/moveにはoverwriteを提供しない。

`--verify` は型・サイズ・rcloneが公開する内容hashで比較する。共通hashがない場合は成功としない。
明示的な `verify --download` ならremote bytesをストリームしてSHA-256を計算するが、全量転送が発生する。

## 閉じたdirectoryの一括取得・照合

`download-tree` は、書込みが終わったraw-file directoryを有限の1操作で取得する。
既存の `download` / `verify`（単一ファイルを含む）の挙動は変えない。

```bash
.venv/bin/python .agents/skills/tennis-drive/scripts/drive.py download-tree \
  outputs/task/evaluate/example/run-id ./recovered/run-id \
  --expected-manifest ./expected-files.json --progress-file ./download-progress.json
```

期待manifestは `schema_version: 1` と `files` objectを持つJSON。
`files` はdirectory内の相対ファイルpathから、信頼する元の記録のSHA-256文字列への対応とする。
期待値を取得後のbytesから作り直さない。`--expected-manifest` を省略した場合もDriveとの全内容照合は必須だが、
過去の期待集合・SHA-256との一致を証明したことにはならない。来歴の意味検証は呼出し側が引き続き行う。

祖先の各階層を1回ずつ解決し、subtreeを再帰listingしてpath・ID・size・hashを集め、
directoryの `copy` を1processで実行する。全ローカルbytesをPythonでhashし、再度の階層解決と再帰listingで
転送中の変更を拒否する。rclone process数は新規取得で `2 × root込みの階層数 + 3`、
既存directoryの再照合ではcopyがなく1回少ない。ファイル数に比例する別process起動やstatはない。
これはHTTPリクエスト数や転送量を一定にする保証ではない。

祖先・subtreeの重複名、期待ファイルの不足・余分、型・size・hash・IDの変化を拒否する。
対応する内容hashがないファイルも成功にしない。`--expected-id` はdirectoryの既知IDを追加照合し、
同名候補を選ぶ引数にはしない。これは並行writerとのlockではないため、書込み中のtreeには使わない。

新規directoryは同じ親の一時directoryへ取得し、照合が全部通ってから正式名にrenameする。
既存destinationは全byteと完全な集合を再検証し、上書き・結合・静かな修復をしない。
失敗時は正式な新規destinationを作らない。強制終了後に残る `.fetching-*` は成功印ではない。
再実行前に部分成果を確認する。`--overwrite` は提供しない。

`--progress-file` はdestinationの外へ置くローカルJSONで、Driveへは公開しない。
`operation` は `listing / downloading / verifying / readback / completed / failed`。
`completed_files / total_files / completed_bytes / total_bytes` は現在の下位工程の件数・byte数で、
download中は同じrclone processのJSON stats、verify中は読んだローカルbytesから更新する。
verifyの開始時は件数を0へ戻す。`updated_at / last_progress_at / elapsed_seconds` も記録する。
同じ下位工程では最大1秒に1回、工程遷移と終了は即時にローカルへ保存する。
タイムアウト・失敗は非0で、存在や転送件数だけをverifiedとして扱わない。

Pythonの呼出し側は `scripts/drive_tree.py` の `download_tree` を使い、同じ検証を通せる。
`expected_files` に相対path→SHA-256、`progress` にkeyword callbackを渡す。
callbackはローカル処理であり、各ファイルでDrive通信を追加しない。

### 検証するテスト

局所的な拒否条件・RecordingBackendによる呼出し数・streaming statsは
`tests/unit/skills/test_tennis_drive.py`、実rcloneとlocal aliasのCLI接続は
`tests/integration/skills/test_tennis_drive_cli.py` に置く。配置・予算は
[tests README](../../../../tests/README.md#テスト作成方針)に従う。

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 .venv/bin/python -m pytest -n 0 \
  tests/unit/skills/test_tennis_drive.py tests/integration/skills/test_tennis_drive_cli.py
```

## 容量の範囲

quotaはアカウント全体、inspectは対象subtreeのサイズである。

## 同名pathと同一性確認

pathの各構成要素に同名項目がないことを検査する。`--expected-id` は既知のIDが変わっていないことの
補助確認で、同名のうちどれかを選ぶ引数ではない。この検査は他クライアントとの排他制御ではない。
曖昧なpathでは操作を停止し、list/inventoryのDrive IDを提示して対象を特定する。同じ対象を並行して整理しない。

## 失敗後の復旧

失敗後はJSONのerror、stderr、source/destinationのinspectとinventoryで部分完了を確認する。
転送先の存在だけで成功扱いにせず、入力manifestと照合する。同じ操作を盲目的に繰り返さず、
必要な残ファイルだけを転送し直し、verifyが通ってから完了とする。
`--timeout-seconds` は1回のrclone呼び出し上限。中断された転送先が存在し得る。
認証失敗では認証済みrclone設定の所在と有効性を確認し、token内容を表示しない。
