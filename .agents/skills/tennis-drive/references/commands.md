# CLIの利用と復旧

入口はrepo rootの `.venv/bin/python .agents/skills/tennis-drive/scripts/drive.py`。
`--remote-root`、`--rclone-config`、`--rclone-bin`、`--timeout-seconds` はsubcommandより前に指定する。

## 接続設定

既存の設定ファイルを `--rclone-config /private/path/rclone.conf` または `RCLONE_CONFIG` で明示する。
両方あれば `--rclone-config` が優先される。設定ファイルがない場合や指定がない場合は操作を失敗させ、
専用設定・rcloneの既定設定へ自動的に切り替えない。各操作の結果JSONの `rclone_config` は
選択した設定の絶対pathを返す。設定未指定のエラーでは `null` になる。
内容は表示せず、認証手順は [authentication.md](authentication.md) を参照する。

多数の小さな不変ファイルを扱う呼出元は、`drive_core.RcloneSession(backend)`を
context managerとして使える。同じ明示設定で、認証付き・loopback限定の`rcd`を1つ
起動し、`call(endpoint, **parameters)`でRC APIを呼ぶ。正常終了・例外の双方で停止する。
これは接続の再利用だけを担当し、path・重複・ID・内容hash・上書き防止・完了確認は
呼出元が保証する。RCの`copyfile`は`Immutable`だけでは既存先を保護しないため、
`IgnoreExisting`と転送後の新鮮な内容照合を組み合わせる。既定CLIの動作は変更しない。

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
