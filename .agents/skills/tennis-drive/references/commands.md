# CLIの利用と復旧

入口はrepo rootの `.venv/bin/python .agents/skills/tennis-drive/scripts/drive.py`。
`--remote-root`、`--rclone-bin`、`--timeout-seconds` はsubcommandより前に指定する。
以下のpathは例。実際の対象をlist/inspectで確定して使う。

```bash
.venv/bin/python .agents/skills/tennis-drive/scripts/drive.py list --path ckpt --max-depth 2
.venv/bin/python .agents/skills/tennis-drive/scripts/drive.py search --path outputs --name '*.ckpt' --limit 50
.venv/bin/python .agents/skills/tennis-drive/scripts/drive.py inspect data/example-v1
.venv/bin/python .agents/skills/tennis-drive/scripts/drive.py quota

.venv/bin/python .agents/skills/tennis-drive/scripts/drive.py upload ./data/example-v1 data/example-v1 --dry-run
.venv/bin/python .agents/skills/tennis-drive/scripts/drive.py upload ./data/example-v1 data/example-v1 --verify
.venv/bin/python .agents/skills/tennis-drive/scripts/drive.py download outputs/task/train/example/run-id ./recovered/run-id --verify
.venv/bin/python .agents/skills/tennis-drive/scripts/drive.py verify ./data/example-v1 data/example-v1

.venv/bin/python .agents/skills/tennis-drive/scripts/drive.py inventory --path outputs/task --output /tmp/task-inventory-unique.json
.venv/bin/python .agents/skills/tennis-drive/scripts/drive.py copy staging/candidate data/example-v2 --dry-run
.venv/bin/python .agents/skills/tennis-drive/scripts/drive.py move staging/draft staging/renamed --expected-id VERIFIED_ID
.venv/bin/python .agents/skills/tennis-drive/scripts/drive.py trash staging/obsolete --recursive --expected-id VERIFIED_ID --dry-run
```

ディレクトリ転送はその中身を指定先へコピーする。既定stdoutはJSON、転送ログはstderr。
list/searchの `truncated: true` は全件を返していない。範囲を狭めるかlimitを変える。
完全な観測をファイルへ残す場合はinventoryを使う。同名項目もID付きで記録する。
quotaはアカウント全体、inspectは対象subtreeのサイズである。

既存転送先へのupload/downloadは `--overwrite` が必要。これは既存ファイルの更新であり、
コピー元にない宛先ファイルを消す同期ではない。入力版の更新には新しい版へのコピーを優先する。
copy/moveにはoverwriteを提供しない。

`--verify` は型・サイズ・rcloneが公開する内容hashで比較する。共通hashがない場合は成功としない。
明示的な `verify --download` ならremote bytesをストリームしてSHA-256を計算するが、全量転送が発生する。

pathの各構成要素に同名項目がないことを検査する。`--expected-id` は既知のIDが変わっていないことの
補助確認で、同名のうちどれかを選ぶ引数ではない。この検査は他クライアントとの排他制御ではない。

失敗後はJSONのerror、stderr、source/destinationのinspectとinventoryで部分完了を確認する。
必要な残ファイルだけを転送し直し、verifyが通ってから完了とする。
`--timeout-seconds` は1回のrclone呼び出し上限。中断された転送先が存在し得る。
認証失敗では認証済みrclone設定の所在と有効性を確認し、token内容を表示しない。
