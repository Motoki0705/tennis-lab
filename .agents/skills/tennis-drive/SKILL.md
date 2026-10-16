---
name: tennis-drive
description: Manage tennis-lab datasets, checkpoints, and experiment artifacts on Google Drive through a JSON CLI. Use for finding, transferring, verifying, inventorying, reorganizing, or explicitly trashing project storage, and when preparing Colab inputs or recovering outputs.
---

# Tennis Lab Drive

Drive管理方針の正本は [references/storage-policy.md](references/storage-policy.md)。
新しい配置、資産の採用、整理・移行、保持方針を判断する前に読む。
ローカルのタスク出力規約は同文書から参照する。規約を別READMEへ複製しない。

入口は、このスキルの `scripts/drive.py`。repo rootからは次のように呼ぶ。

```bash
.venv/bin/python .agents/skills/tennis-drive/scripts/drive.py \
  --rclone-config "$HOME/.config/tennis-lab/rclone-drive.conf" list --path .
```

Python標準ライブラリと認証済みのrcloneを使う。Colabでは利用可能なPythonで同じスクリプトを実行できる。
認証設定の指定方法と結果の確認は [接続設定](references/commands.md#接続設定) を読む。
設定内容を会話・ログ・成果物へ出さない。
初回設定・専用OAuthへの移行は [references/authentication.md](references/authentication.md)。
既定rootは `gdrive:tennis_lab`。`TENNIS_LAB_DRIVE_REMOTE` または `--remote-root remote:project-root` で明示的に変えられる。

## 操作の選び方

- 発見: `list` / `search` で相対pathとDrive IDを確認。`inspect` はファイル情報やディレクトリの総容量を返す。
- 準備: `mkdir` で保存先ディレクトリを作る。例は [操作例](references/commands.md#操作例)。
- 転送: `upload` / `download` に正確なコピー先を指定し、通常は `--verify` を付ける。
- 回収: ユーザーの依頼に応じて対象を選び、`download --verify`。全量ローカル回収を既定にしない。
- 整理: `inventory --output ...` で変更前の記録を残す。`copy` / `move` は検証付きで、既存先へ上書き・結合しない。改名も `move`。
- 削除: 明示的に依頼された対象だけ `trash`。ディレクトリには対象subtreeを示す `--recursive` が必要。ゴミ箱の全削除・期限による自動削除は行わない。
- 容量: `quota` / `inspect`。使い分けは [容量の範囲](references/commands.md#容量の範囲)。

引数・復旧手順は [references/commands.md](references/commands.md) を必要時に読む。
各コマンドの `--help` を使い、未対応のフラグを推測しない。

## 結果の扱い

既定stdoutはJSON（`schema_version: 1`）、転送ログはstderr。終了コード0が成功、
1が操作失敗、2が引数エラー、3が `verify` の不一致。`planned` は未実行、`verified` は実際の内容確認を表す。

対象の特定と `--expected-id` は [同名pathと同一性確認](references/commands.md#同名pathと同一性確認) を読む。

タイムアウトや途中失敗では [失敗後の復旧](references/commands.md#失敗後の復旧) を読む。
内容ハッシュを得られないファイルは未検証として扱う。Google Docs等の編集・変換はこのraw-file操作の対象外。

## Colabとの連携

学習中からDriveへcheckpoint・ログ・設定を保存する。終了時の一括転送だけにしない。
共通training runnerの `run.artifact_store` を使うときも、保存先とrun IDを明示する。
Colabの起動・環境・実行はColabスキルが担当し、Driveの配置判断はこのスキルに集約する。
