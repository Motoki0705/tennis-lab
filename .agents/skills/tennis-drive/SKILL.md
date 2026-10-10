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
.venv/bin/python .agents/skills/tennis-drive/scripts/drive.py list --path .
```

Python標準ライブラリと認証済みのrcloneを使う。Colabでは利用可能なPythonで同じスクリプトを実行できる。
認証は既存のrclone設定または `RCLONE_CONFIG`。内容を会話・ログ・成果物へ出さない。
既定rootは `gdrive:tennis_lab`。`TENNIS_LAB_DRIVE_REMOTE` または `--remote-root remote:project-root` で明示的に変えられる。

## 操作の選び方

- 発見: `list` / `search` で相対pathとDrive IDを確認。`inspect` はファイル情報やディレクトリの総容量を返す。
- 転送: `upload` / `download` に正確なコピー先を指定し、通常は `--verify` を付ける。
- 回収: ユーザーの依頼に応じて対象を選び、`download --verify`。全量ローカル回収を既定にしない。
- 整理: `inventory --output ...` で変更前の記録を残す。`copy` / `move` は検証付きで、既存先へ上書き・結合しない。改名も `move`。
- 削除: 明示的に依頼された対象だけ `trash`。ディレクトリには対象subtreeを示す `--recursive` が必要。ゴミ箱の全削除・期限による自動削除は行わない。
- 容量: `quota` はアカウント全体、`inspect <path>` はプロジェクト配下の容量。

引数・復旧手順は [references/commands.md](references/commands.md) を必要時に読む。
各コマンドの `--help` を使い、未対応のフラグを推測しない。

## 結果の扱い

既定stdoutはJSON（`schema_version: 1`）、転送ログはstderr。終了コード0が成功、
1が操作失敗、2が引数エラー、3が `verify` の不一致。`planned` は未実行、`verified` は実際の内容確認を表す。

同名項目が複数あるpathは操作を停止する。`list` / `inventory` のDrive IDを提示して対象を特定する。
`--expected-id` はmove/copy/trash時の同一性確認であり、曖昧なpathから1件を自動選択する機能ではない。
検査と操作は他のDriveクライアントに対する原子的ロックではない。同じ対象を並行して整理しない。

タイムアウトや途中失敗の後はsource/destinationをinspectしてから続きを決める。
転送先の存在だけで成功扱いにせず、入力manifestや `verify` の結果を確認する。
内容ハッシュを得られないファイルは未検証として扱う。Google Docs等の編集・変換はこのraw-file操作の対象外。

## Colabとの連携

学習中からDriveへcheckpoint・ログ・設定を保存する。終了時の一括転送だけにしない。
共通training runnerの `run.artifact_store` を使うときも、保存先とrun IDを明示する。
Colabの起動・環境・実行はColabスキルが担当し、Driveの配置判断はこのスキルに集約する。
