# ローカルGoogle Drive操作ツール

`scripts/colab/scripts/` は、ローカルPCからGoogle Drive上の `tennis_lab` を確認・転送するためのCLI群です。Colab内での実行やDriveのファイルシステムマウントは前提にしていません。すべてのDrive操作は `rclone` を通じて行います。

## 初期設定

Python 3と[rclone](https://rclone.org/)が必要です。初回だけGoogle Driveリモートを `gdrive` という名前で設定します。

```bash
rclone config
rclone lsd gdrive:tennis_lab
```

OAuthトークンはrcloneの設定ファイルに保存されます。認証情報をこのリポジトリへ配置する必要はありません。既定の操作ルートは次のとおりです。

```text
gdrive:tennis_lab
```

別のリモート名やルートを使う場合だけ、環境変数で変更できます。安全のためDrive全体ではなく、必ず操作専用のサブディレクトリを指定してください。

```bash
export TENNIS_LAB_DRIVE_REMOTE='my-drive:projects/tennis_lab'
```

Drive側の引数はすべて、このルートからの相対パスです。絶対パス、`..`、別のrcloneリモートを示す `:` は拒否されます。

## 一覧と検索

```bash
bash scripts/colab/scripts/list_drive.sh
bash scripts/colab/scripts/list_drive.sh --path data --max-depth 3 --limit 500
bash scripts/colab/scripts/list_drive.sh --path outputs --format json

bash scripts/colab/scripts/search_drive.sh --name '*.ckpt' --type file
bash scripts/colab/scripts/search_drive.sh \
  --path data --name 'scene_*' --type directory --format json
```

`--max-depth` は開始ディレクトリ直下を深さ1として数えます。検索はrcloneで取得した一覧に対して、ファイルまたはディレクトリのベース名を大文字・小文字を区別するglobで照合します。結果が `--limit` を超えた場合は打ち切りを明示します。

## アップロードとダウンロード

転送先は指定したパスそのものとして扱います。

```bash
bash scripts/colab/scripts/upload_to_drive.sh \
  data/sample.mp4 data/sample.mp4 \
  --dry-run

bash scripts/colab/scripts/upload_to_drive.sh \
  outputs/checkpoints outputs/checkpoints \
  --verify --format json

bash scripts/colab/scripts/download_from_drive.sh \
  data/sample.mp4 downloads/sample.mp4 \
  --verify
```

既存の転送先はデフォルトで拒否します。競合するファイルを更新する場合だけ `--overwrite` を明示します。ディレクトリに `--overwrite` を指定しても、転送元に存在しない転送先ファイルは削除しません。このツールは `rclone sync` やDrive上の削除操作を行いません。

`--dry-run` はrclone自身のdry-runを実行し、Driveを書き換えません。`--verify` は転送後にローカルとDriveで共通して取得できるハッシュを照合します。ローカルの転送ツリーにsymlinkが含まれる場合は、曖昧な複製を避けるため拒否します。

## 検査と転送後検証

```bash
bash scripts/colab/scripts/inspect_file.sh outputs/model.ckpt
bash scripts/colab/scripts/inspect_file.sh \
  outputs/model.ckpt --checksum --format json

bash scripts/colab/scripts/verify_transfer.sh \
  outputs/model.ckpt outputs/model.ckpt
```

`inspect_file.sh --checksum` は、Driveバックエンドがrcloneへ公開するMD5、SHA-1、SHA-256などのハッシュを返します。ディレクトリ全体の比較には `verify_transfer.sh` を使用します。

`verify_transfer.sh` は内容が一致すれば終了コード0、不一致なら終了コード3、設定・接続・操作エラーなら終了コード1を返します。共通ハッシュがないファイルを内容まで確認する場合は `--download` を指定します。この場合はDrive上の該当ファイルを読み込み、ローカルでSHA-256を計算します。

## AIへ結果を返す

`--format json` を付けると、結果をそのままAIへ渡せます。ログを残す場合は `tee` を組み合わせます。

```bash
bash scripts/colab/scripts/search_drive.sh \
  --name '*.tar.zst' --format json | tee drive-search-result.json
```

テストでは、rcloneのlocalバックエンドを一時ディレクトリへ向けることで、Google Driveを変更せずに実際のrcloneコマンドを検証しています。
