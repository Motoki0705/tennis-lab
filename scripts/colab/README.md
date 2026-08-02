# Colab utilities

`scripts/colab/scripts/` は、AI が提案したファイル操作を人間が Colab 上で確認・実行するための小さなCLI群です。Google Driveの操作対象は、既定で `/content/drive/MyDrive/tennis_lab` 配下に限定されます。

## 前提

ColabでGoogle Driveをマウントしてから実行します。

```python
from google.colab import drive

drive.mount("/content/drive")
```

各コマンドはPython 3の標準ライブラリだけを使用します。Drive側の引数は、すべて `tennis_lab` からの相対パスです。絶対パス、`..`、symlinkを経由するパスは拒否されます。一覧にはsymlink自体も表示されますが、追跡はしません。

## 一覧と検索

```bash
bash scripts/colab/scripts/list_drive.sh
bash scripts/colab/scripts/list_drive.sh --path data --max-depth 3 --limit 500
bash scripts/colab/scripts/list_drive.sh --path checkpoints --format json

bash scripts/colab/scripts/search_drive.sh --name '*.ckpt' --type file
bash scripts/colab/scripts/search_drive.sh --path data --name 'scene_*' --type directory --format json
```

`--max-depth` は開始ディレクトリ直下を深さ1として数えます。検索名は大文字・小文字を区別するglobです。結果が `--limit` を超えると打ち切られ、JSONの `truncated` またはstderrの警告で通知されます。

## アップロードとダウンロード

転送先はディレクトリとして解釈せず、指定したパスそのものとして扱います。たとえば次のアップロード先は `data/sample.mp4` です。

```bash
bash scripts/colab/scripts/upload_to_drive.sh \
  /content/tennis-lab/data/sample.mp4 data/sample.mp4 \
  --dry-run

bash scripts/colab/scripts/upload_to_drive.sh \
  /content/tennis-lab/data/sample.mp4 data/sample.mp4 \
  --verify --format json

bash scripts/colab/scripts/download_from_drive.sh \
  data/sample.mp4 /content/downloads/sample.mp4 \
  --verify
```

既存の転送先はデフォルトで拒否されます。置換する場合だけ `--overwrite` を明示します。ファイルとディレクトリのどちらにも対応していますが、転送ツリー内のsymlinkは曖昧なデータ複製を避けるため拒否します。

`--verify` は一時領域へコピーした内容をSHA-256で検証してから指定先へ反映します。大きなデータでは読み取り時間が増える点に注意してください。`--dry-run` は作成・上書きを行いません。

## 検査と転送後検証

```bash
bash scripts/colab/scripts/inspect_file.sh checkpoints/model.ckpt
bash scripts/colab/scripts/inspect_file.sh checkpoints/model.ckpt --checksum --format json

bash scripts/colab/scripts/verify_transfer.sh \
  /content/tennis-lab/checkpoints/model.ckpt checkpoints/model.ckpt
```

`verify_transfer.sh` は内容が一致すれば終了コード0、不一致なら終了コード3、操作エラーなら終了コード1を返します。ディレクトリは各ファイルの相対パス、サイズ、SHA-256を比較します。

## AIへ結果を返す

`--format json` を付けると、結果をそのままAIへ渡しやすくなります。記録も残す場合はColab側で `tee` を組み合わせます。

```bash
bash scripts/colab/scripts/search_drive.sh \
  --name '*.tar.zst' --format json | tee /content/drive-search-result.json
```

テストなどで別のルートを使う場合だけ `TENNIS_LAB_DRIVE_ROOT` を指定できます。

```bash
TENNIS_LAB_DRIVE_ROOT=/tmp/mock-drive \
  bash scripts/colab/scripts/list_drive.sh --format json
```
