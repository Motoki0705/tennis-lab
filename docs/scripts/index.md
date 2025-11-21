# scripts/ の設計方針

`scripts/` ディレクトリは、`src/cli/` への薄いラッパとして機能する Bash / Python スクリプト群をまとめる場所である。

- **目的**
  - よく使うコマンドを短いシェルスクリプトにまとめ、毎回長い `uv run python ...` を書かなくてよいようにする。
  - チーム内で同じ実行方法を共有しやすくする。

- **共通ルール（Bash ラッパ）**
  - 先頭に `#!/usr/bin/env bash` と `set -euo pipefail` を付ける。
  - `SCRIPT_DIR` / `ROOT_DIR` を用いてプロジェクトルートに `cd` してから実行する。
  - Python コマンドは `uv run python ...` 経由で呼び出す。
  - 設定ファイルパスなどは環境変数（`CONFIG`, `CONFIG_PATH`, `DATASET_CFG`, `RUNS_DIR` など）で上書き可能にする。

- **サブドキュメント**
  - `train.md` … 学習ジョブ起動用スクリプト
  - `tennis_data_pipeline.md` … テニスデータ生成・前処理パイプライン
  - `tensorboard.md` … TensorBoard 関連スクリプト
  - `visualization.md` … 可視化系スクリプト

各ファイルでは、対応する `src/cli/` の関係・環境変数・代表的な実行例を記載する。
