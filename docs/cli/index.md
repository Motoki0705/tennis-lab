# CLI 仕様の概要

このディレクトリでは、`src/cli/` 配下の CLI エントリポイントの仕様と、対応する `scripts/` ラッパの使い方をまとめる。

- **共通方針**
  - すべての CLI は Python スクリプト（`uv run python ...`）として実行する。
  - 実運用では、`scripts/` 配下の Bash ラッパから呼び出す運用を推奨する。
  - 設定は原則として YAML (`configs/*.yaml`) + `--set key=value` で上書きする。

- **タスク別ドキュメント**
  - `scene_model.md` … シーンモデル学習タスクの CLI 仕様
  - `tennis_multi_cam_3d_pose.md` … テニス multi-cam 3D pose タスクの CLI 仕様

各タスクごとのファイルでは、

- 関連する `src/cli/<task>/` 配下のスクリプト一覧
- 主要な引数と設定ファイル
- 代表的な実行例（`uv run` / `scripts/` ラッパ）

を記述する。
