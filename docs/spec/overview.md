# docs/spec の地図（Overview）

このドキュメントは、`docs/spec/` 以下の仕様ドキュメントの**地図**として機能する。

- 「何をしたいときに、どの spec を読めばよいか」をレイヤー別に整理する。
- **実行コマンド**は `cli/` / `scripts/` に集約し、それ以外は基本的に **アーキテクチャ・フォーマット** の説明に集中する。

---

## 1. Dataset / バッチ表現

### 1.1 シーン系 Dataset 全体

- `datasets/scene_datasets.md`

### 1.2 テニスシミュレータ / シーン JSON 仕様

- `tennis_multi_cam_3d_pose/tennis_simulator.md`
  - テニスシミュレータが出力する `scene_*.json` のフィールド・座標系・カメラ情報などの詳細仕様。
  - `scene_datasets.md` から参照される、テニス専用の「シーンフォーマット」定義。

### 1.3 SceneModel 用 Dataset

- `datasets/scene_model.md`
  - DanceTrack ベースの SceneModel 用 Dataset / バッチ表現の仕様。
  - DancetrackDataset / TrackingSample / TargetFrame / collate_tracking / SceneBatch など。

---

## 2. Models（SceneModel）

### 2.1 SceneModel アーキテクチャ

- `models/scene_model.md`
  - SceneModel の構造をレイヤーごとに説明する。
    - バックボーン
    - ポジショナルエンコーディング
    - テンポラルモジュール
    - デコーダ
    - ヘッド
    - ビルド / 設定層
  - 実際の学習フローや CLI 実行は、`training/scene_model.md` / `cli/scene_model.md` を参照。

---

## 3. Training（SceneModel）

### 3.1 SceneModel Training パイプライン

- `training/scene_model.md`
  - SceneModel の学習パイプラインを構成要素ごとに整理。
    - ConfigLoader と `task == "scene_model"` 分岐
    - DataModule（Dataset + DataLoader）
    - LightningModule（ロス計算・最適化・ログ）
    - コールバック / ロギング
    - Head Adapter / Denoiser などの補助モジュール
  - Trainer が `fit(datamodule, model)` するまでの流れを俯瞰。

- テニス multi-cam 3D pose のトレーニングフローは、よりタスク特化した説明として:
  - `training/tennis_multi_cam_3d_pose.md`
  を参照する。

---

## 4. CLI 仕様

CLI はすべて `src/cli/` 以下の Python スクリプトとして実装されており、
実行方法や引数は `docs/spec/cli/` にまとめてある。

### 4.1 CLI 全体の方針

- `cli/index.md`
  - `src/cli/` の設計方針と共通ルール。
  - `--config` + `--set key=value` による設定の考え方。

### 4.2 SceneModel 向け CLI

- `cli/scene_model.md`
  - `src/cli/scene_model/train.py` の仕様と使い方。
  - 代表的な `uv run` 例と、`scripts/train/run_train_scene_model.sh` との対応関係。

### 4.3 テニス multi-cam 3D pose 向け CLI

- `cli/tennis_multi_cam_3d_pose.md`
  - 以下の CLI を一括して説明:
    - `train.py`
    - `build_dataset.py`
    - `gen_scenes.py`
    - `preprocess_memmap.py`
    - `render_scene.py`
  - 各 CLI の役割・主要引数・`scripts/` ラッパとの対応関係を記載。

---

## 5. scripts（実行ラッパ）

`scripts/` の Bash / Python スクリプトは、日常的に使うコマンドを短く保つための **薄いラッパ** として設計されている。

### 5.1 scripts 全体の方針

- `scripts/index.md`
  - `scripts/` ディレクトリの目的と共通ルール。
  - shebang / `set -euo pipefail` / `SCRIPT_DIR` / `ROOT_DIR` / `uv run python ...` など。

### 5.2 学習ジョブ起動

- `scripts/train.md`
  - `scripts/train/run_train_scene_model.sh`
  - `scripts/train/run_train_tennis_multi_cam_3d_pose.sh`
  - `CONFIG` 環境変数や `--set` を使った実行例。

### 5.3 テニスデータパイプライン

- `scripts/tennis_data_pipeline.md`
  - `scripts/build/build_tennis_dataset.sh`
  - `scripts/build/preprocess_tennis_memmap.sh`
  - `DATASET_CFG` / `DATASET_ROOT` / `DATASET_NAME` などの env とコマンド例。

### 5.4 TensorBoard / ログ収集

- `scripts/tensorboard.md`
  - `scripts/tensorboard/run_tensorboard.sh`
  - `scripts/tensorboard/collect_tensorboard_summaries.sh`
  - `scripts/tensorboard/collect_and_summarize.py`

### 5.5 可視化系ツール

- `scripts/visualization.md`
  - `scripts/tools/render_tennis_augmented.sh`
  - `scripts/vis/render_c3d_markers.py`

---

## 6. テニス multi-cam 3D pose 専用の設計ドキュメント

タスク固有の詳細設計・アーキテクチャは、次の design/spec にまとまっている。

- `design/tennis/tennis_multi_cam_3d_pose_system.md`
  - テニス multi-cam 3D pose システム全体の設計（データフロー・モデル・トレーニング戦略など）。
- `spec/training/tennis_multi_cam_3d_pose.md`
  - テニスタスクのトレーニングパイプライン仕様（ConfigLoader 分岐・DataModule / LightningModule・CLI など）。

---

## 7. 読み方のガイド

- **まず全体像を知りたい**: `design/tennis/tennis_multi_cam_3d_pose_system.md`
- **SceneModel 系の構造と学習を知りたい**:
  - モデル: `models/scene_model.md`
  - トレーニング: `training/scene_model.md`
  - Dataset / バッチ表現: `datasets/scene_model.md`
- **テニス multi-cam 3D pose のデータまわりを知りたい**:
  - シーン JSON: `tennis_multi_cam_3d_pose/tennis_simulator.md`
  - シーン Dataset 全体: `datasets/scene_datasets.md`
- **どうやって動かすか（実行コマンド）を知りたい**:
  - CLI: `cli/*.md`
  - scripts: `scripts/*.md`
