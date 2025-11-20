# SceneModel Training パイプライン仕様

本書は、SceneModel の学習パイプラインを、構成要素と責務の観点からまとめる。

- コード:
  - `src/training/utils/config.py`（ConfigLoader）
  - `src/training/scene_model/*`（DataModule, LightningModule など）
- 設定:
  - `configs/scene_model.yaml`（タスク全体の設定）
  - 各種 includes: datasets/models/training/logging

ここでは **アーキテクチャ** を説明し、実行コマンドは

- `docs/spec/cli/scene_model.md`
- `docs/spec/scripts/train.md`

を参照する前提とする。

---

## 1. ConfigLoader とタスク分岐

- 実装: `src/training/utils/config.py:ConfigLoader`
- 役割:
  - トップレベル YAML（および `--set key=value`）から `DictConfig` を構築
  - `cfg.task` に応じて、適切な DataModule / LightningModule をインスタンス化する
- SceneModel の場合:
  - `task == "scene_model"` の分岐で
    - `DancetrackDataModule`
    - `SceneModelLightningModule`
  を構築する。

---

## 2. DataModule

- 実装: `src/training/scene_model/datamodule.py`
- 役割:
  - Dancetrack 系 Dataset の構築と DataLoader の定義
- 主な責務:
  - `setup(stage)` で train/val/test Dataset を用意
  - `train_dataloader`, `val_dataloader`, `test_dataloader` を返す
- Dataset:
  - `DancetrackDataset`（scene ベース tracking）
  - collate 関数: `collate_tracking` → `SceneBatch`

DataModule は **データの在り方とローディング** に集中し、モデル構造やロス定義には関与しない。

---

## 3. LightningModule

- 実装: `src/training/scene_model/lightning.py:SceneModelLightningModule`
- 役割:
  - SceneModel 本体とオプティマイザ／スケジューラ／ロス計算をまとめ、学習ループに接続する。

主なメソッド:

- `__init__(cfg: DictConfig)`
  - モデル設定・最適化設定などを受け取り、SceneModel インスタンスを構築
- `training_step(batch, batch_idx)`
  - `SceneBatch` を受け取り、順伝播とロス計算を行う
- `validation_step(batch, batch_idx)` / `test_step(batch, batch_idx)`
  - 検証・テスト時の評価指標を計算
- `configure_optimizers()`
  - Optimizer / LR Scheduler を返す

---

## 4. コールバックとロギング

- コールバック:
  - 代表例: early stopping, model checkpoint, learning rate monitor など
  - 設定: `configs/logging/*.yaml` / `configs/training/*.yaml` など
- ロギング:
  - TensorBoard / CSV ロガーなどを通じて
    - loss
    - 評価指標
    - 学習進行状況
  を記録する。

詳細な TensorBoard サマリ生成フローは `docs/spec/scripts/tensorboard.md` を参照。

---

## 5. Head Adapter / Denoiser などの補助モジュール

SceneModel 向けには、タスクやバックボーンに応じて追加の補助モジュールが存在する。

- Head Adapter:
  - 役割: モデルの中間表現を最終タスク（検出／追跡）のヘッドが扱いやすい形に変換する。

- DINO Denoiser など:
  - 役割: DETR 系の学習安定化のために、
    - 追加のノイズ付きクエリ
    - 特殊なロス計算
    を導入する。

これらは主に LightningModule 内で組み立てられ、設定ファイルで有効／無効や詳細パラメータを制御する。

---

## 6. パイプライン全体の流れ（概略）

1. トップレベル YAML を読み込み、`ConfigLoader` で `cfg` を構築
2. `cfg.task == "scene_model"` に応じて
   - `DataModule` と `SceneModelLightningModule` を生成
3. Trainer（PyTorch Lightning）が
   - `fit(datamodule=..., model=...)`
   を実行
4. DataModule が Dataset / DataLoader を提供し、LightningModule がロス計算とログを担当

---

## 7. 関連ドキュメント

- モデル構造: `docs/spec/models/scene_model.md`
- Dataset / バッチ表現:
  - `docs/spec/datasets/scene_datasets.md`
  - `docs/spec/datasets/collate_tracking.md`
- 実行方法:
  - `docs/spec/cli/scene_model.md`
  - `docs/spec/scripts/train.md`
