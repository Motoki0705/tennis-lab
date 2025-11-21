# テニス multi-cam 3D pose システム Overview

このドキュメントは、テニス multi-cam 3D pose タスク全体の **鳥瞰図** を提供する。

- シミュレータ → データセット生成 → memmap → Dataset → Training → Model → 可視化
- 各ステップの詳細仕様は既存の docs に委譲し、ここでは **コンポーネントと依存関係** を整理する。

Quickstart で 1 周したい場合は `docs/tennis_multi_cam_3d_pose/quickstart.md` を参照。

---

## 1. 全体像

テニス multi-cam 3D pose タスクの典型的なフロー:

```text
[1] シミュレータ
    ↓ gen_scenes.py
[2] raw scenes (scene_*.json)
    ↓ build_dataset.py
[3] dataset_root/dataset_name/{scenes,index}
    ↓ preprocess_memmap.py
[4] dataset_root/dataset_name/arrays/*.npz
    ↓ TennisSceneWindowDataset
[5] DataModule (TennisPoseDataModule)
    ↓
[6] Model (TennisDETR v1/v2)
    ↓
[7] 学習・評価・可視化
```

対応する主なコードと docs は次の通り:

- シミュレータ: `src/tennis/sim/*`, `src/cli/tennis_multi_cam_3d_pose/gen_scenes.py`
  - docs: `docs/tennis_multi_cam_3d_pose/tennis_simulator.md`
- データセット生成: `src/cli/tennis_multi_cam_3d_pose/build_dataset.py`
  - docs: `docs/datasets/tennis_multi_cam_3d_pose_dataset.md`
- memmap 前処理: `src/cli/tennis_multi_cam_3d_pose/preprocess_memmap.py`
  - docs: `docs/datasets/tennis_multi_cam_3d_pose_dataset.md`
- Dataset 実装: `src/datasets/tennis/scene_dataset.py:TennisSceneWindowDataset`
  - docs: `docs/datasets/tennis_multi_cam_3d_pose_dataset.md`
- DataModule / LightningModule / Training パイプライン:
  - `src/training/tennis_multi_cam_3d_pose/datamodule.py:TennisPoseDataModule`
  - `src/training/tennis_multi_cam_3d_pose/lightning.py:TennisDetrModule` (v1)
  - `src/training/tennis_multi_cam_3d_pose/lightning_v2.py:TennisDetrV2Module` (v2)
  - docs: `docs/training/tennis_multi_cam_3d_pose.md`
- モデル本体:
  - `src/models/tennis_multi_cam_3d_pose/model.py:TennisDETR` (v1)
  - `src/models/tennis_multi_cam_3d_pose/model_v2.py:TennisDETR_v2` (v2)
  - docs: `docs/models/tennis_mvpose.md`, `docs/models/tennis_mvpose_v2.md`
- 可視化:
  - `src/cli/tennis_multi_cam_3d_pose/render_scene.py`
  - `src/visualize/tennis_multi_cam_3d_pose.py`, `src/visualize/tennis_render.py`
  - docs: `docs/scripts/visualization.md`

---

## 2. コンフィグと CLI の関係

テニスタスクの学習は、トップレベル YAML と CLI で制御される。

### 2.1 トップレベル YAML

- v1: `configs/tennis_multi_cam_3d_pose.yaml`
- v2: `configs/tennis_multi_cam_3d_pose_v2.yaml`

どちらも共通して:

```yaml
task: tennis_multi_cam_3d_pose
experiment_name: ...

includes:
  dataset: datasets/tennis_multi_cam_3d_pose_sim.yaml
  model: models/tennis_mvpose(_v2).yaml
  training: training/tennis_mvpose(_v2).yaml
  logging: logging/tennis_mvpose.yaml
```

`load_cfg()` が `includes` を展開し、`cfg.dataset`, `cfg.model`, `cfg.training`, `cfg.logging` を構成する。詳細は `docs/configs/index.md` を参照。

### 2.2 CLI エントリポイント

- v1 学習: `src/cli/tennis_multi_cam_3d_pose/train.py`
- v2 学習: `src/cli/tennis_multi_cam_3d_pose/train_v2.py`

どちらも共通して:

1. `--config` でトップレベル YAML を指定
2. `--set key=value` で個別パラメータを上書き
3. `ConfigLoader(cfg)` を用いて DataModule / LightningModule / Trainer を構築

scripts ラッパ:

- v1: `scripts/train/run_train_tennis_multi_cam_3d_pose.sh`
- v2: `scripts/train/run_train_tennis_multi_cam_3d_pose_v2.sh`

詳細は `docs/cli/tennis_multi_cam_3d_pose.md`, `docs/scripts/train.md` を参照。

---

## 3. Dataset レイヤ

### 3.1 シーン JSON

- 形式: `scene_*.json`
- 生成元: シミュレータ (`gen_scenes.py`)
- 内容: フレーム列 `frames[t]`、カメラ列 `cameras[v]`、プレーヤー/ラケット/コートの 2D/3D キーポイント、カメラパラメータなど。

仕様の詳細は `docs/tennis_multi_cam_3d_pose/tennis_simulator.md` を参照。

### 3.2 データセットビルド & index

- CLI: `build_dataset.py`
- 出力:
  - `scenes/{train,val,test}/scene_*.json`
  - `index/{split}_index.jsonl` （1 ウィンドウ 1 行）

仕様の詳細は `docs/datasets/tennis_multi_cam_3d_pose_dataset.md` を参照。

### 3.3 memmap (arrays/*.npz)

- CLI: `preprocess_memmap.py`
- 出力: `arrays/{train,val,test}/scene_*.npz`
- `TennisSceneWindowDataset` が読み込む配列群（`keypoints_2d`, `pose_3d_gt`, `camera_*` など）を格納。

`TennisSceneWindowDataset` の想定するテンソル形状や v2 用 GT 自動生成についても、`docs/datasets/tennis_multi_cam_3d_pose_dataset.md` を参照。

---

## 4. Training レイヤ

### 4.1 DataModule

- 実装: `src/training/tennis_multi_cam_3d_pose/datamodule.py:TennisPoseDataModule`
- 役割:
  - `TennisSceneWindowDataset` を train/val/test 用に構築
  - DataLoader を設定（バッチサイズ・num_workers など）

### 4.2 LightningModule

- v1: `TennisDetrModule`
- v2: `TennisDetrV2Module`

主な役割:

- 入力バッチ（`keypoints_2d`, `player_mask`, `court_2d`, `pose_3d_gt` など）からモデルを呼び出し
- v1/v2 それぞれに応じたロスを計算
- ロギングやチェックポイント保存を行う

詳細は `docs/training/tennis_multi_cam_3d_pose.md` を参照。

---

## 5. Model レイヤ

### 5.1 v1: TennisDETR

- 実装: `src/models/tennis_multi_cam_3d_pose/model.py:TennisDETR`
- 設定クラス: `TennisDetrConfig`
- 出力:
  - `pose_3d[B, Q, T, J, 3]`
  - `exist_conf[B, Q, 1]`

仕様詳細は `docs/models/tennis_mvpose.md` を参照。

### 5.2 v2: TennisDETR_v2

- 実装: `src/models/tennis_multi_cam_3d_pose/model_v2.py:TennisDETR_v2`
- 設定クラス: `TennisDetrV2Config`
- 分離出力:
  - `canonical_pose`, `root_trans`, `root_rot`, `global_pose`, `exist_conf`

仕様詳細は `docs/models/tennis_mvpose_v2.md` を参照。

---

## 6. 可視化レイヤ

### 6.1 シーンのレンダリング

- CLI: `src/cli/tennis_multi_cam_3d_pose/render_scene.py`
- バックエンド: `src/visualize/tennis_multi_cam_3d_pose.py`, `src/visualize/tennis_render.py`

役割:

- `scene_*.json` を読み込み、コート線・プレーヤー骨格・ラケットを描画した動画を出力

### 6.2 データ拡張の可視化

- scripts: `scripts/tools/render_tennis_augmented.sh`
- Dataset: `TennisSceneWindowDataset`（augment 付き）

役割:

- 2D アフィン拡張後のサンプルをレンダリングし、augmentation の挙動を確認する

詳細は `docs/scripts/visualization.md` を参照。

---

## 7. どのドキュメントを読むべきか

- **全体像を掴みたい**:
  - 本ドキュメント（Overview）
  - `docs/tennis_multi_cam_3d_pose/quickstart.md`
- **データフォーマット・Dataset を理解したい**:
  - `docs/tennis_multi_cam_3d_pose/tennis_simulator.md`
  - `docs/datasets/tennis_multi_cam_3d_pose_dataset.md`
- **学習パイプラインを理解したい**:
  - `docs/training/tennis_multi_cam_3d_pose.md`
- **モデルアーキテクチャを理解したい**:
  - `docs/models/tennis_mvpose.md`
  - `docs/models/tennis_mvpose_v2.md`
- **実行コマンドを把握したい**:
  - `docs/cli/tennis_multi_cam_3d_pose.md`
  - `docs/scripts/*.md`
