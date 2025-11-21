# テニス multi-cam 3D pose Quickstart

このドキュメントでは、テニス multi-cam 3D pose タスクを **シミュレーション → データセット生成 → memmap 前処理 → 学習 → 可視化** の一周で体験することを目的とする。

- 「とりあえず動かしてみたい」「全体の流れだけ掴みたい」人向け
- 各ステップの仕様や詳細は、対応する `docs/` の spec にリンクする

---

## 0. 前提

- 必要な依存は `pyproject.toml` に従ってインストール済みであること
- `uv` 経由で Python を実行する前提
- 作業ディレクトリはリポジトリルートとする

```bash
# 例
cd /path/to/tennis-lab
```

---

## 1. テニスシーンをシミュレータで生成する

シミュレータから `scene_*.json` を生成する。

- CLI: `src/cli/tennis_multi_cam_3d_pose/gen_scenes.py`
- 詳細仕様: `docs/tennis_multi_cam_3d_pose/tennis_simulator.md`

```bash
uv run python src/cli/tennis_multi_cam_3d_pose/gen_scenes.py \
  --out data/tennis_autogen/raw_scenes \
  --num_scenes 100
```

出力例:

```text
data/tennis_autogen/raw_scenes/
  scene_000000.json
  scene_000001.json
  ...
```

---

## 2. 学習用データセットを構築する（train/val/test + index）

生成したシーンから、train/val/test に分割されたデータセットとインデックスを作成する。

- CLI: `src/cli/tennis_multi_cam_3d_pose/build_dataset.py`
- scripts ラッパ: `scripts/build/build_tennis_dataset.sh`
- 詳細仕様: `docs/datasets/tennis_multi_cam_3d_pose_dataset.md`

### 2.1 scripts ラッパで実行

```bash
./scripts/build/build_tennis_dataset.sh
```

代表的な設定 YAML:

- `configs/tennis/build_tennis_dataset_sim.yaml`（scripts 内で参照）

実行後のディレクトリ構造（例）:

```text
data/tennis_autogen/
  sim_fps60_dur3p0_C4_P1-20_T10/
    meta.json
    scenes/
      train/scene_000000.json
      val/scene_000000.json
      test/scene_000000.json
    index/
      train_index.jsonl
      val_index.jsonl
      test_index.jsonl
```

---

## 3. memmap 前処理（npz 中間表現）

JSON から学習時に高速に読み込める npz/memmap 形式を生成する。

- CLI: `src/cli/tennis_multi_cam_3d_pose/preprocess_memmap.py`
- scripts ラッパ: `scripts/build/preprocess_tennis_memmap.sh`
- 詳細仕様: `docs/datasets/tennis_multi_cam_3d_pose_dataset.md` の memmap セクション

### 3.1 scripts ラッパで実行

```bash
./scripts/build/preprocess_tennis_memmap.sh
```

主な環境変数:

- `DATASET_CFG`: 使用するデータセット設定 YAML
- `DATASET_ROOT`: データセットルート（例: `data/tennis_autogen`）
- `DATASET_NAME`: データセット名（例: `sim_fps60_dur3p0_C4_P1-20_T10`）

出力例:

```text
data/tennis_autogen/
  sim_fps60_dur3p0_C4_P1-20_T10/
    arrays/
      train/scene_000000.npz
      val/scene_000000.npz
      test/scene_000000.npz
```

---

## 4. 学習を走らせる（v1 / v2）

`TennisSceneWindowDataset` → `TennisPoseDataModule` → `TennisDETR (v1/v2)` というパイプラインで学習を実行する。

- training spec: `docs/training/tennis_multi_cam_3d_pose.md`
- model spec: `docs/models/tennis_mvpose.md`, `docs/models/tennis_mvpose_v2.md`
- CLI spec: `docs/cli/tennis_multi_cam_3d_pose.md`

### 4.1 v1 モデルの学習

```bash
./scripts/train/run_train_tennis_multi_cam_3d_pose.sh
```

代表的な設定:

- トップレベル: `configs/tennis_multi_cam_3d_pose.yaml`
  - `includes.dataset`: `configs/datasets/tennis_multi_cam_3d_pose_sim.yaml`
  - `includes.model`: `configs/models/tennis_mvpose.yaml`
  - `includes.training`: `configs/training/tennis_mvpose.yaml`
  - `includes.logging`: `configs/logging/tennis_mvpose.yaml`

### 4.2 v2 モデルの学習

```bash
./scripts/train/run_train_tennis_multi_cam_3d_pose_v2.sh
```

代表的な設定:

- トップレベル: `configs/tennis_multi_cam_3d_pose_v2.yaml`
  - `includes.dataset`: `configs/datasets/tennis_multi_cam_3d_pose_sim.yaml`
  - `includes.model`: `configs/models/tennis_mvpose_v2.yaml`
  - `includes.training`: `configs/training/tennis_mvpose_v2.yaml`
  - `includes.logging`: `configs/logging/tennis_mvpose.yaml`

`--set key=value` で設定をその場で上書きできる:

```bash
./scripts/train/run_train_tennis_multi_cam_3d_pose_v2.sh \
  --set training.trainer.max_epochs=50 \
  --set model.cfg.intra_layers=4
```

Config システムの詳細は `docs/configs/index.md` を参照。

---

## 5. TensorBoard でログを確認する

学習結果は `runs/` 以下に保存される。

- scripts ドキュメント: `docs/scripts/tensorboard.md`

### 5.1 TensorBoard の起動

```bash
./scripts/tensorboard/run_tensorboard.sh
```

`RUNS_DIR` で対象ディレクトリを切り替えられる:

```bash
RUNS_DIR=runs/tennis_multi_cam_3d_pose \
  ./scripts/tensorboard/run_tensorboard.sh --port 7007
```

### 5.2 ログのサマリ生成

```bash
./scripts/tensorboard/collect_tensorboard_summaries.sh runs/tennis_multi_cam_3d_pose
```

詳細な使い方は `docs/scripts/tensorboard.md` を参照。

---

## 6. シーンと推論結果を可視化する

### 6.1 シーン JSON の可視化

- CLI: `src/cli/tennis_multi_cam_3d_pose/render_scene.py`
- 可視化ユーティリティ: `src/visualize/tennis_multi_cam_3d_pose.py`
- scripts ドキュメント: `docs/scripts/visualization.md`

```bash
uv run python src/cli/tennis_multi_cam_3d_pose/render_scene.py \
  --scene data/tennis_autogen/sim_fps60_dur3p0_C4_P1-20_T10/scenes/val/scene_000000.json \
  --out outputs/vis/scene_000000.mp4
```

### 6.2 データ拡張後のサンプル可視化

```bash
./scripts/tools/render_tennis_augmented.sh --num-samples 8 --split val
```

内部で `TennisSceneWindowDataset` を用いて、データ拡張込みの 2D/3D 情報をレンダリングする。詳細は `docs/scripts/visualization.md` を参照。

---

## 7. 次に読むべきドキュメント

Quickstart で一周したあとに、より深く理解したい場合は次を参照:

- **データフォーマットと Dataset**: `docs/datasets/tennis_multi_cam_3d_pose_dataset.md`
- **学習パイプライン仕様**: `docs/training/tennis_multi_cam_3d_pose.md`
- **モデルアーキテクチャ**:
  - v1: `docs/models/tennis_mvpose.md`
  - v2: `docs/models/tennis_mvpose_v2.md`
- **Config/YAML システム**: `docs/configs/index.md`
- **CLI 全体像**: `docs/cli/tennis_multi_cam_3d_pose.md`
