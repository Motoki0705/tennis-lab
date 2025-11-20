# CLI Spec: tennis_multi_cam_3d_pose

`tennis_multi_cam_3d_pose` タスク向けの CLI は、主に以下のスクリプトから構成される。

- `src/cli/tennis_multi_cam_3d_pose/train.py`
- `src/cli/tennis_multi_cam_3d_pose/build_dataset.py`
- `src/cli/tennis_multi_cam_3d_pose/gen_scenes.py`
- `src/cli/tennis_multi_cam_3d_pose/preprocess_memmap.py`
- `src/cli/tennis_multi_cam_3d_pose/render_scene.py`

本ドキュメントでは、実行に必要な最小限の情報と、`scripts/` ラッパとの対応関係をまとめる。

## 1. 共通事項

- **タスク名**: すべての CLI は `cfg.task == "tennis_multi_cam_3d_pose"` を前提とする。
- **設定ファイル**:
  - トップレベル: `configs/tennis_multi_cam_3d_pose.yaml`
  - データセット: `configs/datasets/tennis_multi_cam_3d_pose_sim.yaml`
- **実行方法**: `uv run python ...` が基本。

---

## 2. train.py

- **実装**: `src/cli/tennis_multi_cam_3d_pose/train.py`
- **役割**: テニス multi-cam 3D pose モデルの学習を起動する。

### 2.1 直接実行例

```bash
uv run python src/cli/tennis_multi_cam_3d_pose/train.py \
  --config configs/tennis_multi_cam_3d_pose.yaml \
  --set training.trainer.max_epochs=5
```

### 2.2 scripts/ ラッパとの対応

通常は `scripts/train/run_train_tennis_multi_cam_3d_pose.sh` を通じて起動する。

```bash
./scripts/train/run_train_tennis_multi_cam_3d_pose.sh
```

- 内部では `uv run python src/cli/tennis_multi_cam_3d_pose/train.py` を呼び出す。
- `CONFIG` 環境変数で YAML を差し替え可能:

```bash
CONFIG=configs/tennis_multi_cam_3d_pose_debug.yaml \
  ./scripts/train/run_train_tennis_multi_cam_3d_pose.sh
```

追加の `--set` オプションはそのままシェルスクリプトに渡す。

---

## 3. build_dataset.py

- **実装**: `src/cli/tennis_multi_cam_3d_pose/build_dataset.py`
- **役割**: シミュレータ出力から学習用のインデックス付きデータセットを構築する。
- 詳細な仕様は `docs/spec/datasets/tennis_multi_cam_3d_pose_sim.md` を参照。

### 3.1 直接実行例

```bash
uv run python src/cli/tennis_multi_cam_3d_pose/build_dataset.py \
  --dataset_root data/tennis_autogen \
  --num_scenes_train 500 --num_scenes_val 100 --num_scenes_test 100
```

### 3.2 scripts/ ラッパ

```bash
./scripts/build/build_tennis_dataset.sh
```

環境変数 `CONFIG_PATH` を上書きすることで、別のデータセット設定 YAML を指定できる。

---

## 4. gen_scenes.py

- **実装**: `src/cli/tennis_multi_cam_3d_pose/gen_scenes.py`
- **役割**: テニス用シミュレータから `scene_*.json` を生成する。
- 詳細なシーン仕様は `docs/spec/tennis/tennis_simulator.md` を参照。

### 4.1 直接実行例

```bash
uv run python src/cli/tennis_multi_cam_3d_pose/gen_scenes.py \
  --out data/tennis_autogen/raw_scenes \
  --num_scenes 100
```

---

## 5. preprocess_memmap.py

- **実装**: `src/cli/tennis_multi_cam_3d_pose/preprocess_memmap.py`
- **役割**: JSON シーンから npz/memmap 形式の配列を生成する。
- 詳細な仕様は `docs/spec/tennis/tennis_memmap_preprocess.md` を参照。

### 5.1 scripts/ ラッパ

```bash
./scripts/build/preprocess_tennis_memmap.sh
```

- `DATASET_CFG`, `DATASET_ROOT`, `DATASET_NAME` などの環境変数で、対象データセットを切り替える。

---

## 6. render_scene.py

- **実装**: `src/cli/tennis_multi_cam_3d_pose/render_scene.py`
- **役割**: 単一のシーン JSON を読み込み、`src/visualize/tennis_multi_cam_3d_pose.py` を通じて動画としてレンダリングする。

### 6.1 直接実行例

```bash
uv run python src/cli/tennis_multi_cam_3d_pose/render_scene.py \
  --scene data/tennis_autogen/scenes/train/scene_000000.json \
  --out outputs/vis/scene_000000.mp4
```
