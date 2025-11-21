# CLI Spec: tennis_multi_cam_3d_pose

`tennis_multi_cam_3d_pose` タスク向けの CLI は、主に以下のスクリプトから構成される。

- `src/cli/tennis_multi_cam_3d_pose/train.py` (v1用)
- `src/cli/tennis_multi_cam_3d_pose/train_v2.py` (v2/v2.5/v3共用)
- `src/cli/tennis_multi_cam_3d_pose/build_dataset.py`
- `src/cli/tennis_multi_cam_3d_pose/gen_scenes.py`
- `src/cli/tennis_multi_cam_3d_pose/preprocess_memmap.py`
- `src/cli/tennis_multi_cam_3d_pose/render_scene.py`

本ドキュメントでは、実行に必要な最小限の情報と、`scripts/` ラッパとの対応関係をまとめる。

## 1. 共通事項

- **タスク名**: すべての CLI は `cfg.task == "tennis_multi_cam_3d_pose"` を前提とする。
- **設定ファイル**:
  - トップレベル: `configs/tennis_multi_cam_3d_pose.yaml` (v1用)
  - トップレベルv2: `configs/tennis_multi_cam_3d_pose_v2.yaml` (v2用)
  - トップレベルv2.5: `configs/tennis_multi_cam_3d_pose_v2_5.yaml` (v2.5用)
  - トップレベルv3: `configs/tennis_multi_cam_3d_pose_v3.yaml` (v3用)
  - データセット: `configs/datasets/tennis_multi_cam_3d_pose_sim.yaml`
- **実行方法**: `uv run python ...` が基本。

---

## 2. train.py (v1用)

- **実装**: `src/cli/tennis_multi_cam_3d_pose/train.py`
- **役割**: テニス multi-cam 3D pose モデル v1 の学習を起動する。

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

## 3. train_v2.py (v2/v2.5/v3共用)

- **実装**: `src/cli/tennis_multi_cam_3d_pose/train_v2.py`
- **役割**: テニス multi-cam 3D pose モデル v2/v2.5/v3 の学習を起動する。`experiment_name` に基づいてバージョンを自動判定する。

### 3.1 直接実行例

#### v2
```bash
uv run python src/cli/tennis_multi_cam_3d_pose/train_v2.py \
  --config configs/tennis_multi_cam_3d_pose_v2.yaml \
  --set training.trainer.max_epochs=5
```

#### v2.5
```bash
uv run python src/cli/tennis_multi_cam_3d_pose/train_v2.py \
  --config configs/tennis_multi_cam_3d_pose_v2_5.yaml \
  --set training.trainer.max_epochs=5
```

#### v3
```bash
uv run python src/cli/tennis_multi_cam_3d_pose/train_v2.py \
  --config configs/tennis_multi_cam_3d_pose_v3.yaml \
  --set training.trainer.max_epochs=5
```

### 3.2 scripts/ ラッパとの対応

通常はバージョン専用のシェルスクリプトを通じて起動する。

```bash
# v2
./scripts/train/run_train_tennis_multi_cam_3d_pose_v2.sh

# v2.5
./scripts/train/run_train_tennis_multi_cam_3d_pose_v2_5.sh

# v3
./scripts/train/run_train_tennis_multi_cam_3d_pose_v3.sh
```

- 内部ではいずれも `uv run python src/cli/tennis_multi_cam_3d_pose/train_v2.py` を呼び出す。
- `CONFIG` 環境変数で YAML を差し替え可能:

```bash
CONFIG=configs/tennis_multi_cam_3d_pose_v3_debug.yaml \
  ./scripts/train/run_train_tennis_multi_cam_3d_pose_v3.sh
```

### 3.3 各バージョンの特徴

#### v2
- **階層エンコーダ**: intra → inter → temporal の3段階処理
- **分離出力**: canonical_pose, root_trans, root_rot, global_pose
- **専用損失**: 4要素損失 + マッチング重み
- **自動GT生成**: 既存のpose_3dからv2用GTを自動生成

#### v2.5
- **v2との互換性**: 同じ損失・I/O・可視化ロジック
- **カメラ・時間埋め込み**: エンコーダ入力トークンに明示的に付与
- **役割**: v2 との直接比較用（埋め込みの寄与を評価）

#### v3
- **track-aware temporal encoder**: Decoder出力に対し、クエリごとに時間軸TransformerEncoderを適用
- **v2との互換性**: 同じ損失・I/O・可視化ロジック
- **役割**: 時間一貫性の改善と将来の拡張向けアーキテクチャ

詳細なモデル仕様は以下を参照:
- `docs/models/tennis_mvpose_v2.md` (v2)
- `docs/models/tennis_mvpose_v3.md` (v3)
- v2.5 は v2 とほぼ同じ仕様

---

## 4. build_dataset.py

- **実装**: `src/cli/tennis_multi_cam_3d_pose/build_dataset.py`
- **役割**: シミュレータ出力から学習用のインデックス付きデータセットを構築する。
- 詳細な仕様は `docs/spec/datasets/tennis_multi_cam_3d_pose_sim.md` を参照。

### 4.1 直接実行例

```bash
uv run python src/cli/tennis_multi_cam_3d_pose/build_dataset.py \
  --dataset_root data/tennis_autogen \
  --num_scenes_train 500 --num_scenes_val 100 --num_scenes_test 100
```

### 4.2 scripts/ ラッパ

```bash
./scripts/build/build_tennis_dataset.sh
```

環境変数 `CONFIG_PATH` を上書きすることで、別のデータセット設定 YAML を指定できる。

---

## 5. gen_scenes.py

- **実装**: `src/cli/tennis_multi_cam_3d_pose/gen_scenes.py`
- **役割**: テニス用シミュレータから `scene_*.json` を生成する。
- 詳細なシーン仕様は `docs/spec/tennis/tennis_simulator.md` を参照。

### 5.1 直接実行例

```bash
uv run python src/cli/tennis_multi_cam_3d_pose/gen_scenes.py \
  --out data/tennis_autogen/raw_scenes \
  --num_scenes 100
```

---

## 6. preprocess_memmap.py

- **実装**: `src/cli/tennis_multi_cam_3d_pose/preprocess_memmap.py`
- **役割**: JSON シーンから npz/memmap 形式の配列を生成する。
- 詳細な仕様は `docs/spec/tennis/tennis_memmap_preprocess.md` を参照。

### 6.1 scripts/ ラッパ

```bash
./scripts/build/preprocess_tennis_memmap.sh
```

- `DATASET_CFG`, `DATASET_ROOT`, `DATASET_NAME` などの環境変数で、対象データセットを切り替える。

---

## 7. render_scene.py

- **実装**: `src/cli/tennis_multi_cam_3d_pose/render_scene.py`
- **役割**: 単一のシーン JSON を読み込み、`src/visualize/tennis_multi_cam_3d_pose.py` を通じて動画としてレンダリングする。

### 7.1 直接実行例

```bash
uv run python src/cli/tennis_multi_cam_3d_pose/render_scene.py \
  --scene data/tennis_autogen/scenes/train/scene_000000.json \
  --out outputs/vis/scene_000000.mp4
```
