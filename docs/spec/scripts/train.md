# scripts: train

学習ジョブ起動用のスクリプトは `scripts/train/` にまとめられている。

- `scripts/train/run_train_scene_model.sh`
- `scripts/train/run_train_tennis_multi_cam_3d_pose.sh` (v1用)
- `scripts/train/run_train_tennis_multi_cam_3d_pose_v2.sh` (v2用)

## 1. run_train_scene_model.sh

- **役割**: シーンモデル（`task=scene_model`）の学習を起動する。
- **内部で呼び出す CLI**: `src/cli/scene_model/train.py`

### 1.1 使い方

```bash
./scripts/train/run_train_scene_model.sh
```

- 既定の設定ファイル: `configs/scene_model.yaml`
- `CONFIG` 環境変数で YAML を差し替え可能:

```bash
CONFIG=configs/scene_model_debug.yaml ./scripts/train/run_train_scene_model.sh
```

その他のオプション（`--set ...` など）は、シェルスクリプトにそのまま渡す:

```bash
./scripts/train/run_train_scene_model.sh \
  --set training.trainer.max_epochs=5
```

## 2. run_train_tennis_multi_cam_3d_pose.sh (v1用)

- **役割**: テニス multi-cam 3D pose タスク v1 の学習を起動する。
- **内部で呼び出す CLI**: `src/cli/tennis_multi_cam_3d_pose/train.py`

### 2.1 使い方

```bash
./scripts/train/run_train_tennis_multi_cam_3d_pose.sh
```

- 既定の設定ファイル: `configs/tennis_multi_cam_3d_pose.yaml`
- `CONFIG` 環境変数で YAML を差し替え可能:

```bash
CONFIG=configs/tennis_multi_cam_3d_pose_debug.yaml \
  ./scripts/train/run_train_tennis_multi_cam_3d_pose.sh
```

追加の `--set` オプションなども、そのまま引数として渡すことで CLI に伝播する。

---

## 3. run_train_tennis_multi_cam_3d_pose_v2.sh (v2用)

- **役割**: テニス multi-cam 3D pose タスク v2（階層エンコーダ + 分離出力）の学習を起動する。
- **内部で呼び出す CLI**: `src/cli/tennis_multi_cam_3d_pose/train_v2.py`

### 3.1 使い方

```bash
./scripts/train/run_train_tennis_multi_cam_3d_pose_v2.sh
```

- 既定の設定ファイル: `configs/tennis_multi_cam_3d_pose_v2.yaml`
- `CONFIG` 環境変数で YAML を差し替え可能:

```bash
CONFIG=configs/tennis_multi_cam_3d_pose_v2_debug.yaml \
  ./scripts/train/run_train_tennis_multi_cam_3d_pose_v2.sh
```

### 3.2 v2の特徴

- **階層エンコーダ**: intra → inter → temporal の3段階処理
- **分離出力**: canonical_pose, root_trans, root_rot, global_pose
- **専用損失**: 4要素損失 + マッチング重み
- **自動GT生成**: 既存のpose_3dからv2用GTを自動生成

詳細なモデル仕様は `docs/spec/models/tennis_mvpose_v2.md` を参照。

### 3.3 パラメータ上書き例

```bash
# エポック数と学習率を変更
./scripts/train/run_train_tennis_multi_cam_3d_pose_v2.sh \
  --set training.trainer.max_epochs=100 \
  --set training.optimizer.lr=5.0e-5

# 階層エンコーダの深さを変更
./scripts/train/run_train_tennis_multi_cam_3d_pose_v2.sh \
  --set model.cfg.intra_layers=4 \
  --set model.cfg.inter_layers=4 \
  --set model.cfg.temporal_layers=2
```
