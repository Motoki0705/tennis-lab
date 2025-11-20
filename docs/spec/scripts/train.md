# scripts: train

学習ジョブ起動用のスクリプトは `scripts/train/` にまとめられている。

- `scripts/train/run_train_scene_model.sh`
- `scripts/train/run_train_tennis_multi_cam_3d_pose.sh`

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

## 2. run_train_tennis_multi_cam_3d_pose.sh

- **役割**: テニス multi-cam 3D pose タスクの学習を起動する。
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
