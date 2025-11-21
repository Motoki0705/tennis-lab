# scripts: tennis data pipeline

テニス multi-cam 3D pose タスク向けのデータ生成・前処理パイプラインは、主に次のスクリプトから構成される。

- `scripts/build/build_tennis_dataset.sh`
- `scripts/build/preprocess_tennis_memmap.sh`

これらは、それぞれ `src/cli/tennis_multi_cam_3d_pose/build_dataset.py` と `src/cli/tennis_multi_cam_3d_pose/preprocess_memmap.py` の薄いラッパであり、詳細仕様は:

- `docs/spec/datasets/tennis_multi_cam_3d_pose_sim.md`
- `docs/spec/tennis/tennis_dataset_build.md`
- `docs/spec/tennis/tennis_memmap_preprocess.md`

を参照する。

## 1. build_tennis_dataset.sh

- **役割**: シミュレータ出力から学習用インデックス付きデータセットを構築する。
- **内部で呼び出す CLI**: `src/cli/tennis_multi_cam_3d_pose/build_dataset.py`

### 1.1 使い方

```bash
./scripts/build/build_tennis_dataset.sh
```

- 既定の設定ファイル: `configs/tennis/build_tennis_dataset_sim.yaml`
- `CONFIG_PATH` 環境変数で YAML を差し替え可能:

```bash
CONFIG_PATH=configs/tennis/build_tennis_dataset_small.yaml \
  ./scripts/build/build_tennis_dataset.sh
```

## 2. preprocess_tennis_memmap.sh

- **役割**: JSON シーンから npz/memmap 形式の配列を生成する。
- **内部で呼び出す CLI**: `src/cli/tennis_multi_cam_3d_pose/preprocess_memmap.py`

### 2.1 使い方

```bash
./scripts/build/preprocess_tennis_memmap.sh
```

主要な環境変数:

- `DATASET_CFG` (既定: `configs/datasets/tennis_multi_cam_3d_pose_sim.yaml`)
- `DATASET_ROOT` (既定: `data/tennis_autogen`)
- `DATASET_NAME` (既定: `sim_fps60_dur3p0_C4_P1-20_T10`)

例:

```bash
DATASET_NAME=sim_fps60_dur3p0_C4_P1-40_T10 \
  ./scripts/build/preprocess_tennis_memmap.sh --overwrite
```

memmap 形式の詳細なフォーマットは `docs/spec/tennis/tennis_memmap_preprocess.md` を参照する。
