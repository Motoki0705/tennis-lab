# PLCS E2E テスト

PLCS (Player Location from Court Skeleton) のE2Eテストドキュメントです。

## テストファイル一覧

| ファイル | 説明 |
|---------|------|
| `test_data_validation.py` | 生成データとデータセットのスキーマ検証 |
| `test_generate_dataset.py` | データセット生成スクリプトのテスト |
| `test_train.py` | 学習スクリプトのテスト |
| `test_visualize.py` | 可視化スクリプトのテスト |

---

## test_data_validation.py

生成されたデータがPLCSの型定義（`src/plcs/data/types.py`）に準拠することを検証します。

### TestPLCSGeneratedSceneValidation

NPZシーンファイルの検証テスト。

| テスト | 検証内容 |
|--------|---------|
| `test_scene_meta_schema` | シーンメタデータが `PLCSSceneMeta` に準拠 |
| `test_camera_params_schema` | カメラパラメータが `PLCSCameraParams` に準拠 |
| `test_position_data_shapes` | 位置・回転データの形状が正しい |
| `test_rotation_normalized` | 回転値が [-1, 1] 範囲（sin/cos） |
| `test_camera_keypoint_data_shapes` | カメラキーポイントデータの形状 |
| `test_visibility_ratio_in_range` | 可視性比率が [0, 1] 範囲内 |

**要件**:
- `position`: (T, 3) 正規化コート位置
- `rotation`: (T, 2) [sin(yaw), cos(yaw)]、値は [-1, 1]
- `human_kp_uv`: (T, 17, 2) 人体キーポイント
- `court_kp_uv`: (T, 20, 2) コートキーポイント

### TestPLCSDatasetSampleValidation

`SceneDataset` サンプルの検証テスト。

| テスト | 検証内容 |
|--------|---------|
| `test_sample_has_required_keys` | 必須キーの存在 |
| `test_sample_schema_validation` | `PLCSFrameBatch` スキーマへの準拠 |
| `test_sample_tensor_shapes` | テンソル形状の検証 |
| `test_sample_rotation_normalized` | 回転が [-1, 1] 範囲 |
| `test_sample_visibility_masks_valid` | 可視性マスクが 0/1 |
| `test_sample_tensor_dtypes` | テンソルが float32/float64 |

**要件**:
- `human_kp`: (34,) = 17キーポイント × 2座標（フラット化）
- `court_kp`: (40,) = 20キーポイント × 2座標（フラット化）
- `human_vis`: (17,), 0/1
- `court_vis`: (20,), 0/1
- `position`: (3,)
- `rotation`: (2,), [-1, 1]

### TestPLCSDataLoaderBatchValidation

`DataLoader` バッチの検証テスト。

| テスト | 検証内容 |
|--------|---------|
| `test_batch_shapes_consistent` | バッチ次元の整合性 |
| `test_batch_tensor_shapes` | バッチ化されたテンソルの形状 |

**要件**:
- `human_kp`: (B, 34)
- `court_kp`: (B, 40)
- `position`: (B, 3)
- `rotation`: (B, 2)

---

## test_generate_dataset.py

データセット生成スクリプト（`src.plcs.scripts.generate_dataset`）のテスト。

| テスト | 検証内容 |
|--------|---------|
| `test_plcs_generate_dataset` | スクリプトが正常終了し、必要なファイルが生成される |
| `test_plcs_generate_dataset_custom_settings` | カスタム設定での生成 |

### 使用される設定オーバーライド

```bash
# 標準テスト
simulation.num_scenes=10
simulation.num_cameras=2
simulation.human_visibility_threshold=0.3

# カスタム設定テスト
simulation.num_scenes=5
simulation.num_cameras=1
```

### 出力検証

- `{output_dir}/scenes/scene_*.npz` が1つ以上存在
- `{output_dir}/stats.json` が存在
- `{output_dir}/scenes_meta.json` が存在
- `{output_dir}/meta.json` が存在

---

## test_train.py

学習スクリプト（`src.plcs.scripts.train`, `src.plcs.scripts.train_sequence`）のテスト。

| テスト | GPU要否 | 検証内容 |
|--------|--------|---------|
| `test_plcs_train_basic` | ✓ CUDA | フレームモデルの基本学習 |
| `test_plcs_train_sequence` | ✓ CUDA | シーケンスモデルの学習 |
| `test_plcs_train_with_custom_params` | ✓ CUDA | カスタムパラメータでの学習 |
| `test_plcs_train_dry_run` | CPU | フレームモデルのドライラン |
| `test_plcs_train_sequence_dry_run` | CPU | シーケンスモデルのドライラン |

### モデルの種類

| スクリプト | モデル | 説明 |
|-----------|--------|------|
| `train` | `PLCSModel` | 単一フレーム入力で位置・回転を予測 |
| `train_sequence` | `PLCSSequenceModel` | 時系列入力で位置・回転を予測 |

### 出力検証

- `{output_dir}/checkpoints/*.ckpt` が存在
- `{output_dir}/config.yaml` が存在

### ドライランモード

```bash
# フレームモデル
uv run python -m src.plcs.scripts.train run.dry_run=true

# シーケンスモデル
uv run python -m src.plcs.scripts.train_sequence run.dry_run=true
```

---

## test_visualize.py

可視化スクリプト（`src.plcs.scripts.visualize`）のテスト。

| テスト | 検証内容 |
|--------|---------|
| `test_plcs_visualize_ground_truth` | グラウンドトゥルース可視化 |
| `test_plcs_visualize_predict` | フレームモデル予測可視化 |
| `test_plcs_visualize_sequence_predict` | シーケンスモデル予測可視化 |
| `test_plcs_visualize_different_views` | 2Dビューの可視化 |

### 可視化モード

| モード | 説明 | 必要なもの |
|--------|------|-----------|
| `visualize` | グラウンドトゥルースの表示 | シーンファイル |
| `predict` | フレームモデル予測の表示 | シーンファイル + フレームモデルチェックポイント |
| `predict-seq` | シーケンスモデル予測の表示 | シーンファイル + シーケンスモデルチェックポイント |

### ビューオプション

| ビュー | 説明 |
|--------|------|
| `3d` | 3D空間でのプレーヤー位置表示 |
| `2d` | コート平面（上から見た図） |

### 出力検証

- 指定された出力パスに PNG ファイルが生成される

---

## フィクスチャの使用

すべてのテストは `tests/e2e/fixtures/plcs_fixtures.py` のヘルパーを使用：

```python
from tests.e2e.fixtures.plcs_fixtures import (
    create_minimal_plcs_dataset,           # 最小データセット生成
    create_minimal_plcs_checkpoint,        # フレームモデルチェックポイント
    create_minimal_plcs_sequence_checkpoint, # シーケンスモデルチェックポイント
    make_minimal_plcs_scene,               # 単一シーン生成
)
```

### 生成されるテストデータ

- **シーン**: 64フレーム、1カメラ、5ジョイント
- **データセット**: 10シーン（train 70% / val 15% / test 15%）
- **チェックポイント**: 
  - フレームモデル: hidden_dim=256, num_layers=4
  - シーケンスモデル: hidden_dim=256, num_layers=4, max_seq_len=120

---

## テストの実行

```bash
# PLCSテストのみ実行
uv run pytest tests/e2e/plcs -v

# データ検証テストのみ
uv run pytest tests/e2e/plcs/test_data_validation.py -v

# GPUテストをスキップ
uv run pytest tests/e2e/plcs -v -m "not cuda"

# シーケンスモデル関連のテスト
uv run pytest tests/e2e/plcs -v -k "sequence"
```
