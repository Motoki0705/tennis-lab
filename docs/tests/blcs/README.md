# BLCS E2E テスト

BLCS (Ball Location from Court Skeleton) のE2Eテストドキュメントです。

## テストファイル一覧

| ファイル | 説明 |
|---------|------|
| `test_data_validation.py` | 生成データとデータセットのスキーマ検証 |
| `test_generate_dataset.py` | データセット生成スクリプトのテスト |
| `test_train.py` | 学習スクリプトのテスト |
| `test_visualize.py` | 可視化スクリプトのテスト |

---

## test_data_validation.py

生成されたデータがBLCSの型定義（`src/blcs/data/types.py`）に準拠することを検証します。

### TestBLCSGeneratedSceneValidation

NPZシーンファイルの検証テスト。

| テスト | 検証内容 |
|--------|---------|
| `test_scene_meta_schema` | シーンメタデータが `BLCSSceneMeta` に準拠 |
| `test_camera_params_schema` | カメラパラメータが `BLCSCameraParams` に準拠 |
| `test_trajectory_data_shapes` | 3D軌道データの形状が正しい（ball_pos_world, ball_pos_norm, ball_vel_world） |
| `test_camera_projection_data_shapes` | カメラ投影データの形状（ball_uv, ball_visible, court_kp_uv） |
| `test_visibility_ratio_in_range` | 可視性比率が [0, 1] 範囲内 |

**要件**:
- `ball_pos_world`: (T, 3) ワールド座標
- `ball_pos_norm`: (T, 3) 正規化座標
- `ball_vel_world`: (T, 3) 速度ベクトル
- `ball_uv`: (T, 2) 正規化UV座標 [0, 1]
- `court_kp_uv`: (20, 2) コートキーポイントUV

### TestBLCSDatasetSampleValidation

`BallTrajectoryDataset` サンプルの検証テスト。

| テスト | 検証内容 |
|--------|---------|
| `test_sample_has_required_keys` | 必須キー（ball_uv, ball_mask, court_kp, etc.）の存在 |
| `test_sample_schema_validation` | `BLCSSample` スキーマへの完全準拠 |
| `test_sample_uv_normalized` | UV座標が [0, 1] 範囲に正規化 |
| `test_sample_tensor_dtypes` | テンソルが float32/float64 |
| `test_seq_len_matches_tensor_lengths` | `seq_len` がテンソル長と一致 |

**要件**:
- `ball_uv`: (T, 2), float, [0, 1]
- `ball_mask`: (T,), 0/1
- `court_kp`: (20, 2), float, [0, 1]
- `seq_len` == `ball_uv.shape[0]`

### TestBLCSDataLoaderBatchValidation

`DataLoader` バッチの検証テスト。

| テスト | 検証内容 |
|--------|---------|
| `test_batch_schema_validation` | `BLCSBatch` スキーマへの準拠 |
| `test_batch_shapes_consistent` | バッチ内の形状整合性（B, T_max が統一） |
| `test_batch_padding_valid` | パディングが `seq_len` に基づき正しく適用 |

**要件**:
- すべての時間次元テンソルが同じ `T_max` を持つ
- `seq_len[i] <= T_max` for all samples

---

## test_generate_dataset.py

データセット生成スクリプト（`src.blcs.scripts.generate_dataset`）のテスト。

| テスト | 検証内容 |
|--------|---------|
| `test_blcs_generate_dataset` | スクリプトが正常終了し、シーンファイルと meta.json が生成される |
| `test_blcs_generate_dataset_minimal_samples` | 最小設定での高速生成テスト |

### 使用される設定オーバーライド

```bash
# 標準テスト
sampling.per_from_cell_samples=10
generator.num_cameras_sampled=2
generator.max_attempts_per_cell=50

# 最小テスト
sampling.per_from_cell_samples=5
generator.num_cameras_sampled=1
generator.ball_visibility_threshold=0.5
```

### 出力検証

- `{output_dir}/scenes/scene_*.npz` が1つ以上存在
- `{output_dir}/meta.json` が存在

---

## test_train.py

学習スクリプト（`src.blcs.scripts.train`）のテスト。

| テスト | GPU要否 | 検証内容 |
|--------|--------|---------|
| `test_blcs_train_basic` | ✓ CUDA | 基本学習が正常終了、チェックポイントが生成 |
| `test_blcs_train_with_custom_params` | ✓ CUDA | カスタムパラメータでの学習 |
| `test_blcs_train_dry_run` | CPU | ドライランモードでデータ確認のみ実行 |

### GPU テストの要件

```python
@pytest.mark.cuda  # CUDAがない環境ではスキップ
def test_blcs_train_basic(tmp_path: Path) -> None:
    ...
```

### 出力検証

- `{output_dir}/checkpoints/*.ckpt` が存在
- `{output_dir}/config.yaml` が存在

### ドライランモード

```bash
uv run python -m src.blcs.scripts.train run.dry_run=true
```

- GPU不要
- バッチ情報を表示して終了
- チェックポイントは生成しない

---

## test_visualize.py

可視化スクリプト（`src.blcs.scripts.visualize`）のテスト。

| テスト | 検証内容 |
|--------|---------|
| `test_blcs_visualize_ground_truth` | グラウンドトゥルース可視化（mode=visualize） |
| `test_blcs_visualize_predict` | モデル予測可視化（mode=predict） |
| `test_blcs_visualize_different_views` | 2Dビューの可視化 |
| `test_blcs_visualize_camera_view` | カメラビューの可視化 |

### 可視化モード

| モード | 説明 | 必要なもの |
|--------|------|-----------|
| `visualize` | グラウンドトゥルースの表示 | シーンファイル |
| `predict` | モデル予測の表示 | シーンファイル + チェックポイント |

### ビューオプション

| ビュー | 説明 |
|--------|------|
| `3d` | 3D空間での軌道表示 |
| `2d` | 2D平面（上から見た図） |
| `camera` | カメラ視点（camera=N で指定） |

### 出力検証

- 指定された出力パスに PNG ファイルが生成される

---

## フィクスチャの使用

すべてのテストは `tests/e2e/fixtures/blcs_fixtures.py` のヘルパーを使用：

```python
from tests.e2e.fixtures.blcs_fixtures import (
    create_minimal_blcs_dataset,    # 最小データセット生成
    create_minimal_blcs_checkpoint, # 最小チェックポイント生成
    make_minimal_blcs_scene,        # 単一シーン生成
)
```

### 生成されるテストデータ

- **シーン**: 30フレーム、1カメラ、パラボリック軌道
- **データセット**: 10シーン（train 70% / val 15% / test 15%）
- **チェックポイント**: デフォルト設定の BLCSModel

---

## テストの実行

```bash
# BLCSテストのみ実行
uv run pytest tests/e2e/blcs -v

# データ検証テストのみ
uv run pytest tests/e2e/blcs/test_data_validation.py -v

# GPUテストをスキップ
uv run pytest tests/e2e/blcs -v -m "not cuda"

# 特定のテストクラス
uv run pytest tests/e2e/blcs/test_data_validation.py::TestBLCSGeneratedSceneValidation -v
```
