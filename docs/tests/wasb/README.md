# WASB E2E テスト

WASB (Where's the Ball) のE2Eテストドキュメントです。

## テストファイル一覧

| ファイル | 説明 |
|---------|------|
| `test_data_validation.py` | データセット構造とサンプルのスキーマ検証 |
| `test_generate_dataset.py` | データセット生成スクリプトのテスト（多くはスキップ） |
| `test_tools.py` | ツールスクリプト（バックボーン抽出、エンコーディング）のテスト |
| `test_train.py` | 学習スクリプトのテスト |
| `test_visualize.py` | 可視化スクリプトのテスト |

---

## test_data_validation.py

WASBデータセットの構造とサンプルの検証テストです。

### TestWASBDatasetValidation

データセットディレクトリ構造の検証。

| テスト | 検証内容 |
|--------|---------|
| `test_dataset_structure` | ゲーム/クリップディレクトリ構造が正しい |
| `test_label_csv_format` | Label.csvのフォーマットが正しい |

**要件**:
- ディレクトリ構造: `{root}/{game}/{clip}/`
- 各クリップに `Label.csv` が存在
- 各クリップに画像ファイル（.jpg/.png）が存在
- Label.csv ヘッダー: `file name`, `visibility`, `x-coordinate`, `y-coordinate`

### TestWASBBallDetectionSampleValidation

`BallDetectionSequenceDataset` サンプルの検証。

| テスト | 検証内容 |
|--------|---------|
| `test_sample_has_required_keys` | `BallDetectionSample` の必須キーが存在 |
| `test_sample_tensor_shapes` | テンソル形状が正しい |
| `test_sample_normalized_targets_in_range` | 正規化ターゲットが [0, 1] 範囲 |
| `test_sample_heatmap_shape_matches_frame` | ヒートマップサイズがフレームサイズと整合 |
| `test_sample_visibility_values` | 可視性値が 0, 1, 2 のいずれか |
| `test_sample_tensor_dtypes` | テンソルが適切なdtype |
| `test_sample_metadata_types` | メタデータ（match, clip）が文字列 |

**要件** (`BallDetectionSample` TypedDict):

| キー | 形状 | 説明 |
|-----|------|------|
| `frames` | (T, C, H, W) | 入力フレーム（T=frames_in, C=3） |
| `targets_px` | (frames_out, 2) | ピクセル座標でのターゲット位置 |
| `targets_norm` | (frames_out, 2) | 正規化座標でのターゲット位置 [0, 1] |
| `target_heatmaps` | (frames_out, H', W') | ターゲットヒートマップ |
| `visibility` | (frames_out,) | 可視性フラグ (0=不可視, 1=可視, 2=オクルージョン) |
| `scores` | (frames_out,) | 信頼度スコア |
| `match` | str | マッチID |
| `clip` | str | クリップID |

### TestWASBDataLoaderBatchValidation

バッチデータの検証。

| テスト | 検証内容 |
|--------|---------|
| `test_batch_shapes_consistent` | バッチ次元が一貫している |
| `test_batch_frames_shape` | バッチ化されたフレームが5D (B, T, C, H, W) |

---

## test_generate_dataset.py

データセット生成スクリプトのテスト。

> ⚠️ **注意**: WASBのデータセット生成は複雑な依存関係があるため、多くのテストがスキップされています。

| テスト | 状態 | 理由 |
|--------|------|------|
| `test_wasb_download_videos_status` | スキップ | meta.json のセットアップが必要 |
| `test_wasb_generate_dataset_batch_mode` | スキップ | ビデオ処理が複雑 |
| `test_wasb_clip_sampling_generate_samples` | スキップ | 既存データセット構造が必要 |

---

## test_tools.py

WASBツールスクリプトのテスト。

| テスト | 状態 | 検証内容 |
|--------|------|---------|
| `test_extract_dinov3_backbone` | スキップ | DinoV3バックボーンの抽出（実チェックポイント必要） |
| `test_encode_dinov3_patch_tokens` | スキップ | パッチトークンエンコーディング（実チェックポイント必要） |
| `test_encode_dinov3_patch_tokens_without_checkpoint` | 実行可 | デフォルトモデルでのエンコーディング |

### スキップ理由

実際のWASBチェックポイントには特定のモデルアーキテクチャ（DinoV3バックボーン等）が含まれている必要があり、モックチェックポイントでは代替できません。

---

## test_train.py

学習スクリプトのテスト。

| テスト | GPU要否 | 検証内容 |
|--------|--------|---------|
| `test_wasb_ball_detection_train_dry_run` | CPU | ボール検出モデルのドライラン |
| `test_wasb_ball_detection_train_fast_dev` | ✓ CUDA | ボール検出モデルの高速開発実行 |
| `test_wasb_trajectory_train_dry_run` | CPU | 軌道補完モデルのドライラン |
| `test_wasb_event_detection_train_dry_run` | CPU | イベント検出モデルのドライラン |

### モデルの種類

| スクリプト | モデル | 説明 |
|-----------|--------|------|
| `train.ball_detection` | DinoV3 FPN Heatmap | ヒートマップベースのボール検出 |
| `train.trajectory` | TrajectoryBiLSTM | 軌道補完（欠損フレームの補間） |
| `train.event_detection` | イベント検出器 | バウンス/ヒット等のイベント検出 |

### ドライランモード設定例

```bash
uv run python -m src.wasb.scripts.train.ball_detection \
    run.dry_run=true \
    run.gpus=0 \
    data.root_dir=./data/tennis \
    data.train_matches=[game1] \
    data.val_matches=[game1]
```

### 学習設定の縮小（テスト用）

```python
# モデルサイズを縮小してテスト高速化
model.high_channels=16
model.low_channels=32
model.num_stages=2
model.num_high_blocks=1
model.num_low_blocks=1
model.transformer_kwargs.d_model=32
model.transformer_kwargs.num_heads=4
model.transformer_kwargs.dim_ff=64
model.transformer_kwargs.depth=1
data.batch_size=1
```

---

## test_visualize.py

可視化スクリプトのテスト。

| テスト | 状態 | 検証内容 |
|--------|------|---------|
| `test_wasb_ball_video` | スキップ | ビデオでのボール検出可視化（実チェックポイント必要） |
| `test_wasb_ball_video_ensemble` | スキップ | アンサンブル予測可視化（実チェックポイント必要） |
| `test_wasb_trajectory_visualize` | 実行可 | 軌道補完の可視化 |
| `test_wasb_save_one_sample_visuals` | 実行可 | 単一サンプルの可視化保存 |

### 可視化スクリプト

| スクリプト | 説明 |
|-----------|------|
| `visualize.ball_video` | ビデオ上にボール検出結果を描画 |
| `visualize.ball_video_ensemble` | 複数モデルのアンサンブル結果を描画 |
| `visualize.trajectory` | 軌道補完結果の可視化 |
| `visualize.save_one_sample_visuals` | データセットサンプルの可視化 |

---

## フィクスチャの使用

すべてのテストは `tests/e2e/fixtures/wasb_fixtures.py` のヘルパーを使用：

```python
from tests.e2e.fixtures.wasb_fixtures import (
    create_minimal_wasb_dataset,        # 最小データセット生成
    create_minimal_wasb_checkpoint,     # ボール検出チェックポイント
    create_minimal_trajectory_checkpoint, # 軌道補完チェックポイント
    create_minimal_video,               # テスト用合成ビデオ生成
)
```

### 生成されるテストデータ

- **データセット構造**: 
  - `game1/Clip1/`, `game1/Clip2/`
  - 各クリップに130フレーム（軌道データセットに十分な長さ）
  - 合成ボール画像（放物線運動）
  - 対応する `Label.csv`

- **合成ビデオ**: 
  - 1280×720、30fps
  - 黒背景に白円（ボール）が放物線で移動

- **チェックポイント**:
  - 軌道補完: TrajectoryBiLSTM (hidden_dim=64, num_layers=2)

---

## テストの実行

```bash
# WASBテストのみ実行
uv run pytest tests/e2e/wasb -v

# データ検証テストのみ
uv run pytest tests/e2e/wasb/test_data_validation.py -v

# スキップされないテストのみ
uv run pytest tests/e2e/wasb -v --ignore-glob="*skip*"

# 学習テスト（ドライランのみ、GPUなし）
uv run pytest tests/e2e/wasb/test_train.py -v -m "not cuda"
```
