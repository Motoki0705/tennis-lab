# BLCS (Ball Localization in Court System)

BLCS は、テニスコート座標系におけるボールの 3D 軌道を、2D のボール観測（画像座標）と
コート情報から推定するためのタスク実装です。

## 目的 / 想定入出力

- **入力**: 2D ボール観測（画像座標シーケンス）+ コートキーポイント
- **出力**: コート座標系の 3D 位置（軌道）

### 入力形式

| モード | 入力テンソル | 形状 | 説明 |
|--------|-------------|------|------|
| 単一カメラ | `ball_uv` | `(B, T, 2)` | 2Dボール観測シーケンス |
| 単一カメラ | `court_kp` | `(B, 20, 2)` | 2Dコートキーポイント |
| マルチビュー | `ball_uv` | `(B, N, T, 2)` | 複数カメラからの2Dボール観測 |
| マルチビュー | `court_kp` | `(B, N, T, 20, 2)` | 複数カメラからの2Dコートキーポイント |

### 出力形式

**Predictor 返り値（`BLCSPredictor.predict()`）:**

推論結果は `dict[str, torch.Tensor]` 形式で返されます。全てのテンソルは CPU 上にあります。

| Predictor | キー | 形状 | 型 | 説明 |
|-----------|------|------|-----|------|
| `BLCSPredictor` | `position` | `(B, T, 3)` | `torch.Tensor` | 3D軌道。デフォルト（`denormalize=True`）ではメートル単位、`denormalize=False` の場合は正規化座標 |
| `BLCSPredictor` | `velocity` | `(B, T, 3)` | `torch.Tensor` | 速度ベクトル。デフォルト（`denormalize=True`）かつモデルが出力する場合は m/s 単位。モデルにより含まれない場合あり |

`BLCSPredictor` は `blcs` / `blcs_query` / `blcs_multiview` を単一クラスで扱います。  
`BLCSMultiViewPredictor` は後方互換のためのエイリアスとして残しています。

**注意**:
- すべてのテンソルは CPU に配置されます（統合側での device 変換は不要）
- バッチ次元は常に保持されます

## 実行コマンド

### データ生成

```bash
uv run python -m src.blcs.scripts.generate_dataset
```

### 学習

```bash
# 単一カメラモデル
uv run python -m src.blcs.scripts.train

# マルチビューモデル（複数カメラ統合）
uv run python -m src.blcs.scripts.train \
    model=multiview \
    data=multiview

# マルチビュー学習のカスタム設定例
uv run python -m src.blcs.scripts.train \
    model=multiview \
    data=multiview \
    data.num_views=4 \
    data.min_cameras=2 \
    training.max_epochs=100
```

### ハイパーパラメータ探索

```bash
UV_CACHE_DIR=agents_workspace/tmp_cache/uv_cache \
RUN_ROOT=outputs/blcs/sweep_$(date +%Y-%m-%d_%H-%M-%S) \
bash src/blcs/scripts/run_hparam_sweep.sh
```

## データローディング最適化

- `data.cache_max_scenes`: シーンNPZのLRUキャッシュ容量。
- `data.scene_sampler_mode`: `none | scene | mixed | chunked` を指定。
- `data.chunk_max_scenes`: `chunked` モード時のチャンク内シーン数上限。

### 可視化

統一された単一エントリポイント `python -m src.blcs.scripts.visualize` で全ての可視化タスクを実行できます。

```bash
# Ground Truthシーンの可視化（単一カメラ、デフォルト設定）
uv run python -m src.blcs.scripts.visualize

# 3Dアニメーションビューで可視化
uv run python -m src.blcs.scripts.visualize \
    visualization.animation_view=3d

# 特定のシーンを可視化
uv run python -m src.blcs.scripts.visualize \
    visualization.scene_path=data/blcs/scenes/scene_000003.npz

# シーン情報の表示
uv run python -m src.blcs.scripts.visualize \
    visualization.info=true

# 単一カメラモデルによる予測と可視化
uv run python -m src.blcs.scripts.visualize \
    visualization.mode=predict \
    visualization.checkpoint=outputs/blcs/single/logs/version_0/checkpoints/last.ckpt

# マルチビューモデルによる予測と可視化
uv run python -m src.blcs.scripts.visualize \
    visualization.mode=predict \
    visualization.cameras=all \
    visualization.checkpoint=outputs/blcs/multiview/logs/version_0/checkpoints/last.ckpt

# アニメーションを保存
uv run python -m src.blcs.scripts.visualize \
    visualization.animation_view=3d \
    visualization.save=outputs/blcs/animations/scene_3d.mp4
```

**注意:** 
- 全ての可視化はアニメーションベースです（`visualization.view` は廃止されました）
- `visualization.animation_view` は `2d_camera` または `3d` のみサポートします
- マルチビューの場合は `visualization.cameras=all` または特定カメラリスト（例: `"0,1,2"`）を指定します

## モデルアーキテクチャ

### 単一カメラモデル (`BLCSModel`)

単一カメラからの2D観測シーケンスを入力として3D軌道を推定。
Temporal Encoder + MLP ベースの構造。

実装: `src/blcs/models/blcs_model.py`

### マルチビューモデル (`BLCSMultiViewModel`)

複数カメラからの観測を統合して推定。複数視点からの2D観測を融合することで、
単一視点より高精度な3D軌道復元を狙う構成。

実装: `src/blcs/models/blcs_multiview_model.py`

### クエリモデル (`BLCSQueryModel`)

クエリベースでボール軌道を推定するモデル。

実装: `src/blcs/models/blcs_query_model.py`
