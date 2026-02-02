# blcs_fixtures.py

BLCSテスト用のデータ生成ユーティリティです。

**ファイル**: `tests/e2e/fixtures/blcs_fixtures.py`

## 関数一覧

| 関数 | 説明 |
|------|------|
| `make_minimal_blcs_scene` | 単一シーンの生成 |
| `create_minimal_blcs_dataset` | データセット全体の生成 |
| `create_minimal_blcs_checkpoint` | モデルチェックポイントの生成 |

---

## `make_minimal_blcs_scene`

最小限のBLCSシーンを生成します。

### シグネチャ

```python
def make_minimal_blcs_scene(*, scene_id: str = "scene_000000") -> BLCSSceneData
```

### 生成されるシーン

| プロパティ | 値 |
|-----------|-----|
| フレーム数 | 30（30fpsで1秒） |
| 軌道 | 放物線（サーブのシミュレーション） |
| カメラ数 | 1 |
| ball_uv | ランダム [0, 1] |
| court_kp_uv | ランダム [0, 1] |
| 可視性 | すべて可視 |

### 使用例

```python
from tests.e2e.fixtures.blcs_fixtures import make_minimal_blcs_scene

scene = make_minimal_blcs_scene(scene_id="test_scene")

# アクセス例
print(scene.ball_pos_world.shape)  # (30, 3)
print(scene.cameras[0].ball_uv.shape)  # (30, 2)
```

---

## `create_minimal_blcs_dataset`

最小限のBLCSデータセットを生成します。

### シグネチャ

```python
def create_minimal_blcs_dataset(
    output_dir: Path | str,
    num_scenes: int = 10,
) -> Path
```

### 生成される内容

```
{output_dir}/
├── scenes/
│   ├── scene_000000.npz
│   ├── scene_000001.npz
│   └── ...
├── train.txt      # 70%
├── val.txt        # 15%
├── test.txt       # 15%
└── (metadata files)
```

### 使用例

```python
from tests.e2e.fixtures.blcs_fixtures import create_minimal_blcs_dataset

dataset_dir = create_minimal_blcs_dataset(tmp_path / "blcs_data", num_scenes=5)

# 分割ファイルの確認
train_ids = (dataset_dir / "train.txt").read_text().strip().split("\n")
```

---

## `create_minimal_blcs_checkpoint`

BLCSモデルのチェックポイントを生成します。

### シグネチャ

```python
def create_minimal_blcs_checkpoint(checkpoint_path: Path | str) -> Path
```

### モデル設定

| パラメータ | 値 |
|-----------|-----|
| `hidden_dim` | 256 |
| `num_layers` | 6 |
| `num_heads` | 8 |
| `dropout` | 0.1 |
| `max_seq_len` | 120 |
| `use_cross_attention` | True |
| `predict_velocity` | False |

### チェックポイント形式

```python
{
    "state_dict": {...},
    "hyper_parameters": {...},
    "epoch": 0,
    "global_step": 0,
    "pytorch-lightning_version": "...",
}
```

### 使用例

```python
from tests.e2e.fixtures.blcs_fixtures import create_minimal_blcs_checkpoint

checkpoint_path = create_minimal_blcs_checkpoint(tmp_path / "model.ckpt")

# 予測テストに使用
from src.blcs.api import BLCSPredictor
predictor = BLCSPredictor.load_from_checkpoint(checkpoint_path)
```
