# plcs_fixtures.py

PLCSテスト用のデータ生成ユーティリティです。

**ファイル**: `tests/e2e/fixtures/plcs_fixtures.py`

## 関数一覧

| 関数 | 説明 |
|------|------|
| `make_minimal_plcs_scene` | 単一シーンの生成 |
| `create_minimal_plcs_dataset` | データセット全体の生成 |
| `create_minimal_plcs_checkpoint` | フレームモデルチェックポイントの生成 |
| `create_minimal_plcs_sequence_checkpoint` | シーケンスモデルチェックポイントの生成 |

---

## `make_minimal_plcs_scene`

最小限のPLCSシーンを生成します。

### シグネチャ

```python
def make_minimal_plcs_scene(*, scene_id: str = "scene_000000") -> SceneData
```

### 生成されるシーン

| プロパティ | 値 |
|-----------|-----|
| フレーム数 | 64（シーケンスモデルに十分） |
| ジョイント数 | 5 |
| カメラ数 | 1 |
| human_kp_uv | ランダム [0, 1]、形状 (64, 17, 2) |
| court_kp_uv | ランダム [0, 1]、形状 (64, 20, 2) |
| rotation | [0.0, 1.0]（yaw=0） |

### 使用例

```python
from tests.e2e.fixtures.plcs_fixtures import make_minimal_plcs_scene

scene = make_minimal_plcs_scene(scene_id="test_scene")

print(scene.position.shape)  # (64, 3)
print(scene.rotation.shape)  # (64, 2)
```

---

## `create_minimal_plcs_dataset`

最小限のPLCSデータセットを生成します。

### シグネチャ

```python
def create_minimal_plcs_dataset(
    output_dir: Path | str,
    num_scenes: int = 10,
) -> Path
```

### 生成される内容

```
{output_dir}/
├── scenes/
│   ├── scene_000000.npz
│   └── ...
├── train.txt      # 70%
├── val.txt        # 15%
└── test.txt       # 15%
```

### 使用例

```python
from tests.e2e.fixtures.plcs_fixtures import create_minimal_plcs_dataset

dataset_dir = create_minimal_plcs_dataset(tmp_path / "plcs_data", num_scenes=5)
```

---

## `create_minimal_plcs_checkpoint`

フレームモデル（PLCSModel）のチェックポイントを生成します。

### シグネチャ

```python
def create_minimal_plcs_checkpoint(checkpoint_path: Path | str) -> Path
```

### モデル設定

| パラメータ | 値 |
|-----------|-----|
| `hidden_dim` | 256 |
| `num_layers` | 4 |
| `num_heads` | 8 |
| `dropout` | 0.1 |
| `use_transformer` | True |
| `use_combined_head` | False |

### 使用例

```python
from tests.e2e.fixtures.plcs_fixtures import create_minimal_plcs_checkpoint

checkpoint_path = create_minimal_plcs_checkpoint(tmp_path / "model.ckpt")
```

---

## `create_minimal_plcs_sequence_checkpoint`

シーケンスモデル（PLCSSequenceModel）のチェックポイントを生成します。

### シグネチャ

```python
def create_minimal_plcs_sequence_checkpoint(checkpoint_path: Path | str) -> Path
```

### モデル設定

| パラメータ | 値 |
|-----------|-----|
| `hidden_dim` | 256 |
| `num_layers` | 4 |
| `num_heads` | 8 |
| `dropout` | 0.1 |
| `max_seq_len` | 120 |

### 使用例

```python
from tests.e2e.fixtures.plcs_fixtures import create_minimal_plcs_sequence_checkpoint

checkpoint_path = create_minimal_plcs_sequence_checkpoint(tmp_path / "seq_model.ckpt")
```
