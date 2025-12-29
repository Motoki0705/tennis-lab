# テストフィクスチャ

E2Eテストで使用するテストデータ生成ユーティリティのドキュメントです。

## 概要

`tests/e2e/fixtures/` には各タスク用のテストデータ生成ヘルパーがあります：

| ファイル | タスク | 主な機能 |
|---------|--------|---------|
| `blcs_fixtures.py` | BLCS | シーン、データセット、チェックポイント生成 |
| `plcs_fixtures.py` | PLCS | シーン、データセット、チェックポイント生成 |
| `wasb_fixtures.py` | WASB | データセット、チェックポイント、ビデオ生成 |

---

## BLCS フィクスチャ

### `make_minimal_blcs_scene(scene_id="scene_000000") -> BLCSSceneData`

最小限のBLCSシーンを生成します。

**生成されるシーン**:
- 30フレーム（30fpsで1秒）
- 単純な放物線軌道（サーブのシミュレーション）
- 1カメラ
- ランダムなUV座標（[0, 1]範囲）

**使用例**:
```python
from tests.e2e.fixtures.blcs_fixtures import make_minimal_blcs_scene

scene = make_minimal_blcs_scene(scene_id="test_scene")
```

### `create_minimal_blcs_dataset(output_dir, num_scenes=10) -> Path`

最小限のBLCSデータセットを生成します。

**生成される内容**:
- `scenes/scene_*.npz`: 指定数のシーンファイル
- `train.txt`, `val.txt`, `test.txt`: 分割ファイル（70/15/15%）
- データセットメタデータ

**使用例**:
```python
from tests.e2e.fixtures.blcs_fixtures import create_minimal_blcs_dataset

dataset_dir = create_minimal_blcs_dataset(tmp_path / "blcs_data", num_scenes=5)
```

### `create_minimal_blcs_checkpoint(checkpoint_path) -> Path`

最小限のBLCSモデルチェックポイントを生成します。

**モデル設定**:
- `hidden_dim=256`
- `num_layers=6`
- `num_heads=8`
- `max_seq_len=120`
- `use_cross_attention=True`

**使用例**:
```python
from tests.e2e.fixtures.blcs_fixtures import create_minimal_blcs_checkpoint

checkpoint_path = create_minimal_blcs_checkpoint(tmp_path / "model.ckpt")
```

---

## PLCS フィクスチャ

### `make_minimal_plcs_scene(scene_id="scene_000000") -> SceneData`

最小限のPLCSシーンを生成します。

**生成されるシーン**:
- 64フレーム（シーケンスモデル用に十分な長さ）
- 5ジョイント
- 1カメラ
- ランダムなキーポイントUV座標

**使用例**:
```python
from tests.e2e.fixtures.plcs_fixtures import make_minimal_plcs_scene

scene = make_minimal_plcs_scene(scene_id="test_scene")
```

### `create_minimal_plcs_dataset(output_dir, num_scenes=10) -> Path`

最小限のPLCSデータセットを生成します。

**生成される内容**:
- `scenes/scene_*.npz`: 指定数のシーンファイル
- `train.txt`, `val.txt`, `test.txt`: 分割ファイル（70/15/15%）

**使用例**:
```python
from tests.e2e.fixtures.plcs_fixtures import create_minimal_plcs_dataset

dataset_dir = create_minimal_plcs_dataset(tmp_path / "plcs_data", num_scenes=5)
```

### `create_minimal_plcs_checkpoint(checkpoint_path) -> Path`

フレームモデル（PLCSModel）のチェックポイントを生成します。

**モデル設定**:
- `hidden_dim=256`
- `num_layers=4`
- `num_heads=8`
- `use_transformer=True`

### `create_minimal_plcs_sequence_checkpoint(checkpoint_path) -> Path`

シーケンスモデル（PLCSSequenceModel）のチェックポイントを生成します。

**モデル設定**:
- `hidden_dim=256`
- `num_layers=4`
- `num_heads=8`
- `max_seq_len=120`

---

## WASB フィクスチャ

### `create_minimal_wasb_dataset(output_dir) -> Path`

最小限のWASBデータセットを生成します。

**生成される内容**:
- `game1/Clip1/`, `game1/Clip2/`: 2つのクリップ
- 各クリップに130フレームの合成画像（.jpg）
- 各クリップに `Label.csv`

**合成画像**:
- 1280×720 黒背景
- 白円（ボール）が放物線で移動
- Label.csv に対応する座標

**使用例**:
```python
from tests.e2e.fixtures.wasb_fixtures import create_minimal_wasb_dataset

dataset_dir = create_minimal_wasb_dataset(tmp_path / "tennis")
```

### `create_minimal_wasb_checkpoint(checkpoint_path) -> Path`

最小限のWASBチェックポイントを生成します。

> ⚠️ **注意**: これは簡易チェックポイントであり、実際のWASBモデルとは互換性がない場合があります。

### `create_minimal_trajectory_checkpoint(checkpoint_path) -> Path`

軌道補完モデル（TrajectoryBiLSTM）のチェックポイントを生成します。

**モデル設定**:
- `hidden_dim=64`
- `num_layers=2`
- `dropout=0.1`

**使用例**:
```python
from tests.e2e.fixtures.wasb_fixtures import create_minimal_trajectory_checkpoint

checkpoint_path = create_minimal_trajectory_checkpoint(tmp_path / "trajectory.ckpt")
```

### `create_minimal_video(video_path, num_frames=100, width=1280, height=720, fps=30) -> Path`

テスト用の合成ビデオを生成します。

**生成されるビデオ**:
- 黒背景に白円（ボール）
- ボールは放物線で移動
- MP4形式（mp4v コーデック）

**使用例**:
```python
from tests.e2e.fixtures.wasb_fixtures import create_minimal_video

video_path = create_minimal_video(tmp_path / "test.mp4", num_frames=50)
```

---

## pytest フィクスチャとしての使用

これらのヘルパーは pytest の `@pytest.fixture` と組み合わせて使用できます：

```python
import pytest
from pathlib import Path
from tests.e2e.fixtures.blcs_fixtures import create_minimal_blcs_dataset

@pytest.fixture
def blcs_dataset_dir(tmp_path: Path) -> Path:
    """最小BLCSデータセットを作成するフィクスチャ"""
    return create_minimal_blcs_dataset(tmp_path / "blcs_data", num_scenes=5)

def test_something(blcs_dataset_dir: Path) -> None:
    """データセットディレクトリを使用するテスト"""
    assert (blcs_dataset_dir / "scenes").exists()
```

---

## 新しいフィクスチャの追加

新しいフィクスチャを追加する際は：

1. 適切な `{task}_fixtures.py` ファイルに関数を追加
2. 最小限のデータで高速に動作するよう設計
3. docstring に生成される内容を記載
4. `tests/e2e/fixtures/__init__.py` にエクスポートを追加
5. 必要に応じて `conftest.py` に pytest フィクスチャを追加
