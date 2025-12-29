# wasb_fixtures.py

WASBテスト用のデータ生成ユーティリティです。

**ファイル**: `tests/e2e/fixtures/wasb_fixtures.py`

## 関数一覧

| 関数 | 説明 |
|------|------|
| `create_minimal_wasb_dataset` | データセット全体の生成 |
| `create_minimal_wasb_checkpoint` | ボール検出チェックポイントの生成 |
| `create_minimal_trajectory_checkpoint` | 軌道補完チェックポイントの生成 |
| `create_minimal_video` | テスト用合成ビデオの生成 |

---

## `create_minimal_wasb_dataset`

最小限のWASBデータセットを生成します。

### シグネチャ

```python
def create_minimal_wasb_dataset(output_dir: Path | str) -> Path
```

### 生成される内容

```
{output_dir}/
├── game1/
│   ├── Clip1/
│   │   ├── 0.jpg
│   │   ├── 1.jpg
│   │   ├── ...
│   │   └── Label.csv
│   └── Clip2/
│       ├── 0.jpg
│       ├── ...
│       └── Label.csv
```

### 合成画像

- サイズ: 1280×720
- 背景: 黒
- ボール: 白円、放物線で移動
- フレーム数: 130（軌道データセットに十分）

### Label.csv 形式

```csv
file name,visibility,x-coordinate,y-coordinate,status,score
0.jpg,1,640,360,0,0.0
1.jpg,1,644,358,0,0.0
...
```

### 使用例

```python
from tests.e2e.fixtures.wasb_fixtures import create_minimal_wasb_dataset

dataset_dir = create_minimal_wasb_dataset(tmp_path / "tennis")

# データセット読み込み
from src.wasb.data.ball_detection_dataset import BallDetectionSequenceDataset
dataset = BallDetectionSequenceDataset(
    root_dir=dataset_dir,
    matches=["game1"],
    ...
)
```

---

## `create_minimal_wasb_checkpoint`

ボール検出モデルのチェックポイントを生成します。

### シグネチャ

```python
def create_minimal_wasb_checkpoint(checkpoint_path: Path | str) -> Path
```

### 注意

> ⚠️ これは簡易チェックポイントであり、実際のWASBモデル（DinoV3バックボーン等）とは互換性がない場合があります。スクリプトの起動テストには使用できますが、推論結果は無効です。

### 使用例

```python
from tests.e2e.fixtures.wasb_fixtures import create_minimal_wasb_checkpoint

checkpoint_path = create_minimal_wasb_checkpoint(tmp_path / "model.ckpt")
```

---

## `create_minimal_trajectory_checkpoint`

軌道補完モデル（TrajectoryBiLSTM）のチェックポイントを生成します。

### シグネチャ

```python
def create_minimal_trajectory_checkpoint(checkpoint_path: Path | str) -> Path
```

### モデル設定

| パラメータ | 値 |
|-----------|-----|
| `hidden_dim` | 64 |
| `num_layers` | 2 |
| `dropout` | 0.1 |

### 使用例

```python
from tests.e2e.fixtures.wasb_fixtures import create_minimal_trajectory_checkpoint

checkpoint_path = create_minimal_trajectory_checkpoint(tmp_path / "trajectory.ckpt")
```

---

## `create_minimal_video`

テスト用の合成ビデオを生成します。

### シグネチャ

```python
def create_minimal_video(
    video_path: Path | str,
    num_frames: int = 100,
    width: int = 1280,
    height: int = 720,
    fps: int = 30,
) -> Path
```

### 生成されるビデオ

- フォーマット: MP4（mp4vコーデック）
- 内容: 黒背景に白円（ボール）が放物線で移動

### 使用例

```python
from tests.e2e.fixtures.wasb_fixtures import create_minimal_video

video_path = create_minimal_video(
    tmp_path / "test.mp4",
    num_frames=50,
    fps=30,
)
```
