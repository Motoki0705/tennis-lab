# テニス Pose シミュレーションデータセット仕様（Spec）

本書は、テニス用シミュレータ出力（`scene_*.json`）、自動生成 CLI（`build_tennis_dataset.py`）、および学習用 Dataset（`TennisSceneWindowDataset`）の仕様をまとめる。

---

## 1. スコープ

- シミュレータ: `src/cli/gen_tennis_pose_scenes.py`, `src/tennis/sim/*`
- データセットビルダー CLI: `src/cli/build_tennis_dataset.py`
- 学習 Dataset: `src/datasets/tennis/scene_dataset.py:TennisSceneWindowDataset`

目的は、**テニス用シミュレータから学習用テンソル `[T,V,M,J,2]` / `[T,M,J,3]` を得るまでの流れ**をコードを読まずに把握できるようにすること。

---

## 2. データセット自動生成 CLI: `build_tennis_dataset.py`

### 2.1 入力引数

実装: `src/cli/build_tennis_dataset.py:1`

| 引数 | 型 / 既定値 | 説明 |
| --- | --- | --- |
| `--dataset_root` | `str`, `"data/tennis_autogen"` | 生成するデータセットのルートディレクトリ |
| `--dataset_name` | `str`, `None` | サブディレクトリ名。未指定時はパラメータから自動生成 |
| `--num_scenes_train` | `int`, `100` | train split のシーン数 |
| `--num_scenes_val` | `int`, `20` | val split のシーン数 |
| `--num_scenes_test` | `int`, `20` | test split のシーン数 |
| `--fps` | `int`, `60` | シミュレータの FPS |
| `--duration` | `float`, `3.0` | 1 シーンの長さ（秒） |
| `--num_cameras` | `int`, `4` | カメラ数 |
| `--asset_root` | `str`, `"data/raw/3dtennisds"` | 3DTennisDS アセットルート |
| `--min_players` | `int`, `1` | シーン毎の最小プレーヤー数 |
| `--max_players` | `int`, `20` | シーン毎の最大プレーヤー数 |
| `--window_T` | `int`, `10` | インデックス上の時間ウィンドウ長（フレーム） |
| `--window_stride` | `int`, `5` | ウィンドウ間ストライド（フレーム） |
| `--seed` | `int`, `1234` | ベース乱数シード（train/val/test に +0/+1/+2） |
| `--overwrite` | flag | 既存ディレクトリがあっても上書きする |

追加制約:
- `--max_players >= --min_players`
- `window_T > 0`, `window_stride > 0`

### 2.2 ディレクトリとファイル出力

`<dataset_root>/<dataset_name>/` 配下に以下を生成する。

```text
<dataset_root>/<dataset_name>/
  meta.json
  scenes/
    train/scene_000000.json
    val/scene_000000.json
    test/scene_000000.json
  index/
    train_index.jsonl
    val_index.jsonl
    test_index.jsonl
```

- `scenes/*/*.json` は `tennis_simulator.md` で定義されたシーン JSON。
- `meta.json` は生成条件とメタデータ:
  - `fps`, `duration_sec`, `num_cameras`, `asset_root`
  - `min_players`, `max_players`
  - `window_T`, `window_stride`
  - `seed` と split ごとの seed（`splits.train/val/test`）
  - `created_at`（UTC ISO 8601）、`git_commit`（取得できた場合）
- `index/<split>_index.jsonl` は 1 行 1 ウィンドウの JSON:

| フィールド | 型 | 説明 |
| --- | --- | --- |
| `scene_path` | str | データセットルートからの相対パス（例: `scenes/train/scene_000000.json`） |
| `scene_id` | str | シーン ID（JSON 内 `scene_id`） |
| `t_start` | int | ウィンドウ開始フレーム index（0-based, inclusive） |
| `t_end` | int | ウィンドウ終了フレーム index（exclusive） |
| `num_frames` | int | `t_end - t_start`（`<= window_T`） |
| `num_cameras` | int | シーン内のカメラ数 |
| `max_players_in_window` | int | 当該ウィンドウ内で観測された最大プレーヤー数 |

ウィンドウ生成ロジック:
- シーンの総フレーム数を `T_total` とし、`t = 0, window_stride, ...` とずらしながら `[t, min(t+window_T, T_total))` を列挙。
- 最後が `T_total` に満たない場合も 1 ウィンドウとして出力（`num_frames < window_T`）。

---

## 3. 学習 Dataset: `TennisSceneWindowDataset`

実装: `src/datasets/tennis/scene_dataset.py:TennisSceneWindowDataset`

### 3.1 コンストラクタ引数

```python
TennisSceneWindowDataset(
    dataset_root: str | Path,
    dataset_name: str,
    split: str,
    window_T: int,
    max_cameras: int,
    max_players: int,
    num_joints: int = 20,
    use_memmap: bool = False,
    min_cameras: int | None = None,
    augment_2d: bool = False,
)
```

- `dataset_root` / `dataset_name`: `build_tennis_dataset.py` で生成されたディレクトリを指す。
- `split`: `"train"`, `"val"`, `"test"` のいずれか。
- `window_T`: DataLoader が期待する固定時間長。`index.num_frames <= window_T` が前提。
- `max_cameras`: バッチ整形時のカメラ次元上限。memmap では npz に保存されたカメラのうち高々 `max_cameras` 本が使用される。
- `max_players`: 1 フレームあたりのプレーヤー数上限（通常 20）。
- `num_joints`: プレーヤー 1 人あたりのキーポイント数（20; pose 17 + racket 3）。
- `use_memmap`: `true` の場合は `arrays/<split>/scene_*.npz` から読み込み、`false` の場合は JSON から直接テンソルを構築する。
- `min_cameras`: 1 サンプルあたりの使用カメラ数の下限。省略時は `max_cameras` と同じになり、常に `max_cameras` 本使用する。
- `augment_2d`: `true` の場合、train split において 2D 座標（`keypoints_2d`, `court_2d`）にランダムアフィン変換によるデータ拡張を行う。

前提条件:
- `<dataset_root>/<dataset_name>/index/<split>_index.jsonl` が存在し、正常にパースできること。

### 3.2 `__len__` と `__getitem__`

- `__len__()` は index 行数（= ウィンドウ数）を返す。
- `__getitem__(i)` は i 行目の `_WindowRecord` を元に、対応するシーン JSON からウィンドウ `[t_start:t_end)` を切り出し、以下の dict を返す。

#### 3.2.1 出力テンソルと形状

- `keypoints_2d: Float[T, V, M, J, 2]`
  - 2D キーポイント（pose17 + racket3 = 20 点）を画像座標 `[x, y]` から `[-1, 1]` に正規化。
  - `T = window_T`。`index.num_frames < window_T` の場合、末尾フレームはゼロ埋めパディング。
  - `V = max_cameras`, `M = max_players`, `J = num_joints`。
  - 実際のカメラ数 / プレーヤー数が少ない場合、余剰分はゼロ。
  - memmap/JSON いずれのパスでも、元シーンのカメラから `K` 本をランダムサンプリングして先頭 `K` スロットに詰める設計とする（`min_cameras <= K <= max_cameras`）。
- `player_mask: Bool[T, V, M]`
  - そのフレーム・カメラにおいて m 番プレーヤーが観測されていれば True。
  - 2D/3D GT どちらの有効性判定にも利用可能。
- `court_2d: Float[V, 20, 2]`
  - `frames[0].cam_v.court_keypoints_2d.points[:20]` を `[-1, 1]` 正規化したもの。
  - コートがフレーム間で変化しない前提で、ウィンドウ全体に共通の値。
- `pose_3d_gt: Float[T, M, J, 3]`
  - 3D GT（pose 17 + racket 3）を結合した 20 点のコート座標系 `(x,y,z)`。
  - 各フレームの `player_joints_3d`（17×3）と `racket_points_3d`（3×3）から構成。
  - プレーヤー m が存在しない場合はゼロ。
- `exist_3d_gt: Bool[T, M]`
  - フレーム t にプレーヤー m の 3D GT が存在する場合 True。
- メタ情報:
  - `scene_id: Long[1]`（Python の `hash(scene_id)` を格納）
  - `t_start: Long[1]`, `t_end: Long[1]`

#### 3.2.2 座標系と正規化

- 2D:
  - JSON 内 `image_size = [w, h]` に対して、
    - `u_norm = (u / w) * 2 - 1`
    - `v_norm = (v / h) * 2 - 1`
  - court/player/racket すべて同じ変換。
- 3D:
  - JSON の値をそのまま使用（メートル単位のコート座標系）。追加の正規化は行わない。

#### 3.2.3 カメラサンプリングと 2D データ拡張

- カメラサンプリング:
  - 各サンプルごとに、元シーンのカメラ数 `V_src` から `K` 本をランダムにサンプリングする。
    - `K` は `min_cameras`〜`min(max_cameras, V_src)` の一様乱数。
  - 選ばれた `K` 本は出力テンソルのカメラ次元先頭 `K` スロットに詰められ、残り `max_cameras - K` スロットはゼロパディング＋`player_mask=False` となる。
  - memmap パスと JSON パスで同一の挙動となる。
- 2D データ拡張（`augment_2d=True` かつ `split=="train"` の場合）:
  - ビューごとにランダムなアフィン変換（小さな回転・スケーリング・平行移動）を `[-1, 1]` 正規化座標上でサンプリングし、`keypoints_2d` と `court_2d` に適用する。
  - 3D GT（`pose_3d_gt`）およびカメラパラメータ（`camera_C`, `camera_R`, `camera_intr`）は変更しない。あくまで「画像上の見え方」を揺らすための拡張である。

---

## 4. DataModule: `TennisPoseDataModule`

実装: `src/training/tennis/datamodule.py:TennisPoseDataModule`

### 4.1 コンストラクタと設定

```python
TennisPoseDataModule(
    dataset_cfg: DictConfig | Mapping[str, Any] | None,
    debug_cfg: DictConfig | Mapping[str, Any] | None,
)
```

- `dataset_cfg` は通常 YAML から読み込まれた dict で、代表例は `configs/datasets/tennis_pose_sim.yaml`。
  - 主要キー:
    - `root`, `name`（`build_tennis_dataset.py` の出力に対応）
    - `window_T`, `max_cameras`, `max_players`, `num_joints`
    - `loader.train/val/test.{batch_size,num_workers,shuffle,pin_memory,...}`
- `debug_cfg`:
  - `seed`: DataLoader のシャッフルに使う乱数シード。

### 4.2 `setup` と DataLoader

- `setup(stage)`:
  - `stage in (None, "fit")` で `train_dataset` と `val_dataset` を構築。
  - `stage == "test"` で `test_dataset` を構築。
- `train_dataloader/val_dataloader/test_dataloader`:
  - 引数:
    - `batch_size`, `num_workers`, `pin_memory`, `drop_last`, `persistent_workers` は `dataset_cfg.loader.*` を参照。
  - 返却:
    - 各 step で `dict[str, Tensor]`（`TennisSceneWindowDataset.__getitem__` の返り値）をバッチ化したもの。
  - collate は標準の PyTorch collate を使用（キーごとにスタック）。

---

## 5. 想定ワークフロー

1. シミュレーションシーンの生成（必要に応じて）:

```bash
python src/cli/gen_tennis_pose_scenes.py --out data/sim_scenes --num_scenes 100
```

2. 学習用データセットの自動構築:

```bash
python src/cli/build_tennis_dataset.py \
  --dataset_root data/tennis_autogen \
  --num_scenes_train 500 --num_scenes_val 100 --num_scenes_test 100 \
  --fps 60 --duration 3.0 --num_cameras 4 \
  --min_players 1 --max_players 20 \
  --window_T 10 --window_stride 5 \
  --seed 1234
```

3. `configs/datasets/tennis_pose_sim.yaml` の `name` を `--dataset_name` もしくは自動生成されたディレクトリ名に合わせることで、`TennisPoseDataModule` がこのデータセットを読み込む。
