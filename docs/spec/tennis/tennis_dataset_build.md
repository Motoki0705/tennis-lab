# テニス Pose データセット自動生成仕様（Build Spec）

本書は、テニス用シミュレータシーンから学習用データセットを自動生成する CLI `build_tennis_dataset.py` と、そのコンフィグ駆動実行についてまとめる。

---

## 1. スコープ

- シミュレータ: `docs/spec/tennis/tennis_simulator.md` で定義されるテニスシーン JSON
- データセットビルダー CLI: `src/cli/build_tennis_dataset.py`
- 出力ディレクトリ構造・メタ情報・インデックス形式
- コンフィグファイル: `configs/tennis/build_tennis_dataset_sim.yaml`

詳細なデータセット仕様（`TennisSceneWindowDataset` 向けの前提条件など）は `docs/spec/datasets/tennis_pose_sim.md` を参照。

---

## 2. CLI: `build_tennis_dataset.py`

### 2.1 基本説明

- 役割: テニスシミュレータ出力（シーン JSON）から、train/val/test 分割されたシーン群とインデックス、およびメタ情報をまとめて生成する。
- 実装: `src/cli/build_tennis_dataset.py`

### 2.2 引数一覧

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
| `--config` | `str`, `None` | YAML コンフィグへのパス。指定時、その内容をデフォルト値として読み込み、CLI 引数で上書き可能 |

追加制約:
- `--max_players >= --min_players`
- `window_T > 0`, `window_stride > 0`

### 2.3 コンフィグ駆動実行

`build_tennis_dataset.py` は `--config` オプションをサポートしており、YAML ファイルに主要パラメータをまとめておくことで、毎回長いコマンドラインを叩かずに済む。

- 動作イメージ:
  - `--config` で読み込まれた YAML のキーが、同名の CLI 引数（`dataset_root`, `num_scenes_train`, ...）の **デフォルト値** として適用される。
  - 同じキーを CLI でも指定した場合は、**CLI 側が優先**される。

#### 2.3.1 代表的なコンフィグファイル

パス: `configs/tennis/build_tennis_dataset_sim.yaml`

```yaml
# 出力位置
dataset_root: data/tennis_autogen
# TennisPose 用データセット名（学習側の configs/datasets/tennis_pose_sim.yaml と揃える）
dataset_name: sim_fps60_dur3p0_C4_P1-20_T10

# シーン数
num_scenes_train: 500
num_scenes_val: 100
num_scenes_test: 100

# シミュレーション条件
fps: 60
duration: 3.0
num_cameras: 4
asset_root: data/raw/3dtennisds
min_players: 1
max_players: 20

# インデックス条件
window_T: 10
window_stride: 5

# 乱数シードと上書き挙動
seed: 1234
overwrite: false
```

#### 2.3.2 実行コマンド例

```bash
python src/cli/build_tennis_dataset.py \
  --config configs/tennis/build_tennis_dataset_sim.yaml
```

- デフォルトでは `overwrite: false` のため、同じ `dataset_root/dataset_name` に既にデータが存在し、かつ空でない場合はエラー終了する。
- 既存データセットを作り直したい場合は、CLI 側で `--overwrite` を明示的に指定する:

```bash
python src/cli/build_tennis_dataset.py \
  --config configs/tennis/build_tennis_dataset_sim.yaml \
  --overwrite
```

---

## 3. 出力ディレクトリ構造

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
- `meta.json` は生成条件とメタデータを保持する:
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

ウィンドウ生成ロジックの詳細や学習時の前提条件は `docs/spec/datasets/tennis_pose_sim.md` を参照。

---

## 4. ワークフローまとめ

1. シミュレーションシーンの生成（必要に応じて）:

   ```bash
   python src/cli/gen_tennis_pose_scenes.py --out data/sim_scenes --num_scenes 100
   ```

2. データセットの自動生成（コンフィグベース）:

   ```bash
   python src/cli/build_tennis_dataset.py \
     --config configs/tennis/build_tennis_dataset_sim.yaml
   ```

3. 学習側設定 (`configs/datasets/tennis_pose_sim.yaml`) の `root` / `name` を上記と揃えることで、`TennisPoseDataModule` からこのデータセットを読み込める。
