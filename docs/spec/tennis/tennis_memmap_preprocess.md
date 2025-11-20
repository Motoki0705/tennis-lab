# テニス Pose memmap 前処理仕様（Preprocess Spec）

本書は、テニス用データセットの JSON シーンを **npz/memmap 形式の中間表現** に変換する CLI
`src/cli/preprocess_tennis_memmap.py` の仕様をまとめる。

目的は、`build_tennis_dataset.py` が生成した JSON ベースのデータセットを、
学習時に高速に読み込める形（npz + memmap）へ事前変換する挙動を、コードを読まずに把握できるようにすること。

---

## 1. スコープ

- 入力:
  - `build_tennis_dataset.py` によって生成されたデータセットディレクトリ
  - 具体的には `<dataset_root>/<dataset_name>/scenes/<split>/scene_*.json`
- 出力:
  - `<dataset_root>/<dataset_name>/arrays/<split>/scene_*.npz`
  - 各 npz には、2D/3D キーポイント・プレーヤーマスク・カメラパラメータなど、
    学習に必要なテンソルをすべて含む
- 利用側:
  - `src/datasets/tennis/scene_dataset.py:TennisSceneWindowDataset`
    - `use_memmap=true` のとき、これらの npz を `np.load(..., mmap_mode="r")` で参照する

シミュレータ仕様は `docs/spec/tennis/tennis_simulator.md`、
データセットビルド仕様は `docs/spec/tennis/tennis_dataset_build.md`、
学習用データセット仕様は `docs/spec/datasets/tennis_pose_sim.md` を参照。

---

## 2. CLI: `preprocess_tennis_memmap.py`

### 2.1 役割

- `build_tennis_dataset.py` の出力（JSON シーン）を 1 シーン単位で読み込み、
  テンソル化・正規化した配列を npz に書き出す。
- 学習時は JSON を毎回パースする代わりに、この npz から **スライスだけ行う**ことで、
  データローディングの CPU 負荷と I/O を削減することが目的。

### 2.2 引数一覧

実装: `src/cli/preprocess_tennis_memmap.py:1`

| 引数 | 型 / 既定値 | 説明 |
| --- | --- | --- |
| `--dataset_root` | `str`, `"data/tennis_autogen"` | `build_tennis_dataset.py` が生成したデータセットルート |
| `--dataset_name` | `str`, **必須** | `<dataset_root>/<dataset_name>` のサブディレクトリ名 |
| `--max_cameras` | `int`, `4` | 1 シーンあたりの最大カメラ数（npz に保持する上限。本番学習時にはこの中からさらにサブサンプリングされ得る） |
| `--max_players` | `int`, `20` | 1 フレームあたりの最大プレーヤー数（`dataset.max_players` と揃える） |
| `--num_joints` | `int`, `20` | プレーヤー 1 人あたりのキーポイント数（pose 17 + racket 3 = 20） |
| `--splits` | `str`, `"train,val,test"` | 前処理対象の split 名（カンマ区切り） |
| `--overwrite` | flag | 既に npz が存在する場合でも上書きする |

追加制約:

- `dataset_dir = Path(dataset_root) / dataset_name` が存在しない場合は `SystemExit`。
- 各シーン JSON 内の `num_cameras` が `max_cameras` を超える場合は `ValueError`。
- カメラメタデータやフレーム配列が不正な場合も `ValueError`。

### 2.3 実行コマンド例

代表的な実行例:

```bash
python src/cli/preprocess_tennis_memmap.py \
  --dataset_root data/tennis_autogen \
  --dataset_name sim_fps60_dur3p0_C4_P1-20_T10 \
  --max_cameras 4 \
  --max_players 20 \
  --num_joints 20
```

特定 split のみ処理したい場合:

```bash
python src/cli/preprocess_tennis_memmap.py \
  --dataset_root data/tennis_autogen \
  --dataset_name sim_fps60_dur3p0_C4_P1-20_T10 \
  --splits train
```

既存の npz を作り直したい場合（上書き）:

```bash
python src/cli/preprocess_tennis_memmap.py \
  --dataset_root data/tennis_autogen \
  --dataset_name sim_fps60_dur3p0_C4_P1-20_T10 \
  --overwrite
```

---

## 3. 入出力構造

### 3.1 入力ディレクトリ構造

前提として、`build_tennis_dataset.py` で以下のような構造が作成されている:

```text
<dataset_root>/<dataset_name>/
  scenes/
    train/scene_000000.json
    val/scene_000000.json
    test/scene_000000.json
  index/
    train_index.jsonl
    val_index.jsonl
    test_index.jsonl
  meta.json
```

本 CLI は `index` ではなく、`scenes/<split>/scene_*.json` を直接読み込む。

### 3.2 出力ディレクトリ構造

各 split ごとに、以下の npz を生成する:

```text
<dataset_root>/<dataset_name>/
  arrays/
    train/
      scene_000000.npz
      scene_000001.npz
      ...
    val/
      scene_000000.npz
      ...
    test/
      scene_000000.npz
      ...
```

- npz は **非圧縮**（`np.savez`）で保存され、
  `np.load(path, mmap_mode="r")` によるメモリマップ読み込みを前提とする。

---

## 4. npz 内の配列仕様

実装: `src/cli/preprocess_tennis_memmap.py:_process_scene_json`

あるシーン JSON から生成される npz には、少なくとも以下の配列が含まれる:

| キー | 形状 | dtype | 説明 |
| --- | --- | --- | --- |
| `keypoints_2d` | `[T, V, M, J, 2]` | `float32` | 正規化済み 2D キーポイント（pose 17 + racket 3 = J=20） |
| `player_mask` | `[T, V, M]` | `bool` | 該当 `(t,v,m)` に実プレーヤーが存在するか |
| `court_2d` | `[V, 20, 2]` | `float32` | 正規化済みコート 2D キーポイント（各カメラ v について 20 点） |
| `pose_3d_gt` | `[T, M, J, 3]` | `float32` | 正規化済み 3D GT キーポイント（pose+racket, J=20） |
| `exist_3d_gt` | `[T, M]` | `bool` | `(t,m)` に 3D GT が存在するか |
| `camera_C` | `[V, 3]` | `float32` | カメラ中心（世界座標系） |
| `camera_R` | `[V, 3, 3]` | `float32` | world→camera 回転行列 |
| `camera_intr` | `[V, 3]` | `float32` | 簡易内部パラメータ（例: `[f, cx, cy]`） |
| `image_size` | `[V, 2]` | `int32` | 各カメラの画像サイズ `[width, height]` |

ここで:

- `T`: シーン内の総フレーム数（`frames` の長さ）
- `V`: `max_cameras`（CLI 引数の上限値）
- `M`: `max_players`（CLI 引数の上限値）
- `J`: `num_joints`（通常 20）

### 4.1 2D キーポイント正規化 (`keypoints_2d`, `court_2d`)

内部関数 `_normalize_2d(points, width, height)` によって、
ピクセル座標 `[u_px, v_px]` を `[-1, 1]` レンジへ線形変換する:

- 入力: `points[..., 2]` はピクセル座標
- 変換:
  - `u_norm = (u_px / width) * 2 - 1`
  - `v_norm = (v_px / height) * 2 - 1`
- `width <= 0` または `height <= 0` の場合は変換をスキップしてそのままコピー

`keypoints_2d` の構成:

- 元 JSON ではフレーム t, カメラ v ごとに
  - `player_keypoints_2d.joints[m]: [17, 2]`
  - `racket_keypoints_2d.points[m]: [3, 2]`
- 本 CLI では、各プレーヤー m について
  1. `pose_np: [17, 2]` をゼロ初期化
  2. JSON の実長に合わせて `pose_np[:n_pose]` 部分だけコピー
  3. 同様に `racket_np: [3, 2]` にラケット 2D をコピー
  4. `combined: [20, 2] = concat(pose_np, racket_np)`
  5. `_normalize_2d(combined, width, height)` を適用

- 結果は `keypoints_2d[t, v, m, :, :]` に格納される。
- プレーヤー数が `M` を超える場合は `M` までに切り詰められる。
- 対応する `player_mask[t, v, m]` は True に設定される。

`court_2d` の構成:

- 各カメラ v について、フレーム 0 の
  - `frames[0].cam_v.court_keypoints_2d.points[:20]`
  を `_normalize_2d` したものを `court_2d[v, :, :]` に格納。
- 以降のフレームではコートは一定とみなす。

### 4.2 3D キーポイント正規化 (`pose_3d_gt`, `exist_3d_gt`)

3D GT は、コート座標系 (メートル) から **無次元の正規化座標** へ変換される。
使用する定数は `src/tennis/geometry/court.py` からの:

- `HALF_DOUBLES_WIDTH`
- `HALF_LENGTH`
- `NET_HEIGHT_POST`

変換式:

- 元の世界座標 `(x_w, y_w, z_w)` に対して:
  - `x_n = x_w / HALF_DOUBLES_WIDTH`
  - `y_n = y_w / HALF_LENGTH`
  - `z_n = z_w / NET_HEIGHT_POST`

実装では、各フレーム t, プレーヤー m について:

1. `player_joints_3d[m]: [17, 3]`
2. `racket_points_3d[m]: [3, 3]`
3. それぞれをゼロ初期化された配列にコピー
4. `combined3d: [20, 3] = concat(pose3d_np, racket3d_np)`
5. 上記のスケーリングを各次元に適用
6. `pose_3d_gt[t, m, :, :] = combined3d`
7. `exist_3d_gt[t, m] = True`

プレーヤー数が `M` を超える場合は `M` までに切り詰められ、
`player_joints_3d` が list でない場合などはスキップされる。

---

## 5. カメラパラメータと画像サイズ

各シーン JSON には、カメラごとに以下の情報が含まれている想定:

- `cameras[v].image_size = [w, h]`
- `cameras[v].camera_C: [3]`
- `cameras[v].camera_R: [3, 3]`
- `cameras[v].camera_intr: [3]`

本 CLI では、これらを次の配列にまとめる:

- `image_size[v, :] = [w, h]`
- `camera_C[v, :] = camera_C`
- `camera_R[v, :, :] = camera_R`
- `camera_intr[v, :] = camera_intr`

これらは `TennisSceneWindowDataset` 経由でバッチに渡され、
将来的な **3D→2D 再投影ベースの可視化 / デバッグ** に利用されることを想定している。

---

## 6. 学習時の利用方法（概要）

学習用 Dataset 側では、`use_memmap` フラグにより読み込みパスを切り替える:

- `use_memmap = false` の場合:
  - `index/<split>_index.jsonl` と `scenes/<split>/scene_*.json` を直接読み、
    各バッチで JSON パース + Python ループによりテンソル生成を行う。
- `use_memmap = true` の場合:
  - 本 CLI が生成した `arrays/<split>/scene_*.npz` を `np.load(..., mmap_mode="r")` で開き、
    既にテンソル化された配列から `[t_start:t_end]` をスライスして PyTorch Tensor に変換するだけで済む。

この設計により、
- 一度 memmap 前処理を行えば、以降の学習では I/O と CPU オーバーヘッドを大幅に削減できる
- JSON 仕様 (`tennis_simulator.md`) と学習仕様 (`tennis_pose_sim.md`) の間に、
  安定した中間フォーマット（npz/memmap）を挟むことができる

---

## 7. ワークフローまとめ

1. **データセット生成**（既存）

   ```bash
   python src/cli/build_tennis_dataset.py \
     --config configs/tennis/build_tennis_dataset_sim.yaml
   ```

2. **memmap 前処理**（本 CLI）

   ```bash
   python src/cli/preprocess_tennis_memmap.py \
     --dataset_root data/tennis_autogen \
     --dataset_name sim_fps60_dur3p0_C4_P1-20_T10
   ```

3. **学習実行**（例）

   - `configs/datasets/tennis_pose_sim.yaml` で
     - `root: data/tennis_autogen`
     - `name: sim_fps60_dur3p0_C4_P1-20_T10`
     - `use_memmap: true`
     を設定。
   - その上で `train_tennis_pose.py` を実行することで、
     memmap ベースの高速データローディングを利用できる。
