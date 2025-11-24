# テニス multi-cam 3D pose Dataset 仕様

本書は、テニス multi-cam 3D pose タスクにおける Dataset 仕様をまとめる。

- シミュレータ出力（`scene_*.json`）
- データセットビルド CLI による train/val/test 分割と index
- 必要に応じた memmap 前処理（npz 中間表現）
- `TennisSceneWindowDataset` が前提とするディレクトリ構造とテンソル形状
- **v1/v2両モデル対応**: v2用GTデータの自動生成機能

CLI や scripts による実行方法は `docs/spec/cli/tennis_multi_cam_3d_pose.md` および
`scripts/tennis_data_pipeline.md` を参照し、本ドキュメントでは **フォーマットと前提条件** に集中する。

---

## 1. シーン JSON の前提

- 仕様: `docs/spec/tennis_multi_cam_3d_pose/tennis_simulator.md`
- 各シーンには、フレーム列 `frames[t]` とカメラ列 `cameras[v]` が含まれる。
- プレーヤーごとの 2D/3D キーポイント、ラケット、コートキーポイント、カメラパラメータなどが格納される。

---

## 2. データセットビルド（JSON + index）

- 役割: シーン JSON 群から、train/val/test 分割されたシーンとインデックスを生成する。
- CLI 実装: `src/cli/tennis_multi_cam_3d_pose/build_dataset.py`
- 代表的な設定ファイル: `configs/tennis/build_tennis_dataset_sim.yaml`

出力ディレクトリ構造は次の通り:

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

- `meta.json` には、fps / duration / num_cameras などの共通メタデータと乱数シード情報を保持する。
- `index/<split>_index.jsonl` は 1 ウィンドウ 1 行の JSON ラインであり、少なくとも次のフィールドを持つ:
  - `scene_path`: ルートからの相対パス（例: `scenes/train/scene_000000.json`）
  - `scene_id`: シーン ID
  - `t_start`, `t_end`: ウィンドウの時間範囲
  - `num_frames`: ウィンドウの実フレーム数（`<= window_T`）
  - `num_cameras`: シーン内のカメラ数
  - `max_players_in_window`: 当該ウィンドウで観測された最大プレーヤー数

ウィンドウ生成は、全フレーム長 `T_total` に対して

- `t = 0, window_stride, 2*window_stride, ...` と進めながら
- `[t, min(t+window_T, T_total))` を列挙する

というシンプルなスライディングウィンドウで行う。

---

## 3. memmap 前処理（npz 中間表現）

- 役割: JSON シーンから、学習時に高速に読み込める npz/memmap 形式を生成する。
- CLI 実装: `src/cli/tennis_multi_cam_3d_pose/preprocess_memmap.py`
- 入力: 上記ビルド済みディレクトリ（`scenes/*/*.json`）
- 出力:

```text
<dataset_root>/<dataset_name>/
  arrays/
    train/scene_000000.npz
    val/scene_000000.npz
    test/scene_000000.npz
```

 各 npz には、少なくとも以下の配列が含まれる想定である（代表例）:

 - `keypoints_2d`: `[T, V, M, J, 2]` — 正規化済み 2D キーポイント
 - `player_mask`: `[T, V, M]` — プレーヤー存在マスク
 - `court_2d`: `[V, 20, 2]` — 正規化済みコート 2D キーポイント
 - `pose_3d_gt`: `[T, M, J, 3]` — 正規化済み 3D GT キーポイント
 - `exist_3d_gt`: `[T, M]` — 3D GT の有無
 - `camera_C`, `camera_R`, `camera_intr`, `image_size` などのカメラ情報

  これらの配列の座標系および正規化は次の通りである:

  - **2D キーポイント (`keypoints_2d`, `court_2d`)**
    - 元のピクセル座標 `[u, v]` を画像サイズ `(w, h)` でスケーリングし、`[-1, 1]` に線形マッピングする。
    - 実装: `src/cli/tennis_multi_cam_3d_pose/preprocess_memmap.py::_normalize_2d` および `src/datasets/tennis/scene_dataset.py::_getitem_from_json`。
    - 具体的には `u_norm = (u / w) * 2 - 1`, `v_norm = (v / h) * 2 - 1`。
    - 可視化などでピクセル座標に戻す場合は `src/training/utils/tennis_projection.py::norm_to_px` を利用できる。

  - **3D ポーズ (`pose_3d_gt` とそれに由来する v2 用 GT)**
    - シミュレータのワールド座標系は ITF 規格ベースのコート寸法 (単位メートル) に従う。
    - 各関節のワールド座標 `[x, y, z]` をテニスコートの代表長さでスケーリングし、無次元化している。
    - 実装: `src/cli/tennis_multi_cam_3d_pose/preprocess_memmap.py::_process_scene_json` および `src/datasets/tennis/scene_dataset.py::_getitem_from_json`。
    - 具体的には `x_norm = x / HALF_DOUBLES_WIDTH`, `y_norm = y / HALF_LENGTH`, `z_norm = z / NET_HEIGHT_POST` (`src/tennis/geometry/court.py` を参照)。
    - `src/training/utils/tennis_projection.py::denorm_pose3d` で `[HALF_DOUBLES_WIDTH, HALF_LENGTH, NET_HEIGHT_POST]` を掛けることでメートル単位に戻せる。

  - **マスク (`player_mask`, `exist_3d_gt`)**
    - これらはブール値 (`True` / `False`) の存在フラグであり，数値的な正規化は行わない。

  - **カメラ情報 (`camera_C`, `camera_R`, `camera_intr`, `image_size`)**
    - `camera_C`: シミュレータのワールド座標系におけるカメラ中心 `[x, y, z]` (メートル)。
    - `camera_R`: ワールド座標 → カメラ座標への回転行列 (3x3)。
    - `camera_intr`: `[f, cx, cy]` という形式の内部パラメータで，`f` はピクセル単位の焦点距離，`cx, cy` は主点座標 (ピクセル)。
    - `image_size`: `[w, h]` (ピクセル)。
    - これらは正規化せず，`src/tennis/sim/schema.py` / `src/tennis/sim/generator.py` で生成された値をそのまま保持し，`src/training/utils/tennis_projection.py::project_world_points` などで利用する。

### 3.1 v2用GTデータ（オプション）

v2モデル（階層エンコーダ + 分離出力）用のGTデータも含めることができる:

- `canonical_pose_gt`: `[T, M, J, 3]` — ルート相対・回転なしの正規化ポーズ
- `root_trans_gt`: `[T, M, 3]` — コート上の正規化ルート位置
- `root_rot_gt`: `[T, M, 2]` — ルート回転（cos, sin）
- `global_pose_gt`: `[T, M, J, 3]` — 再構成された絶対座標ポーズ

  これらの配列の座標系および正規化は次の通りである:
  - **`canonical_pose_gt`, `root_trans_gt`, `root_rot_gt`, `global_pose_gt`**
    - いずれも上記の正規化済み `pose_3d_gt` から `_decompose_pose_for_v2` / `_decompose_pose_for_v2_torch` により派生する。
    - `root_trans_gt`: 関節インデックス 0 (腰) の絶対位置 `[x, y, z]` をそのまま保持（スケールは `pose_3d_gt` と同じ）。
    - `canonical_pose_gt`: 各フレーム・プレーヤーごとに root を原点に平行移動し，左右肩 (11, 12 番) のベクトルから推定した yaw を打ち消した座標。
    - `root_rot_gt`: 上記 yaw 角を `(cosθ, sinθ)` で表現した 2 次元ベクトル。
    - `global_pose_gt`: 元の `pose_3d_gt` をコピーしたもの（スケール・座標系は `pose_3d_gt` と同一）。

**注意**: v2用GTデータが存在しない場合、`TennisSceneWindowDataset`は既存の`pose_3d_gt`から自動的にv2用GTを生成する。したがって、既存データセットでv2モデルの学習が可能。

これにより、学習時には JSON パースと Python ループを避け、

- `np.load(path, mmap_mode="r")` で npz を開き
- 時間ウィンドウ `[t_start:t_end)` をスライスするだけで

PyTorch Tensor を構築できる。

---

## 4. TennisSceneWindowDataset の前提

- 実装: `src/datasets/tennis/scene_dataset.py:TennisSceneWindowDataset`
- 代表的なコンストラクタ引数（簡略）:

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

- `dataset_root` / `dataset_name`: 上記ビルド出力を指す。
- `split`: `"train" | "val" | "test"`。
- `window_T`, `max_cameras`, `max_players`, `num_joints`: テンソル形状を決めるハイパーパラメータ。
- `use_memmap`: true の場合は `arrays/<split>/scene_*.npz` を利用し、false の場合は JSON から直接テンソルを構築する。
- `min_cameras`: 1 サンプルあたりの最小カメラ数。指定時、`min_cameras <= K <= max_cameras` を満たすようにビュー数 `K` がサンプリングされる。
- `augment_2d`: true かつ train split の場合、2D キーポイントとコート 2D にランダムアフィン変換を適用する。

この Dataset は、上記の index / scenes / arrays を前提として、

- `__len__` = ウィンドウ数（index の行数）
- `__getitem__` = 1 ウィンドウ分のテンソル dict

を返すよう設計されている。

### 4.1 出力テンソル

`__getitem__` は以下のテンソルを含む辞書を返す:

**基本入力データ**:
- `keypoints_2d`: `[T, K, M, J, 2]` — 選択されたカメラの2Dキーポイント
- `player_mask`: `[T, K, M]` — プレーヤー存在マスク
- `court_2d`: `[K, 20, 2]` — コート2Dキーポイント
- `camera_*`: カメラパラメータ群

**v1用GTデータ**:
- `pose_3d_gt`: `[T, M, J, 3]` — 3DポーズGT
- `exist_3d_gt`: `[T, M]` — 存在フラグGT

**v2用GTデータ（条件付き）**:
- `canonical_pose_gt`: `[T, M, J, 3]` — v2用canonicalポーズGT
- `root_trans_gt`: `[T, M, 3]` — v2用ルート位置GT
- `root_rot_gt`: `[T, M, 2]` — v2用ルート回転GT
- `global_pose_gt`: `[T, M, J, 3]` — v2用グローバルポーズGT

**注意**: v2用GTデータは、
1. npzファイルに存在する場合はそのまま読み込まれ
2. 存在しない場合は`pose_3d_gt`から自動生成される

これにより、既存データセットでのv1/v2両モデルの学習がシームレスに可能。
