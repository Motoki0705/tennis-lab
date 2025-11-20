# テニス multi-cam 3D pose Dataset 仕様

本書は、テニス multi-cam 3D pose タスクにおける Dataset 仕様をまとめる。

- シミュレータ出力（`scene_*.json`）
- データセットビルド CLI による train/val/test 分割と index
- 必要に応じた memmap 前処理（npz 中間表現）
- `TennisSceneWindowDataset` が前提とするディレクトリ構造とテンソル形状

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
