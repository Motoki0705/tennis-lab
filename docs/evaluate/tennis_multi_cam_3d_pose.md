# Tennis multi-cam 3D pose 評価 / 可視化

本書は、テニス用 multi-cam 3D pose モデル（v1/v2/v2.5/v3 共通）の **評価・可視化パイプライン** をまとめる。

- 評価 CLI: `src/evaluate/tennis_multi_cam_3d_pose.py`
- シェルラッパ: `scripts/evaluate/run_eval_tennis_multi_cam_3d_pose*.sh`
- 入力: `TennisSceneWindowDataset`（train / val / test split）
- 出力: 予測 3D ポーズをカメラに再投影した mp4 動画

---

## 1. 全体像

### 1.1 やっていること

1. 学習時と同じトップレベル YAML（v1: `configs/tennis_multi_cam_3d_pose.yaml`, v2: `_v2.yaml`, v2.5: `_v2_5.yaml`, v3: `_v3.yaml`）を読み込む
2. `ConfigLoader` で `TennisPoseDataModule` / `TennisDetrModule` or `TennisDetrV2Module` or `TennisDetrV25Module` or `TennisDetrV3Module` を構築
3. `runs/` 以下から、`experiment_name` に対応する checkpoint (`*.ckpt`) を自動探索
4. 指定した split (`train` / `val` / `test`) のウィンドウをサンプリング
5. モデルで 3D 予測 (`pose_3d`, `exist_conf`) を計算
6. 予測 3D をカメラパラメータで再投影して、各プレーヤの 2D キーポイント列を構築
7. `src.visualize.tennis_render.render_pose2d_frame` で各フレームを描画し、
      `src.visualize.video_io.write_video` で mp4 に書き出し

※ 目的は **定量評価ではなく視覚的な挙動確認**。

---

## 2. 評価 CLI: `src/evaluate/tennis_multi_cam_3d_pose.py`

### 2.1 主要引数

- `--config` (必須)
  - 学習時と同じトップレベル YAML。
  - 例: `configs/tennis_multi_cam_3d_pose_v2.yaml`
- `--set key=value ...`
  - `load_cfg` と同じ dotlist 形式でオーバーライド。
  - 例: `--set dataset.name=sim_fps60_dur3p0_C4_P1-20_T10`
- `--splits`
  - 評価対象 split。デフォルト: `train test`
  - `train` / `val` / `test` から 1 つ以上を指定可能。
- `--output-dir`
  - 動画の出力先ディレクトリ。
  - デフォルト: `outputs/tennis_eval_videos`
- `--num-samples`
  - 各 split ごとにレンダリングするウィンドウ数。デフォルト: `4`
- `--start-index`
  - 各 split ごとの開始インデックス。デフォルト: `0`
- `--camera-index`
  - 可視化に用いるカメラ ID。
  - 負値の場合は「そのウィンドウで少なくとも 1 人写っている最初のカメラ」を自動選択。
- `--fps`
  - 出力動画のフレームレート。デフォルト: `30`
- `--runs-dir`
  - TensorBoard / checkpoint のベースディレクトリ。デフォルト: `runs`
- `--checkpoint`
  - 明示的な `*.ckpt` パス。未指定時は `runs/<experiment_name>/version_*/checkpoints/` から自動探索。
- `--exist-threshold`
  - `exist_conf >= threshold` を満たす Query だけを可視化対象プレーヤとして描画。
  - 未指定時は `cfg.logging.visualizer.exist_threshold` を参照し、なければ `0.5`。
- `--device`
  - 推論に使うデバイス (`cpu`, `cuda`, `cuda:0` など)。未指定時は `cuda` があれば `cuda`, なければ `cpu`。

### 2.2 checkpoint 探索ロジック

`--checkpoint` 未指定時は、

1. `cfg.experiment_name`（なければ `"tennis_multi_cam_3d_pose"`）を取得
2. `runs/<experiment_name>/version_*/checkpoints/` を列挙
3. 以下の優先順位で 1 つ選択:
   - `epoch=*val_total=*.ckpt` のうち最後のもの
   - `last.ckpt`
   - 上記が無ければ `*.ckpt` のうち最後のもの

v1 (`experiment_name: tennis_mvpose_dev`) / v2 (`experiment_name: tennis_mvpose_dev_v2`) / v2.5 (`experiment_name: tennis_mvpose_dev_v2_5`) / v3 (`experiment_name: tennis_mvpose_dev_v3`) いずれにも対応。

---

## 3. シェルラッパ `scripts/evaluate/*.sh`

### 3.1 v1 用: `run_eval_tennis_multi_cam_3d_pose.sh`

- デフォルト設定:
  - `CONFIG=configs/tennis_multi_cam_3d_pose.yaml`
  - `RUNS_DIR=runs`
  - `OUTPUT_DIR=outputs/tennis_eval_videos`
- 実行例:

```bash
./scripts/evaluate/run_eval_tennis_multi_cam_3d_pose.sh \
  --splits train test \
  --num-samples 4
```

### 3.2 v2 用: `run_eval_tennis_multi_cam_3d_pose_v2.sh`

- デフォルト設定:
  - `CONFIG=configs/tennis_multi_cam_3d_pose_v2.yaml`
  - `RUNS_DIR=runs`
  - `OUTPUT_DIR=outputs/tennis_eval_videos`
- 実行例:

```bash
./scripts/evaluate/run_eval_tennis_multi_cam_3d_pose_v2.sh \
  --splits train test \
  --num-samples 4
```

### 3.3 v2.5 用: `run_eval_tennis_multi_cam_3d_pose_v2_5.sh`

- デフォルト設定:
  - `CONFIG=configs/tennis_multi_cam_3d_pose_v2_5.yaml`
  - `RUNS_DIR=runs`
  - `OUTPUT_DIR=outputs/tennis_eval_videos`
- 実行例:

```bash
./scripts/evaluate/run_eval_tennis_multi_cam_3d_pose_v2_5.sh \
  --splits train test \
  --num-samples 4
```

### 3.4 v3 用: `run_eval_tennis_multi_cam_3d_pose_v3.sh`

- デフォルト設定:
  - `CONFIG=configs/tennis_multi_cam_3d_pose_v3.yaml`
  - `RUNS_DIR=runs`
  - `OUTPUT_DIR=outputs/tennis_eval_videos`
- 実行例:

```bash
./scripts/evaluate/run_eval_tennis_multi_cam_3d_pose_v3.sh \
  --splits train test \
  --num-samples 4
```

`CONFIG` / `RUNS_DIR` / `OUTPUT_DIR` は環境変数で上書き可能。

---

## 4. ワークフロー例

### 4.1 v2/v2.5/v3 モデルの学習 → 評価

#### v2
1. 学習

```bash
./scripts/train/run_train_tennis_multi_cam_3d_pose_v2.sh
```

2. 評価 & 可視化

```bash
./scripts/evaluate/run_eval_tennis_multi_cam_3d_pose_v2.sh \
  --splits train test \
  --num-samples 8 \
  --start-index 0
```

- `runs/tennis_mvpose_dev_v2/version_*/checkpoints/*.ckpt` から checkpoint が自動選択される。

#### v2.5
1. 学習

```bash
./scripts/train/run_train_tennis_multi_cam_3d_pose_v2_5.sh
```

2. 評価 & 可視化

```bash
./scripts/evaluate/run_eval_tennis_multi_cam_3d_pose_v2_5.sh \
  --splits train test \
  --num-samples 8 \
  --start-index 0
```

- `runs/tennis_mvpose_dev_v2_5/version_*/checkpoints/*.ckpt` から checkpoint が自動選択される。

#### v3
1. 学習

```bash
./scripts/train/run_train_tennis_multi_cam_3d_pose_v3.sh
```

2. 評価 & 可視化

```bash
./scripts/evaluate/run_eval_tennis_multi_cam_3d_pose_v3.sh \
  --splits train test \
  --num-samples 8 \
  --start-index 0
```

- `runs/tennis_mvpose_dev_v3/version_*/checkpoints/*.ckpt` から checkpoint が自動選択される。

- `outputs/tennis_eval_videos/` 配下に、`<split>_idxXXXXXX_sceneY_tAAAA-BBBB_pred.mp4` が生成される。

### 4.2 特定 checkpoint / split だけを見る

```bash
CKPT="runs/tennis_mvpose_dev_v3/version_2/checkpoints/epoch=004-val_total=0.000.ckpt" \
./scripts/evaluate/run_eval_tennis_multi_cam_3d_pose_v3.sh \
  --splits test \
  --checkpoint "${CKPT}" \
  --num-samples 4 \
  --start-index 100
```

- テスト split から index 100〜103 のウィンドウを評価し、mp4 を出力する。
- 同様の形式で v1/v2/v2.5/v3 いずれの評価も可能。

---

## 5. 実装メモ（内部仕様）

- Dataset は `TennisSceneWindowDataset` をそのまま利用
  - `keypoints_2d` / `player_mask` / `court_2d` / `camera_*` / `image_size` を LightningModule に渡す
- モデル出力は v1/v2/v2.5/v3 共通で `outputs["pose_3d"]` / `outputs["exist_conf"]` を使用
- 3D ポーズは `HALF_DOUBLES_WIDTH` / `HALF_LENGTH` / `NET_HEIGHT_POST` で[m]スケールに戻したあと、
  カメラパラメータで再投影して 2D を得る
- 再投影した 2D 座標をもとに、`render_pose2d_frame()` でフレーム画像を描画し、
  `write_video()` で mp4 を生成している
