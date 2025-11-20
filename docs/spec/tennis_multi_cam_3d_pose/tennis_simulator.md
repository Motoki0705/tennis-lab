# テニスシミュレーター仕様書（Spec）

本書は `src/tennis/sim/` 配下のアセットローダーとシーンジェネレーター、およびそれを利用する CLI (`src/cli/gen_tennis_pose_scenes.py`) の仕様をまとめる。目的は **3DTennisDS アセットを用いた多人数・多視点シーン生成の挙動を実装詳細を読まずに把握できること** である。

---

## 1. スコープとゴール

- 3DTennisDS の C3D モーションを読み込み、ViTPose+racket 20 点スケルトンにリターゲットした **クリーンな 3D/2D キーポイントシーン** を生成する。
- 各シーンは 1..20 人のプレーヤーを同一コートに配置し、フェンス上にサンプリングした複数カメラからの 2D 投影を含む。
- **ノイズや欠損は生成時には付与しない**。DataLoader がオンザフライでノイズ注入する前提。
- 生成結果は JSON（`scene_XXXXX.json`）として書き出し、`src/tennis/sim/schema.py` のバリデーションに合格する必要がある。

---

## 2. 主要モジュール

### 2.1 `src/tennis/sim/assets.py` — TennisAssetLibrary

| 項目 | 内容 |
| --- | --- |
| 入力 | 3DTennisDS C3D ファイル群（`data/raw/3dtennisds/**/.c3d`） |
| 依存 | `ezc3d`, `numpy`, `torch` |
| 主なクラス | `TennisAssetLibrary`, `AssetClip`, `AssetSample` |

**処理フロー**
1. `TennisAssetLibrary(root, min_frames, max_files)` が指定ディレクトリ配下の C3D を列挙。`max_files` で読み込みテスト用に制限可能。
2. `_load_clip(path)` が `ezc3d` でマーカー列をロード。
   - Plug-in Gait のマーカー群を `COCO_TO_MARKERS` マップで ViTPose 17 関節へ平均化。
   - ラケットマーカー `RH*`, `DOL` を `RACKET_MARKERS` に沿って 3 点へ集約。
   - 骨盤中心（左右 Hip の平均）を抽出し、ルート平行移動を原点化。
   - 欠損 (`NaN` 含む) フレームを除外し、`min_frames` 未満なら無効クリップとしてスキップ。
3. クリップは `AssetClip` としてキャッシュされ、`sample_sequence(frames_total, target_fps, rng)` により以下を返す：
   - 指定フレーム数・FPS にリサンプリングされた `joints[T,17,3]`, `racket[T,3,3]`, `pelvis[T,3]`（ペルビス軌跡は後段の平行移動に使用）。
   - クリップ長不足時は `np.tile` で複製し、ターゲットスパンを満たす。スタート時刻は乱数でオフセット。

### 2.2 `src/tennis/sim/generator.py` — TennisPoseSceneGenerator

| 項目 | 内容 |
| --- | --- |
| 入力 | `GenConfig`（FPS、シーン長、カメラ数、プレーヤー数範囲、アセットパス等） |
| 依存 | `TennisAssetLibrary`, `torch`, `src/tennis/geometry.court`, `src/tennis/geometry.skeleton` |
| 主なメソッド | `_build_cameras`, `_build_players`, `_sample_player_origins`, `_place_sample`, `generate_scene` |

**主要設定 (`GenConfig`)**
- `fps`, `duration_sec`, `num_cameras`: シーン時間分解能
- `asset_root`, `asset_min_frames`, `asset_max_files`: アセット読み込み制御
- `min_players`, `max_players`: シーン毎にサンプリングする人数レンジ（1〜20）
- `player_min_separation`, `spawn_margin_{x,y}`: コート上配置時の最近接距離とコート端マージン
- `seed`: 乱数シード（`random.Random` と `torch.manual_seed` を初期化）

**処理フロー**
1. **カメラ構築**: `_sample_camera()` がフェンスの near/far/left/right 辺から一様サンプル。LookAt 先は `(0,0,0.5)` 固定。`make_look_at_camera()` 内では `torch.cross(..., dim=0)` と `math.tan` で姿勢/焦点距離を算出。
2. **プレーヤー構築**: `_build_players(frames_total)` がランダム人数を決め、`TennisAssetLibrary.sample_sequence` で各プレーヤーの軌跡を取得。
3. **配置（アンカーサンプリング）**: `_sample_player_origins(count)` がフェンス内矩形から anchor (x,y,0) を抽出し、`player_min_separation` を満たすまでランダムサーチ。失敗時は RuntimeError。
4. **回転/平行移動**: `_place_sample(sample, anchor, yaw)` がランダム yaw（-π〜π）を Z 回り回転行列 `_rotation_matrix_z` で適用し、ペルビス軌跡＋anchor を加算してワールド座標へ変換。
5. **フレームループ** (`generate_scene`):
   - `player_joints_3d` / `racket_points_3d` は各プレーヤー分のリスト（`num_players` を含める）。
   - 各カメラで `project_points` を通して court/player/racket を 2D 投影。結果は **multi-player 形式**：
     ```jsonc
     "player_keypoints_2d": {
       "joints": [ [17x2], [17x2], ... ],
       "visibility": [ [17], [17], ... ]
     }
     ```
     いずれも `num_players` 個のサブ配列を持つ。visibility は `project_points` の z>0 判定のみで、ノイズ/欠損なし。
6. **出力**: `scene = {scene_id, fps, num_cameras, cameras, frames}` を生成し、`write_scene_json()` で保存する直前に `validate_scene_dict(scene)` を必須実行。

---

## 3. シーン JSON スキーマ（抜粋）

| フィールド | 型 | 説明 |
| --- | --- | --- |
| `scene_id` | str | CLI が付与する一意 ID（例: `0`, `1`） |
| `fps` | int | シーン共通フレームレート |
| `num_cameras` | int | `cameras` の長さ |
| `cameras[i]` | `{id: str, image_size: [w, h]}` | 内部で持っていた `Camera` から公開属性のみ抜粋 |
| `frames[t].num_players` | int | プレーヤー数。`player_joints_3d` 長さと一致必須 |
| `frames[t].player_joints_3d` | `list[list[17][3]]` | 各プレーヤーの 3D ViTPose 座標（メートル） |
| `frames[t].racket_points_3d` | `list[list[3][3]]` | 各プレーヤーのラケット 3 点 3D |
| `frames[t].cam_k.player_keypoints_2d` | 辞書 | `joints`: `num_players` 個の `[17][2]`、`visibility`: `num_players` 個の `[17]`（0/1） |
| `frames[t].cam_k.racket_keypoints_2d` | 辞書 | `points`: `num_players x [3][2]`、`visibility`: 同型 |
| `frames[t].cam_k.court_keypoints_2d` | 辞書 | `points`: `[20][2]`, `visibility`: `[20]` |

整合性は `src/tennis/sim/schema.py` が検証する。違反時は ValueError。

---

## 4. CLI (`src/cli/gen_tennis_pose_scenes.py`)

| 引数 | デフォルト | 説明 |
| --- | --- | --- |
| `--out` | 必須 | シーンを書き出すディレクトリ |
| `--num_scenes` | 5 | 生成するシーン数 |
| `--num_cameras` | 4 | カメラ台数 |
| `--fps` | 60 | フレームレート |
| `--duration` | 3.0 | シーン長（秒） |
| `--asset_root` | `data/raw/3dtennisds` | 3DTennisDS ルート |
| `--min_players / --max_players` | 1 / 20 | 1 シーンあたりの人数レンジ |
| `--seed` | 1234 | 乱数シード |

CLI は以下を行う：
1. 引数を `GenConfig` に反映。
2. `for scene_id in range(num_scenes)` で `generate_scene` → `validate_scene_dict` → `write_scene_json`。
3. 生成成功時 `[tennis-gen] Wrote N scene(s) ...` を出力。例外発生時 `[gen-error] ...` を stderr に書き終了コード 1。

---

## 5. 制約・拡張ポイント

- **依存ライブラリ**: `ezc3d` が未インストールの場合、`TennisAssetLibrary` 初期化が RuntimeError で失敗する。CLI 以前に `uv pip install ezc3d` などで依存解決すること。
- **アセット要件**: `data/raw/3dtennisds` 直下に TP1〜TPn ディレクトリと `.c3d` が存在すること。欠損フレームが多いクリップはスキップされる。
- **人数配置失敗**: `player_min_separation` が大きすぎると `_sample_player_origins` が制限回数で失敗し RuntimeError。CLI で `--min_players` を下げるか距離パラメータを見直す。
- **ノイズ/欠損**: 生成物は理想値のみ。訓練用ノイズは DataLoader (`src/datasets/tennis_pose.py` 実装予定) で注入する。
- **将来的な拡張**:
  1. カメラ姿勢揺らぎや height ランダム化を追加（`GenConfig` 拡張）。
  2. クリップメタデータ（球種、選手 ID）を scene JSON に含める。
  3. 物理的衝突回避や複数ボール軌跡を導入し、リアルな対戦シナリオに近付ける。

以上の仕様を満たすことで、3DTennisDS アセットから一貫したシミュレーションデータセットを再現性高く生成できる。
