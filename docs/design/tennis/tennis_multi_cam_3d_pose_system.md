# Tennis Pose System 設計（Simulator → Dataset → Tennis-DETR）

本書は、`src/tennis/sim/` のシミュレーションと `src/cli/build_tennis_dataset.py`、および `src/models/tennis/TennisDETR` を一貫した **学習システム**としてまとめる設計ドキュメントである。

---

## 1. ゴールと全体像

- ゴール
  - 3DTennisDS 由来のシミュレーションから **マルチビュー・マルチプレーヤー 2D/3D キーポイント**を生成。
  - それを用いて、Detection Transformer ベースの **Tennis-DETR モデル**で 3D ポーズ（20 点: pose 17 + racket 3）を学習。
  - データセット生成〜学習までを **再現性の高い自動パイプライン**として運用。
- パイプライン
  1. Simulator: `gen_tennis_pose_scenes.py` がクリーンな JSON シーンを生成。
  2. Dataset Builder: `build_tennis_dataset.py` が train/val/test に分割し、シーン + インデックス + メタを構築。
  3. Dataset/DataModule: JSON + index から `[B, T, V, M, J, 2]` テンソルを組み立て、ノイズもここで注入。
  4. Model: `TennisDETR` が全カメラ・全時間の検出をトークン列として処理し、Association + 3D Lifting を同時に解く。

---

## 2. シミュレーション出力と生データ

シミュレーション仕様の詳細は `docs/spec/tennis/tennis_simulator.md` を参照し、本節では **学習に必要な部分だけ**を抜粋する。

### 2.1 シーン JSON 構造（抜粋）

- トップレベル
  - `scene_id: str`
  - `fps: int`
  - `num_cameras: int`
  - `cameras: list[{id: str, image_size: [w, h]}]`
  - `frames: list[Frame]`
- 各フレーム `Frame`
  - `num_players: int`
  - `player_joints_3d: list[list[17][3]]`  — pose 17 点
  - `racket_points_3d: list[list[3][3]]`   — racket 3 点
  - 各カメラごとに:
    - `cam_k.court_keypoints_2d: {points: [20][2], visibility: [20]}`
    - `cam_k.player_keypoints_2d: {joints: [num_players][17][2], visibility: [num_players][17]}`
    - `cam_k.racket_keypoints_2d: {points: [num_players][3][2], visibility: [num_players][3]}`

### 2.2 学習側での 2D 表現

- モデル側では、プレーヤー 1 人あたりの 2D キーポイントを
  - `pose 17 点 + racket 3 点 = 20 点`
  として扱う。
- よって、フレーム t, カメラ v, プレーヤー m に対して:
  - `player_keypoints_2d[t, v, m]: [20, 2]`
  - 上記をバッチ化すると Tensor 形状は:
    - `keypoints_2d: [B, T, V, M, J, 2]` （`J=20`）
    - `player_mask: [B, T, V, M]` （有効な検出 = True, パディング = False）

---

## 3. データセット自動生成システム

学習用データセットは、専用 CLI `build_tennis_dataset.py` によって自動生成する。

### 3.1 ディレクトリ構成とバージョニング

`data/tennis_autogen/<dataset_name>/` 以下に、以下の構造で保存する。

```text
data/tennis_autogen/
  <dataset_name>/
    config.yaml            # （将来拡張）生成に使った主要設定
    meta.json              # メタ情報（シード、シーン数、git hash など）
    scenes/
      train/scene_000000.json
      val/scene_000000.json
      test/scene_000000.json
    index/
      train_index.jsonl    # 1 行 = 1 シーンの時間ウィンドウ
      val_index.jsonl
      test_index.jsonl
```

- `dataset_name` は CLI 引数または自動生成:
  - 例: `sim_fps60_dur3p0_C4_P1-20_T10`
- `meta.json` には少なくとも以下を出力:
  - `fps`, `duration_sec`, `num_cameras`, `asset_root`
  - `min_players`, `max_players`
  - `window_T`, `window_stride`
  - `seed` と split ごとの seed（train/val/test）
  - 生成したシーン数（split ごと）
  - `created_at`（UTC ISO 8601）と `git_commit`（取得できれば）

### 3.2 CLI: `build_tennis_dataset.py`

- 主な引数（`src/cli/build_tennis_dataset.py` 参照）:
  - 出力:
    - `--dataset_root`（デフォルト `data/tennis_autogen`）
    - `--dataset_name`（省略可）
    - `--overwrite`（既存ディレクトリの上書き可否）
  - シミュレーション:
    - `--num_scenes_train/val/test`
    - `--fps`, `--duration`, `--num_cameras`
    - `--asset_root`, `--min_players`, `--max_players`
    - `--seed`（train/val/test に +0/+1/+2 で分割）
  - インデックス:
    - `--window_T`（時間ウィンドウ長; フレーム数）
    - `--window_stride`（ウィンドウ間ストライド; フレーム数）

#### 3.2.1 シーン生成フェーズ

- split ごと（train/val/test）に:
  - `GenConfig` を構築して `TennisPoseSceneGenerator` をインスタンス化。
  - `scene_id = "<split>_<i>"` として `num_scenes` 回 `generate_scene` を呼ぶ。
  - 生成した dict を `validate_scene_dict` に通し、`scenes/<split>/scene_XXXXXX.json` として保存。

#### 3.2.2 インデックス生成フェーズ

- 各 split に対して `index/<split>_index.jsonl` を作成。
- 1 行 = 1 シーン内の時間ウィンドウ:

| フィールド | 説明 |
| --- | --- |
| `scene_path` | Dataset ルートからの相対パス（例: `scenes/train/scene_000000.json`） |
| `scene_id` | JSON 内の `scene_id` |
| `t_start` | ウィンドウ開始フレーム index（0-based, inclusive） |
| `t_end` | ウィンドウ終了フレーム index（exclusive） |
| `num_frames` | `t_end - t_start` |
| `num_cameras` | `scene["num_cameras"]` |
| `max_players_in_window` | 該当フレーム範囲で観測された最大プレーヤー数 |

- ウィンドウ生成:
  - シーン内総フレーム数を `T_total` とし、`t = 0, window_stride, ...` で `[t, t+window_T)` を作成。
  - `t+window_T > T_total` の場合は `t_end = T_total` とし、短いウィンドウとして扱う。

### 3.3 再現性とリラン時の挙動

- デフォルトでは、`dataset_root/dataset_name` に既にファイルが存在し、かつ空でない場合はエラー終了。
- `--overwrite` を付与した場合のみ、既存ディレクトリを再利用して上書き生成する。
- `meta.json` の内容により、後から
  - 「この学習はどの設定で生成したデータか」
  - 「同じ条件のデータセットを再生成できるか」
  を追跡可能。

### 3.4 マルチカメラ構成と 2D データ拡張の設計

本システムでは、**シーン生成段階**と**ローディング段階**の 2 段階で多様性を確保し、オーバーフィットを避ける。

#### 3.4.1 シーン生成段階（build_tennis_dataset.py 側）

- 各シーンに対して、まず **20 箇所のカメラ位置をランダムサンプリング**する。
  - コート周囲の合理的な範囲内で位置・向きを乱数で決定し、「撮影レイアウトそのもの」の多様性を確保する。
- サンプリングした各カメラ位置ごとに、**使用するアセット（コート・背景・観客など）をランダムに選択**してレンダリングする。
  - これにより、同一シミュレーションシーンでも「見た目（テクスチャ・背景・環境）」が大きく変化する。
- こうして得られた **20 カメラからのレンダリング結果を 1 シーンとして保存**する。
  - シーン JSON には `num_cameras=20` と各カメラのメタデータ・2D キーポイントが含まれる。

#### 3.4.2 memmap 前処理段階

- memmap/npz 前処理では、**保存済み 20 カメラすべての情報を 1 つの npz にまとめる**。
  - `keypoints_2d[T, 20, M, J, 2]`, `player_mask[T, 20, M]`, `court_2d[20, 20, 2]` など、時間 × カメラ × プレーヤーの配列をそのまま格納する。
- これにより、学習時には JSON を解釈することなく、「20 カメラの生配列」から直接サンプリングできる基盤が整う。

#### 3.4.3 ローダーによるカメラ数・配置のサンプリング

- `TennisSceneWindowDataset`（`use_memmap=True` を想定）は、npz に保存された 20 カメラを読み込みつつ、**ローダー側で使用カメラ集合を動的に決める**。
  - コンフィグで `{min_cam, max_cam}` を持ち、各サンプルごとに `K ~ Uniform(min_cam, max_cam)` をサンプリング。
  - 20 カメラのインデックスからランダムに `K` 本を選び、その部分のみを `keypoints_2d`, `player_mask`, `court_2d` 等に反映する。
- バッチ整形の都合上、`max_cameras` は依然としてテンソル次元の上限として維持する。
  - 実際に選ばれた `K < max_cameras` の分については、残りのスロットをマスク付きパディングとして扱う。
- この設計により、**学習中の各イテレーションで「カメラ数」と「カメラ配置」が毎回変化**し、
  - 特定のカメラセットに過剰適応することを防ぎ、
  - 現実の「カメラ数が足りない/欠ける」ケースにもロバストなモデルを目指す。

#### 3.4.4 2D 座標に対するアフィン変換によるデータ拡張

- ローダーは、カメラサンプリング後の `keypoints_2d`（および整合性を保つための `court_2d`）に対して、**2D アフィン変換ベースのデータ拡張**を施す。
  - 例: 小さな回転、スケーリング、平行移動、軽微なせん断など。
  - これらは、すでに画素座標から `[-1, 1]` の正規化空間に写像された後の座標に対して適用する想定とし、変換が画面外に極端にはみ出さないようにパラメータ範囲を制御する。
- 2D 変換は **ビューごと・フレームごとに独立して適用**できるが、同一ビュー内のプレーヤー・コートには同一変換を適用することで、幾何整合性を保つ。
- 3D GT（`pose_3d_gt`）はコート座標系の値であり、2D アフィン変換とは切り離して扱う。
  - 2D 変換は「画像上の見え方」を揺らすだけで、**3D 空間上の位置関係はそのまま**にする。

#### 3.4.5 全体としての効果

- シーン生成時:
  - ランダムな 20 カメラ配置 × ランダムアセットにより、**視点と外観の大きな多様性**を確保。
- memmap 前処理時:
  - 20 カメラを余さず npz に格納し、**後段で柔軟にサブサンプリングできる基盤**を用意。
- ローダー時:
  - `{min_cam, max_cam}` の範囲でカメラ数と組み合わせを毎回変えることで、**カメラ構成に依存しないモデル**を目指す。
  - 2D アフィン変換により、同じポーズ・同じカメラであっても「画面内での向き・スケール・位置」が毎回揺らぎ、**画面配置へのオーバーフィットを抑える**。

この 2 段階（生成 + ロード）のランダム性により、最終的には **多様なカメラ数・カメラ配置・見え方** に対してロバストな 3D ポーズ推定モデルを育てることを狙う。

---

## 4. Dataset / DataModule 設計

Dataset / DataModule は、`build_tennis_dataset.py` の出力前提で設計する。

### 4.1 Dataset: `TennisSceneWindowDataset`（想定）

- 入力:
  - `index_path`: `index/train_index.jsonl` 等。
  - `window_T`: モデル側が期待する最大フレーム長。
  - `max_cameras`: バッチ整形時のカメラ上限。
  - `max_players`: 1 画像あたりのプレーヤー上限（通常 20）。
- `__getitem__(idx)` の処理:
  1. index の 1 行を読み取り、該当 `scene_path`, `t_start`, `t_end` を取得。
  2. シーン JSON をロードし、`frames[t_start:t_end]` を切り出し。
  3. 次のテンソルを構築:
     - `keypoints_2d: [T, V, M, J, 2]`
     - `player_mask: [T, V, M]`
     - 必要に応じて `court_2d: [V, 20, 2]` も返す。
  4. `T < window_T` の場合は時間方向にパディング（末尾ゼロ埋め + マスク）。

### 4.2 LightningDataModule: `TennisPoseDataModule`（想定）

- コンストラクタ引数:
  - `dataset_root`, `dataset_name`
  - `window_T`, `max_cameras`, `max_players`
  - DataLoader 設定: `batch_size`, `num_workers`, `prefetch_factor` など。
  - ノイズ設定: `noise_std`, `drop_prob`, `occlusion_ratio`, `camera_dropout_prob` 等。
- `setup(stage)`:
  - `train`, `val`, `test` それぞれに `TennisSceneWindowDataset` を構築。
  - 存在しない場合に `build_tennis_dataset.py` 相当のロジックで **自動生成するオプション**も将来的に追加可能（例: `auto_build=True`）。
- DataLoader 内でのノイズ注入:
  - 2D 座標にガウスノイズを加える。
  - 一部キーポイント/カメラ/プレーヤーをドロップ（`visibility` を 0 にする）。
  - これらは **シミュレーション JSON を汚さずに** On-the-fly で行う。

---

## 5. モデルアーキテクチャ: Tennis-DETR

アーキテクチャ名: **Tennis-DETR (Detection Transformer)**
コアアイデアは、**全カメラ・全時間の検出結果をフラットなトークン列として扱い、Encoder/Decoder 型 Transformer の Attention で「同一人物の結合 (Association)」と「3D 復元 (Lifting)」を同時に解く**こと。

### 5.1 記号と定数

- `B`: バッチサイズ（クリップ数）
- `T`: 時間ウィンドウサイズ（例: 10 フレーム）
- `V`: カメラ台数（例: 4）
- `M`: 1 画像あたりの最大検出人数（パディング上限。例: 5〜20）
- `J`: プレーヤー 1 人あたりのキーポイント数（20; pose 17 + racket 3）
- `D`: トークン特徴次元（例: 256）
- `Q`: 出力したい最大プレーヤー数（例: 4〜6）

### 5.2 入力トークン化 (Input Tokenization)

1. **Raw Input**
   - 2D ポーズ（pose+racket 20 点）:
     - `keypoints_2d: [B, T, V, M, J, 2]`
     - `player_mask: [B, T, V, M]`（True=有効, False=パディング）
   - コート 2D:
     - `court_2d: [B, V, 20, 2]`（20 点コートキーポイント）
2. **インスタンスベクトル化**
   - 各 `(b, t, v, m)` について:
     - `J×2` を flatten → `MLP: J*2 → D` (`joint_mlp`)。
     - 対応カメラのコート `[20,2]` を flatten → `MLP: 20*2 → D` (`court_mlp`)。
     - これらを加算して `token = pose_embed + court_embed`。
3. **位置埋め込み (Positional Encoding)**
   - `camera_embed[v]`, `time_embed[t]` を学習パラメータとして保持し、トークンに加算:
     - `token = token + camera_embed[v] + time_embed[t]`
4. **Flatten + Padding Mask**
   - `L = T * V * M`
   - `tokens: [B, L, D]`
   - `key_padding_mask: [B, L]` を `player_mask == False` の位置で True に設定。

### 5.3 Transformer Encoder

- 実装: `nn.TransformerEncoder` (`batch_first=True`)。
- 入力: `tokens: [B, L, D]`, `src_key_padding_mask: [B, L]`。
- 出力: `memory: [B, L, D]`
  - 「どのトークンがどのトークンと同一人物っぽいか」「コート上で位置的に整合するか」を Self-Attention によって学習。

### 5.4 Transformer Decoder + Object Queries

**目的**: 冗長なトークン列 `memory` から「プレーヤースロット」ごとの特徴を抽出。

1. **ベース Query**
   - 学習可能パラメータ: `query_embed: [Q, D]`
2. **時間付き Query**
   - `time_embed[t]` を再利用し、各プレーヤー × 各時刻用 Query を構成:
     - `queries_time[b, q, t] = query_embed[q] + time_embed[t]`
     - Shape: `[B, Q, T, D]`
   - 次元を平坦化して Decoder に入力:
     - `queries_flat: [B, Q*T, D]`
3. **Decoder**
   - 実装: `nn.TransformerDecoder` (`batch_first=True`)。
   - 入力: `tgt=queries_flat`, `memory`, `memory_key_padding_mask=key_padding_mask`。
   - 出力: `decoder_out: [B, Q*T, D]` → reshape → `decoder_out_time: [B, Q, T, D]`

### 5.5 3D 出力ヘッド

1. **3D Pose Head**
   - 入力: `decoder_out_time: [B, Q, T, D]`
   - 各 `(q, t)` ごとに `MLP: D → (J*3)` を適用。
   - 出力: `pose_3d: [B, Q, T, J, 3]`
     - **正規化コート座標系の `(x, y, z)`** を出力する。
       - シミュレータの 3D は `src/tennis/geometry/court.py` におけるコート座標系（メートル）で表現されるが、学習時には数値スケールを整えるために正規化する。
       - 正規化の例（世界座標 `(x_w, y_w, z_w)` → 学習座標 `(x_n, y_n, z_n)`）:
         - `x_n = x_w / HALF_DOUBLES_WIDTH`
         - `y_n = y_w / HALF_LENGTH`
         - `z_n = z_w / NET_HEIGHT_POST`
         - ここで `HALF_DOUBLES_WIDTH`, `HALF_LENGTH`, `NET_HEIGHT_POST` はいずれも `src/tennis/geometry/court.py` の定数。
       - 実際のプレーエリアはフェンスを含めても `x_n, y_n` のレンジが概ね `[-2, 2]` 程度に収まるスケールとなる。
       - 学習時の損失はこの正規化空間で計算し、評価時や可視化時にメートルスケールへ戻す場合は
         - `x_w = x_n * HALF_DOUBLES_WIDTH`
         - `y_w = y_n * HALF_LENGTH`
         - `z_w = z_n * NET_HEIGHT_POST`
         で逆変換する。
2. **Existence Head**
   - 時間方向に平均 pooling: `exist_feat = decoder_out_time.mean(dim=2)` → `[B, Q, D]`
   - `exist_head: D → 1` を通し、`exist_conf = sigmoid(exist_logit)` → `[B, Q, 1]`
   - 推論時には `exist_conf < 0.5` の Query を「空スロット」と見なして無視。

### 5.6 PyTorch Lightning での学習

#### 5.6.1 LightningModule（概略）

```python
class TennisDetrModule(pl.LightningModule):
    def __init__(self, model_cfg, optim_cfg):
        super().__init__()
        self.model = TennisDETR(model_cfg)
        self.save_hyperparameters()

    def forward(self, batch):
        return self.model(
            player_kpts_2d=batch["keypoints_2d"],
            player_mask=batch["player_mask"],
            court_kpts_2d=batch["court_2d"],
        )
```

- `training_step` では、**Hungarian matching によるセットベース損失**を用いて Query と GT プレーヤーの対応付けを行ったうえで、3D ポーズ L1 / 存在 BCE / 速度正則化を計算する。
- `configure_optimizers` では AdamW とし、**linear warmup 付き Cosine スケジューラ**を設定する（詳細は後述）。

#### 5.6.2 セットベース損失（Hungarian Matching）

- 目的:
  - 各バッチにおいて、**Query スロット `Q` 個と GT プレーヤー `M` 人を 1 対 1 で最適対応**させる。
  - プレーヤー数が変動しても、Query の解釈が安定し、学習が進みやすくなる（DETR と同様の set-based training）。
- 入出力テンソル:
  - 予測:
    - `pose_pred: [B, Q, T, J, 3]` （TennisDETR の出力）
    - `exist_logit: [B, Q, 1]`
  - GT:
    - `pose_3d_gt: [B, T, M, J, 3]`（DataLoader が返す）
    - `exist_3d_gt: [B, T, M]`（True = そのフレームで存在）
- マッチング対象の GT プレーヤー:
  - 時間方向に OR を取って「そのウィンドウ内で一度でも出現したプレーヤー」を対象とする:
    - `exist_any: [B, M] = exist_3d_gt.any(dim=1)`
  - `exist_any[b, m] == False` のプレーヤーはマッチング対象外（完全な背景）として扱う。
- コスト行列の定義（バッチごとに計算）:
  - まず `(b, q, m)` ごとの 3D L1 コストを計算:
    - `pose_cost[b, q, m] = mean_{t,j} |pose_pred[b,q,t,j,:] - pose_3d_gt[b,t,m,j,:]|`
  - 存在ロジットもコストに含める:
    - Query `q` がプレーヤー `m` に割り当てられる場合の存在コスト:
      - `exist_cost[b, q, m] = BCEWithLogits(exist_logit[b,q,0], target=1.0)`
  - 総コスト:
    - `total_cost[b, q, m] = λ_pose_match * pose_cost[b,q,m] + λ_exist_match * exist_cost[b,q,m]`
    - `λ_pose_match` / `λ_exist_match` は `training.loss` に別途ハイパーパラメータとして定義（例: 1.0 / 1.0）。
- Hungarian matching:
  - 各バッチ `b` について `total_cost[b]`（形状 `[Q, M]`）を CPU 上に移し、Hungarian アルゴリズム（`scipy.optimize.linear_sum_assignment` もしくは PyTorch 実装）で最小コストの 1 対 1 対応を求める。
  - 結果は `assignment[b]` として
    - `matched_queries[b, k]`, `matched_targets[b, k]`
    - `k=0..K-1`（`K <= min(Q, M)`）を返す。
- 最終的な損失計算:
  - 3D ポーズ損失:
    - マッチしたペアに対してのみ L1 を計算:
      - `pose_loss = mean_k mean_{t,j} |pose_pred[b, q_k, t, j, :] - pose_3d_gt[b, m_k, t, j, :]|`
    - 未割り当て Query にはポーズ損失をかけない（空スロット扱い）。
  - 存在損失:
    - マッチした Query にはターゲット `1.0`、未割り当て Query にはターゲット `0.0` を割り当て:
      - `exist_target[b, q, 0] = 1.0` if `q` is in `matched_queries[b]` else `0.0`
    - `exist_loss = BCEWithLogits(exist_logit, exist_target)` をバッチ平均で計算。
  - 速度正則化（任意）:
    - 既存と同様、`vel = pose_pred[:, :, 1:] - pose_pred[:, :, :-1]` として L2 を計算。
  - 合成損失:
    - `total = λ_pose * pose_loss + λ_exist * exist_loss + λ_vel * vel_loss`
    - ここで `λ_pose, λ_exist, λ_vel` は `training.loss` にて設定。

実装上は、既存の `_compute_loss` 内で「先頭 M 人を先頭 Q Query に単純対応」させていた部分を削除し、上記のマッチング処理に置き換える。

#### 5.6.3 学習率スケジューラ（Linear Warmup + Cosine）

- 目的:
  - Transformer 系モデルで学習初期の発散を防ぐため、**学習率を徐々に立ち上げる warmup** を導入し、その後 **Cosine decay** で滑らかに減衰させる。
- コンフィグ設計（例: `configs/training/tennis_mvpose.yaml`）:
  ```yaml
  training:
    max_steps: 7000
    optimizer:
      lr: 1.0e-4
      weight_decay: 1.0e-4
    scheduler:
      name: cosine_with_warmup
      warmup_steps: 500        # 線形に 0 → base_lr へ
      min_lr_ratio: 0.1        # 終了時の lr = base_lr * 0.1
  ```
- `TennisDetrModule.configure_optimizers` の挙動:
  1. Optimizer は従来通り AdamW を使用。
  2. `training.max_steps` と `training.scheduler.*` からスケジューラ設定を読み取り、`LambdaLR` 等で以下のスケジュールを構成:
     - ステップ `s` に対して
       - `s < warmup_steps`: `lr_scale = s / warmup_steps`
       - `s >= warmup_steps`:
         - `progress = (s - warmup_steps) / max(1, max_steps - warmup_steps)`
         - `cos = 0.5 * (1 + cos(pi * progress))`
         - `lr_scale = min_lr_ratio + (1 - min_lr_ratio) * cos`
     - 実際の学習率は `lr = base_lr * lr_scale`。
  3. Lightning には `"interval": "step"` で登録し、1 ステップ毎に更新。

これにより、初期は安定した学習を行いつつ、後半は CosineAnnealingLR と同等の挙動で徐々に学習率を減衰させる。

#### 5.6.4 学習戦略

- バッチあたりのメモリ制約を考慮し、
  - `window_T`, `max_cameras`, `max_players` を適切に設定。
  - 勾配累積で見かけのバッチサイズを拡大。
- Mixed precision (`precision=16-mixed`) と勾配クリップ（例: 1.0）を有効化。
- 検証指標:
  - MPJPE / P-MPJPE（3D joints）
  - ラケット tip error
  - Query レベルの existence 精度。

---

## 6. 学習環境の構築

Tennis Pose システムの学習環境は、既存の SceneModel 向けインフラを再利用しつつ、以下のコンポーネントで構成する。

### 6.1 コンフィグ構成
- トップレベル YAML: `configs/tennis_multi_cam_3d_pose.yaml`
  - 役割: `task=tennis_multi_cam_3d_pose` を指定し、dataset/model/training サブ設定をインクルードするハブ。
  - 例:
    ```yaml
    task: tennis_multi_cam_3d_pose
    experiment_name: tennis_mvpose_dev

    includes:
      dataset: configs/datasets/tennis_multi_cam_3d_pose_sim.yaml
      model: configs/models/tennis_mvpose.yaml
      training: configs/training/tennis_mvpose.yaml
      logging: configs/logging/tennis_mvpose.yaml
    ```
- データセット設定: `configs/datasets/tennis_multi_cam_3d_pose_sim.yaml`
  - 主なキー:
    - `root`, `name`（`build_tennis_dataset.py` で作った dataset_name）
    - `window_T`, `max_cameras`, `max_players`, `num_joints`
    - `loader.train/val/test.{batch_size,num_workers,pin_memory,...}`
  - 例:
    ```yaml
    root: data/tennis_autogen
    name: sim_fps60_dur3p0_C4_P1-20_T10
    window_T: 10
    max_cameras: 4
    max_players: 20
    num_joints: 20

    loader:
      train:
        batch_size: 4
        num_workers: 8
        shuffle: true
      val:
        batch_size: 2
        num_workers: 4
        shuffle: false
    ```
- モデル設定: `configs/models/tennis_mvpose.yaml`
  - `TennisDetrConfig` に対応するフィールドを定義（`D_model`, `encoder_layers`, `decoder_layers`, `num_queries` など）。
- トレーニング設定: `configs/training/tennis_mvpose.yaml`
  - Lightning Trainer 用設定（`max_epochs`, `accelerator`, `devices`, `precision`, `gradient_clip_val` など）。

### 6.2 CLI エントリと ConfigLoader

- CLI: `src/cli/tennis_multi_cam_3d_pose/train.py`
  - すでに P0 スカフォールドがあり、`--config` と `--set` で YAML を読み込む。
  - P1 以降では、SceneModel と同様に:
    1. `load_cfg` で DictConfig を取得。
    2. `ConfigLoader(cfg)` を使って:
       - `build_datamodule()` → `TennisPoseDataModule`
       - `build_lit_module()` → `TennisDetrModule`（LightningModule）
       - `build_trainer()` → `Trainer`
    3. `trainer.fit(module, datamodule=dm)` を実行。
- `ConfigLoader.build_datamodule` / `build_lit_module`
  - `task == "tennis_multi_cam_3d_pose"` の分岐で、Tennis 用の DataModule / LightningModule を返すように拡張する。

### 6.3 依存関係と環境準備

- Python パッケージ
  - 既存プロジェクトの環境に加え、以下が必須:
    - `lightning` または `pytorch_lightning`
    - `torch`（GPU 利用時は CUDA 対応ビルド）
    - `ezc3d`（シミュレータで 3DTennisDS を読むため）
    - ロギング用: `tensorboard` / `wandb`（任意）
- 推奨セットアップ手順（例）:
  1. 仮想環境作成（`uv`, `venv`, `conda` など）。
  2. `uv pip install -r requirements.txt` または `pip install -e .` でローカルインストール。
  3. `uv pip install ezc3d lightning tensorboard` 等で追加依存を導入。
- GPU/マルチ GPU
  - Trainer 側で `accelerator=gpu`, `devices=N` を設定。
  - DDP を利用する場合でも、DataModule / Dataset は現設計のままで対応可能。

### 6.3.1 memmap / npz 前処理による高速化

現状の `TennisSceneWindowDataset` は、各バッチでシーン JSON をロード・パースし、Python ループで `[T,V,M,J,2]` テンソルを構築しているため、データローディングがボトルネックになりやすい。これを改善するために、**コンフィグで切り替え可能な memmap ベースの前処理パス**を導入する。

- フラグ: `dataset.use_memmap: bool`
  - `false`（既定）: 現行通り JSON からオンザフライでテンソルを構築。
  - `true` : 事前に npz/memmap に変換された中間フォーマットから読み込む。
- npz レイアウト（1 シーンあたり）:
  - `keypoints_2d: float32[T, V, M, J, 2]`
  - `player_mask: bool[T, V, M]`
  - `court_2d: float32[V, 20, 2]`
  - `pose_3d_gt: float32[T, M, J, 3]`
  - `exist_3d_gt: bool[T, M]`
  - 必要ならメタデータ（`fps`, `scene_id` など）も含める。
- 読み込み:
  - `np.load(path, mmap_mode="r")` で各シーンファイルを開き、`__getitem__` では単に `[t_start:t_end]` をスライスして `torch.from_numpy` するだけにする。
  - これにより、学習中は JSON パースや Python のネスト dict 操作を完全に回避できる。

このフラグは `configs/datasets/tennis_pose_sim.yaml` に追加し、memmap 前処理を終えた環境では `use_memmap: true` に切り替えるだけで高速パスに乗る設計とする。

### 6.4 実行フローとコマンド例

1. **データセット生成**
   - 例:
     ```bash
     python src/cli/build_tennis_dataset.py \
       --dataset_root data/tennis_autogen \
       --num_scenes_train 500 --num_scenes_val 100 --num_scenes_test 100 \
       --fps 60 --duration 3.0 --num_cameras 4 \
       --min_players 1 --max_players 20 \
       --window_T 10 --window_stride 5 \
       --seed 1234
     ```
   - 生成された `dataset_name` を `configs/datasets/tennis_pose_sim.yaml` の `name` に設定。
2. **学習実行**
   - 例:
     ```bash
     python src/cli/train_tennis_pose.py \
       --config configs/tennis_pose.yaml \
       --set training.trainer.max_epochs=50
     ```
   - 必要に応じて `--set dataset.name=... training.trainer.devices=2` などでオーバーライド。
3. **再現実験**
   - 同じ `dataset_root` / `dataset_name` / `meta.json` と YAML を用いれば、別環境でもほぼ同条件で再学習可能。

### 6.5 ロギング・チェックポイント・監視

  - ロギング:
  - 既存の SceneModel と同様、`configs/logging/*.yaml` を利用し、
    - `TensorBoardLogger` で loss/メトリクス・学習率を記録。
    - 必要であれば `WandbLogger` に切り替え可能。
  - **GT / Pred レンダリングの保存**:
    - `validation_step` 内で、少数バッチについて 2D / 3D の可視化を行う。
      - 入力 2D キーポイント（カメラごと）の scatter/render。
      - **カメラ内部・外部パラメータを用いた 3D→2D 再投影**による GT / Pred オーバーレイ。

#### 6.5.1 カメラパラメータを用いたデバッグレンダリング設計

- 目的:
  - 予測 3D（正規化座標系）をシミュレーション時と同じカメラ幾何で 2D に再投影し、**GT の 2D 検出とのズレを視覚的に確認**できるようにする。
  - これにより、「3D が正しいのに可視化だけおかしい」「そもそも 3D が崩れている」といったケースを切り分けやすくする。

- カメラ内部・外部パラメータの取得:
  - シミュレータ側では、`src/tennis/geometry/court.py` の `Camera` dataclass と `make_look_at_camera` を用いて
    - カメラ中心 `C: (3,)`
    - 回転行列 `R: (3,3)`（world→camera）
    - 焦点距離 `f`, 画素中心 `cx, cy`, 画像サイズ `w, h`
    を構築し、`project_points(cam, xyz)` で 3D→2D 投影を行っている。
  - データセット構築時（`build_tennis_dataset.py` / memmap 前処理）で、各カメラについてこれらのパラメータを JSON / npz に書き出し、`TennisSceneWindowDataset` が
    - `camera_C: float32[V, 3]`
    - `camera_R: float32[V, 3, 3]`
    - `camera_intr: float32[V, 3]`（例: `[f, cx, cy]`）
    - `image_size: int32[V, 2]`
    を `batch` に含められるように拡張する。

- 3D GT / Pred の正規化と逆変換:
  - 学習時には 3D を正規化コート座標系 `(x_n, y_n, z_n)` で扱う（5.5 節参照）。
  - デバッグ投影時には、まず正規化座標から世界座標 `(x_w, y_w, z_w)` へ戻す:
    - `x_w = x_n * HALF_DOUBLES_WIDTH`
    - `y_w = y_n * HALF_LENGTH`
    - `z_w = z_n * NET_HEIGHT_POST`
  - これにより、シミュレータが使っているものと同一の座標系に戻る。

- 3D→2D 再投影パイプライン:
  1. LightningModule の `_render_debug_images`（または専用フック）で、
     - `pose_3d_gt_norm: [B, T, M, J, 3]`（正規化）
     - `pose_3d_pred_norm: [B, Q, T, J, 3]`（正規化）
     を受け取る。
  2. 上記を世界座標に逆変換して
     - `pose_3d_gt_world`, `pose_3d_pred_world`
     を得る。
  3. 任意の `(b, t, v)` を選び、プレーヤーごとに `project_points(cam_v, xyz_world)` を適用:
     - `xyz_world: [N, 3]`（J=20 なら `N=20`）
     - 返り値 `uv_px: [N, 2]` を画像ピクセル座標として利用。
  4. 2D GT（`keypoints_2d`）も、[-1,1] → ピクセルへの逆変換で座標を復元:
     - `u_px = (u_norm + 1) * 0.5 * (w - 1)`
     - `v_px = (v_norm + 1) * 0.5 * (h - 1)`
  5. 共通のキャンバス（`H_vis x W_vis`）に
     - コート 2D（`court_2d` を同様にピクセル復元）
     - GT 2D ポーズ（color A）
     - Pred 2D（再投影）ポーズ（color B）
     を描画する。

- 実装構成:
  - 新しい汎用レンダリングモジュールを `src/visualize/tennis_render.py` / もしくは `src/visualize/tennis_pose_render.py` に切り出す。
    - 例:
      - `render_pose2d_frame(width, height, court_points, court_visibility, player_poses, ...)`
      - `project_and_render_3d_pose(cam, pose_3d_world, image_size, ...)`
  - `src/visualize/tennis_pose.py` はこのモジュールを利用する薄いラッパとし、
  - LightningModule (`TennisDetrModule`) の `_render_debug_images` からも同じレンダリング関数を呼び出すように統一する。

- TensorBoard への保存方針:
  - GT と Pred を**別タグ**で保存して比較しやすくする:
    - `"val/pose2d_gt"`: 再投影なしの純粋な GT 2D overlay。
    - `"val/pose2d_pred_reproj"`: Pred 3D をカメラ幾何で再投影した overlay。
  - 必要であれば、`make_comparison_grid` で GT/Pred 並べた画像を `"val/pose2d_gt_vs_pred_reproj"` として追加する。
      - カメラ視点の 2D オーバーレイ（court + player + racket）についても同様に `add_image` で保存。
- チェックポイント:
  - `ModelCheckpoint` コールバックを利用し、`val/Mpjpe` 等の指標でベストモデルを保存。
  - 保存先は `runs/tennis_multi_cam_3d_pose/<experiment_name>/checkpoints` を想定。
- モニタリング:
  - 学習中に GPU メモリ使用量・データローダーのスループットを確認し、
    - ボトルネックに応じて `num_workers` や `window_T`, `batch_size` を調整。

### 6.6 実験管理と命名規則

- 実験名 (`experiment_name`) は、少なくとも以下を含めるとよい:
  - データセットバージョン（例: `simv1`）
  - ウィンドウ長 / カメラ数（例: `T10C4`）
  - モデルサイズ（例: `D256L6x6`）
  - 例: `tennis_mvpose_simv1_T10C4_D256L6x6`
- `runs/tennis_pose/<experiment_name>/` 以下に:
  - TensorBoard ログ
  - チェックポイント
  - 実験で使用した `config_dump.yaml`（実行時のマージ済み設定）を保存しておくと再現が容易。

---

## 7. 拡張アイデア

1. **実データ混合**
   - `index` に `source` カラムを追加し、`sim` / `real` を混在させる。
   - DataModule 側で sim:real 比率を制御。
2. **ボール・インパクト検出**
   - Tennis-DETR に追加ヘッドを設け、impact クラスやボール 3D を同時に推定。
3. **Camera Drop Augmentation**
   - DataLoader レベルでランダムにカメラを無効化し、少数カメラへのロバスト性を強化。
4. **オンライン生成モード**
   - 大規模学習用に、データセットをフル保存せず「学習中にシーン生成 + 即学習」するモードを追加。ただし基本はオフライン生成を推奨。

5. **memmap 前処理スクリプト**
   - `src/cli/preprocess_tennis_memmap.py` のような CLI を追加し、`build_tennis_dataset.py` が生成した JSON シーン＋index をもとに npz/memmap を一括生成する。
   - 役割:
     1. `index/train_index.jsonl` 等を読み、出現する `scene_path` を列挙。
     2. 各シーン JSON に対して:
        - 現行 `TennisSceneWindowDataset` と同等のロジックで 2D/3D テンソルを numpy 配列として構築。
        - 上記レイアウトで `arrays/<split>/scene_000000.npz` などに保存。
     3. `TennisSceneWindowDataset` は `use_memmap=true` のとき、`scenes/...` ではなく `arrays/...` を読む。
   - これにより、学習ループ中の DataLoader は `np.load(..., mmap_mode="r")` + スライス + `torch.from_numpy` だけで済み、I/O と CPU 負荷の大部分を前処理フェーズに移せる。

この設計に従うことで、シミュレーションから Tennis-DETR による 3D ポーズ推定までの流れを、再現性・拡張性の高い形で一元管理できる。
