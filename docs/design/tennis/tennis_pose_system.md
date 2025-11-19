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
     - コート座標系の `(x, y, z)`。pelvis 原点ローカルで学習し、必要なら絶対座標に変換。
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

- `training_step` では、GT 3D（`[B, Q, T, J, 3]`）との L1/SmoothL1 損失に加え、骨長一貫性や速度正則化、`exist_conf` とのクロスエントロピーなどを組み合わせる。
- `configure_optimizers` では AdamW + CosineAnnealingLR 等の標準設定を使用。

#### 5.6.2 学習戦略

- バッチあたりのメモリ制約を考慮し、
  - `window_T`, `max_cameras`, `max_players` を適切に設定。
  - 勾配累積で見かけのバッチサイズを拡大。
- Mixed precision (`precision=16-mixed`) と勾配クリップ（例: 1.0）を有効化。
- 検証指標:
  - MPJPE / P-MPJPE（3D joints）
  - ラケット tip error
  - Query レベルの existence 精度。

---

## 6. 拡張アイデア

1. **実データ混合**
   - `index` に `source` カラムを追加し、`sim` / `real` を混在させる。
   - DataModule 側で sim:real 比率を制御。
2. **ボール・インパクト検出**
   - Tennis-DETR に追加ヘッドを設け、impact クラスやボール 3D を同時に推定。
3. **Camera Drop Augmentation**
   - DataLoader レベルでランダムにカメラを無効化し、少数カメラへのロバスト性を強化。
4. **オンライン生成モード**
   - 大規模学習用に、データセットをフル保存せず「学習中にシーン生成 + 即学習」するモードを追加。ただし基本はオフライン生成を推奨。

この設計に従うことで、シミュレーションから Tennis-DETR による 3D ポーズ推定までの流れを、再現性・拡張性の高い形で一元管理できる。
