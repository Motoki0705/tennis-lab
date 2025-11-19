# Tennis-DETR モデル仕様（Spec）

本書は、テニス用マルチビュー 3D ポーズ推定モデル `TennisDETR` と、その設定クラス `TennisDetrConfig` の仕様をまとめる。

実装:
- モデル本体: `src/models/tennis/model.py:TennisDETR`
- 設定クラス: `src/models/tennis/config.py:TennisDetrConfig`

---

## 1. 目的

- 入力:
  - マルチカメラ・マルチプレーヤーの 2D キーポイント（pose17 + racket3 = 20 点）:
    - `keypoints_2d[B, T, V, M, J, 2]`
  - プレーヤー存在マスク:
    - `player_mask[B, T, V, M]`（True=観測有り）
  - コート 2D キーポイント:
    - `court_kpts_2d[B, V, 20, 2]`
- 出力:
  - 各 Query（プレーヤースロット）ごとの 3D ポーズ時系列:
    - `pose_3d[B, Q, T, J, 3]`
  - Query の存在確率:
    - `exist_conf[B, Q, 1]`

ここで:
- `B`: バッチサイズ
- `T`: 時間ウィンドウ長
- `V`: カメラ数
- `M`: 1 フレームあたり最大プレーヤー数（パディング含む）
- `J`: キーポイント数（20）
- `Q`: Query 数（最大プレーヤー数スロット、通常は `M` 以上）

---

## 2. 設定クラス: `TennisDetrConfig`

```python
@dataclass(slots=True)
class TennisDetrConfig:
    D_model: int = 256
    dim_feedforward: int = 1024
    nheads: int = 8
    encoder_layers: int = 6
    decoder_layers: int = 6
    dropout: float = 0.1

    num_joints: int = 20
    num_court_points: int = 20
    num_queries: int = 20

    max_cameras: int = 8
    max_frames: int = 32
```

- `max_cameras`, `max_frames` は Embedding テーブルの上限であり、実際の `V`, `T` がこれ以下である必要がある。
- `num_joints=20` は「pose17 + racket3」の合計数。

YAML からの設定例は `configs/models/tennis_mvpose.yaml` を参照。

---

## 3. 入出力と内部処理フロー

### 3.1 `forward(player_kpts_2d, player_mask, court_kpts_2d)`

署名:

```python
def forward(
    self,
    player_kpts_2d: Tensor,  # [B, T, V, M, J, 2]
    player_mask: Tensor,     # [B, T, V, M]
    court_kpts_2d: Tensor,   # [B, V, P, 2]
) -> Mapping[str, Tensor]:
```

戻り値:
- `"pose_3d": Float[B, Q, T, J, 3]`
- `"exist_conf": Float[B, Q, 1]`（Sigmoid で 0〜1）

### 3.2 トークン化

1. 2D キーポイント:
   - `player_kpts_2d` を `[B, T, V, M, J*2]` に reshape。
   - `joint_mlp: Linear(J*2 → D) → GELU → Linear(D → D)` で `joint_embed[B,T,V,M,D]`。
2. コート 2D:
   - `court_kpts_2d` を `[B, V, P*2]` に reshape (`P=20`)。
   - `court_mlp: Linear(P*2 → D) → GELU → Linear(D → D)` で `court_embed[B,V,D]`。
   - `[B,1,V,1,D]` に reshape & broadcast して `[B,T,V,M,D]` に展開。
3. カメラ / 時間埋め込み:
   - `camera_embed: Embedding(max_cameras, D)`
     - `cam_ids = 0..V-1` を `[1,1,V,1,D]` に展開して加算。
   - `time_embed: Embedding(max_frames, D)`
     - `time_ids = 0..T-1` を `[1,T,1,1,D]` に展開して加算。
4. 結果:
   - `tokens[B,T,V,M,D] = joint_embed + court_embed + camera_embed + time_embed`
   - `[B, L, D]` に reshape（`L = T*V*M`）。
   - `player_mask` から `key_padding_mask[B,L]` を生成（無効位置=True）。

### 3.3 Encoder

- 実装: `nn.TransformerEncoder` (`batch_first=True`)。
- 入力: `tokens[B,L,D]`, `src_key_padding_mask[B,L]`。
- 出力: `memory[B,L,D]`。

### 3.4 時間付き Query と Decoder

1. ベース Query:
   - `query_embed: Embedding(num_queries, D)` → `[Q,D]`。
2. 時間付き Query:
   - `query_base[B,Q,T,D] = query_embed[q] + time_embed[t]` を broadcast。
   - `[B, Q*T, D]` に reshape → `queries_flat`。
3. Decoder:
   - 実装: `nn.TransformerDecoder` (`batch_first=True`)。
   - 入力: `tgt=queries_flat`, `memory`, `memory_key_padding_mask=key_padding_mask`。
   - 出力: `decoder_out[B,Q*T,D]` → reshape → `decoder_out_time[B,Q,T,D]`。

### 3.5 出力ヘッド

- 3D ポーズ:

  ```python
  pose_head: LayerNorm(D) → Linear(D → J*3)
  pose_3d = pose_head(decoder_out_time)  # [B,Q,T,J*3] → reshape → [B,Q,T,J,3]
  ```

- 存在確率:
  - `exist_feat = decoder_out_time.mean(dim=2)` → `[B,Q,D]`
  - `exist_head: LayerNorm(D) → Linear(D → 1)`
  - `exist_conf = sigmoid(exist_head(exist_feat))` → `[B,Q,1]`

制約:
- `player_kpts_2d.shape[1] <= max_frames`
- `player_kpts_2d.shape[2] <= max_cameras`
- これを超える場合は `ValueError`。

---

## 4. 学習時の利用（TennisDetrModule との関係）

`TennisDETR` は LightningModule `TennisDetrModule`（`src/training/tennis/lightning.py`）経由で学習される。

- 入力バッチ:
  - `TennisPoseDataModule` が `TennisSceneWindowDataset` の出力をバッチ化した dict。
  - `TennisDetrModule.forward` はこの dict をそのまま `TennisDETR` に渡す。
- 損失計算（概要）:
  - 3D GT: `pose_3d_gt[B,T,M,J,3]`, `exist_3d_gt[B,T,M]` を `pose_3d[B,Q,T,J,3]` / `exist_conf[B,Q,1]` にアライン。
  - 「先頭 M プレーヤー ↔ 先頭 Q Query」の素朴なアサインでマスク付き L1 / BCE を計算（将来的にハンガリアン等に置き換え可能）。

---

## 5. 注意点

- Query 数 `Q` は「同時に扱いたい最大プレーヤー数以上」に設定しておく必要がある。現在の既定設定では、データセット側の `max_players=20` に合わせて `num_queries=20` としている。
- `max_frames`, `max_cameras` は「想定より少し多め」に設定しておき、実際の `T`, `V` はその範囲内に収める。
- 3D 出力はコート座標系であり、シミュレータと同じスケール（メートル）で扱われる。カメラ投影や画像上へのレンダリングは上位の可視化モジュールの責任とする。
