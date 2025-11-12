# Decoder 仕様（再帰・時間変形可能）

実装: `src/models/scene_model/decoder.py:1`

## 目的
- 時間コンテキストとパッチ特徴から、クエリ軌跡 `[B,T,Q,D]` を生成する。
- 各時刻で再帰的にクエリを更新し、時間窓内から空間・時間的にサンプリングして集約する。

## モジュール構成
- `QueryMergeLayer(D)`
  - 直前クエリ `q_prev[B,Q,D]` と初期クエリ `q_init[B,Q,D]` のソフトマックス重み付き和。
  - 文脈 `ctx[B,D]` と絶対時間ベクトル `e_abs[B,2K]->Linear-> [B,D]` を加法バイアス。
- `RecurrentTemporalDeformableDecoder(D,Q,L,M,k, ...)`
  - 初期クエリ `init_queries[Q,D]`（学習可能）。
  - `build_kv` で時刻 t を中心に `F=2k+1` の時間窓を構築。
  - `TemporalDeformableDecoderLayer` を `L` 回適用して `q` を更新。
- `TemporalDeformableDecoderLayer`
  - 参照点 `ref_xy = sigmoid(Linear(LN(q))) ∈ [0,1]^2`（グリッド正規化座標）。
  - 相対時間特徴 `rel = Linear(RelTimePE(delta_sec))` を条件に、各 τ（窓内フレーム）ごとに M 個のオフセットを生成。
  - 空間は双線形、時間は線形で三線形サンプリングし、L1 距離に基づく減衰を含めて重み付き和。

## 形状とワークフロー
### build_kv
- 入力: `patch_tokens[B,T,Np,D]`, `t_center:int`
- 内部: `pos2d[1,Np,D]` を加法、`AbsTimePE(tau/fps)->Linear[D]` を加法
- 窓インデックス: `tau ∈ [t-k, t+k]`
- 範囲外 τ は `feats=0`, `mask=True`
- 出力:
  - `kv_feats[B,F,H',W',D]`（`Np=H'*W'` を 2D へ整形）
  - `kv_mask[B,F,H',W']`（範囲外 True）
  - `delta_sec[B,Q,F]`（`(tau-t_center)/fps` を放送）

### 層内更新（擬似コード）
```python
qn = LN(q)  # [B,Q,D]
rel = rel_proj( RelTimePE(delta_sec) )  # [B,Q,F,D]
psi = concat(qn.unsqueeze(2).expand(-1,-1,F,-1), agg_rel)
if offset_mode == 'per_tau':
    agg_rel = rel                     # [B,Q,F,D]
else:
    agg_rel = rel.mean(dim=2, keep=True).expand(-1,-1,F,-1)

delta = delta_mlp(psi).view(B,Q,F,M,3)  # xyΔ, tΔ
logits = alpha_mlp(psi).view(B,Q,F,M)
ref = sigmoid(ref_xy(qn)).unsqueeze(2).expand(-1,-1,F,-1)
coords = clamp(ref.unsqueeze(-2) + tanh(delta[...,:2]), 0, 1)  # [B,Q,F,M,2]
delta_t = tanh(delta[...,2])                                   # [B,Q,F,M]

sampled = sample_trilinear(kv_feats, kv_mask, coords, delta_t) # [B,Q,F,M,D]

lam = softplus(λ)
weights = softmax(logits - lam * abs(delta_sec).unsqueeze(-1), dim=-1)
agg = (weights.unsqueeze(-1) * sampled).sum(dim=-2).sum(dim=2) # [B,Q,D]

out = q + Linear(agg)
out = out + FFN(LN(out))
```

### sample_trilinear
- 時間: 目標 `target = base(=0..F-1) + delta_t` を `floor/ceil` で補間。
- 空間: `torch.grid_sample`（`align_corners=False`, `padding_mode='border'`）。
- マスク: 範囲外フレームは 0、マスクを乗算。

## 例外/制約
- `Number of patches mismatch`（`Np != H'*W'`）→ `ValueError`
- `offset_mode`: 実装は `per_tau`（τごと条件）と `pooled`（平均条件）に対応。

