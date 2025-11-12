# シーンモデル 仕様（概要）

本ディレクトリは `src/models/scene_model/` 実装の仕様（Spec）をまとめる。コードを見ずに役割や入出力、ワークフロー、テンソル形状を把握できることを目的とする。

## 目的と範囲
- 単眼動画からボール/プレーヤーの役割・存在確率・3D位置・（任意で）SMPLパラメータを時系列で推定するエンドツーエンド推論器。
- 実装は以下の構成要素で構成される。
  - ViT ベースの画像バックボーン（`backbone.py`）
  - RoPE を用いた時間エンコーダ（`temporal.py`）
  - 絶対/相対時間の位置埋め込み（`positional.py`）
  - 再帰的・時間変形可能デコーダ（`decoder.py`）
  - 予測ヘッド（`head.py`）
  - 構築ユーティリティ（`build.py`）
  - 構成ハイパーパラメータ（`config.py`）

## 入出力（契約）
- 入力: `frames` は Tensor `[B, T, 3, H, W]`、`H=W=img_size`、`float32`。
- 出力: 辞書（PredictionHead に準拠）。代表例
  - `role_logits`: `[B, T, Q, 2]`
  - `exist_conf`: `[B, T, Q, 1]`（sigmoid 済）
  - `ball_xyz`, `player_xyz`: `[B, T, Q, 3]`
  - `smpl`: `[B, T, Q, smpl_param_dim]`（`smpl_param_dim=0` の場合は空テンソル）

## パイプライン（擬似コード）
```python
# 1) 画像エンコード
patch_tokens[B,T,Np,D], cls_tokens[B,T,D] = ViTBackbone(frames)

# 2) 時間エンコード（CLS のみ）
z[B,T,D] = TemporalEncoderRoPE(cls_tokens)

# 3) デコード（逐時 t=0..T-1）
prev = init_queries[Q,D]
for t in range(T):
    if tbptt_detach: prev = detach(prev)
    ctx = Linear(z[:,t]) # [B,D]
    abs_vec = Linear(AbsTimePE(t/fps)) # [B,D]
    q0 = QueryMergeLayer(prev, init_queries, ctx, abs_vec) # [B,Q,D]

    kv_feats[B,F,H',W',D], kv_mask[B,F,H',W'], delta_sec[B,Q,F] = build_kv(patch_tokens, t)

    q = q0
    for _ in range(L):
        q = TemporalDeformableDecoderLayer(
            q, kv_feats, kv_mask, RelTimePE, Linear, delta_sec
        )

    collect q

decoded[B,T,Q,D] = concat_t(collected)
outputs = PredictionHead(decoded)
```

## 形状の定義
- `D`: `SceneConfig.D_model`
- `Q`: クエリ数 `SceneConfig.num_queries`
- `Np`: パッチ数 `(img_size/patch_size)^2`
- `F`: 時間窓長 `2*k+1`
- `H'×W'`: パッチグリッド `img_size/patch_size`

## 主要ファイルの仕様
- 詳細は各モジュールの仕様を参照。
  - config: `docs/spec/models/scene_model/config.md`
  - backbone: `docs/spec/models/scene_model/backbone.md`
  - temporal: `docs/spec/models/scene_model/temporal.md`
  - positional: `docs/spec/models/scene_model/positional.md`
  - decoder: `docs/spec/models/scene_model/decoder.md`
  - head: `docs/spec/models/scene_model/head.md`
  - build: `docs/spec/models/scene_model/build.md`

## エッジケースと例外
- 入力が `[B,T,3,H,W]` でない: `ValueError("frames must be [B,T,3,H,W]")`
- パッチ数不一致: `ValueError("Number of patches mismatch")`
- RoPE の head_dim 奇数: `ValueError("head_dim must be even for RoPE")`

