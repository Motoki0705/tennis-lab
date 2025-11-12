# Prediction Head 仕様

実装: `src/models/scene_model/head.py:1`

## 目的
- デコーダの埋め込み `[B,T,Q,D]` をタスク別の構造化出力に変換する。

## PredictionHead
- 構成: 各分岐は `LayerNorm(D) -> Linear`
  - `role_logits`: `[B,T,Q,2]`
  - `exist_conf`: `[B,T,Q,1]`（sigmoid）
  - `ball_xyz`: `[B,T,Q,3]`
  - `player_xyz`: `[B,T,Q,3]`
  - `smpl`: `[B,T,Q,smpl_param_dim]`（`smpl_param_dim==0` のとき空テンソル）

## BBoxHead（オプション）
- `build.py` 経由で選択可能な DETR 風ヘッド。
- 出力:
  - `cls_logits`: `[B,T,Q,num_classes]`
  - `exist_conf`: `[B,T,Q,1]`
  - `bbox_center`: `[B,T,Q,2]`（`sigmoid` により 0〜1 正規化）
  - `bbox_size`: `[B,T,Q,2]`（`sigmoid` を `clamp(min=1e-3)` した 0〜1 正規化）

## 例外/備考
- `smpl_param_dim=0` の場合、内部 `_ZeroProjection` が `[...,0]` 形状の空テンソルを返し、以降の shape を壊さない。
