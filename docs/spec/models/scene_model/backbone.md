# Backbone 仕様（ViT）

実装: `src/models/scene_model/backbone.py:1`

## 目的
- フレーム列 `[B,T,3,H,W]` をパッチトークン `[B,T,Np,D]` と CLS トークン `[B,T,D]` に変換する。

## 構成
- `PatchEmbed(img_size, patch_size, dim)`
  - Conv2d(k=stride=`patch_size`) でパッチ埋め込み。
  - 出力: `[BT, D, H', W'] -> [BT, Np, D]` に整形。
- `ViTBackbone`
  - 位置埋め込み `pos_embed: [1, Np+1, D]` と学習可能 `cls_token: [1,1,D]`。
  - `TransformerEncoder(depth, heads, d_model=D)`（batch_first）。
  - LayerNorm 後、CLS とパッチを切り出し元の `[B,T,...]` に整形。

## 入出力
- 入力: `frames[B,T,3,H,W]`（H=W=`img_size`）。
- 出力: `patch_tokens[B,T,Np,D]`, `cls_tokens[B,T,D]`（`Np=(H/patch_size)*(W/patch_size)`）。

## 擬似コード
```python
flat = frames.view(B*T, 3, H, W)
patches = PatchEmbed(flat)  # [BT, Np, D]
cls = expand(cls_token, [BT, 1, D])
tokens = (cls + patches_with_pos)  # 加法で pos_embed を付与
encoded = TransformerEncoder(tokens)
encoded = LayerNorm(encoded)
cls_tokens = encoded[:,0].view(B,T,D)
patch_tokens = encoded[:,1:].view(B,T,Np,D)
```

## 例外
- 入力次元が 5 でない → `ValueError("frames must be [B,T,3,H,W]")`
- `img_size % patch_size != 0` → `ValueError`

