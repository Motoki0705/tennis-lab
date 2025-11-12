# Temporal Encoder 仕様（RoPE）

実装: `src/models/scene_model/temporal.py:1`

## 目的
- CLS トークン系列 `[B,T,D]` に対し、RoPE 付き自己注意を複数層適用し、時間的文脈を埋め込む。

## 構成
- `TemporalEncoderRoPE(dim, depth, heads)`
  - `TemporalBlock` を `depth` 回スタック。
- `TemporalBlock(dim, heads)`
  - 前正規化（LN）→ QKV 線形 → 形状 `[B,heads,T,head_dim]`。
  - RoPE 適用（cos/sin キャッシュ `_rope_cache(T, head_dim)`）。
  - `scaled_dot_product_attention(q, k, v)` → 出力 `[B,T,D]`。
  - 残差 + MLP（FFN）で更新。

## 形状
- 入力: `x[B,T,D]`
- 出力: `x'[B,T,D]`
- 制約: `D % heads == 0` かつ `head_dim = D/heads` が偶数（RoPE の 2 分割に必要）。

## 擬似コード
```python
normed = LN(x)
q,k,v = Linear(normed).chunk(3,-1)
q = q.view(B,T,H,hd).transpose(1,2)
k = k.view(B,T,H,hd).transpose(1,2)
v = v.view(B,T,H,hd).transpose(1,2)
cos, sin = rope_cache(T, hd)
q = apply_rope(q, cos, sin)
k = apply_rope(k, cos, sin)
attn = SDPA(q,k,v)  # [B,H,T,hd]
attn = attn.transpose(1,2).reshape(B,T,D)
out = x + Linear(attn)
out = out + FFN(LN(out))
```

## 例外
- `D % heads != 0` → `ValueError("dim must be divisible by heads")`
- `head_dim` が奇数 → `ValueError("head_dim must be even for RoPE")`

