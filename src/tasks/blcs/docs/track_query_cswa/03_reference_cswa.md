# Component 3: gather-based reference CSWA

## 1. Responsibility

このコンポーネントは、Component 2 の token-level compressor が生成した compressed K/V に対して、各 query 時刻から固定半径の compressed entries だけを gather し、self-attention を計算する CPU/PyTorch correctness reference を提供する。

責務は次の二層に分ける。

```text
src/utils/models/components/cswa.py
  - Q projection
  - TokenLevelKVCompressor の呼び出し
  - query/compressed-key RoPE
  - output projection
  - backend executor の construction-time 解決

src/utils/models/components/ops/compressed_time_local/
  - query時刻 -> compressed-window index layout
  - gather-based reference attention executor
  - backend resolver
```

`attention.py` は dense MHA/GQA/cross-attention のまま維持する。CSWA は query length `T` と K/V length `Tc` が異なり、`[N,T]` validity から compressed mask を構築するため、dense self-attention と同じ public contract に押し込まない。

## 2. Definition

入力を

```text
X:           [N,T,D]
state_valid: [N,T]
```

とする。compressor ratioを `m`、compressed window radiusを `r` とすると、

```text
Tc = ceil(T / m)
Wc = 2r + 1
center(t) = floor(t / m)
Omega(t) = {center(t)-r, ..., center(t)+r} intersect [0,Tc)
```

である。

head `h` のquery `q_{t,h}` と compressed key/value `kc_{c,h}, vc_{c,h}` に対して

```text
score(t,c,h) = <RoPE(q_{t,h}, t), RoPE(kc_{c,h}, pos_c)> / sqrt(Dh)
alpha(t,c,h) = softmax over c in Omega(t) and compressed_valid[c]
out(t,h) = sum_c alpha(t,c,h) vc_{c,h}
```

を計算する。参照先は compressed K/V のみである。raw/uncompressed K/V を局所 branch として併用しない。

## 3. Public API

```python
@dataclass(frozen=True, slots=True)
class CSWAConfig:
    dim: int
    n_heads: int
    head_dim: int
    rope_dim: int
    attn_dropout: float
    compression_ratio: int
    window_radius: int
    backend: Literal["reference", "cuda"]


class CompressedSlidingWindowSelfAttention(nn.Module):
    def forward(
        self,
        x: Tensor,             # [N,T,D]
        *,
        freqs_cis: Tensor,     # query positions, broadcastable to [N,T,H,rope_dim/2]
        state_valid: Tensor,   # [N,T], bool
    ) -> Tensor:               # [N,T,D]
        ...
```

内部 executor の contract は attention projection から独立させる。

```python
def reference_compressed_time_local_attention(
    query: Tensor,             # [N,H,T,Dh], RoPE適用済み
    key: Tensor,               # [N,H,Tc,Dh], RoPE適用済み
    value: Tensor,             # [N,H,Tc,Dh]
    *,
    query_valid: Tensor,       # [N,T]
    key_valid: Tensor,         # [N,Tc]
    compression_ratio: int,
    window_radius: int,
    dropout_p: float = 0.0,
    training: bool = False,
) -> Tensor:                   # [N,H,T,Dh]
    ...
```

executor は model dimension、linear projection、compressor object を知らない。

## 4. Projection and RoPE

`cswa.py` は次を保持する。

```text
Wq: D -> H*Dh
TokenLevelKVCompressor: D -> compressed K/V
Wo: H*Dh -> D
RotaryFrequencyComputer for compressed positions
```

query は既存 MHA と同様に

```text
q = Wq(x).view(N,T,H,Dh)
```

とする。query RoPE は caller が渡した `freqs_cis` を使う。compressed key RoPE は compressor の deterministic `positions[Tc]` から `RotaryFrequencyComputer` で生成する。query と compressed key で異なる長さ・位置を持つため、同一 frequency tensor を流用しない。

RoPE は各 head の先頭 `rope_dim` にだけ適用し、残りの `head_dim-rope_dim` はそのまま連結する。この挙動は既存 MHA/GQA と一致させる。

## 5. Window layout

`build_compressed_sliding_window_layout()` は data-independent な次の tensor を返す。

```python
def build_compressed_sliding_window_layout(
    *,
    query_len: int,
    key_len: int,
    compression_ratio: int,
    window_radius: int,
    device: torch.device,
) -> tuple[Tensor, Tensor]:
    # indices: [T,Wc] long, boundary外はclamp済み
    # index_valid: [T,Wc] bool, boundary内だけtrue
```

定義は

```text
centers[t] = floor(t/m)
offsets = [-r, ..., r]
raw_indices[t,w] = centers[t] + offsets[w]
index_valid = 0 <= raw_indices < Tc
indices = clamp(raw_indices, 0, Tc-1)
```

である。`query_len`, `key_len`, `m`, `r` の組み合わせが矛盾し、`key_len != ceil(query_len/m)` の場合は reject する。layout cache を導入する場合は `(T,Tc,m,r,device)` の immutable tensorだけを保持し、inputやautograd graphを保持しない。

## 6. Gather-based reference execution

K/V を window layout で gather する。

```text
gathered_key:   [N,H,T,Wc,Dh]
gathered_value: [N,H,T,Wc,Dh]
```

key keep mask は

```text
window_keep[n,t,w]
  = index_valid[t,w]
    and key_valid[n, indices[t,w]]
    and query_valid[n,t]
```

とする。

reference executor は次のどちらかで実装できるが、外部 contract は固定する。

1. `[N*H*T,1,Dh]` query と `[N*H*T,Wc,Dh]` gathered K/V を `scaled_dot_product_attention` に渡す。
2. `einsum -> masked softmax -> einsum` の明示実装。

まず PyTorch SDPA 版を production reference とし、明示実装を test oracle に使う。`Wc` だけに対してattentionするため、dense `[T,Tc]` score/maskを production pathで生成しない。

## 7. Empty-row policy

- invalid query の output は zero。
- valid query については、その query 自身を含む current compressed blockが validになるため、少なくとも1 keyが存在することを compressorとの結合 invariant とする。
- それでも valid query に key がない入力は contract violation として `RuntimeError` を送出し、global keyへのsilent fallbackをしない。
- all-invalid sample は output zero。SDPA に all-false row を渡す前に分離または safe row を作り、最後にquery maskでzero化する。

この policy により、局所windowが空のときだけglobal attentionへ切り替わるような隠れたarchitecture変更を防ぐ。

## 8. Dropout semantics

- `dropout_p` は attention probability にだけ適用する。
- evaluation 時は常に `0.0`。
- training 時のreference/CUDA比較では同じ乱数列への依存を避けるため、parity testは原則 `dropout_p=0` とする。
- dropout behavior自体は seed固定で再現可能性とshape/finiteを別testする。

## 9. Complexity

```text
compressor: O(N*T*D*H*Dh) を含むlinear projectionと O(N*Tc*2m*H*Dh) reduction
attention:  O(N*H*T*Wc*Dh)
working set: O(N*H*T*Wc*Dh)
```

`Wc` は `T` に依存しない固定値であり、dense global attention の `O(T^2)` scoreをCSWA stageでは生成しない。ただし reference gather tensorは明示的な `T*Wc` working setを持つ。Component 6で fused CUDA化する際は、このmaterializationを削減対象とする。

## 10. Validation

construction:

- `dim == n_heads * head_dim`
- `rope_dim > 0`, even, `rope_dim <= head_dim`
- `0 <= attn_dropout < 1`
- `compression_ratio >= 2`
- `window_radius >= 0`
- backend は `reference` または `cuda` のみ

runtime:

- `x=[N,T,D]`, floating, `T>0`
- `state_valid=[N,T]`, bool, same device
- `freqs_cis` は query length `T` と rope dimensionを満たす
- executorのQ/K/Vは同一batch/head/head-dim/dtype/device
- `Tc=ceil(T/m)`

CUDA backend指定時にextensionが利用不能ならconstructionまたはfirst callで明示的に失敗し、referenceへfallbackしない。

## 11. Files owned by this Implementer

```text
src/utils/models/components/cswa.py
src/utils/models/components/ops/compressed_time_local/__init__.py
src/utils/models/components/ops/compressed_time_local/api.py
src/utils/models/components/ops/compressed_time_local/layout.py
src/utils/models/components/ops/compressed_time_local/reference.py
tests/unit/utils/models/components/test_cswa.py
tests/unit/utils/models/components/ops/compressed_time_local/test_layout.py
tests/unit/utils/models/components/ops/compressed_time_local/test_reference.py
```

Component 2の`compressor.py`を利用するが変更しない。`block.py`、BLCS files、CUDA/C++ filesは変更しない。

## 12. Required tests

1. layout oracle: `T<m`, exact multiple, partial tail, `r=0`, window larger than `Tc`。
2. gathered index/boundary mask oracle。
3. explicit dense masked-attention oracleとのforward一致。
4. dense oracleとのinput/parameter backward一致。
5. padding key value invariance。
6. invalid query zero、all-invalid zero、valid-query/no-key reject。
7. first/middle/last query window。
8. non-contiguous Q/K/Vとmodel input。
9. float32、autocast低精度のfinite。
10. small double `gradcheck`。
11. dropout offの決定性、dropout onのshape/finite。
12. query/compressed-key RoPE positionが別系列であること。
13. production pathがdense `[T,Tc]` tensorを作らないことをmock/profileで確認。
14. backend resolverがunknown backendとmissing CUDA extensionをfail-fastすること。

## 13. Completion handoff

terminal handoffには、公開API、dense oracleとの最大誤差、forward/backward結果、focused test command、reference working-set riskを含める。block/model integrationやCUDA性能を完了扱いにしない。
