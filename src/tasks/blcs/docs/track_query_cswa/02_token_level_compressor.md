# Component 2: token-level KV compressor

## 1. Responsibility

`src/utils/models/components/compressor.py` は、正規化済み temporal tokens `X[N,T,D]` から、短い key/value sequence `Kc,Vc[N,H,Tc,Dh]` を生成する。query projection、attention score、sliding-window gather、output projectionは担当しない。

DeepSeek-V4 の token-level compressor を参考に、学習可能な channel-wise gated pooling と previous/current block overlap を採用する。ただし DeepSeek-V4 の MLA latent をそのまま前提にせず、BLCS の MHA key/value を連結した固定幅 latent として圧縮する。

## 2. Non-goals

- lightning indexer
- Top-k compressed entry selection
- raw/uncompressed KV branch
- decode cache または autoregressive incremental state
- runtime `T` ごとの parameter 生成
- GQA support。v1 は `n_kv_heads = n_heads` の MHA contract に限定する

## 3. Public API

```python
@dataclass(frozen=True, slots=True)
class TokenLevelCompressorConfig:
    dim: int
    n_heads: int
    head_dim: int
    compression_ratio: int
    overlap: bool


@dataclass(slots=True)
class CompressedKV:
    key: Tensor              # [N,H,Tc,Dh]
    value: Tensor            # [N,H,Tc,Dh]
    state_valid: Tensor      # [N,Tc], bool
    positions: Tensor        # [Tc], float32


class TokenLevelKVCompressor(nn.Module):
    def forward(
        self,
        x: Tensor,           # [N,T,D]
        state_valid: Tensor, # [N,T], bool
    ) -> CompressedKV:
        ...
```

`x` は CSWA attention norm 後の token とする。compressor 内で再度 model-level norm を加えない。gating と weighted reduction の numerical stability のため、内部 accumulation は `float32` とする。

## 4. Static parameterization

```text
m      = compression_ratio
KVDim  = 2 * H * Dh
branches = 2 when overlap=True
```

v1 は DeepSeek-V4 の ratio-4 overlap path を一般化し、`overlap=True` を必須とする。`m >= 2` を validation する。

parameter shapes:

```text
W_kv:   D -> branches * KVDim
W_gate: D -> branches * KVDim
APE:    [m, branches * KVDim]
```

`W_kv` は key/value latent、`W_gate` は channel-wise pooling logits を生成する。`APE` は block 内 offset ごとの learned bias で、同一 block 内の位置を識別する。parameter shape は `D,H,Dh,m` の construction-time config だけで決まり、runtime `T` には依存しない。

## 5. Compression layout

```text
Tc = ceil(T / m)
I_c = {cm, ..., min((c+1)m-1, T-1)}
I_{c-1} = previous block; c=0ではempty
```

compressed token `c` の source は、previous block の overlap branch と current block の normal branchを連結した最大 `2m` token とする。

```text
source(c) = [(I_{c-1}, branch=0), (I_c, branch=1)]
```

reference implementationは、constructor parameterではない deterministic layout tensorをruntime device上で構築または `(T,m,device)` 単位でcacheする。cacheは tensor shape/layout だけを保持し、input data や autograd graph を保持しない。

末尾 block が `m` 未満でも、source layoutを `2m` 幅へpaddingし、`source_valid` で除外する。tensorを黙ってdropして `floor(T/m)` にしない。

## 6. Gated pooling

入力から

```text
KV_raw = W_kv(X)   -> [N,T,2,KVDim]
G_raw  = W_gate(X) -> [N,T,2,KVDim]
```

を生成する。各 raw frame の block offset `o=t mod m` に対して

```text
G_raw[n,t,b,:] += APE[o,b,:]
```

を加える。

compressed index `c` で source を gather し、channel ごとに source axis上の masked softmaxを行う。

```text
alpha_{c,j,d}
  = softmax_j(Gather(G_raw)_{c,j,d} + source_mask_{c,j})

KV_c[c,d]
  = sum_j alpha_{c,j,d} Gather(KV_raw)_{c,j,d}
```

`j` は最大 `2m` source、`d` は `KVDim` channel である。`KV_c` を

```text
[N,Tc,2,H,Dh] -> key/value [N,H,Tc,Dh]
```

へ分割する。

同じ pooling weightを key と value 全体で共有するのではなく、`KVDim` channelごとに logitsを持つ。これは DeepSeek-V4 compressor の channel-wise gated pooling に対応する。

## 7. Mask semantics

raw validityは

```text
raw_valid[n,t] = state_valid[n,t]
```

である。source validityは raw validity、sequence boundary、previous/current branch boundary の論理積とする。

```text
compressed_valid[n,c] = any_j source_valid[n,c,j]
```

all-invalid compressed rowでは softmaxを呼ばず、`key=value=0`, `compressed_valid=false` とする。valid query `t` の current block `floor(t/m)` は必ずその `t` を含むため、CSWA側では valid query が少なくとも一つの valid local compressed key を持つ。この invariant を両 component の test で固定する。

padding tokenの `x` はprojection前にzero化し、padding値を変更しても valid compressed outputが変化しないことを保証する。

## 8. Compressed positions

v1 は data-dependent weighted centroid を用いず、静的な block centerを使う。

```text
pos_c = min(c*m + (m-1)/2, T-1)
```

`positions` は `[Tc]` の `float32` tensor とする。理由は次のとおり。

- query/key RoPE layoutをinput dataから分離できる
- sampleごとの frequency tensorを不要にできる
- reference/CUDA parityを単純化できる
- `torch.compile` と static scheduling に適する

learned/weighted compressed positionは将来のablationであり、v1の実装範囲外とする。

## 9. Initialization

- `W_kv` は既存 attention projection と同等の標準 linear initializationを用いる。
- `W_gate.weight` は小さい値またはzeroで初期化し、初期 pooling が APE 主導の安定した分布になるようにする。
- `APE` は zero initを基本とし、初期状態を有効sourceのchannel-wise uniform averageにする。
- 特定の source tokenだけへ極端に集中するbiasを初期化時に与えない。

initializationはtestで決定性を固定し、parameter construction時にglobal dtype/deviceを暗黙変更しない。

## 10. Numerical and memory policy

- projection outputはinput dtypeでよいが、gate logits、softmax、weighted sumは`float32`。
- output K/Vはinput dtypeへ戻す。
- explicit expanded copy `[N,Tc,2m,KVDim]` はreferenceの小規模実装では許容するが、production referenceは`gather`とviewを用い、不要なrepeatを避ける。
- dense `[T,T]` または `[T,Tc]` tensorを生成しない。
- non-contiguous `x` とmaskを受け入れる。
- `Tc=ceil(T/m)` と `2m` source width以外のhidden shapeをruntime dataから変更しない。

## 11. Validation

- `dim,n_heads,head_dim > 0`
- `dim` と `n_heads*head_dim` の関係はcaller configと一致すること。v1では `dim == n_heads*head_dim` を要求する
- `compression_ratio >= 2`
- `overlap is True`
- `x.shape == [N,T,dim]`, `T>0`
- `state_valid.shape == [N,T]`, bool, same device
- floating input dtypeのみ

unsupported optionを無視せず、constructorでrejectする。

## 12. Files owned by this Implementer

```text
src/utils/models/components/compressor.py
tests/unit/utils/models/components/test_compressor.py
```

必要なlayout helperは `compressor.py` 内のprivate functionとして開始する。reference CSWAのwindow layoutやCUDA ops directoryは変更しない。public exportはintegratorまたはComponent 4が追加する。

## 13. Required tests

1. constructor/runtime validation。
2. `T` が `m` の倍数、非倍数、`T<m`。
3. first block、middle block、last partial block のsource index oracle。
4. all-valid、sparse-valid、single-valid、all-invalid。
5. padding value invariance。
6. loop-based channel-wise compression oracleとのforward一致。
7. deterministic block positions。
8. key/value splitとhead axis ordering。
9. non-contiguous input。
10. float32 backward、small double `gradcheck`。
11. autocast下のfinite output/gradient。
12. `state_dict` round-trip。
13. raw source countに関係なくparameter countが一定であること。

## 14. Completion handoff

terminal handoffには、public API、layoutの具体例、oracle test結果、実行command、reference memory上の既知riskを記載する。CSWA attention、block dispatch、CUDA speedupは完了扱いにしない。
