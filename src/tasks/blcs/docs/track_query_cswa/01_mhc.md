# Component 1: fixed-width mHC

## 1. Responsibility

`src/utils/models/components/mhc.py` は、固定本数の residual streams を 1 stream へ読み出し、外部 sublayer が生成した更新量を streams へ書き戻す汎用 component とする。attention、BLCS、camera、time の概念は持たない。

BLCS では

```text
num_streams = P = Q = model.num_queries
```

として構築する。input の `P` を見て layer、parameter、buffer を動的生成してはならない。

## 2. Paper との対応

mHC の基本式を

```text
X_next = H_res X + H_post^T F(H_pre X)
```

とする。

```text
X:      [...,P,D]
H_pre:  [...,1,P]
H_res:  [...,P,P]
H_post: [...,1,P]
```

DeepSeek/mHC の residual mixing は doubly stochastic matrix へ射影される。BLCS でも `H_res` の valid submatrix に Sinkhorn iteration を適用する。一方、camera candidate streams は順序付き residual channels ではなく候補集合なので、係数生成器は BLCS 利用時にも安全な permutation-equivariant 形式にする。mHC の constrained pre/residual/post topology は維持し、DeepSeek の flatten-based coefficient generator はそのままコピーしない。

## 3. Public API

推奨 API は次のとおり。

```python
@dataclass(frozen=True, slots=True)
class MHCConfig:
    dim: int
    num_streams: int
    coefficient_dim: int
    sinkhorn_iters: int
    eps: float
    residual_identity_bias: float
    update_scale_init: float


@dataclass(slots=True)
class MHCState:
    residual_mix: Tensor  # [...,P,P]
    post_weights: Tensor  # [...,P,1]
    valid_mask: Tensor    # [...,P]


class ManifoldConstrainedHyperConnection(nn.Module):
    def pre(
        self,
        streams: Tensor,      # [...,P,D]
        valid_mask: Tensor,   # [...,P]
    ) -> tuple[Tensor, MHCState]:
        # projected: [...,1,D]
        ...

    def post(
        self,
        update: Tensor,       # [...,1,D], residual-free update
        residual: Tensor,     # [...,P,D], same tensor used by pre
        state: MHCState,
    ) -> Tensor:
        # [...,P,D]
        ...
```

`pre()` と `post()` を分離する理由は、両者の間に temporal block を挿入するためである。`post()` は `update` を完全な block output として扱わない。caller は必ず `forward_update()` の戻り値を渡す。

`MHCState` に `pre_weights` を保存する必要はない。backward graph は `projected` から保持され、post に必要なのは residual/post maps と mask だけである。実装上必要な tensor が増える場合も、state は tensor-only とし、Python callable や runtime module を格納しない。

## 4. Shape validation

construction time:

- `dim > 0`
- `num_streams > 0`
- `coefficient_dim > 0`
- `sinkhorn_iters > 0`
- `eps > 0` かつ finite
- `residual_identity_bias >= 0` かつ finite
- `update_scale_init` は finite

runtime:

- `streams.ndim >= 2`
- `streams.shape[-2:] == (num_streams, dim)`
- `valid_mask.shape == streams.shape[:-1]`
- `valid_mask.dtype == torch.bool`
- `streams` と `valid_mask` は同一 device
- `post()` の residual shape/device/dtype は `pre()` の contract と一致
- `update.shape == residual.shape[:-2] + (1, dim)`

不正 shape を broadcast で受け入れない。

## 5. Permutation-equivariant coefficient generator

mask 済み stream を

```text
X0_p = m_p X_p
u_p  = phi(RMSNorm(X0_p)) in R^C
u_bar = sum_p m_p u_p / max(sum_p m_p, 1)
```

とする。`phi` は全 stream で共有する小さな MLP である。

read/post logits は共有関数から生成する。

```text
a_p = f_pre([u_p, u_bar])
c_p = f_post([u_p, u_bar])
```

residual logits は共有 pair function とする。

```text
r_pq = <W_q u_p, W_k u_q>/sqrt(C)
       + f_pair([u_p, u_q, u_bar])
       + beta * 1[p=q]
```

この生成方法では stream permutation `Pi` に対し、

```text
H_pre(Pi X)  = H_pre(X) Pi^T
H_res(Pi X)  = Pi H_res(X) Pi^T
H_post(Pi X) = H_post(X) Pi^T
```

が成立する設計となる。候補 slot の数値 index を意味的 ID として学習しない。

## 6. Read map

read weights は valid streams 上の masked softmax とする。

```text
A_p = exp(a_p) m_p / sum_j exp(a_j) m_j
z   = sum_p A_p X0_p
```

shape は `A=[...,1,P]`, `z=[...,1,D]` とする。all-invalid row では `A=0`, `z=0` を明示的に返し、softmax of all `-inf` を実行しない。

## 7. Residual map と masked Sinkhorn

pair validity を

```text
M_pq = m_p and m_q
```

とする。valid submatrix に対して log-domain Sinkhorn を行う。

```text
L^(0) = r
L^(k+1/2)_pq = L^k_pq - logsumexp_j(L^k_pj)
L^(k+1)_pq   = L^(k+1/2)_pq - logsumexp_i(L^(k+1/2)_iq)
B = exp(L^(K))
```

実装では invalid entry を各 reduction から除外する。all-invalid sample と invalid row/column に対して `logsumexp(-inf)` を実行しない。

期待 contract:

```text
B_pq = 0 if not M_pq
sum_q B_pq = 1 for every valid p
sum_p B_pq = 1 for every valid q
```

valid stream が一つだけなら `B=[1]` になる。padding stream の row/column は zero とし、post の最後にも output mask を掛ける。

## 8. Write map と initialization

post weight は

```text
C_p = 2 * sigmoid(c_p) * m_p
```

とし、`C=[...,P,1]` とする。最終更新は

```text
X_next = B X0 + g * C Delta z
```

である。`g` は learnable scalar または per-channel parameter とし、`update_scale_init` で初期化する。v1 の既定値は `0.0` とし、新規 path が初期 step で residual stream を急変させないようにする。

residual logits の動的部分は zero-centered、対角項は `residual_identity_bias` で初期化し、初期 `B` を near-identity にする。`B` 自体を parameter として保持せず、毎 forward で constrained map を計算する。

## 9. Mask semantics

- `streams` は最初に `valid_mask` で zero 化する。
- padding stream の任意値を変更しても projected/update後の valid output は変化してはならない。
- all-invalid element の `pre()` output、`post()` output はすべて zero。
- mask false stream へ gradient が流れないことを検証する。
- `candidate_mask` と visibility のどちらを `valid_mask` に使うかは caller の責務であり、mHC は意味を推測しない。

## 10. Numerical policy

- coefficient logits と Sinkhorn reduction は `float32` で実行する。
- residual/update tensor は入力 dtype に戻す。
- autocast 下でも `NaN`/`Inf` を生成しない。
- `eps` は division と normalization のみに使用し、invalid row を eps で擬似的に valid にしない。
- non-contiguous input を受け入れ、必要箇所だけ `reshape`/`contiguous` を使う。

## 11. Files owned by this Implementer

```text
src/utils/models/components/mhc.py
tests/unit/utils/models/components/test_mhc.py
```

必要なら component package export を追加できるが、`block.py`、`cswa.py`、BLCS model/data/config は変更しない。CUDA code は Component 6 の別 ownership である。

## 12. Required tests

1. construction/runtime validation。
2. shape preservation: 複数 leading dimensions。
3. valid submatrix の row/column sum。
4. all-valid、部分mask、single-valid、all-invalid。
5. padding 値 invariance。
6. stream permutation equivariance。
7. `update_scale=0` と near-identity residual initialization。
8. `pre -> arbitrary delta -> post` の forward/backward finite。
9. `gradcheck` 用の小さい double precision case。
10. state/residual mismatch の fail-fast。
11. state dict round-trip と parameter count が runtime `P` に依存して変化しないこと。

## 13. Completion handoff

Implementer は terminal handoff に、変更 path、public API、実行した focused command、Sinkhorn invariant の結果、既知の numerical risk だけを返す。model integration、CUDA performance、全体 architecture の完了を主張しない。
