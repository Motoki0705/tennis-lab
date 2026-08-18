# Component 4: `forward_update()` and block dispatch

## 1. Responsibility

`src/utils/models/components/block.py` に、Transformer block が計算した更新量だけを返す `forward_update()` を追加する。同時に construction-time attention option として CSWA を統合し、MHA/GQA と CSWA の異なる runtime contract を fail-fast に分岐する。

このコンポーネントは mHC を直接実装せず、BLCS stage も知らない。mHC caller が二重 residual を避けるために利用できる汎用 block API を提供する。

## 2. Residual semantics

既存 block は

```text
A(x) = Attention(RMSNorm(x))
x_attn = x + A(x)
F(x) = FFN(RMSNorm(x_attn))
Block(x) = x_attn + F(x) = x + A(x) + F(x)
```

である。

新 API は

```text
DeltaBlock(x) = A(x) + F(x)
Block(x) = x + DeltaBlock(x)
```

を返す。

重要なのは、`x_attn = x + A(x)` を削除しないことである。FFN は従来どおり attention residual 後の state を入力とする。除去するのは戻り値に含まれる最外側の `x` だけである。

推奨実装は次の形とする。

```python
def forward_update(
    self,
    x: Tensor,
    *,
    freqs_cis: Tensor,
    attn_mask: Tensor | None = None,
    state_valid: Tensor | None = None,
) -> Tensor:
    attn_output = self._attention_forward(
        self.attn_norm(x),
        freqs_cis=freqs_cis,
        attn_mask=attn_mask,
        state_valid=state_valid,
    )
    x_attn = x + attn_output
    ffn_output = self.ffn(self.ffn_norm(x_attn))
    return cast(Tensor, attn_output + ffn_output)


def forward(...) -> Tensor:
    return cast(Tensor, x + self.forward_update(...))
```

既存 `forward()` の演算順序を変えず、数値回帰を最小化する。

## 3. Configuration contract

`TransformerBlockConfig.attention_type` を次へ拡張する。

```python
attention_type: Literal["mha", "gqa", "cswa"]
n_kv_heads: int | None
cswa: CSWAConfig | None
```

validation matrix:

| attention type | `n_kv_heads` | `cswa` |
|---|---:|---|
| `mha` | must be `None` | must be `None` |
| `gqa` | required | must be `None` |
| `cswa` | must be `None` | required |

`CSWAConfig.dim/n_heads/head_dim/rope_dim/attn_dropout` は parent `TransformerBlockConfig` と完全一致しなければならない。重複値の不一致を一方へ黙って合わせない。

v1 の CSWA は MHA head layoutだけをサポートする。`n_kv_heads`、GQA、MLA latent option を混在させない。

## 4. Runtime argument contract

block の public signature は既存 call site を壊さないよう keyword-only を維持する。

```python
def forward(
    self,
    x: Tensor,
    *,
    freqs_cis: Tensor,
    attn_mask: Tensor | None = None,
    state_valid: Tensor | None = None,
) -> Tensor:
    ...
```

attention typeごとの引数は次のとおり。

| type | required | prohibited |
|---|---|---|
| MHA/GQA | `attn_mask` | `state_valid` |
| CSWA | `state_valid` | `attn_mask` |

- MHA/GQA は既存の boundary-prepared boolean keep mask `[N,T,T]` を使う。
- CSWA は raw time validity `[N,T]` を受け取り、compressorとcompressed-window maskを内部で構築する。
- prohibited argumentが渡された場合、`del` や無視をせず `ValueError` とする。
- required argumentが欠けた場合も明示的に失敗する。

同じ contract を `forward_update()` に適用する。

## 5. Internal typing

attention field は protocol または明示 union とする。

```python
self.attn: (
    MultiHeadSelfAttention
    | GroupedQuerySelfAttention
    | CompressedSlidingWindowSelfAttention
)
```

異なる forward signature を `type: ignore` で押し込むのではなく、private `_attention_forward()` で construction-time type/modeを分岐する。runtime tensor値でmodule implementationを切り替えない。

より厳密にする場合、次の private adapterを construction時に作成してもよい。

```text
_DenseSelfAttentionInvocation
_CSWAInvocation
```

ただし新たな抽象化が block より複雑になる場合は enum/type branch を優先する。

## 6. Backward compatibility

- 既存 MHA/GQA block の constructor callは、`cswa=None` を明示する設定更新を除き意味を変えない。
- 既存 `forward(x, freqs_cis=..., attn_mask=...)` は同じshape・演算順序・state dict keyを維持する。
- MHA/GQA parameter名を変更しない。
- `forward_update()` 追加だけで既存 checkpoint keyに差分を生じさせない。
- `forward()` を `x + forward_update()` へ書き換えた結果が既存実装と tolerance 内で一致することを回帰testする。

CSWA blockは新規architectureであり、旧checkpointとの互換性を要求しない。

## 7. Export policy

```text
src/utils/models/components/__init__.py
src/utils/models/__init__.py
```

では、repository-wide public API として必要なものだけをexportする。

推奨:

```text
CSWAConfig
CompressedSlidingWindowSelfAttention
TransformerBlockConfig
TransformerBlock
```

`reference_compressed_time_local_attention` や layout helper は specialized ops package ownershipのままとし、`src.utils.models` rootへexportしない。`MHCConfig`/mHC classのroot exportはComponent 1との統合時に決めるが、不要なpublic surfaceを広げない。

## 8. Files owned by this Implementer

```text
src/utils/models/components/block.py
src/utils/models/components/__init__.py
src/utils/models/__init__.py
tests/unit/utils/models/components/test_block.py
```

既存 block test pathが別名ならそれを更新する。`attention.py`、`mhc.py`、`compressor.py`、`cswa.py`、BLCS filesは変更しない。ただし import/export整合に必要な最小変更は ownership handoffでparentに明示する。

## 9. Required tests

1. 既存式を直接計算した oracle と新 `forward()` の一致。
2. `forward(x) == x + forward_update(x)`。
3. `forward_update()` が `attn_output + ffn_output` であり、`x` を再加算しないこと。
4. FFN入力が `x + attn_output` のままであること。
5. MHA/GQAの既存shape、gradient、state dict key回帰。
6. attention type/config validation matrix。
7. MHA/GQAに`state_valid`を渡した場合のreject。
8. CSWAに`attn_mask`を渡した場合のreject。
9. required argument欠落のreject。
10. CSWA blockの`forward`/`forward_update` shapeとfinite gradient。
11. train/eval dropout semantics。
12. serialization round-trip。
13. mypy上で型無視を追加せずにunion dispatchできること。

## 10. Integration invariant

BLCS camera pathは次を使用する。

```python
projected, mhc_state = mhc.pre(...)
update = object_temporal_block.forward_update(projected, ...)
camera_tokens = mhc.post(update, residual=camera_tokens, state=mhc_state)
```

BLCS track-query temporal pathは通常のblock stateを必要とするため、次を使用する。

```python
queries = query_temporal_block(queries, ...)
```

この違いを block 側で自動推測しない。

## 11. Completion handoff

terminal handoffには、変更API、既存MHA/GQAの最大数値差、`forward=x+update` test、focused commands、公開surfaceの変更を記載する。mHC integrationやBLCS stage完了を主張しない。
