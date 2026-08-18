# BLCS Track Query: fixed-width mHC + Hybrid CSWA

Issue: [#753](https://github.com/Motoki0705/tennis-lab/issues/753)

このディレクトリは、`BLCSTrackQueryModel` を fixed-width mHC と hybrid CSWA へ更新するための設計正本である。実装時に判断が分かれた場合は、Issue の Acceptance checklist、本文書、各コンポーネント設計書の順に解釈する。同じ契約を別文書へ複製せず、共通事項は本文書、コンポーネント固有事項は対応する設計書だけで管理する。

## 1. 固定する設計判断

1. observation candidate axis は常に `P = Q = model.num_queries` とする。実候補数は `candidate_mask[B,V,T,Q]` で表し、runtime shape に応じて parameter や module を作り直さない。
2. mHC は DeepSeek の式 `H_res X + H_post^T F(H_pre X)` を、camera candidate stream に適用する。BLCS では `n_hc = P = Q` とする。
3. CSWA は token-level KV compressor と compressed sliding-window attention のみで構成する。lightning indexer、Top-k routing、raw/uncompressed sliding-window KV branch は実装しない。
4. temporal mode は stage index `s` に対して固定する。

   ```text
   s % 4 = 0,1,2: CSWA
   s % 4 = 3:     Global MHA
   ```

   object temporal path と track-query temporal path は、同じ stage で必ず同じ mode を使う。
5. camera path の mHC post へ渡すのは Transformer block の完全な出力ではなく `forward_update()` の更新量である。これにより residual を二重加算しない。
6. backend は construction time に `reference` または `cuda` を明示する。CUDA extension が利用不能なときに reference へ黙って切り替えない。
7. CPU/reference candidate を完成・検証してから CUDA 最適化へ進む。CUDA work package は一度に一つだけ実行する。

## 2. 記号とテンソル契約

| 記号 | 意味 |
|---|---|
| `B` | batch size |
| `V` | camera/view 数 |
| `T` | frame 数 |
| `Q` | persistent track query 数 |
| `P` | camera observation candidate stream 数。常に `P=Q` |
| `D` | model hidden dimension |
| `H` | attention head 数 |
| `Dh` | head dimension |
| `m` | compressor ratio |
| `Tc` | compressed sequence length `ceil(T/m)` |
| `r` | compressed-window radius |
| `Wc` | compressed window width `2r+1` |

主要 state は次の shape を持つ。

```text
camera tokens C_s: [B,V,T,Q,D]
track queries S_s: [B,T,Q,D]
candidate mask:    [B,V,T,Q]
frame mask:        [B,T]
```

padding 値は計算結果へ影響してはならない。mask が false の token は、各コンポーネント境界で zero に戻す。mask と実データが矛盾する入力は model 内で推測せず、data/model-I/O boundary で reject する。

## 3. Stage の定式化

stage `s` の temporal operator を

```text
A_s = CSWA   if s % 4 < 3
A_s = Global if s % 4 = 3
```

とする。1 stage は次の順序でのみ実行する。

```text
C_s [B,V,T,Q,D]
  -> mHC pre: Q streams -> 1 object stream
  -> object temporal A_s over T
  -> mHC post: 1 update -> Q streams + constrained residual mixing
  -> spatial self-attention over Q + VQ at each frame
  -> query temporal A_s over T
  -> (C_{s+1}, S_{s+1})
```

数式では、camera stream `X in R^{Q x D}` に対して

```text
z       = H_pre(X, mask) X
Delta z = TemporalBlock.forward_update(z)
X'      = H_res(X, mask) X + H_post(X, mask)^T Delta z
```

とする。その後、各 `(b,t)` で

```text
U_{b,t} = concat(S_{b,t}, vec_{v,q}(C'_{b,v,t,q}))
U'      = SpatialBlock(U)
```

を計算し、query 部を `[B*Q,T,D]` に変形して stage mode の temporal block を適用する。

最終出力 head は既存契約を維持する。

```text
position:        [B,T,Q,3]
presence_logits: [B,T,Q]
```

## 4. Repository 配置

```text
src/utils/models/components/
├── mhc.py
├── compressor.py
├── cswa.py
├── block.py
└── ops/
    ├── mhc/                       # profiler gate通過時のみCUDA実装
    ├── token_compressor/          # profiler gate通過時のみCUDA実装
    └── compressed_time_local/     # reference + optional CUDA

src/tasks/blcs/
├── data/
├── model_io/
├── configuration.py
├── configs/model/
└── models/
    ├── blcs_track_query_model.py
    └── components/track_query_stage.py
```

`attention.py` は dense MHA/GQA/cross-attention の責務を維持する。CSWA は query length `T` と key/value length `Tc` が異なり、valid-mask 契約も dense self-attention と異なるため `cswa.py` に分離する。ただし `TransformerBlockConfig.attention_type` は construction-time option として `cswa` を追加する。

## 5. Component ownership と統合順序

各 Implementer は自分の production path と対応 test だけを変更する。共有の implementation/preflight/seal artifact は parent または明示された integrator だけが書く。

| 順序 | Component | 正本 | 主 ownership | 依存 |
|---:|---|---|---|---|
| 1 | fixed-width mHC | `01_mhc.md` | `mhc.py`, mHC unit tests | なし |
| 2 | token-level compressor | `02_token_level_compressor.md` | `compressor.py`, compressor tests | なし |
| 3 | gather reference CSWA | `03_reference_cswa.md` | `cswa.py`, `ops/compressed_time_local/reference.py`, tests | 2 |
| 4 | `forward_update()` と block dispatch | `04_forward_update.md` | `block.py`, exports, block tests | 3 の公開API |
| 5 | BLCS model/data/config integration | `05_model_integration.md` | `src/tasks/blcs/**`, integration tests | 1–4 |
| 6 | CUDA optimization packages | `06_cuda_optimization.md` | packageごとの専用ops path | CPU candidate |

production component は上表の順に candidate へ統合する。CUDA package は mHC、compressor、CSWA、integrated benchmark を別 agent ownership とし、GPU execution は必ず直列化する。

## 6. 共通品質契約

- すべての public tensor contract は shape、dtype、device、mask semantics を docstring と test で固定する。
- `reshape` 前後の軸順を暗黙に扱わず、`B,V,T,Q,D` と `B,T,V,Q,D` の変換箇所を限定する。
- all-invalid sample、末尾padding、非contiguous input、mixed valid count を test する。
- mask false の入力値を変更しても valid output が変化しないことを property test する。
- `float32` reference を correctness oracle とし、低精度 backend は dtype 別 tolerance を明示する。
- architecture/config/checkpoint の非互換を silent migration しない。旧 checkpoint は strict load error とする。
- performance のために correctness check を弱めない。custom CUDA kernel は reference parity を通過してから dispatch に登録する。

## 7. Workflow bootstrap、agent orchestration、待機

Issue #753 は `.agents/skills/issue-subagent-workflow` で実行する。candidate fingerprint の基準を `main` に固定し、設計書をcandidate scopeへ含めるため、開始順序を次に固定する。

```text
1. cleanなmainをcheckoutする
2. main上でIssue #753を初期化し、base_revisionをfreezeする
3. 失敗済み.codex/tasks/issue-753が残る場合だけ--refresh-issueで再生成する
4. feat/blcs-track-query-cswaをcheckoutする
5. frozen base_revisionがHEADのancestorでなければfeatureへ取り込む
6. 本ディレクトリの存在を確認してexploration以降へ進む
```

upstream Issue 更新後の回復に `--refresh-issue` を使い、frozen `issue.json`、`issue.md`、`state.toml` を手編集しない。feature branch上で先に初期化してdocs commitをbaseline外へ落とさない。

Draft delivery PR #755 はユーザーが明示的に要求した早期PRであり、Validator PASS前から存在してよい。旧design PR #754はsupersededとしてclosed済みで、workflow packagingには使用しない。PR #755の存在やDraft解除を完了根拠にせず、最終headで `capture-pr`、required checks、`finalize-pr`、最終workflow checkまで実行する。

全 child は `fork_turns="none"` と `spawn-contracts.md` のmandatory terminal-only footerを完全一致で使う。spawn 後は parent が独立作業を終えた時点で、利用可能な最大 timeout の blocking wait を一度だけ行う。`list_agents`、短時間 `wait_agent`、status/log/GPU の反復確認、routine progress request は行わない。

GPU job は `.agents/skills/training-queue/SKILL.md` に従い、全 worktree から repo root の同一 `.training_queue/` を参照する。job 完了待ちも status polling ではなく blocking wait とする。CUDA production editsもpackage単位で直列化し、前packageのterminal handoff・統合・CPU checks・queued GPU evidenceが完了するまで次agentをspawnしない。

## 8. 参考資料

- DeepSeek-AI, *DeepSeek-V4: Towards Highly Efficient Million-Token Context Intelligence*, arXiv:2606.19348: https://arxiv.org/abs/2606.19348
- DeepSeek-V4 official inference implementation: https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash/blob/main/inference/model.py
- Xie et al., *mHC: Manifold-Constrained Hyper-Connections*, arXiv:2512.24880: https://arxiv.org/abs/2512.24880
