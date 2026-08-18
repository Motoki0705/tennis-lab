# Component 6: serial CUDA optimization work packages

## 1. Entry gate

CUDA workは、Components 1–5 の CPU/reference candidate が次を満たした後にのみ開始する。

```text
- focused unit/integration tests PASS
- reference forward/backward finite
- dense/oracle parity PASS
- ruff PASS
- mypy PASS
- architecture/config contracts frozen
```

correctness未確定のPython実装とcustom kernelを同時に設計しない。CUDA最適化中にreference semanticsを変更する必要が生じた場合は、CUDA workを停止してparentへinterface blockerを返し、reference componentのownerを通して修正する。

## 2. Serial execution rule

CUDAは次の4 work packageへ分割し、**一度に一つのImplementerだけ**をactiveにする。

```text
6A fixed-width mHC profiling / optional CUDA
  -> terminal handoff + integration + GPU evidence
6B token compressor profiling / optional CUDA
  -> terminal handoff + integration + GPU evidence
6C compressed-window attention CUDA
  -> terminal handoff + integration + GPU evidence
6D integrated dispatch and end-to-end benchmark
```

次のagentは、前agentのterminal handoff、candidate統合、focused CPU checks、queued GPU job完了を確認するまでspawnしない。GPU jobだけでなくCUDA production editsも並列化しない。これにより同一ops loader、extension build cache、共有GPU、benchmark環境の競合を防ぐ。

各agentは独立ownershipを持ち、他packageのCUDA pathを変更しない。共通loader変更が必要な場合は6Aで最小の汎用extension registryを先に作るか、parent integratorがownershipを明示的に引き取る。

## 3. Shared GPU execution contract

すべてのGPU test/benchmarkは `.agents/skills/training-queue/SKILL.md` を読み、repo rootの共有queue経由で実行する。

```bash
REPO_ROOT="$(git rev-parse --show-toplevel)"
export TRAINING_QUEUE_DIR="$REPO_ROOT/.training_queue"
QUEUE="$REPO_ROOT/.agents/skills/training-queue/scripts/training_queue.sh"
```

AI agentはskillのprovider別referenceから自分のsession IDを取得し、すべてのjobに

```text
--provider <provider>
--session <session-id>
--issue 753
```

を付ける。worktree内の `.training_queue/` を作らない。queue workerは常に一つとし、GPU commandをqueue外から直接起動しない。

待機中は短時間の `status`、log tail、`nvidia-smi`、process list、epoch/progress、`list_agents` を反復しない。利用可能な最大timeoutのblocking waitまたはforeground workerを使う。timeoutした場合のみ状態を一度確認し、未完了なら再び長時間待機する。

Issue workflowのcanonical GPU checkを実行する場合、queue jobのcommandとして

```text
python manage_issue_task.py run-check <task> <stage> <gpu-check-id>
```

を登録し、candidate-bound result JSONを通常のworkflow authorityで生成する。queue完了ログだけをcanonical verdictの代用にしない。

## 4. Common CUDA API policy

各optimized componentは construction-time backendを持つ。

```text
backend = reference | cuda
```

- `cuda` 指定時にextensionが未build、load失敗、unsupported dtype/device/shapeの場合は明示的に失敗する。
- referenceへのsilent fallbackを行わない。
- auto backendはv1で提供しない。
- referenceとCUDAでpublic tensor shape、mask semantics、dtype/device、empty-row policyを完全一致させる。
- custom autogradを使う場合、forwardでbackwardに必要な最小tensorだけをsaveする。
- C++/CUDA entrypointでshape、dtype、device、contiguity、index rangeをvalidateする。Python validationだけに依存しない。

既存 `src/utils/models/components/ops/time_local/` の loader、bindings、autograd、reference分離をrepository patternとして参照するが、compressed query/key lengthの違いを無理に既存kernelへ混在させない。

## 5. Work package 6A: fixed-width mHC

### 5.1 Ownership

```text
src/utils/models/components/ops/mhc/__init__.py
src/utils/models/components/ops/mhc/api.py
src/utils/models/components/ops/mhc/_autograd.py
src/utils/models/components/ops/mhc/bindings.cpp
src/utils/models/components/ops/mhc/kernels.cu
対応CUDA tests/benchmarks
```

`mhc.py` のreference semanticsは変更しない。

### 5.2 Profile-first gate

BLCSの`Q`は小さいため、kernel launch overheadがPyTorch op fusionの利得を上回る可能性がある。まず以下を個別profileする。

```text
- coefficient MLP
- masked log-Sinkhorn
- residual B @ X
- post outer product C @ Delta z
- combined pre/post
```

`torch.compile`を利用する場合はreference候補として別計測し、eager PyTorch、compiled PyTorch、custom CUDAを同一条件で比較する。

### 5.3 Optional fusion scope

GOの場合のみ、次を候補とする。

```text
- fixed P=Q specialized masked Sinkhorn
- residual matmul + post broadcast add + output mask fusion
- backward scatter/reduction fusion
```

coefficient MLPまで巨大kernelへ融合して保守性を落とさない。`Q`ごとのruntime JIT codegenやmodule再構築を行わず、supported fixed widthsを明示する。

### 5.4 GO/NO-GO

representative BLCS shapeのforward+backwardで、最良PyTorch referenceに対して少なくとも`1.10x` speedupを示し、peak memoryを悪化させず、parityを満たす場合のみCUDA backendを登録する。それ以外はproduction CUDA filesを追加せず、profile JSONとNO-GO理由を残す。NO-GOは失敗ではない。

## 6. Work package 6B: token-level compressor

### 6.1 Ownership

```text
src/utils/models/components/ops/token_compressor/__init__.py
src/utils/models/components/ops/token_compressor/api.py
src/utils/models/components/ops/token_compressor/_autograd.py
src/utils/models/components/ops/token_compressor/bindings.cpp
src/utils/models/components/ops/token_compressor/kernels.cu
対応CUDA tests/benchmarks
```

### 6.2 Optimization target

referenceの主なmaterializationは

```text
KV/G projection
-> previous/current 2m source gather
-> channel-wise masked softmax
-> weighted reduction
```

である。linear projectionはPyTorch GEMMを維持し、CUDA kernelはprojection後のlayout/gate-softmax/reductionを融合することを第一候補とする。

```text
input:
  kv_raw   [N,T,2,KVDim]
  gate_raw [N,T,2,KVDim]
  state_valid [N,T]
output:
  compressed_kv [N,Tc,KVDim]
  compressed_valid [N,Tc]
```

kernelは`m`をconstruction-time supported integerとして扱い、partial tail、first blockのempty previous branch、all-invalid rowをreferenceと同一に処理する。

### 6.3 GO/NO-GO

forwardとbackwardの両方を比較する。少なくともtarget shape 2件で最良referenceより`1.10x`以上、peak temporary memory削減または非増加、全mask case parityを満たす場合に登録する。小さい`T`だけ高速でlong-contextが遅い実装をdefaultにしない。

## 7. Work package 6C: compressed-window attention

### 7.1 Ownership

```text
src/utils/models/components/ops/compressed_time_local/_autograd.py
src/utils/models/components/ops/compressed_time_local/bindings.cpp
src/utils/models/components/ops/compressed_time_local/kernels.cu
src/utils/models/components/ops/compressed_time_local/api.py のCUDA resolver部分
対応CUDA tests/benchmarks
```

reference/layout filesの意味を変更しない。

### 7.2 Optimization target

referenceの

```text
K/V window gather [N,H,T,Wc,Dh]
-> flattened SDPA
```

という明示materializationを削減する。kernelはquery `t` から `center=floor(t/m)` を直接計算し、compressed K/Vの`Wc` entriesだけをloadしてonline softmaxを行う。

```text
Q: [N,H,T,Dh]
K: [N,H,Tc,Dh]
V: [N,H,Tc,Dh]
query_valid: [N,T]
key_valid: [N,Tc]
output: [N,H,T,Dh]
```

`T != Tc` をfirst-class contractとし、既存same-length local-attention kernelをshape hackで再利用しない。boundary clampされたindexをvalidとして誤利用せず、index validityとkey validityを別に適用する。

backwardはQ/K/V gradientを実装し、同じcompressed keyが複数query windowから参照されるためK/V gradientを正しくaccumulateする。必要ならatomic addを使うが、determinism/performance trade-offを記録する。

### 7.3 GO/NO-GO

このpackageは主要optimization candidateである。referenceよりforward、forward+backward、peak memoryの全てを測る。少なくともtarget long-context shapeで`1.20x`以上、OOM threshold改善またはpeak memory削減、dtype別parityを満たすことをdefault CUDA registration条件とする。

## 8. Work package 6D: dispatch and integrated benchmark

### 8.1 Ownership

```text
component backend wiringの最小変更
CUDA integration tests
reproducible benchmark module/result schema
Issue workflow evidence
```

6Dは6A–6Cのkernel内部を変更しない。failureが見つかった場合はaffected packageへfocused failure bundleを返す。

### 8.2 Compared candidates

同一model width、batch、views、frames、queries、dtypeで次を比較する。

```text
A. existing/global-only track-query temporal attention
B. hybrid CSWA reference
C. hybrid CSWA CUDA-enabled
```

architecture AとB/Cはparameter countも記録し、単なるkernel latencyだけでなくmodel stage全体を測る。

### 8.3 Shape matrix

実装時点のsmall/base/large configから実shapeを読み取り、少なくとも次を含める。

```text
- smoke: small N/T for correctness
- configured training shape
- configured inference shape
- long-context T in {512, 1024, 2048} where supported
- object path N=B*V
- query path N=B*Q
- Q fixed to each evaluated model config
- float32 and the production mixed-precision dtype
```

GPU memory上実行不能なshapeを黙って除外せず、OOMをresultとして記録する。

### 8.4 Measurement protocol

- fixed seedと固定mask densityを記録する。
- warm-up後に複数iterationを測る。
- 各iteration前後でCUDA synchronizeする。
- latencyはmedianとp95を記録する。
- forward-onlyとforward+backwardを分離する。
- `torch.cuda.reset_peak_memory_stats()` 後のpeak allocated/reservedを記録する。
- throughputはframes/sまたはtokens/sで定義を固定する。
- reference/CUDAで同じautocast、compile、dropout設定を使う。
- benchmark中に他GPU jobを実行しない。

resultはmachine-readable JSONに、git commit、candidate fingerprint、GPU名、PyTorch/CUDA version、shape、dtype、backend、warmup、iterations、latency、throughput、memory、parityを保存する。

## 9. Correctness tolerances

最終値はdtypeとkernel reduction orderを踏まえてtestで明示する。初期基準:

```text
float32 forward:  atol=1e-5, rtol=1e-4
float32 backward: atol=2e-5, rtol=2e-4
float16/bfloat16: component別にreference誤差分布から設定
```

低精度で大きな固定toleranceを先に許可しない。random、boundary-heavy、sparse-mask、all-invalid casesの最大/平均誤差を記録してから設定する。NaN/Infはtoleranceに関係なくFAIL。

## 10. Per-package required evidence

各CUDA agentのterminal handoffは次だけを含む。

```text
- ownership paths
- GO or NO-GO
- reference/CUDA forward and backward parity
- exact queued GPU job IDs and outcomes
- benchmark JSON path and headline metrics
- supported dtype/shape restrictions
- unresolved correctness/performance risks
```

routine progress、compile中、percentage、予想完了時刻をparentへ送らない。parentはterminal handoffまで長時間waitし、次packageをspawnしない。

## 11. Final acceptance

CUDA phaseの完了条件は「全componentにkernelが存在すること」ではない。次を満たすことを完了とする。

```text
- 6A–6Cそれぞれに独立したprofile/parity/GO-NO-GO evidenceがある
- 採用kernelはreference parityとspeed gateを通過する
- 不採用kernelはproduction dispatchに残らない
- backend指定はexplicitでsilent fallbackがない
- GPU executionがtraining queueで直列化されている
- integrated benchmarkがglobal-only、hybrid reference、hybrid CUDAを比較している
- final candidateのcanonical CPU/GPU checksとIssue validationがPASSする
```
