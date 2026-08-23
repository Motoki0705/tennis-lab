# Compressed time-local attention benchmark results

This is the canonical retained performance record for the compressed
time-local attention operator. The BLCS-specific benchmark drivers and raw JSON
artifacts used to produce it were removed on 2026-08-21; no runnable benchmark
is shipped with the production package.

## Recorded environment and protocol

- Recorded: 2026-08-19 for GitHub Issue #753
- GPU: NVIDIA GeForce RTX 5060 Ti, compute capability 12.0
- Software: Python 3.11.15, PyTorch 2.11.0+cu130, CUDA 13.0
- Shape: `N=16, H=4, T=2048, Tc=512, Dh=64`, compression ratio 4,
  window radius 4, valid density 0.875
- Timing: 2 warmups and 5 synchronized iterations; medians below
- Candidates: gathered SDPA reference and fused CUDA online softmax

## Results

| dtype | direction | candidate | median / p95 (ms) | peak allocated / reserved (bytes) | speedup |
|---|---|---|---:|---:|---:|
| float32 | forward | reference | 6.599 / 6.889 | 1,055,467,520 / 1,084,227,584 | 1.000x |
| float32 | forward | CUDA | 0.995 / 1.028 | 101,491,200 / 111,149,056 | 6.632x |
| float32 | forward-backward | reference | 17.905 / 18.063 | 1,163,470,848 / 1,184,890,880 | 1.000x |
| float32 | forward-backward | CUDA | 3.165 / 3.366 | 235,709,440 / 245,366,784 | 5.657x |
| bfloat16 | forward | reference | 9.257 / 9.320 | 1,340,090,368 / 1,365,245,952 | 1.000x |
| bfloat16 | forward | CUDA | 1.000 / 1.074 | 59,548,160 / 77,594,624 | 9.257x |
| bfloat16 | forward-backward | reference | 21.773 / 21.982 | 1,401,300,992 / 1,415,577,600 | 1.000x |
| bfloat16 | forward-backward | CUDA | 3.253 / 3.461 | 227,320,832 / 262,144,000 | 6.693x |

CUDA/reference parity passed for every row. Maximum observed errors were
`5.9604645e-7` forward and `1.4305115e-6` backward for float32, and
`1.5258789e-5` forward and `0.0625` backward for bfloat16. The admission gate
required at least 1.2x speedup, reduced peak allocation, and parity in both
dtypes and directions. The recorded decision was **GO**.

## Later production optimizations

The production implementation subsequently added the following without
changing the public attention contract:

- one shared compressed KV head broadcast over query heads (MQA layout);
- a compile-visible dispatcher for the explicit CUDA operator;
- packed query/KV/gate projection;
- full-head query/key RoPE inside the CUDA kernel;
- removal of redundant invalid-query output masking.

An ABBA comparison at the configured training maximum
`B=1,V=3,T=1024,Q=4,D=256`, bfloat16 forward-backward produced these retained
end-to-end medians after those changes:

| execution | global-only | hybrid CUDA | C vs A | peak allocated A / C |
|---|---:|---:|---:|---:|
| eager | 265.679 ms | 272.004 ms | 0.977x | 4,184,039,424 / 4,063,334,400 B |
| compiled | 103.720 ms | 100.933 ms | 1.028x | 2,521,415,680 / 2,488,655,872 B |

These end-to-end rows are optimization diagnostics rather than a component
admission gate. They show a small compiled win and lower allocation, but no
material eager speedup. The full model record and rejected ablations live in
[`../../benchmarks.md`](../../benchmarks.md).

## Constraints

- CUDA supports float16, bfloat16, and float32, window radius at most 64, and
  attention dropout 0.
- K/V gradients use float32 atomic accumulation and are not bitwise
  deterministic.
- Results are specific to the recorded hardware and shapes; small-shape launch
  overhead remains workload-dependent.

## 2026-08-23 kernel optimization

The forward kernel now normalizes each local-window softmax once per query row
and reuses the probabilities across value features. Query values are cached in
registers; backward also caches the upstream gradient. Invalid rows are zeroed
inside the kernels, removing full output and query-gradient prefill launches.
Profiling selected int32 row arithmetic for forward and int64 arithmetic for
backward; runtime validation rejects forward dimensions, row counts, or
compression ratios that cannot be represented safely.

Environment: NVIDIA GeForce GTX 1650 (compute capability 7.5), Python 3.11.15,
PyTorch 2.13.0+cu130, CUDA 13.0. A five-iteration profiler run used bfloat16
CSWA with `N=7,T=512,D=256,H=4,Dh=64`, compression ratio 4, window radius 4,
and approximately 0.875 valid density.

| kernel | mean CUDA time (ms) |
|---|---:|
| compressed attention forward | 0.716 |
| compressed attention backward | 1.022 |

CUDA/reference parity, non-contiguous layouts, fused RoPE, invalid rows, and
forward/backward behavior passed the complete operator test matrix after a
fresh compute-capability-7.5 extension build.

### Narrow-head sub-warp packing

The configured BLCS width uses `Dh=16`, where assigning one 32-thread warp to
one attention row left half of the lanes idle. The kernel now packs independent rows
within a warp while keeping at most four features per lane: eight rows for
`Dh=16`, four for `Dh=32`, two for `Dh=64`, and one for wider or uncommon
widths. Forward and backward share the same dispatch, and partial final warps
are covered by fused-RoPE tied-KV parity tests in float16, bfloat16, and
float32.

On the same GTX 1650, a profiler comparison used bfloat16
`N=7,T=1024,D=64,H=4,Dh=16`, compression ratio 4, window radius 4, and
approximately 0.875 valid density. Both candidates used the preceding kernel
optimizations; only row packing changed.

| kernel | one row per warp (ms) | eight rows per warp (ms) | speedup |
|---|---:|---:|---:|
| compressed attention forward | 0.960 | 0.298 | 3.221x |
| compressed attention backward | 1.322 | 0.543 | 2.435x |

The corresponding complete CSWA forward-backward benchmark used 4 warmups and
20 synchronized iterations. The gathered reference measured 10.156 / 11.053
ms median / p95, while CUDA measured 3.262 / 4.037 ms, a 3.113x median
speedup. Peak allocation was 84,805,632 B for the reference and 28,680,192 B
for CUDA.
