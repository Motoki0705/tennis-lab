# Track-query component and integrated benchmark results

This document retains benchmark decisions that span model components and do
not belong exclusively to one operator package. Component-specific results are
kept with the implementations:

- [compressed time-local attention](ops/compressed_time_local/benchmarks.md)
- [token compressor](ops/token_compressor/benchmarks.md)

The BLCS benchmark programs, their tests, raw JSON results, and provenance
snapshots were removed on 2026-08-21. The tables below are the canonical
repository record.

## mHC CUDA prototype

Environment: NVIDIA GeForce RTX 5060 Ti, Python 3.11.15, PyTorch 2.11.0+cu130,
CUDA 13.0. The float32 forward-backward case used the configured training lower
bound, 3 warmups, and 6 synchronized measured iterations.

| candidate | median / p95 (ms) | peak allocated / reserved (bytes) | parity max forward / backward |
|---|---:|---:|---:|
| eager PyTorch | 22.792 / 29.080 | 544,483,328 / 767,557,632 | 0 / 0 |
| compiled PyTorch | 7.223 / 7.472 | 63,086,592 / 673,185,792 | 0 / 1.5259e-5 |
| CUDA prototype | 22.769 / 29.262 | 544,483,328 / 742,391,808 | 0 / 7.6294e-6 |

Parity passed, but the CUDA prototype was only 0.317x as fast as the compiled
reference and increased peak allocation. The decision was **NO-GO**; mHC
remains PyTorch.

## Integrated A/B/C record

The required production-maximum comparison used
`B=1,V=3,T=1024,Q=4,D=256`, 8 stages, bfloat16 forward-backward, dropout 0,
eval mode, one warmup, and three synchronized measured iterations.

| candidate | median / p95 (ms) | throughput (frames/s) | peak allocated / reserved (bytes) | parity |
|---|---:|---:|---:|---|
| A: global-only | 229.987 / 230.170 | 4,452.43 | 4,203,175,936 / 4,408,213,504 | not applicable |
| B: hybrid reference | 293.844 / 295.034 | 3,484.84 | 5,905,436,672 / 6,102,712,320 | pass, self-reference |
| C: hybrid CUDA | 276.950 / 277.340 | 3,697.42 | 5,503,182,848 / 5,721,030,656 | pass vs B |

C/B parity passed with maximum forward error `0.0703125` and input-gradient
error `2.682209e-7` under bfloat16 tolerances `atol=0.065, rtol=0.02`. All
required smoke and maximum-training comparisons passed; the matrix contained
26 complete same-shape A/B/C triplets. This was a feasibility **PASS**, not a
claim that hybrid CUDA beat global-only end to end.

## Later optimization decisions

Subsequent diagnostics used the same configured maximum shape unless noted.

| experiment | result | decision |
|---|---|---|
| compile-visible CUDA dispatch and redundant-output elimination | Three compiled pairs measured A at 76.98 ms and C at 71.33 ms on median, about 1.079x C/A; C allocated 2,260,645,888 B vs A 2,535,872,512 B | retained |
| packed query/KV/gate projection | Compiled ABBA replicate medians: A 76.485/76.807 ms, C 70.748/70.577 ms; eager C remained slower than A | retained for the compiled path |
| fused full-head RoPE in CUDA attention | Compiled ABBA medians: A 103.045/103.759 ms, C 98.204/99.644 ms; CUDA/reference parity tests passed | retained |
| key-centric CUDA loop | Correctness passed, but timing was not materially better and was workload-sensitive | reverted |
| FFN recomputation | Reduced allocation but did not provide a stable latency benefit for the target path | removed |
| SwiGLU W1/W3 packing | Forward-backward medians: separate 272.302 ms, runtime concatenation 269.262 ms (1.011x), persistent gate-up 274.091 ms (0.993x) | reverted; improvement was not material |

After the retained operator changes and removal of FFN recomputation, the ABBA
diagnostic medians were 265.679 ms (A) vs 272.004 ms (C) in eager mode and
103.720 ms (A) vs 100.933 ms (C) when compiled. The hybrid path therefore has a
small compiled advantage and lower measured allocation, but does not meet a
material 1.10x end-to-end speedup gate.

## Limitations

- Results are device-, software-, and shape-specific.
- The global temporal stage remains quadratic and can OOM beyond configured
  context limits.
- Integrated parity covers stage outputs and camera/query input gradients;
  operator tests cover CUDA query/key/value gradients separately.

## 2026-08-23 GPU-1 validation

The optimized token-compressor and compressed-attention operators were
validated on the repository's second physical GPU while GPU 0 remained in use.
Environment: NVIDIA GeForce GTX 1650 (compute capability 7.5), Python 3.11.15,
PyTorch 2.13.0+cu130, CUDA 13.0.

The CSWA comparison used `N=7,T=1024,D=64,H=4,Dh=16`, bfloat16
forward-backward, approximately 0.875 valid density, 4 warmups, and 20
synchronized iterations. The BLCS comparison used
`B=1,V=3,T=512,Q=4,D=64`, four stages, float32 forward-backward, eval mode, 2
warmups, and 8 synchronized iterations. Reference and CUDA candidates shared
identical weights.

| workload | candidate | median / p95 (ms) | peak allocated / reserved (bytes) | speedup |
|---|---|---:|---:|---:|
| default-width CSWA | reference | 10.314 / 25.399 | 86,091,776 / 111,149,056 | 1.000x |
| default-width CSWA | CUDA | 4.920 / 18.897 | 28,688,384 / 31,457,280 | 2.097x |
| BLCS track-query model | reference | 213.322 / 387.541 | 581,164,544 / 629,145,600 | 1.000x |
| BLCS track-query model | CUDA | 175.596 / 289.563 | 530,912,256 / 587,202,560 | 1.215x |

CUDA reduced peak allocation by 66.7% for default-width CSWA and 8.6% for the
full model. A permanent integration test with the configured `Dh=16` compares
the BLCS model's position/presence outputs and ball/court input gradients
against the reference backend; it passed on the same GPU build. Operator-level
measurements remain in the linked package benchmark records above.

### Narrow-head follow-up

After packing multiple narrow-head attention rows into each warp, the default
BLCS workload was rerun with the same shape, model configuration, warmup count,
iteration count, dtype, and paired reference/CUDA weights. The fresh paired
run measured 176.132 / 210.902 ms median / p95 for reference and 136.797 /
154.573 ms for CUDA, increasing the paired median speedup to 1.288x. Peak
allocation was 584,594,944 B for reference and 531,196,928 B for CUDA, a 9.1%
reduction. The operator-level narrow-head timings and dispatch policy are
retained in the linked compressed-attention benchmark record.
