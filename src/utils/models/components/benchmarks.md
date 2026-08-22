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
