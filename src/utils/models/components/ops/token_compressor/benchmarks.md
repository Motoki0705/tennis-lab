# Token compressor benchmark results

This is the canonical retained performance record for token-level KV
compression. The BLCS-specific benchmark drivers and raw JSON artifacts were
removed on 2026-08-21; the production package contains only the operator and
its correctness tests.

## Recorded environment and protocol

- Recorded: 2026-08-19 for GitHub Issue #753
- GPU: NVIDIA GeForce RTX 5060 Ti, compute capability 12.0
- Software: Python 3.11.15, PyTorch 2.11.0+cu130, CUDA 13.0
- dtype and direction: float32 forward-backward
- Compression ratio: 4; valid density: 0.875
- Timing: 3 warmups and 6 synchronized iterations; medians below
- Candidates: eager PyTorch, `torch.compile(mode="reduce-overhead")`, and a
  benchmark-only Triton gather/softmax/reduction prototype

## Results

| shape `(N,T,D,H,Dh)` | candidate | median / p95 (ms) | peak allocated / reserved (bytes) | parity max forward / backward |
|---|---|---:|---:|---:|
| `(24,512,64,4,16)` | eager | 3.242 / 5.250 | 156,163,072 / 192,937,984 | 0 / 0 |
| `(24,512,64,4,16)` | compiled | 1.460 / 1.592 | 7,352,320 / 111,149,056 | 1.1921e-7 / 7.1526e-7 |
| `(24,512,64,4,16)` | Triton prototype | 1.617 / 1.888 | 80,520,192 / 117,440,512 | 1.1921e-7 / 9.5367e-7 |
| `(32,1025,256,4,64)` | eager | 37.927 / 38.198 | 1,493,428,736 / 1,866,465,280 | 0 / 0 |
| `(32,1025,256,4,64)` | compiled | 15.569 / 15.901 | 77,974,016 / 903,872,512 | 1.1921e-7 / 7.1526e-7 |
| `(32,1025,256,4,64)` | Triton prototype | 12.444 / 12.508 | 658,599,936 / 744,488,960 | 1.1921e-7 / 9.5367e-7 |

Against the best PyTorch reference, the prototype achieved 0.903x on the
default object path and 1.251x on the longer context. Peak allocation increased
on both shapes, so it failed the required 1.1x-per-shape and non-increasing
memory gate. The recorded decision for that prototype was **NO-GO**.

## Current implementation

The later production implementation is not the rejected multi-head prototype.
It compresses to one shared `head_dim` latent used as both key and value, and
uses an explicit reference/CUDA pooling resolver. The CUDA path is a Triton
masked-pooling operator with parity, boundary, all-invalid, non-contiguous, and
gradient tests. It supports float16, bfloat16, and float32 inputs at head
dimensions 16, 32, 64, and 128. Unsupported widths fail at construction; they
do not silently select the reference path.

## 2026-08-23 pooling optimization

The backward launch was changed from one program per input token and branch to
one program per compressed output. Each program now computes its masked
softmax statistics once and writes all contributing source gradients. The
forward and backward launches also select one, two, or four warps according to
head width. Boundary-only source gradients are explicitly zero-initialized.

Environment: NVIDIA GeForce GTX 1650 (compute capability 7.5), Python 3.11.15,
PyTorch 2.13.0+cu130, CUDA 13.0. The benchmark used bfloat16 `raw_kv`, float32
gates, `N=7,T=1024,Dh=64`, compression ratio 4, approximately 0.875 valid
density, 8 warmups, and 30 synchronized measured iterations.

| direction | candidate | median / p95 (ms) | peak allocated / reserved (bytes) | speedup |
|---|---|---:|---:|---:|
| forward | gathered reference | 2.319 / 3.204 | 46,846,976 / 67,108,864 | 1.000x |
| forward | optimized Triton | 0.117 / 0.224 | 11,936,768 / 23,068,672 | 19.820x |
| forward-backward | gathered reference | 3.187 / 6.981 | 46,846,976 / 69,206,016 | 1.000x |
| forward-backward | optimized Triton | 0.593 / 0.898 | 17,441,792 / 23,068,672 | 5.376x |

The optimized path reduced peak allocation by 74.5% for forward and 62.8% for
forward-backward in this run. CUDA/reference operator forward, validity,
input-gradient, and gate-gradient parity passed across all four supported
widths; float16 coverage was added with the optimization. Full-compressor
parameter-gradient parity also passed at the configured width 16 and the
established width 64.

The integrated CSWA and BLCS model results are retained in
[`../../benchmarks.md`](../../benchmarks.md).
