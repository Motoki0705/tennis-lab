---
id: group-blcs-compile-training-abba-v4
type: group
title: BLCS Hybrid CSWA CUDA torch.compile 3 epoch ABBA比較
members:
- run-blcs-compile-eager-a-v4
- run-blcs-compile-compiled-a-v4
- run-blcs-compile-compiled-b-v4
- run-blcs-compile-eager-b-v4
parents: []
tags: [blcs, torch-compile, hybrid-cswa, cuda, training-benchmark]
---

## まとめ

Hybrid CSWA CUDAのBLCSをT=1024、3 cameras、batch size 1、gradient accumulation 8、FP32、3 epochでABBA比較した。各compiled runは独立した空Inductor cacheを使用した。

| mode | cold wall平均 | steady epoch平均 | steady batch平均 | peak CUDA allocated | 最終train loss平均 | test loss平均 |
|---|---:|---:|---:|---:|---:|---:|
| eager | 140.08 s | 40.0 s | 500.0 ms | 6.299 GB | 0.100994736 | 0.098784741 |
| compiled | 418.02 s | 21.0 s | 262.5 ms | 5.066 GB | 0.100995764 | 0.098785050 |

観測結果は、steady-stateではcompiledが1.90倍高速、peak CUDA allocatedは19.6%減少した一方、cold-start込み3 epoch wall timeはcompiledが2.98倍長い。loss差は1e-5未満で数値挙動は同等だった。現在のcompile costと固定shape throughputからの粗い外挿では、total wall timeのbreak-evenは約18 epochである。

速度比較datasetは正式disk schemaを満たすが物理的な軌道精度を保証しないため、このgroupは学習loop性能と短期数値同等性のみを根拠とする。またrun別`pred_test.npz`は既存artifact root挙動によりrepro bundleへ保存されず、test metricsは各runのTensorBoard eventとqueue logから採取した。
