---
id: group-i786-normv2-large-cuda-ablation-eb32
type: group
title: BLCS normalization v2 large CUDA ablation（effective batch 32、#786）
issue: 786
members:
- run-i786-normv2-large-cuda-ablation-a-b8-a4-eb32-e100-gpu0
- run-i786-normv2-large-cuda-ablation-b-b8-a4-eb32-e100-gpu0
- run-i786-normv2-large-cuda-ablation-c-b8-a4-eb32-e100-gpu0
- run-i786-normv2-large-cuda-ablation-d-b8-a4-eb32-e100-gpu0
parents: []
tags:
- blcs
- tracking
- normalization-v2
- cuda
- ablation
- effective-batch-32
---

## まとめ

normalization v2、T=128、V=4、seed 42、bf16 mixed、hidden 512、12 stages、CUDA CSWA、micro-batch 8 × accumulation 4（effective batch 32）、100 epochを固定し、FFN modeとmHC writeback位置の2×2 ablationを行った。4 runのtarget、presence、instance ID、frame mask、scene IDは一致しており、同一test splitのcheckpoint replay metricで比較した。

| Variant | FFN | mHC writeback | Params | Position error | Presence F1 | ID switches | Birth / death error |
|---|---|---|---:|---:|---:|---:|---:|
| A | per-attention | after object temporal | 109.4M | **3.5223m** | 0.9646 | 20.16 | 7.10 / 8.53 |
| B | shared | after object temporal | 57.5M | 3.7572m | 0.9639 | **17.04** | 8.77 / 10.52 |
| C | per-attention | layer end | 109.4M | 3.9295m | 0.9713 | 19.92 | 5.98 / 6.22 |
| D | shared | layer end | 57.5M | 4.2109m | **0.9721** | 22.64 | **4.43 / 5.36** |

単一の総合勝者はなく、positionはA、identity continuityはB、presence/lifecycle timingはDが最良だった。同一FFN内のA/CおよびB/D比較では、layer-end mHCはpresence/lifecycleとpeak memoryに有利だがpositionには不利だった。次はA/CとB/Dを複数seedで再現確認し、主目的をpositionに置く場合はAをbaseline、memory/lifecycleを重視する場合はCまたはDをbaselineにする。
