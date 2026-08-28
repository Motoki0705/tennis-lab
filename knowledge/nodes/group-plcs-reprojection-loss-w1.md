---
id: group-plcs-reprojection-loss-w1
type: group
title: PLCS全出力＋reprojection loss weight 1比較
members:
- run-plcs-multiview-all-outputs-beta01
- run-plcs-multiview-all-outputs-beta01-reprojection-w1
parents: []
tags:
- plcs
- reprojection-loss
- canonical-pose
- loss-ablation
---

## まとめ

`PLCSMultiViewModel`でposition・rotation・angle・canonical poseをすべてweight 1、position Smooth-L1 betaを0.1とした対照条件に、reprojection weight 1を追加した2-run比較。両runはseed 42、固定800/100/100 split、2 views、16 frames、effective batch 4、42 epoch/global step 8400で揃えた。

| test metric | baseline | reprojection w1 | delta |
|---|---:|---:|---:|
| position error (m) | 6.605493 | 6.603486 | -0.002007 |
| angular error (deg) | 91.731071 | 88.649841 | -3.081230 |
| raw reprojection loss | 0.105332 | 0.087569 | -0.017764 |
| canonical pose loss | 0.009055 | 0.009150 | +0.000095 |
| Z error (m) | 0.093369 | 0.275332 | +0.181963 |
| angle accuracy 30deg | 0.161875 | 0.180000 | +0.018125 |

weight 1のreprojectionは画像面整合と向きを改善したが、3D positionは実質横ばいでZが悪化した。現時点ではproduction defaultへ直ちに採用せず、複数seed・低weight sweepとdepth拘束の有無で再検証する。
