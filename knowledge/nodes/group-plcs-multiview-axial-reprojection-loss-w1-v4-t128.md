---
id: group-plcs-multiview-axial-reprojection-loss-w1-v4-t128
type: group
title: PLCS axial V=4 T=128 reprojection loss weight 1比較
members:
- run-plcs-multiview-axial-all-outputs-beta01-v4-t128
- run-plcs-multiview-axial-all-outputs-beta01-reprojection-w1-v4-t128
parents: []
tags:
- plcs
- multiview
- axial
- canonical-pose
- reprojection-loss
- loss-ablation
- v4
- t128
---

## まとめ

`PLCSMultiViewAxialModel`を4 views・128 framesで使用し、position・rotation・angle・canonical poseをすべてweight 1、position Smooth-L1 betaを0.1とした2-run比較。両runはGPU #0、seed 42、固定split、batch 4、BF16、50 epochで揃え、差分はreprojection weight 0/1だけである。

| test metric | baseline | reprojection w1 | delta |
|---|---:|---:|---:|
| position error (m) | 1.386235 | 1.352761 | -0.033474 |
| position median (m) | 1.311628 | 1.187532 | -0.124096 |
| angular error (deg) | 66.968224 | 63.600704 | -3.367519 |
| angular median (deg) | 50.819160 | 46.408375 | -4.410786 |
| raw reprojection loss | 0.025546 | 0.023028 | -0.002518 |
| canonical pose loss | 0.008194 | 0.008204 | +0.000011 |
| X / Y / Z error (m) | 0.696738 / 1.025382 / 0.096257 | 0.754972 / 0.960795 / 0.093188 | +0.058234 / -0.064587 / -0.003069 |
| position accuracy 0.5 m | 0.118984 | 0.088828 | -0.030156 |
| position accuracy 1 m | 0.355625 | 0.391094 | +0.035469 |
| position accuracy 2 m | 0.811562 | 0.852031 | +0.040469 |
| angle accuracy 30 deg | 0.303438 | 0.346641 | +0.043203 |

reprojection weight 1は平均・中央値の3D position、向き、画像面整合を改善し、V=2・T=16で見られたZ悪化も再現しなかった。ただし0.5 m以内率、X誤差、position分散は悪化したため全面的な改善ではない。採用判断には複数seedと低weight sweepを行い、meanだけでなく近距離accuracyとtailまで評価する必要がある。

### 可視化

- [baseline（GT vs prediction）](../runs/run-plcs-multiview-axial-all-outputs-beta01-v4-t128/visualization_scene_000233_position_rotation.mp4)
- [reprojection weight 1（GT vs prediction）](../runs/run-plcs-multiview-axial-all-outputs-beta01-reprojection-w1-v4-t128/visualization_scene_000233_position_rotation.mp4)

両動画は同じ`scene_000233`のtest clip（test index 18、source frames 265–392、128 frames）を30 fpsで描画した。選択条件は「reprojection版でscene平均position/angular errorがともに改善するsceneのうち、両runの4指標がtest全体の分布中心に最も近いもの」。このclipではbaselineが1.177687 m / 56.011864度、reprojection版が0.809771 m / 44.879143度である。`pred_test.npz`にはcanonical pose予測が含まれないため、両予測に同一のGT canonical poseを使用し、position・rotationの差だけを可視化している。
