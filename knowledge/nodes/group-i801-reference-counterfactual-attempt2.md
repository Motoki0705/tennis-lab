---
id: group-i801-reference-counterfactual-attempt2
type: group
title: Reference-camera RoPE strict paired検証 Attempt 2 (#801)
issue: 801
members:
- run-i801-a2-blcs-reference-paired
- run-i801-a2-blcs-selector-zero-paired
- run-i801-a2-plcs-d-reference
- run-i801-a2-plcs-d-selector-zero
- run-i801-a2-plcs-reference-paired
- run-i801-a2-plcs-selector-zero-paired
parents:
- group-i801-reference-selector-ablation
tags:
- blcs
- plcs
- reference-camera
- counterfactual
- paired-evaluation
- ablation-d
- camera-view-v2
---

## まとめ

Attempt 2では、dataset v2・D architecture・第三RoPE selectorのmatched比較をPLCSまで拡張し、BLCS/PLCS双方をsame/opposite cameraのstrict paired evaluatorで再評価した。正式証拠は学習2 runと評価4 runであり、失敗runおよびdeterministic parityを満たさないretry3は含めない。

| Task / evidence | Reference selector | Selector-zero | 判定 |
|---|---:|---:|---|
| BLCS same Y-sign | 0.7531 | **0.7586** | zeroが僅差 |
| BLCS opposite Y-sign | **0.7668** | 0.7629 | referenceが僅差 |
| BLCS physical position consistency | **3.0581m** | 3.3259m | referenceが良い |
| PLCS train position error | **5.1151m** | 5.1417m | referenceが僅差 |
| PLCS train Y-sign | 0.6803 | **0.6845** | zeroが僅差 |
| PLCS same Y-sign | 0.6719 | **0.7277** | zeroが良い |
| PLCS opposite Y-sign | **0.7684** | 0.7407 | referenceが良い |
| PLCS physical position consistency | 4.7011m | **4.5009m** | zeroが良い |

全runはseed 42、T128、camera-view v2、`reference_camera_court_rzpi_v1`、`time_camera_reference_selector_v1`を使用した。paired evaluatorは各task 100 test scenesについてscene ID、centered window、6-view集合/順、checkpoint、config、input、physical target、lifecycleをfail-closedに照合し、BLCSは`cam_2`/`cam_0`、PLCSは`camera_2`/`camera_0`をsame/oppositeとして選んだ。

selectorはBLCSで小さな混在差、PLCSではsame/oppositeで明確な効果反転を示した。指定cameraへ出力frameを一意化する信号としての一方向の改善は得られず、production defaultはv1を維持する。次に進めるなら、paired physical consistencyを直接教師化し、reference side/local indexを均衡化した3 seeds以上の検証が必要である。
