---
id: group-i801-reference-selector-ablation
type: group
title: BLCS D reference-camera RoPE selector比較 (#801)
issue: 801
members:
- run-i801-d-reference-seeded
- run-i801-d-selector-zero-seeded
parents:
- run-i786-normv2-large-cuda-ablation-d-b8-a4-eb32-e100-gpu0
tags:
- blcs
- tracking
- camera-view-v2
- reference-camera
- rope
- ablation
- effective-batch-32
---

## まとめ

dataset v2、D architecture、seed 42、T128/V4、micro-batch 8×accumulation 4、bf16 mixed、CUDA CSWA、100 epochを固定し、第三RoPE軸の`reference`と`selector_zero`だけを変更したmatched comparisonである。

| Variant | Position error | Y-sign accuracy | X / Y / Z MAE | Presence F1 | ID switches |
|---|---:|---:|---:|---:|---:|
| reference selector | 3.8135m | 0.8711 | 1.6719 / 2.9425 / 0.6550m | 0.9682 | 24.92 |
| selector-zero | **3.7211m** | **0.8777** | **1.6394 / 2.8525** / 0.6594m | 0.9676 | **24.08** |

reference selectorはzero比でposition errorが`0.0924m`（`2.48%`）高く、Y-sign accuracyも`0.00656`低かった。Z MAE、presence F1、birth/death timingはreferenceがわずかに良いが、対称frame識別への明確な寄与は確認できない。Issueの停止条件に従い、production defaultはv1のまま維持する。

### Matched evidence

- 両runはcommit `b392bbcbab877172b74c190af32b4dcc12366853`、session、dataset path、split、seed、batch、epoch、CUDA条件が一致し、argv差分は`model=`と`run.output_dir=`だけである。
- `pred_test.npz`の100 scenesについて、`target_position`、presence、instance ID、frame mask、scene ID、view ID集合/順、reference stable ID/local index、forward/inverse transform、court/target/RoPE contractがbitwise一致した。これらを連結したdigestは`b210be56cfaf6de65e59c42df2ce9bd030a284e7be8e0919c33a8312f1bfb208`。
- reference local index分布は`[26, 29, 18, 27]`で、index 0固定ではない。local index別position errorはreference=`[3.4188, 2.6704, 3.9984, 3.2895]m`、zero=`[3.4523, 2.4632, 3.6435, 3.3484]m`。
- dataset split seedは42、800/100/100 scenes。train/val/test split SHA-256はそれぞれ`a7889a9c…`、`8bc2b42e…`、`89cd5848…`。reference transformの直交性・inverse・determinant最大誤差はすべて`0.0`。

### PLCS functional evidence

PLCSはfull GPU学習を本Issueの範囲に含めず、同じv2 D model/data contractのCPU forward/backwardとopposite-side counterfactual geometryを検証した。headingをphysical frameへ戻したreference-transform consistency errorは`0.0 rad`、point/vector/world-jointsも`0.0`で、float64 `atol=rtol=1e-9`とfloat32 runtime `atol=1e-6, rtol=1e-5`を満たす。これは学習済みPLCSのheading精度ではなく、paired functional contractの証拠である。

### 除外したattempt

retry3の100 epoch 2本はdataset RNGがentropy seedで、testのview ID/orderとreference local indexが一致しなかったため非因果diagnosticとして正式登録しない。seed修理後のGPU smoke初回は1-stepに対してwarmup 200、2回目はfast-dev-runでTensorBoard loggerが無いままqualitative callbackを有効にしたため停止した。`warmup_steps=0`とqualitative logging無効を明示した3回目は同じCUDA/16-worker training・validation pathをPASSし、その後に本比較を実行した。
