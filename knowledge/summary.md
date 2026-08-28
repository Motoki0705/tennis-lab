# Tennis Lab Knowledge Summary

更新日: 2026-08-28  
調査基準commit: `bb5a7c1460b5b9b6e3cdc934e1f4cdcbecae1387`

この文書は、Tennis Labの学習・実験から得られた**現在の到達点、主要な知見、判断保留事項、次に解くべき課題**を横断的に把握するための要約です。個々の数値、再現手順、因果考察の正本は [`nodes/`](./nodes) のrun / group nodeと [`runs/`](./runs) の再現性bundleです。この文書は正本を置き換えず、研究状況を短時間で理解するための入口として使います。

現行knowledge graphの正式node typeはrunとgroupです。評価契約が異なる実験を同じランキングへ混ぜず、production、benchmark、family、diagnosticを区別して整理します。

## 現在の全体像

現行pipelineでは、2D ball detection、court detection、single-person PLCS、single-ball BLCSにdeploy checkpointがあります。SLCSはend-to-end経路の動作確認までは完了していますが、recording-disjointな汎化baselineはありません。2026-08-28時点の新しいPLCS / BLCS研究結果は、いずれもproduction checkpointを置き換える段階には達していません。

| task | 現在の基準 | 主な固定値 | 現在の判断 |
|---|---|---|---|
| `ball_detection` | [`run-i618-convnext-v2-ft`](nodes/run-i618-convnext-v2-ft.md) | test F1 `0.721789`、precision `0.735656`、recall `0.708436`、距離 `2.176208 px` | offline最高値ではなく、実clipのcoverageと軌道安定性を含めてdeploy継続 |
| `court_detection` | [`run-i621-court-kp512-resume-r4`](nodes/run-i621-court-kp512-resume-r4.md) | val best `2.23 px`、固定checkpoint再評価 `1.708886 px` | KP14 / 512入力のdeploy。独立held-out testは未確立 |
| `plcs` | [`run-deploy-multiview-plcs-i590-courtkp14-v2`](nodes/run-deploy-multiview-plcs-i590-courtkp14-v2.md) | position `0.175284 m`、yaw `6.443357°` | 3–6 camera・court KP14の現行single-person deploy |
| `blcs` | [`run-deploy-multiview-blcs-v3-simfix-c3-6-v2`](nodes/run-deploy-multiview-blcs-v3-simfix-c3-6-v2.md) | position `1.064595 m`、endpoint `2.024551 m` | 3–6 camera・court KP14の現行single-ball deploy |
| `slcs` | [`run-i634-slcs-overfit-dino`](nodes/run-i634-slcs-overfit-dino.md) | player `0.470360 m`、yaw `7.761920°`、ball `1.954024 m` | 同一13 windowのmemorization diagnostic。deploy / 汎化主張は不可 |

pipelineが参照するcheckpointは次です。

| stage | checkpoint |
|---|---|
| court | `court_detection/kp/run-i621-court-kp512-resume-r4-epoch21.ckpt` |
| ball | `ball_detection/run-i618-convnext-v2-ft-epoch13.ckpt` |
| PLCS | `plcs/run-multiview-plcs-i590-courtkp14-epoch197.ckpt` |
| BLCS | `blcs/run-multiview-blcs-v3-simfix-c3-6-epoch129.ckpt` |

## タスク別に確立した知見

### Ball Detection

現行deployはfine-tuning版を維持します。[`run-i618-convnext-v2-scratch`](nodes/run-i618-convnext-v2-scratch.md) はTrackNet test F1 `0.7692`、距離 `2.01 px`でoffline評価では上ですが、実clip coverageが`92.0% → 91.1%`へ下がり、`179.9 px`のteleportを1件発生させました。したがって、単一のF1最高値より実動画上の安定性を優先しています。

3DGS augmentationでは、固定checkpoint・split・decodeによる比較基盤 [`run-i618-3dgs-blcs-real-baseline-v1`](nodes/run-i618-3dgs-blcs-real-baseline-v1.md) が整備されています。simple-sphereを1/12混合したtreatmentは単一seedのgame9で`+0.018454 F1`でしたが、残りseedとgame10 final testが未完了のため、効果は確立していません。

### Court Detection

KP14 detectorは実pipelineで利用可能な水準ですが、`1.708886 px`は`test_dataloader`がvalidation dataを読む条件の再評価値であり、独立testではありません。次の品質更新にはrecording-disjoint test、geometry valid率、line support、処理時間、PLCS / BLCSへのE2E影響が必要です。

court segmentationはKP14とは別契約です。[`group-i524-dinov3-ssl-court`](nodes/group-i524-dinov3-ssl-court.md) では凍結backbone条件で非SSL `0.517 mIoU`からSSL `0.800 mIoU`へ改善しましたが、KP14 deploy modelの置換根拠にはなりません。点・線共同postprocessについては、detectorを固定した正式なpostprocess-only baselineがまだありません。

### PLCS

現行deployは、split trunk、H=0/S=6、position weight 8、補助pose loss無効、court KP14というpipeline互換recipeです。過去のablationでは、positionにはS6/H0、rotationには別の容量配分が有利であり、単一構成が全目的を同時に最適化しないPareto構造が確認されています。[`group-i545-loss-head-tuning`](nodes/group-i545-loss-head-tuning.md) のposition frontier `0.166 m`は有力ですが、現行KP14 deploy以前の契約なので直接置換には使いません。

canonical poseでは、[`run-plcs-canonical-temporal-decomp-beta01-noaug`](nodes/run-plcs-canonical-temporal-decomp-beta01-noaug.md) が平均pose固定から入力依存motionへの移行を確認しました。canonical MPJPEは`0.091136 m`、motion amplitude ratioは`1.174967`、centered Pearsonは`0.795146`です。一方、high-frequency fractionは予測`0.391067`に対してGT `0.068930`であり、motionを復元する代わりにjitterを過剰生成しています。position / rotation headは未学習なのでdeploy精度との比較には使いません。

reprojection lossは一方向な改善ではありません。[`group-plcs-multiview-axial-reprojection-loss-w1-v4-t128`](nodes/group-plcs-multiview-axial-reprojection-loss-w1-v4-t128.md) のV=4/T=128条件では、position `1.386235 → 1.352761 m`、angle `66.968224 → 63.600704°`へ改善しましたが、0.5 m以内率は`0.118984 → 0.088828`へ悪化し、X誤差と分散も増えました。複数seedとweight sweep前にdefaultへ採用しません。

camera-view v2のreference selectorも決着していません。PLCSではreferenceがpositionとID switchesで良い一方、selector-zeroがY-sign、heading、presenceで良く、指標ごとに優位が逆転しました。productionはv1を維持します。

### BLCS

single-ballではmultiview deployが単眼親よりposition `1.845 → 1.065 m`、endpoint `3.408 → 2.025 m`へ改善しました。ただしcamera presetも変わるため、改善全量をview数へ帰属できません。

physics priorはaccuracyとsmoothnessのtrade-offです。[`group-i593-physics-prior`](nodes/group-i593-physics-prior.md) のftCは実clip jerkを`0.280 → 0.106`へ改善しましたが、in-distribution positionを`1.845 → 1.947 m`へ悪化させました。そのため機能は残してもdefault checkpointは置き換えていません。

track-query architectureの[`group-i786-normv2-large-cuda-ablation-eb32`](nodes/group-i786-normv2-large-cuda-ablation-eb32.md) では、positionはA `3.522323 m`、identity continuityはB `17.04 ID switches`、presence / lifecycleはD `0.972115 F1`・birth/death `4.43 / 5.36 frames`が最良でした。単一の総合勝者はありません。また、この群は旧versioned-v2 runtimeで、現行mainとloss beta・artifact schemaが異なるためhistorical family evidenceとして扱います。

[`group-i801-reference-selector-ablation`](nodes/group-i801-reference-selector-ablation.md) のmatched BLCS比較では、selector-zeroがreferenceよりposition `3.721058 < 3.813505 m`、Y-sign accuracy `0.877656 > 0.871094`でした。第三RoPE軸によるreference明示の追加効果は確認できず、production v1を維持します。

学習性能では、[`group-blcs-compile-training-abba-v4`](nodes/group-blcs-compile-training-abba-v4.md) によりcompiled実行がsteady-stateで`1.90×`高速、peak CUDA allocatedが`19.6%`減る一方、cold-start込み3 epochでは`2.98×`遅く、break-evenは約18 epochと分かっています。これはtrajectory精度ではなくruntime baselineです。

multi-ballはsingle-ballと別契約です。短clip diagnosticと、[`run-i648-blcs-lifecycle-v4-large-pointattn32-rope2d-t512-b1-100ep`](nodes/run-i648-blcs-lifecycle-v4-large-pointattn32-rope2d-t512-b1-100ep.md) の512-frame lifecycle baselineも、sequence length・data・training budgetが違うため相互に直接順位付けしません。

### SLCS

SLCSはshared DINOによるend-to-end経路と、split DINOによるyaw / ball側の改善までは確認されています。しかしtrain / val / testが同じ13 windowを共有するmemorization実験であり、モデルの汎化性能は未評価です。最初にrecording_id非重複のsplitを作り、shared DINO、split DINO、DINOなしを同一seed・budgetで比較する必要があります。

## 結果を解釈するための規則

| 区分 | 用途 |
|---|---|
| **production / deploy** | 現行pipelineが参照するcheckpoint。単一metricの最高値だけでは変更しない |
| **benchmark** | 固定split・decode・metricで新施策を比較する起点 |
| **family** | 特定architecture、loss、data contract内の比較。タスク全体へ一般化しない |
| **diagnostic** | overfit、smoke、canonical-onlyなど、経路や仮説の成立だけを確認する実験 |
| **missing** | held-out評価、複数seed、正式runなどが不足し、基準として使えない状態 |

比較時は次を守ります。

1. view数、dataset、target frame、single/multi-object、metricが異なるrunを直接順位付けしない。
2. deploy判断ではbest checkpoint、実clip安定性、下流E2E、処理時間を単一metricより優先する。
3. 単一seedの小差は確立した効果とみなさない。
4. no-op、作業tree取り違え、failed qualification、holdout rejectを正の証拠へ昇格しない。
5. 現行mainとnormalization、loss beta、runtime、artifact schemaが異なるrunはhistorical evidenceと明記する。

## 優先して解くべき課題

| 優先度 | 領域 | 不足している証拠 | 完了条件 |
|---|---|---|---|
| S | Court Detection | recording-disjoint test | KP距離、geometry valid、line support、wall time、下流E2Eを固定評価 |
| S | PLCS canonical motion | jitter抑制とmulti-task再導入 | motion相関を保ち、high-frequency fractionを低下させ、position / rotation併用でもmean collapseしない |
| S | BLCS track-query | 現行main contractでの再現性 | A/B/Dをcurrent loss・schema、3 seed以上で再実行し、position / ID / lifecycleのParetoを確認 |
| A | PLCS reprojection | 単一seed・weight 1のみ | weight `0.1/0.3/1.0`を複数seedで比較し、meanだけでなく0.5 m率・軸別誤差・tailを改善 |
| A | SLCS | 汎化splitがない | recording-disjoint splitで3 seed、欠損率・jerk・calibrationまで報告 |
| A | Ball 3DGS augmentation | campaign未完了 | 残りseedとgame10 final testを固定protocolで完了 |
| B | multi-person / multi-ball | deploy互換E2E評価が不足 | single-object契約と分離したまま、lifecycle・presence・identityを長sequenceで評価 |

## このknowledge directoryの読み方

- [`summary.md`](./summary.md): 現在の到達点と未解決課題を俯瞰する入口。
- [`nodes/`](./nodes): 各runの設定・metric・考察、および関連runをまとめるgroup nodeの正本。
- [`runs/`](./runs): `repro.sh`、`metrics.json`、`pred_test.npz`、収束曲線などの再現性bundle。
- [`README.md`](./README.md): knowledge graphのschema、登録方法、検証手順。
- [`webui/`](./webui): node間の関係と実験結果をグラフとして閲覧するUI。

このsummaryは、pipeline checkpointが変わったとき、同一契約で再現された重要な結果が追加されたとき、評価契約が変わったとき、またはdiagnostic領域に初めてheld-out baselineができたときに更新します。新runが1件追加されるたびに追記するのではなく、研究上の結論または優先順位が変わった場合に更新します。
