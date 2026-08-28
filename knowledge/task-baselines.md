# タスク別ベースライン台帳

更新日: 2026-08-28  
調査基準commit: `bb5a7c1460b5b9b6e3cdc934e1f4cdcbecae1387`

この文書は、`knowledge/nodes/` の全group node、主要なdeploy / benchmark / family run node、および実際のpipeline checkpoint設定を照合して作成した**運用用の派生インデックス**です。正式知識グラフのnodeではなく、各runの数値・再現手順・因果主張の正本は個別nodeです。

現行knowledge graphはrun nodeとgroup nodeだけを正式対象とします。2026-08-28までにproposal / paper nodeは削除されたため、この台帳も削除済みnodeへの参照を持たず、実験として登録済みの証拠だけをbaselineとして扱います。

`src/tasks/base` は共通基盤であり独立した予測タスクではないため、baselineを割り当てません。コアタスクは `ball_detection`、`court_detection`、`plcs`、`blcs`、`slcs` の5つです。multi-person / multi-ball、3DGS synthetic、segmentation、compile性能は出力・データ・評価契約が異なるため、コアbaselineと混ぜず後半の「拡張契約」に分離します。

## baselineの分類と選定規則

| 分類 | 意味 |
|---|---|
| **deploy baseline** | 現行pipelineがcheckpointを参照し、再現可能なdeploy runが存在する比較起点 |
| **benchmark baseline** | 固定split・固定decode・固定metricで比較する評価起点。deployと同一とは限らない |
| **family baseline** | 特定architecture / loss / data family内の対照。タスク全体のbaselineへ一般化しない |
| **diagnostic baseline** | overfit、smoke、短期学習など、経路成立だけを示す比較起点 |
| **missing** | held-out評価または正式runがなく、まだbaselineとして昇格できない状態 |

選定は次の順で行います。

1. `status: done` かつ再現情報と評価値があること。
2. 明示的なdeploy runと [`src/tennis_scene/configs/pipeline.yaml`](../src/tennis_scene/configs/pipeline.yaml) のcheckpoint参照を、単純な最高値より優先すること。
3. best checkpointとlast epochの値が異なる場合は、固定manifestまたはdeploy対象の**best checkpoint評価**を採用すること。
4. view数、court keypoint数、single/multi-object、split、metric、実写/合成が異なるrunを直接順位付けしないこと。
5. no-op、作業tree取り違え、failed qualification、holdout rejectをbaselineへ昇格しないこと。
6. multi-objectiveでは単一の最高値へ潰さず、deploy baselineとPareto challengerを併記すること。
7. normalization、loss beta、artifact schema、runtime contractが現行mainと異なるrunは、数値が再現可能でも**historical family evidence**として明示し、現行mainのbaselineへ読み替えないこと。

## 現行コアbaseline一覧

| task | 現行baseline | 分類 | 主な固定値 | 判定 |
|---|---|---|---|---|
| `ball_detection` | [`run-i618-convnext-v2-ft`](nodes/run-i618-convnext-v2-ft.md) | deploy | test F1 `0.721789`、precision `0.735656`、recall `0.708436`、距離 `2.176208 px` | 実clipのcoverageと軌道安定性を含めて採用 |
| `court_detection` | [`run-i621-court-kp512-resume-r4`](nodes/run-i621-court-kp512-resume-r4.md) | deploy | val best `2.23 px`、固定checkpoint再評価 `1.708886 px` | KP14 / 512入力の現行checkpoint |
| `plcs` | [`run-deploy-multiview-plcs-i590-courtkp14-v2`](nodes/run-deploy-multiview-plcs-i590-courtkp14-v2.md) | deploy | position `0.175284 m`、yaw `6.443357°` | 3–6 camera、court KP14、独立再学習で再現 |
| `blcs` | [`run-deploy-multiview-blcs-v3-simfix-c3-6-v2`](nodes/run-deploy-multiview-blcs-v3-simfix-c3-6-v2.md) | deploy | position `1.064595 m`、endpoint `2.024551 m` | 3–6 camera、court KP14の現行checkpoint |
| `slcs` | [`run-i634-slcs-overfit-dino`](nodes/run-i634-slcs-overfit-dino.md) | diagnostic | player `0.470360 m`、yaw `7.761920°`、ball `1.954024 m` | 同一13 windowのmemorization。deploy / 汎化baselineは未確立 |

## `ball_detection`

### 決定

現行deploy baselineは [`run-i618-convnext-v2-ft`](nodes/run-i618-convnext-v2-ft.md) です。pipelineは `ball_detection/run-i618-convnext-v2-ft-epoch13.ckpt` を参照しています。比較時はnode frontmatterのlast-epoch F1 `0.645535`ではなく、配備したbest checkpointの固定評価を使います。

3DGS augmentation campaignでは、同じcheckpoint・split・decodeをhash固定して再評価した [`run-i618-3dgs-blcs-real-baseline-v1`](nodes/run-i618-3dgs-blcs-real-baseline-v1.md) をbenchmark baselineとします。このnodeが再現した値は次です。

| split | F1 | precision | recall | mean distance | negative-frame FPR |
|---|---:|---:|---:|---:|---:|
| TrackNet game9 val | 0.712628 | 0.725000 | 0.700671 | 2.302407 px | 0.116667 |
| TrackNet game10 test | 0.721789 | 0.735656 | 0.708436 | 2.176208 px | 0.087719 |

### challengerとfamily baseline

- [`run-i618-convnext-v2-scratch`](nodes/run-i618-convnext-v2-scratch.md) はbest checkpointでTrackNet test F1 `0.7692`、距離 `2.01 px`を記録し、offline benchmarkでは上です。しかし実clipでcoverageが `91.1%`（deploy FTは `92.0%`）へ微減し、`179.9 px`のteleportを1件出したため、deploy baselineは置換しません。
- DINOv3 + 3軸RoPE familyの現時点の研究起点は [`run-i579-phase3`](nodes/run-i579-phase3.md)（F1 `0.106`）です。これはfrozen-backbone系のfamily baselineであり、ConvNeXt U-Net deploy baselineと同格に扱いません。
- 3DGS simple-sphereを1/12混合した [`run-i618-3dgs-blcs-half-rate-v1-treatment-s731`](nodes/run-i618-3dgs-blcs-half-rate-v1-treatment-s731.md) は単一seedのgame9でpaired controlを `+0.018454 F1`上回りましたが、残りseedとgame10 final testが未登録なので候補止まりです。

### 置換条件

新しいball detectorを現行baselineへ昇格するには、少なくとも固定game9/game10 manifest、best-checkpoint評価、実clip coverage、teleport / jump、negative-frame FPRを同時に比較します。TrackNet F1だけの上振れでは置換しません。

## `court_detection`

### 決定

現行deploy baselineは [`run-i621-court-kp512-resume-r4`](nodes/run-i621-court-kp512-resume-r4.md) です。pipelineは `court_detection/kp/run-i621-court-kp512-resume-r4-epoch21.ckpt` を参照し、KP14、512入力、subpixel refinement、RANSAC後処理を使用します。

- checkpoint選択時のval best: `2.23 px`
- 2,211 sampleの再評価: `1.708886 px`
- tennis_clip手動GT比較: `0.76 px`

ただし `test_dataloader` が `data_val.json` を読むため、`1.708886 px`は独立held-out testではなくvalidation再評価値です。この制約を隠さず、次のbaseline更新ではrecording-disjoint testを必須にします。

### 別契約

[`group-i524-dinov3-ssl-court`](nodes/group-i524-dinov3-ssl-court.md) のcourt segmentationは別出力契約です。凍結backbone条件で非SSL baseline `0.517 mIoU`に対しSSL treatment `0.800 mIoU`ですが、KP14 deploy modelの代替値として比較しません。

point RANSACを点・線共同残差へ変えるpostprocessは、現行knowledge graphに正式run / groupがありません。detector checkpointとしては `run-i621-court-kp512-resume-r4` を固定できますが、`line_edge_support`、geometry valid率、wall timeを持つ**postprocess-only baseline run**を別途登録するまで、KP14 deploy baselineの改善として扱いません。

### 置換条件

独立testでのkeypoint距離、geometry valid率、line support、実clip E2EのBLCS / PLCS差、処理時間を固定します。detectorとpostprocessを同時変更したrunは、個別寄与を分離できないため主baseline比較には使いません。

## `plcs`

### 決定

現行deploy baselineは [`run-deploy-multiview-plcs-i590-courtkp14-v2`](nodes/run-deploy-multiview-plcs-i590-courtkp14-v2.md) です。pipelineは `plcs/run-multiview-plcs-i590-courtkp14-epoch197.ckpt` を参照します。

| metric | mean / rate | median |
|---|---:|---:|
| player position | 0.175284 m | 0.132863 m |
| player yaw | 6.443357° | 4.797728° |
| position ≤ 0.5 m | 0.962584 | — |
| yaw ≤ 15° | 0.913884 | — |

採用理由は、`multiview_axial_split` H=0/S=6、3–6 camera、court KP14、position weight 8、補助pose loss無効という実pipeline互換recipeを、親 [`run-i590-courtkp14`](nodes/run-i590-courtkp14.md) とは独立に再学習してほぼ再現したためです。

### 研究lineageとPareto challenger

| node | 役割 | position | yaw | deploy baselineにしない理由 |
|---|---|---:|---:|---|
| [`run-i518-exp10`](nodes/run-i518-exp10.md) | split trunk + aux-positionを確立した歴史的architecture baseline | 0.238 m | 9.98° | 初期契約。現行KP14 deploy以前 |
| [`run-i541-parameff-deeppose`](nodes/run-i541-parameff-deeppose.md) | parameter-efficiency frontier、19.5M | 0.202 m | 9.55° | 効率研究用。現行deploy recipeと契約差 |
| [`run-i545-s6-h0-auxoff-posw8`](nodes/run-i545-s6-h0-auxoff-posw8.md) | position frontier | **0.166 m** | 8.46° | court tokenが現行KP14契約へ揃う前の比較 |
| [`run-i560-nocanon-rs-s5-h2`](nodes/run-i560-nocanon-rs-s5-h2.md) | rotation frontier | 0.332 m | **7.54°** | positionとのtrade-offが大きい |
| [`run-i590-courtkp14`](nodes/run-i590-courtkp14.md) | KP14 deploy-compatible親run | 0.189 m | **6.28°** | checkpointがprune済みだったためdeploy再学習で確認 |

[`group-i535-asym-capacity`](nodes/group-i535-asym-capacity.md) で取り違えによりno-opだったrunは、数値が良くてもbaseline候補から除外します。valid rerunでもEX10を上回っていません。

### 2026-08-27以降のfamily / diagnostic baseline

| 契約 | baseline / treatment | 固定結果 | 判定 |
|---|---|---|---|
| canonical-only temporal motion | [`run-plcs-canonical-temporal-decomp-beta01-noaug`](nodes/run-plcs-canonical-temporal-decomp-beta01-noaug.md) | canonical MPJPE `0.091136 m`、motion amplitude ratio `1.174967`、centered Pearson `0.795146`、high-frequency fraction `0.391067`（GT `0.068930`） | 平均pose固定から入力依存motionへ移行したfamily diagnostic。jitterが大きく、position / rotation headは未学習なのでdeploy比較には使わない |
| interleaved V=2 / T=16 reprojection | [`group-plcs-reprojection-loss-w1`](nodes/group-plcs-reprojection-loss-w1.md) | position `6.605493 → 6.603486 m`、angle `91.731071 → 88.649841°`、Z `0.093369 → 0.275332 m` | 2D整合と向きは改善したがdepth悪化。default不採用 |
| axial V=4 / T=128 reprojection | [`group-plcs-multiview-axial-reprojection-loss-w1-v4-t128`](nodes/group-plcs-multiview-axial-reprojection-loss-w1-v4-t128.md) | position `1.386235 → 1.352761 m`、angle `66.968224 → 63.600704°`、0.5 m以内率 `0.118984 → 0.088828` | mean / medianと向きは改善した単一seed challenger。近距離率・X誤差・分散の悪化があり、複数seed前は昇格しない |
| camera-view v2 selector | [`run-i801-a2-plcs-d-reference`](nodes/run-i801-a2-plcs-d-reference.md) / [`selector-zero`](nodes/run-i801-a2-plcs-d-selector-zero.md) | reference: position `5.115064 m`、Y-sign `0.680313`、heading `35.71°`、ID switches `24.76`; zero: `5.141654 m`、`0.684531`、`34.32°`、`36.68` | 指標ごとに優位が逆転。reference-camera selectorの一方向な効果はなく、production v1を維持 |

これらはdataset、出力、loss、sequence length、view数がdeploy PLCSと異なるため、`run-deploy-multiview-plcs-i590-courtkp14-v2`を置き換えません。特にcanonical-only runのposition / rotation値と、camera-view v2 track-query runのmulti-person tracking値をsingle-person deploy値へ直接比較しません。

### multi-personは別契約

[`run-issue-643-multiperson-baseline`](nodes/run-issue-643-multiperson-baseline.md) はtrack query、Hungarian matching、presenceを含むmulti-person diagnostic baselineです。position `0.267685 m`、yaw `41.522079°`、presence F1 `0.762981`ですが、single-person PLCSと出力・データ・metricが違うため直接比較しません。

### 現行比較の境界

PLCSの現行pipeline改善判定では、deploy nodeとepoch197 checkpointを固定します。新しいarchitecture / loss / camera-view contractは、それぞれ同一protocolの対照runをfamily baselineとして持ち、deploy置換を主張する場合にだけ3–6 camera・court KP14・実pipeline互換recipeで再評価します。

## `blcs`

### 決定

現行deploy baselineは [`run-deploy-multiview-blcs-v3-simfix-c3-6-v2`](nodes/run-deploy-multiview-blcs-v3-simfix-c3-6-v2.md) です。pipelineは `blcs/run-multiview-blcs-v3-simfix-c3-6-epoch129.ckpt` を参照します。

| metric | value |
|---|---:|
| mean position error | 1.064595 m |
| X / Y / Z error | 0.426141 / 0.856086 / 0.186533 m |
| mean endpoint error | 2.024551 m |
| position ≤ 1.2 m | 0.759561 |
| endpoint ≤ 1 m | 0.53 |

単眼親 [`run-mono3d-blcs-bcast-v3-simfix`](nodes/run-mono3d-blcs-bcast-v3-simfix.md) に対し、positionは `1.845 → 1.065 m`、endpointは `3.408 → 2.025 m`へ改善しています。ただしcamera presetも変わるため、改善全量をview数へ帰属しません。

### 別目的のchallenger

- [`run-mono3d-blcs-gan-ft-v3`](nodes/run-mono3d-blcs-gan-ft-v3.md) は単眼in-distribution positionを `1.548 m`へ改善しましたが、実clip jerkは `0.280 → 0.278`でほぼ不変です。単眼accuracy challengerであり、multiview deploy baselineの置換候補ではありません。
- [`group-i593-physics-prior`](nodes/group-i593-physics-prior.md) のphysics ftCは実clip ball jerkを `0.280 → 0.106`へ改善した一方、in-distribution positionを `1.845 → 1.947 m`へ悪化させました。機能は採用されてもdefault checkpointはrevertされており、smoothness family baselineとしてのみ保持します。
- [`group-blcs-compile-training-abba-v4`](nodes/group-blcs-compile-training-abba-v4.md) は学習loop性能のbaselineです。compiledはsteady-state `1.90×`高速、peak CUDA allocated `19.6%`減ですが、cold-start込み3 epochは `2.98×`遅く、break-evenは約18 epochです。trajectory精度baselineではありません。

### normalization v2 / track-query family

[`group-i786-normv2-large-cuda-ablation-eb32`](nodes/group-i786-normv2-large-cuda-ablation-eb32.md) はT=128、V=4、effective batch 32、100 epochのsingle-seed architecture familyです。positionはA `3.522323 m`、identity continuityはB `17.04 ID switches`、presence / lifecycleはD `0.972115 F1`・birth/death `4.43 / 5.36 frames`が最良で、単一の総合勝者はありません。さらに旧versioned-v2 runtimeと現行mainではtracking Smooth-L1 / Hungarian betaおよびartifact schemaが異なるため、これはhistorical family evidenceであり、現行mainの再現baselineではありません。

[`group-i801-reference-selector-ablation`](nodes/group-i801-reference-selector-ablation.md) はcamera-view v2上で第三RoPE軸のreference selectorだけを変えたmatched comparisonです。selector-zeroはreferenceよりpositionが `3.721058 < 3.813505 m`、Y-sign accuracyが `0.877656 > 0.871094`で、reference selectorの明確な寄与は確認できませんでした。dataset / target frame自体がv1 deployと異なるため、production defaultはv1のまま維持します。

### multi-ballは別契約

短clipの最初のtrack-query diagnosticは [`run-issue-648-multiball-baseline`](nodes/run-issue-648-multiball-baseline.md)（position `0.337709 m`、presence F1 `0.847570`）です。512-frame lifecycleの現行比較起点は [`run-i648-blcs-lifecycle-v4-large-pointattn32-rope2d-t512-b1-100ep`](nodes/run-i648-blcs-lifecycle-v4-large-pointattn32-rope2d-t512-b1-100ep.md)（position `1.125558 m`、presence F1 `0.751386`）です。前者と後者もsequence length、data、training budgetが違うため、改善・悪化として直接比較しません。

今後multi-ball lifecycle arcを改善する場合はlong lifecycle runを対照に固定します。single-ball deploy trajectoryの後処理を対象とする場合は、現行multiview deploy checkpointから専用baseline runを作り、両契約を混ぜません。

## `slcs`

### 決定

SLCSにはdeploy baselineもrecording-disjointな汎化baselineもありません。既定の全共有trunk (`shared=2`) とDINO inputに対応する [`run-i634-slcs-overfit-dino`](nodes/run-i634-slcs-overfit-dino.md) を、**end-to-end経路のdiagnostic baseline**として採用します。

| metric | value |
|---|---:|
| player position | 0.470360 m |
| player yaw | 7.761920° |
| ball position | 1.954024 m |

このrunはtrain / val / testで同一13 windowを共有する100 epoch memorizationです。したがって、他タスクのdeploy値と並べてモデル品質を主張してはいけません。

[`run-i634-slcs-overfit-split-dino`](nodes/run-i634-slcs-overfit-split-dino.md) は完全分離trunkでyaw `5.751387°`、ball `1.615541 m`へ改善しますが、player positionは `0.512462 m`へ悪化します。これはdiagnostic Pareto challengerであり、汎化baselineへの昇格ではありません。

### baseline昇格条件

1. `recording_id`非重複のtrain / val / testを固定する。
2. shared DINO、split DINO、DINOなしを同一seed・budget・checkpoint selectionで比較する。
3. player position、yaw、ball positionに加え、欠損率、temporal jerk、uncertainty calibrationを報告する。
4. 少なくとも3 seedで平均と分散を記録する。
5. その最初のheld-out runが、SLCSの正式benchmark baselineになる。現状のoverfit値を引き継がない。

## 拡張契約のbaseline

| 契約 | baseline | 現状 |
|---|---|---|
| 3DGS renderer port | [`run-i618-renderer-port-cpu-reference-v1`](nodes/run-i618-renderer-port-cpu-reference-v1.md) | deterministic CPU reference。focused 10 / tennis_scene 195 tests pass |
| 3DGS full-scale ball dataset | [`run-i618-blcs-b00-full-scale-v1`](nodes/run-i618-blcs-b00-full-scale-v1.md) | 4,096 frame、64 clip、2,913 positive、1,183 negativeの生成契約 |
| 3DGS → ball detector効果 | [`run-i618-3dgs-blcs-real-baseline-v1`](nodes/run-i618-3dgs-blcs-real-baseline-v1.md) | immutable checkpoint / split / evaluatorの比較起点 |
| court segmentation | [`run-i524-court-seg-baseline`](nodes/run-i524-court-seg-baseline.md) | 非SSL `0.517 mIoU`。SSL treatmentは `0.800 mIoU` |
| multi-person PLCS | [`run-issue-643-multiperson-baseline`](nodes/run-issue-643-multiperson-baseline.md) | 初期diagnostic。single-personとは非互換 |
| short multi-ball BLCS | [`run-issue-648-multiball-baseline`](nodes/run-issue-648-multiball-baseline.md) | 初期diagnostic。lifecycleとは非互換 |
| long lifecycle multi-ball | [`run-i648-blcs-lifecycle-v4-large-pointattn32-rope2d-t512-b1-100ep`](nodes/run-i648-blcs-lifecycle-v4-large-pointattn32-rope2d-t512-b1-100ep.md) | 512-frame比較起点 |
| BLCS normalization-v2 architecture | [`group-i786-normv2-large-cuda-ablation-eb32`](nodes/group-i786-normv2-large-cuda-ablation-eb32.md) | A=position、B=identity、D=lifecycle。旧runtimeのhistorical family evidence |
| BLCS camera-view v2 selector | [`group-i801-reference-selector-ablation`](nodes/group-i801-reference-selector-ablation.md) | selector-zeroがposition / Y-signで僅差優位。production v1維持 |
| PLCS canonical temporal motion | [`run-plcs-canonical-temporal-decomp-beta01-noaug`](nodes/run-plcs-canonical-temporal-decomp-beta01-noaug.md) | mean-collapse解消のdiagnostic。高周波jitter未解決 |
| PLCS reprojection V=2 / T=16 | [`group-plcs-reprojection-loss-w1`](nodes/group-plcs-reprojection-loss-w1.md) | 向き・2D整合改善、Z悪化 |
| PLCS axial reprojection V=4 / T=128 | [`group-plcs-multiview-axial-reprojection-loss-w1-v4-t128`](nodes/group-plcs-multiview-axial-reprojection-loss-w1-v4-t128.md) | mean / median改善、近距離率・tailにtrade-off |
| BLCS compile性能 | [`group-blcs-compile-training-abba-v4`](nodes/group-blcs-compile-training-abba-v4.md) | eagerをruntime baseline、compiledをtreatmentとする |

正式run / group nodeがない構想は、この台帳へbaselineとして登録しません。まず固定対照・再現情報・評価値を持つrunを登録し、その後にfamilyまたはbenchmark baselineへ昇格します。

## 全group node監査

| group | baseline上の位置づけ |
|---|---|
| [`group-i518-rotation-frontier`](nodes/group-i518-rotation-frontier.md) | `run-i518-exp10`が最初のclean PLCS architecture baseline。現行deployにはsuperseded |
| [`group-i519-mcmc`](nodes/group-i519-mcmc.md) | 全条件悪化。baseline昇格なし |
| [`group-i520-canon-split-ablation`](nodes/group-i520-canon-split-ablation.md) | 分離効果のfamily ablation。trade-offのため主baseline変更なし |
| [`group-i521-velocity`](nodes/group-i521-velocity.md) | velocity-loss familyの比較起点は`run-i521-ex10-vel` |
| [`group-i524-dinov3-ssl-court`](nodes/group-i524-dinov3-ssl-court.md) | court segmentationの別契約。SSL treatmentが勝者 |
| [`group-i525-shared-vs-split`](nodes/group-i525-shared-vs-split.md) | split優位のparam-matched証拠。EX10 lineageを補強 |
| [`group-i530-canonical-head-ablation`](nodes/group-i530-canonical-head-ablation.md) | head familyのbaselineは`run-i530-direct-baseline` |
| [`group-i535-asym-capacity`](nodes/group-i535-asym-capacity.md) | no-op runを棄却。valid runはEX10を更新せず |
| [`group-i536-parameff-frontier`](nodes/group-i536-parameff-frontier.md) | parameter-efficiency winnerは`run-i541-parameff-deeppose` |
| [`group-i539-chunked-capacity`](nodes/group-i539-chunked-capacity.md) | data-rich Phase1 winnerは`run-i539-wide-chunked`。後続i545で更新 |
| [`group-i545-trunk-allocation`](nodes/group-i545-trunk-allocation.md) | position=`S6/H0`、rotation=`S4/H4`のParetoを確立 |
| [`group-i545-loss-head-tuning`](nodes/group-i545-loss-head-tuning.md) | position=`S6/H0 auxoff posw8`、balanced=`S4/H4 posw4` |
| [`group-i551-dinov3-rope`](nodes/group-i551-dinov3-rope.md) | DINO ball family初期baseline。タスク全体では低性能 |
| [`group-i560-nocanon-sweep`](nodes/group-i560-nocanon-sweep.md) | no-canonical rotation frontier。position trade-offあり |
| [`group-i576-camtoken-posw`](nodes/group-i576-camtoken-posw.md) | shared-trunk / readout-split family。deploy置換なし |
| [`group-i579-staged`](nodes/group-i579-staged.md) | DINO ball familyの現行研究baselineはPhase3 |
| [`group-i593-physics-prior`](nodes/group-i593-physics-prior.md) | accuracy / smoothnessの目的別challenger。defaultは置換しない結論 |
| [`group-i634-slcs-dino-overfit`](nodes/group-i634-slcs-dino-overfit.md) | SLCS diagnosticでDINO有効性を確認。汎化主張不可 |
| [`group-i634-slcs-compression-split-ablation`](nodes/group-i634-slcs-compression-split-ablation.md) | SLCS diagnostic Pareto。split DINOがyaw / ball最良 |
| [`group-i786-normv2-large-cuda-ablation-eb32`](nodes/group-i786-normv2-large-cuda-ablation-eb32.md) | BLCS track-query family。A=position、B=identity、D=presence/lifecycle。旧runtimeのためhistorical扱い |
| [`group-i801-reference-selector-ablation`](nodes/group-i801-reference-selector-ablation.md) | matched BLCS selector比較。reference selectorの改善なし、production v1維持 |
| [`group-plcs-reprojection-loss-w1`](nodes/group-plcs-reprojection-loss-w1.md) | V=2 / T=16のreprojection family。向き改善とZ悪化のtrade-off |
| [`group-plcs-multiview-axial-reprojection-loss-w1-v4-t128`](nodes/group-plcs-multiview-axial-reprojection-loss-w1-v4-t128.md) | axial V=4 / T=128 family。mean / median改善だが単一seed・tail悪化あり |
| [`group-blcs-compile-training-abba-v4`](nodes/group-blcs-compile-training-abba-v4.md) | runtime / memory baseline。精度baselineではない |

## 現行knowledge graphの監査境界

2026-08-28時点の正式node typeはrun / groupです。以前のproposal / paper nodeと`baseline_nodes`監査はknowledge graphから削除されたため、この台帳からも撤去しました。issue、PR、研究案だけではbaselineとみなさず、固定対照・再現情報・評価値を持つrunと、それを束ねるgroupだけを監査対象にします。

## 更新ルール

この台帳は次のいずれかが発生したときに更新します。

1. `pipeline.yaml` のcheckpointが変更された。
2. 新runが現行baselineを`supersedes`し、同一評価契約で再現された。
3. evaluation manifest、split、decode、metric定義が変わった。
4. diagnostic taskに初めてheld-out baselineが登録された。
5. 新しいrun / groupが、既存とは異なる評価契約のbenchmarkまたはfamily baselineを確立した。
6. knowledge graphのnode type、schema、runtime contractが変わり、この台帳の参照または比較境界が古くなった。

baseline変更時は、旧nodeを削除せず、旧baseline、置換run、評価契約差、置換理由をこの文書へ残します。単一metricの最高値だけでbaselineを動かしません。
