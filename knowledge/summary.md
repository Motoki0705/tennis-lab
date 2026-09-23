<!-- knowledge-review: ce3b1f56c576c18f79def76a9ac754e68a2e098b7dc6e0336e629ac8e3a44de3 on 2026-09-23 -->
# Tennis Lab Knowledge Summary

更新日: 2026-09-21（実RGB SLCSの全体版比較、コート推定・SfM診断、KP＋LINE下流移行を統合）

実RGB SLCSの130ノードをタスク別保存形式へ統合し、実験結果と採否を確認した。補助CLIの削除は学習結果・固定splitを変更せず、頑健性未達・固定test未評価という判断を維持する。詳細は[結果総括](reports/slcs-real-rgb.md)を参照。

従来の横断調査基準commit: `5f64fbd9c8fffc75295027eb2ece2a2f72eb6f9d`。追加実験の版・差分は各runの再現性bundleを参照。

この文書は、Tennis Labの学習・実験から得られた**現在の到達点、主要な知見、判断保留事項、次に解くべき課題**を横断的に把握するための要約です。個々の数値、再現手順、因果考察の正本は [`nodes/`](./nodes) のrun / group nodeと [`runs/`](./runs) の再現性bundleです。この文書は正本を置き換えず、研究状況を短時間で理解するための入口として使います。

現行knowledge graphの正式node typeはrunとgroupです。評価契約が異なる実験を同じランキングへ混ぜず、production、benchmark、family、diagnosticを区別して整理します。

## 2026-09-23の統合パイプライン確認

[Meiji clip_000の既定Court再生成](nodes/tennis_scene/000003-run-tennis-scene-cleanup-meiji-court-model-20260922.md)では、3camera×1010 frameに共通の完全14点校正frameがなく、後段の3D推論前に停止した。実行時checkpoint SHAは事前記録と一致した。既定hybridへの移行を実動画E2Eの精度保証とみなさない判断を維持する。[画像だけによる領域探索](nodes/court_detection/000032-run-court-meiji-model-only-regions-20260922.md)では、既存b863…checkpointと固定grid選択で27/27のhybrid推論・3camera校正が通った。手動点と外注ballは推論入力にしていない。[全3030frame](nodes/tennis_scene/000004-run-tennis-scene-meiji-court-regions-dino-block-20260922.md)でも全camera・全frameのKP14と校正が成立した。後段は旧DINO拡張のdeprecated APIで停止し、[正規buildと32frame chain確認](nodes/tennis_scene/000005-run-tennis-scene-dino-extension-chain-20260922.md)で演算互換性を確認した。[再実行の人物照合](nodes/tennis_scene/000006-run-tennis-scene-meiji-auto-person-mismatch-20260922.md)で、同じtrack IDでも隣接コート人物を選ぶauto選択を検出し、公開前に中止した。[モデルCourtの5m filter](nodes/tennis_scene/000007-run-tennis-scene-meiji-court-filter-5m-20260922.md)では対象へ戻った一方、cam2遠方選手の境界欠損と、広いROIの投影反転が判明した。[可視ROI修正と10m margin](nodes/tennis_scene/000008-run-tennis-scene-meiji-visible-court-roi-20260922.md)では、cam2遠方選手の直接観測が298→925frameとなり、全cameraで既存対象への軌跡対応を確認した。補間区間は残るため、次は全段生成時の姿勢照合と3D/動画評価を行う。

## 2026-09-21の追加確認

技術報告に伴う5ノードを確認し、実写homography postprocessと合成データ源の幾何的不確実性を更新した。以下の知見は既存deploy checkpointの昇格根拠にはしない。

- **Court detection / 実写homography postprocess**: [指定4写真の保存予測](nodes/court_detection/000028-run-court-supplied-photos-paper-20260918.md)に対し、[PROSAC](nodes/court_detection/000029-run-court-prosac-paper-20260920.md)と[KP・LINE共同推定](nodes/court_detection/000030-run-court-kp-line-hybrid-20260920.md)はいずれも4枚でHを生成した。共同推定はKPのみより予測LINEとの双方向内部整合を高めたが、人手GTがなく、段階間で採用KPも異なるため、実コートへの精度向上率や4写真外への汎化は未確認である。写真CのLINE欠落・対応の曖昧さも残る。次は会場分離の人手GTと固定モデル・解像度で旧H／PROSAC／共同推定の誤差、失敗率、棄却率、処理時間、下流E2Eを同条件比較する。保存出力による再検証は可能だが、ニューラル再推論には[記録済みのcheckpointハッシュ不一致](../paper/court_robustness/README.md)の解消が必要である。
- **Synthetic data / SfM幾何**: [B00の保存トラック診断](nodes/synthetic_data_generation/000022-run-court-sfm-ground-drift-20260919.md)に続き、[B00〜B03の共通2区間診断](nodes/synthetic_data_generation/000023-run-court-sfm-all-scenes-drift-20260920.md)でも、時間分割した共通地面セルに高さ不整合を観測した。SfM driftと整合する兆候だが絶対ドリフト誤差ではなく、符号はシーンごとに異なり、B01の偏りは小さく、B03は支持セルが少ない。地面凹凸・特徴点誤差・三角測量誤差も分離できないため、合成教師の幾何的不確実性として扱う。次は再訪で十分に重なる地面観測と独立地面基準を用意し、長距離構造制約・loop closureの有無を同条件で比較して下流court精度への影響を測る。

## 2026-09-20の追加確認

CIと登録SKILLの整合性を再確認した。保存形式・未完成の記録・summary本文の更新検出を強化した運用上の変更であり、実験結果や以下の研究判断には変更がない。

初回移行時点の全200ノードを7つの機能・研究トピックへ分割した。既存の実験IDと数値・再現bundleは保持している。論文の出典は[Papers](Papers/README.md)に一元化し、GVHMRを利用するPLCS記録に背景研究の参照を追加した。これは過去runが論文の手法を比較検証したという主張ではない。

9月に追加された知見のうち、次の判断を更新する。

- **Court detection**: [run-court-residual-wideline-pose-lora-b8-e20-s42-resume-e5-retry1](nodes/court_detection/000026-run-court-residual-wideline-pose-lora-b8-e20-s42-resume-e5-retry1.md)では親に対してKP平均誤差とline Diceが改善する一方、seg mIoUとpose再投影が悪化した。head構造とline幅を同時変更しており、単一要因の効果は分離できない。次は同一line schemaでheadだけを比較する。
- **PLCS / motion source**: [run-plcs-accad-gvhmr-meiji-1000-v1-train](nodes/plcs/000102-run-plcs-accad-gvhmr-meiji-1000-v1-train.md)は200 epochの学習と混合testを完走したが、向きとposeに改善余地がある。生成教師に対する評価で、実動画の独立3D正解への精度でもGVHMR追加の因果効果でもない。次は固定val/testでACCAD-only対混合train、未見収録holdoutを比較する。
- **Synthetic data / B00外観変換**: [50/100枚・7k/30k比較](nodes/synthetic_data_generation/000021-group-b00-clay-flare-images-steps-v1.md)では、共通評価8視点で100枚30kが今回の実験基準となった。50枚の学習を延ばすだけではSSIM低下と白線の薄れがあり、100枚版にも近いネットのぼけ・線の欠けが残る。[ホスト再起動による中断](nodes/synthetic_data_generation/000016-run-b00-clay-flare-nht-7k-interrupted-v1.md)は失敗として保存した。生成教師に対する単一シーン・単一seedの診断であり、実世界の幾何精度やproduction採用は未検証。次は元画像100枚の同条件対照で変換由来の誤差を分離し、編集範囲と視点間整合性を検討する。
- **統合・生成・UI検証**: 新しいtask区分によりデータ生成・smoke・統合診断を辿れる。これらの完走を推定精度の改善と混同しない。最新の個別条件・残課題はノード本文を正本とする。

以下のproduction/deploy表は**2026-09-04時点の調査記録**を保持している。今回の構造移行ではcheckpointの再評価・昇格をしておらず、9月20日の現行配備状態を保証する表には更新しない。新規結果を以前の異なるsplitと直接ランキングしない。

## 現在の全体像

現行pipelineでは、2D ball detection、court detection、single-person PLCS、single-ball BLCSにdeploy checkpointがあります。SLCSは全体版の収録分離学習・validation比較まで進みましたが、実世界での頑健性は未確立です。実RGB用の教師checkpointは専用生成profileで選択し、以下の従来pipelineのdeployとは区別します。

追加の[実RGB SLCSの実験群](nodes/slcs/000009-group-slcs-real-rgb.md)では、Meiji全体の生成・品質確認とbroadcastとの統合を完了しました。現在の課題は入力欠損・裾誤差・時間的スパイクであり、教師生成の未完了とは区別します。

従来の2026-08-30のBLCS観測ベース2D追跡比較（[#832のgroup](nodes/blcs/000031-group-i832-blcs-observation-tracking.md)、3 run）では、conservative設定が学習用associationの運用選択となりました。ただし単一seed・FP augmentation無効のfamily内比較であり、single-ball deployの置換や実動画での優位を示す結果ではありません。

| task | 現在の基準 | 主な固定値 | 現在の判断 |
|---|---|---|---|
| `ball_detection` | [`run-i618-convnext-v2-ft`](nodes/ball_detection/000009-run-i618-convnext-v2-ft.md) | test F1 `0.721789`、precision `0.735656`、recall `0.708436`、距離 `2.176208 px` | offline最高値ではなく、実clipのcoverageと軌道安定性を含めてdeploy継続 |
| `court_detection` | [`run-i621-court-kp512-resume-r4`](nodes/court_detection/000006-run-i621-court-kp512-resume-r4.md) | val best `2.23 px`、固定checkpoint再評価 `1.708886 px` | 旧KP14 / 512入力deployの比較値。現在のhybrid移行は別評価で、独立held-out testは未確立 |
| `plcs` | [`run-deploy-multiview-plcs-i590-courtkp14-v2`](nodes/plcs/000082-run-deploy-multiview-plcs-i590-courtkp14-v2.md) | position `0.175284 m`、yaw `6.443357°` | 3–6 camera・court KP14の現行single-person deploy |
| `blcs` | [`run-deploy-multiview-blcs-v3-simfix-c3-6-v2`](nodes/blcs/000011-run-deploy-multiview-blcs-v3-simfix-c3-6-v2.md) | position `1.064595 m`、endpoint `2.024551 m` | 3–6 camera・court KP14の現行single-ball deploy |
| `slcs` | [全体版のval5条件](nodes/slcs/000062-run-slcs-full-no-smooth-gap-rgb-val-v2.md) | 全61clipの固定split、60epoch、入力条件・train定数baseline比較 | ball低分散崩壊を脱しRGBの寄与を確認。欠損・裾・時間的スパイクは残り、頑健なdeployとは未認定 |

[pipeline設定](../src/tennis_scene/configs/pipeline.yaml)が参照するcheckpoint（2026-09-21、下流移行後）は次です。

| stage | checkpoint |
|---|---|
| court | `court_detection/hybrid/court-detection-epoch=17.ckpt` |
| ball | `ball_detection/run-i618-convnext-v2-ft-epoch13.ckpt` |
| PLCS | `plcs/real-rgb-meiji-foot-e60-v1.ckpt` |
| BLCS | `blcs/real-rgb-meiji-e60-v1.ckpt` |

## 2026-09-21のKP＋LINE下流移行

[run-court-hybrid-downstream-migration](nodes/court_detection/000031-run-court-hybrid-downstream-migration.md)では、ユーザー指定の残差head checkpointへ共通KP＋LINE推論を接続し、下流もcamera_view_v2へ移行した。8画像のH採用は3例であり、推定完了率の改善や実動画E2E精度は未確立。既定の変更は入力契約の統一であって、旧モデルへの精度優位の証明ではない。B00〜B03の保存alignmentと生成データは再publicationしていない。

以下の既存baseline比較は元のas-of commitに基づく履歴として保持する。現在のpipeline checkpointは下表へ更新し、PLCS/BLCSはMeiji fine-tune版・window128・明示したreference-camera契約を使う。他会場の汎化、独立正解Hでの誤採用率、下流E2E評価を次の課題とする。

## タスク別の主要な知見と判断保留事項

### Ball Detection

現行deployはfine-tuning版を維持します。[`run-i618-convnext-v2-scratch`](nodes/ball_detection/000010-run-i618-convnext-v2-scratch.md) はTrackNet test F1 `0.7692`、距離 `2.01 px`でoffline評価では上ですが、実clip coverageが`92.0% → 91.1%`へ下がり、`179.9 px`のteleportを1件発生させました。したがって、単一のF1最高値より実動画上の安定性を優先しています。

3DGS augmentationでは、固定checkpoint・split・decodeによる比較基盤 [`run-i618-3dgs-blcs-real-baseline-v1`](nodes/ball_detection/000011-run-i618-3dgs-blcs-real-baseline-v1.md) が整備されています。simple-sphereを1/12混合したtreatmentは単一seedのgame9で`+0.018454 F1`でしたが、残りseedとgame10 final testが未完了のため、効果は確立していません。

[Meijiの全scene診断](nodes/tennis_scene/000009-run-tennis-scene-meiji-raw-ball-baseline-20260923.md)ではscene/7動画の構造・decodeは成立したが、Ball欠損と非物理的3D軌道が大きかった。[保存前処理の照合](nodes/ball_detection/000018-run-ball-checkpoint-normalization-meiji-20260923.md)で、公開RGB APIとcheckpointのImageNet正規化の接続漏れを確認した。修正はdataset前処理と実model入力が完全一致し、Meiji選定窓の大誤検出は減ったが、recall改善は一様でなくTrackNet 8frameの4px一致数は5→4だった。前処理復元と精度向上を同一視せず、次は修正後の全区間GPU・3D・動画を再評価する。

[修正版の単発scene](nodes/tennis_scene/000010-run-tennis-scene-meiji-corrected-pipeline-20260923.md)と[独立dataset生成](nodes/tennis_scene/000011-run-tennis-scene-meiji-corrected-dataset-20260923.md)は完了し、両sceneの構造と全14動画の全frame decode、既存SLCS reader受理を確認した。Courtは全区間で成立したが、Ball欠損は62.5/33.9/34.0%、3D ballの負高さ50frame・最大412m/s、PLCS/GVHMR整合残差が残る。窓境界不整合の証拠はなく、2D観測/pose mask急変が異常と同時にある。scene公開の成立を高品質教師や3D精度保証とみなさず、次は観測の同一性・可視性の安定性と独立3D評価を分けて検証する。

### Court Detection

[Meiji全frame処理の時間分解](nodes/court_detection/000033-run-court-meiji-hybrid-cpu-profile-20260923.md)では、3030frameのCourt工程が約116分だったのに対し、3cameraの各1frameでもCPU hybrid geometry単体が1.77–2.58秒を要した。GPU推論だけの所要時間とは扱わない。精度評価と並行して、同じframeごとの推定契約を保つCPU後処理並列化・GPU batch化を検証する価値がある。静止frameの複製や間引きによる結果変更とは区別する。

KP14 detectorは実pipelineで利用可能な水準ですが、`1.708886 px`は`test_dataloader`がvalidation dataを読む条件の再評価値であり、独立testではありません。次の品質更新にはrecording-disjoint test、geometry valid率、line support、処理時間、PLCS / BLCSへのE2E影響が必要です。

court segmentationはKP14とは別契約です。[`group-i524-dinov3-ssl-court`](nodes/court_detection/000005-group-i524-dinov3-ssl-court.md) では凍結backbone条件で非SSL `0.517 mIoU`からSSL `0.800 mIoU`へ改善しましたが、KP14 deploy modelの置換根拠にはなりません。点・線共同postprocessは固定した保存予測4枚に対するdiagnosticまで成立しましたが、held-out人手GT付きの正式なpostprocess-only benchmarkはまだありません。

### PLCS

従来deploy（上記as-of commit）は、split trunk、H=0/S=6、position weight 8、補助pose loss無効、court KP14というpipeline互換recipeです。過去のablationでは、positionにはS6/H0、rotationには別の容量配分が有利であり、単一構成が全目的を同時に最適化しないPareto構造が確認されています。[`group-i545-loss-head-tuning`](nodes/plcs/000067-group-i545-loss-head-tuning.md) のposition frontier `0.166 m`は有力ですが、現行KP14 deploy以前の契約なので直接置換には使いません。

canonical poseでは、[`run-plcs-canonical-temporal-decomp-beta01-noaug`](nodes/plcs/000087-run-plcs-canonical-temporal-decomp-beta01-noaug.md) が平均pose固定から入力依存motionへの移行を確認しました。canonical MPJPEは`0.091136 m`、motion amplitude ratioは`1.174967`、centered Pearsonは`0.795146`です。一方、high-frequency fractionは予測`0.391067`に対してGT `0.068930`であり、motionを復元する代わりにjitterを過剰生成しています。position / rotation headは未学習なのでdeploy精度との比較には使いません。

reprojection lossは一方向な改善ではありません。[`group-plcs-multiview-axial-reprojection-loss-w1-v4-t128`](nodes/plcs/000090-group-plcs-multiview-axial-reprojection-loss-w1-v4-t128.md) のV=4/T=128条件では、position `1.386235 → 1.352761 m`、angle `66.968224 → 63.600704°`へ改善しましたが、0.5 m以内率は`0.118984 → 0.088828`へ悪化し、X誤差と分散も増えました。複数seedとweight sweep前にdefaultへ採用しません。

camera-view v2のreference selectorも決着していません。PLCSではreferenceがpositionとID switchesで良い一方、selector-zeroがY-sign、heading、presenceで良く、指標ごとに優位が逆転しました。このselector比較単独ではv1からの移行根拠になりません。2026-09-21のpipelineは上記の入力契約移行によりv2へ変更しています。

### BLCS

single-ballではmultiview deployが単眼親よりposition `1.845 → 1.065 m`、endpoint `3.408 → 2.025 m`へ改善しました。ただしcamera presetも変わるため、改善全量をview数へ帰属できません。

physics priorはaccuracyとsmoothnessのtrade-offです。[`group-i593-physics-prior`](nodes/blcs/000004-group-i593-physics-prior.md) のftCは実clip jerkを`0.280 → 0.106`へ改善しましたが、in-distribution positionを`1.845 → 1.947 m`へ悪化させました。そのため機能は残してもdefault checkpointは置き換えていません。

track-query architectureの[`group-i786-normv2-large-cuda-ablation-eb32`](nodes/blcs/000019-group-i786-normv2-large-cuda-ablation-eb32.md) では、positionはA `3.522323 m`、identity continuityはB `17.04 ID switches`、presence / lifecycleはD `0.972115 F1`・birth/death `4.43 / 5.36 frames`が最良でした。単一の総合勝者はありません。また、この群は旧versioned-v2 runtimeで、現行mainとloss beta・artifact schemaが異なるためhistorical family evidenceとして扱います。

[`group-i801-reference-selector-ablation`](nodes/blcs/000025-group-i801-reference-selector-ablation.md) のmatched BLCS比較では、selector-zeroがreferenceよりposition `3.721058 < 3.813505 m`、Y-sign accuracy `0.877656 > 0.871094`でした。第三RoPE軸によるreference明示の追加効果は確認できず、production v1を維持します。

観測ベース2D追跡の[`group-i832-blcs-observation-tracking`](nodes/blcs/000031-group-i832-blcs-observation-tracking.md)では、GT lifecycleに依存する旧random-slot入力を、noise後の2D観測からdeterministicに対応付ける入力へ変更して比較しました。3 runは同一の`blcs/multi_object` split、`blcs_track_query`（Q=4）、seed `832`、100 epoch、FP augmentation無効で揃えています。

| run / association | position error (m) ↓ | presence F1 ↑ | ID switches ↓ | duplicate active tracks ↓ | missed GT frames ↓ |
|---|---:|---:|---:|---:|---:|
| [legacy random slot](nodes/blcs/000028-run-i832-blcs-legacy-slot-baseline.md) | 5.434213 | 0.878670 | 0.64 | 80.22 | 26.76 |
| [conservative（距離閾値0.04・保持2 frames）](nodes/blcs/000029-run-i832-blcs-tracker-conservative.md) | 5.430877 | 0.881151 | 0.66 | 86.19 | 24.63 |
| [permissive（距離閾値0.10・保持8 frames）](nodes/blcs/000030-run-i832-blcs-tracker-permissive.md) | 5.435566 | 0.878761 | 0.60 | 83.53 | 25.32 |

表は各runの`metrics.json` / `diagnostic_metrics.json`に基づくtest実測値です。conservativeはposition / F1 / missed GT framesで最良ですが、baseline比のposition改善は約`0.00334 m`にとどまり、ID switchesとduplicate active tracksは増えています。permissiveはID switchesが最良で、F1もbaselineを僅かに上回りますが、positionとprecisionは悪化しています。全runが100 epochを完走し、曲線に発散やNaNは報告されていません。低いID switchesだけでtracking failureが解消したとは判断しません。

#832ではconservativeを運用選択として採用し、[現行association設定](../src/tasks/blcs/configs/data/_observation_tracking.yaml)も`max_distance=0.04`、`max_missed_frames=2`です。ただし、指標間の許容差・重みと複数seed評価は未確定で、因果的・統計的な優位は未確立です。次はこの条件を基準に重複trackとID切替の増加を検証し、FP augmentationを有効にした条件と実検出入力での評価を分けて確認します。

#832のID switchesはpost-#824の定義です。#643 / #648 / #650などのpre-#824の保存値とは直接比較できません。再現時は各bundleのcommitと`uncommitted.patch`を確認します。特にlegacy baselineとconservativeには未commit差分が保存されており、現行mainの設定だけで同じ学習条件になるとは限りません。

学習性能では、[`group-blcs-compile-training-abba-v4`](nodes/blcs/000014-group-blcs-compile-training-abba-v4.md) によりcompiled実行がsteady-stateで`1.90×`高速、peak CUDA allocatedが`19.6%`減る一方、cold-start込み3 epochでは`2.98×`遅く、break-evenは約18 epochと分かっています。これはtrajectory精度ではなくruntime baselineです。

multi-ballはsingle-ballと別契約です。短clip diagnosticと、[`run-i648-blcs-lifecycle-v4-large-pointattn32-rope2d-t512-b1-100ep`](nodes/blcs/000013-run-i648-blcs-lifecycle-v4-large-pointattn32-rope2d-t512-b1-100ep.md) の512-frame lifecycle baselineも、sequence length・data・training budgetが違うため相互に直接順位付けしません。

### SLCS

従来のshared/split DINO比較はtrain / val / testが同じ13 windowを共有するmemorization実験でした。[収録分離パイロット](nodes/slcs/000009-group-slcs-real-rgb.md)で残ったball低分散出力は、[全体版60epoch](nodes/slcs/000072-run-slcs-full-real-rgb-no-ball-smooth-e60-v3.md)では改善し、[同じvalの5入力条件](nodes/slcs/000062-run-slcs-full-no-smooth-gap-rgb-val-v2.md)でtrain平均位置定数との差とRGBの寄与を確認しました。pilotと全体版は更新数・評価clip構成が異なるため、両者の差を単一施策の効果とは呼びません。

[Meiji全件監査](nodes/slcs/000108-run-slcs-meiji-v9-full-qc-v2.md)と[broadcastとの統合](nodes/slcs/000134-run-slcs-real-rgb-full-assembly-v1.md)は完了し、全体版testにはMeijiの別収録もあります。既存基準runの自動testは終端last重みの記録であり、候補選定根拠ではありません。追加探索では自動testを無効にし、validationで採否を決めます。[gap48比較](nodes/slcs/000069-run-slcs-full-real-rgb-gap48-val-v1.md)は全体平均を改善してもbroadcast full/gapが悪化したため基準置換を見送りました。教師は独立実測3D正解ではなく、欠損区間の大誤差と時間的スパイクは未解決です。最新の施策・選定・限界は[実験群](nodes/slcs/000009-group-slcs-real-rgb.md)を参照してください。

なお、[速度整合候補の初回val評価](nodes/slcs/000081-run-slcs-full-real-rgb-velocity-val-interrupted-v1.md)は再起動後に空出力が見つかり採用不可。
その後のユーザーのgoal優先指示でローカル作業を再開し、[新しいval5条件評価](nodes/slcs/000082-run-slcs-full-real-rgb-velocity-val-v2.md)は完走しましたが、位置平均と欠損境界の退行により基準置換を見送りました。Windowsクラッシュの原因は未確定です。

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
6. associationの生成方法、FP augmentation、ID switchesの定義も比較条件に含める。学習入力の運用設定の採用と、production checkpointの更新は別の判断として記録する。

## 優先して解くべき課題

| 優先度 | 領域 | 不足している証拠 | 完了条件 |
|---|---|---|---|
| S | Court Detection | recording-disjoint test | 固定モデル・解像度で旧H／PROSAC／共同推定のKP距離、geometry valid、line support、失敗・棄却率、wall time、下流E2Eを比較 |
| S | PLCS canonical motion | jitter抑制とmulti-task再導入 | motion相関を保ち、high-frequency fractionを低下させ、position / rotation併用でもmean collapseしない |
| S | BLCS観測ベースtracking | #832が単一seed・FP augmentation無効、重複track増加 | 許容差・重みを先に固定し、同一条件で3 seed以上を比較。position / presence / ID / duplicate / missedを併記し、FP有効条件と実検出入力でも評価 |
| A | BLCS track-query architecture | #786が旧runtime・入力契約 | associationとpost-#824 metricを固定し、A/B/Dをcurrent loss・schema、3 seed以上で再実行してposition / ID / lifecycleのParetoを確認 |
| A | PLCS reprojection | 単一seed・weight 1のみ | weight `0.1/0.3/1.0`を複数seedで比較し、meanだけでなく0.5 m率・軸別誤差・tailを改善 |
| A | SLCS | 入力欠損・位置の裾・時間的スパイクと擬似教師の限界 | 固定valのdomain別・高速区間別で施策を比較し、選定後testと複数seedで再現性を確認 |
| A | Ball 3DGS augmentation | campaign未完了 | 残りseedとgame10 final testを固定protocolで完了 |
| B | multi-person / multi-ball | deploy互換E2E評価が不足 | single-object契約と分離したまま、lifecycle・presence・identityを長sequenceで評価 |

## このknowledge directoryの読み方

- [`summary.md`](./summary.md): 現在の到達点と未解決課題を俯瞰する入口。
- [`nodes/`](./nodes): 各runの設定・metric・考察、および関連runをまとめるgroup nodeの正本。
- [`runs/`](./runs): `repro.sh`、`metrics.json`、`pred_test.npz`、収束曲線などの再現性bundle。
- [`README.md`](./README.md): knowledge graphのschema、登録方法、検証手順。
- [`webui/`](./webui): node間の関係と実験結果をグラフとして閲覧するUI。

このsummaryは、pipeline checkpointが変わったとき、同一契約で再現された重要な結果が追加されたとき、評価契約が変わったとき、またはdiagnostic領域に初めてheld-out baselineができたときに更新します。新runが1件追加されるたびに追記するのではなく、研究上の結論または優先順位が変わった場合に更新します。
