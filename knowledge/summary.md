# Tennis Lab Knowledge Summary

更新日: 2026-11-25

研究トピックごとの**現在の判断**（採用中の構成とその根拠、未解決の課題、次の実験）をまとめた入口です。数値・条件・考察の正本は各[ノード](nodes/)、再現手順は[runs](runs/)にあります。2026-11-05までの日付順の記録は、[凍結アーカイブ](history/summary-2026-11-05.md)に原文のまま残しています。

- 標準pipelineのcheckpointと採用根拠: [deployments.yaml](registry/deployments.yaml)（CIがpipeline設定と照合する）
- トピックとsrcモジュールの対応、廃止したトピック: [tasks.yaml](registry/tasks.yaml)
- ノードの絞り込み: `kg_query.py`（[knowledge-control skill](../.agents/skills/knowledge-control/SKILL.md#plan-an-experiment)）

採用構成・評価契約・研究の優先順位が変わったときに、該当する節を書き換えます。runを1件追加するたびの追記はしません。結論を変えないrunは、ノードの `summary_impact: none` と本文で扱います。

## 標準pipeline（tennis_scene）

標準のシーン再構成は、各カメラのコートkeypoint・人物box/pose・ボールの2D観測からカメラを校正し、人物の関節とボールを三角測量します。PLCS・BLCS・SLCSの学習モデルは標準pipelineに入っていません。

- **clip長の上限**: 約30fpsの全長gridを2048点まで許容し、超過は切り詰めず拒否します。[1055点のCPU確認](nodes/tennis_scene/000047-run-frame-sampling-limit-20261117.md)ではcourt_side・camera_alignment・body_view_selectionが完了し、上限4096との出力も一致しました。下流2段は人物対応未生成のためballのみ・暫定ID・制御入力を区別した検証で、全69clipのGPU完走を示していません。
- **明示profileのボール部品**: X3D/CSN＋DPT T32の[scene接続](nodes/tennis_scene/000043-run-scene-x3d-dpt-20261116.md)は、人間が現状の証拠で受入れました（2026-11-16）。学習時validationとの許容差超過は数値のまま保持し、既定i935は維持します。meiji_4camの通常4cameraはcourt推定で停止しており、3D比較は条件付き結果です。各clipの動画確認とcourt停止への対処は人間が判断します。
- **動作確認**: 公開3カメラサンプル（Meiji clip_000）で、既定設定がSceneResult v2の保存まで完走しました（[標準pipeline](nodes/tennis_scene/000028-run-multiview-sample-standard-pipeline-20261021.md)、[クリーン導入](nodes/tennis_scene/000030-run-clean-standard-install-20261021.md)）。1clipの動作確認であり、独立した3D精度や全frameの目視合格は示していません。
- **ボール3Dの平滑化**: 三角測量の後、局所運動のconsensusで外れ値を除き、頑健な2次LOESSをかけます（既定で有効）。[導入時の比較](nodes/tennis_scene/000031-run-scene-ball-loess-20261026.md)では3D有効frameが821→859/1010に増え、隣接移動量のp95も下がりました。一方、元の2D観測への再投影中央値は3.554→4.810 pxに増えました。滑らかさと観測追従の交換関係であり、3D精度の向上とは扱いません。
- **ボール3Dの評価基盤**（2026-11-03）: clip_000の人間参照を使い、[固定入力のCPU replay比較](nodes/tennis_scene/000032-run-ball3d-evaluation-foundation-20261103.md)を作りました。[レビュー修正後](nodes/tennis_scene/000033-run-ball3d-foundation-review-20261103.md)、全方式に共通の768点でのLOVO中央値/p90は、既定が8.97/25.40 px、LOESSなしが9.70/28.02 pxです。共通点は全対象の52.53%に限られ、可用性を条件とする比較です。[誤差分解](nodes/tennis_scene/000034-run-ball3d-error-decomposition-20261103.md)では、top-8候補にargmax失敗の約半数を回収できる候補がある一方、既存の採用maskを保つ3D候補oracleの改善は小さく、校正誤差と同期ずれも分離できていません。
- **6方式の比較と採否**: [統合比較](nodes/tennis_scene/000041-run-ball3d-integrated-comparison-20261104.md)では後半LOVOとcoverageの一貫した改善は得られず、人間はbaseline維持を決定しました。後処理・三角測量側の追加改善はキャンセルされ、検出モデルの学習戦略の改善が必要と判断されました。比較は1 clip・近似校正・推定を含む2D参照に限られます。詳細は[凍結アーカイブ](history/summary-2026-11-05.md)を参照してください。

- **動画一括推論の現行本番構成**: [player証拠と特徴分離のCPU統合](nodes/tennis_scene/000072-run-i1752-player-main-cpu-20261119.md)で、共有コートload・clip単位のplayer side判定・ball両機能off・ダブルス2・DINO/ViTPose fp16を明示します。ballありのbenchmarkも共通設定で選択可能です。保存済み3clipの単独/共有判定は#2270と完全一致しました。全69clipのplayer証拠の採否、新構成のGPU再測定と本番実行は未確認です。
- **動画一括推論の過去GPU証拠**: C4の[共有court_side・明示fp16・ダブルス2](nodes/tennis_scene/000064-run-i1752-shared-fp16-20261118.md)で000/037/027の全114nodeが1計画で完了し、GVHMR配置・scene公開まで通りました。各side2 identity、C3管理7resourceは各1load/unload。CPU AFLinkの都度loadは別に記録しています。[同じfp16での単一027比較](nodes/tennis_scene/000065-run-i1752-shared-fp16-reference-20261118.md)は最終scene29配列・metadata・全離散値がexactですが、Ball補助evidence9配列が従来許容を超え、全component一致は未達です。原因と採否は親レビューに残します。69clipは未実行で、25.27〜25.88時間はballありの短い3clipからの条件付き投影で、現行player-only本番の時間ではありません。C1同期・初回compile等を含み、共有side・精度・profilerの差を単独の高速化率にはしません。

- **TensorRTの人物モデル単体比較**　ViTPose-HはFP16が許容差内で中央値54.423→8.468 ms。INT8・FP8には最大5 pxの基準を超える関節があり、E2E候補はFP16です。HMR2はFP8も単体精度の基準を満たし、中央値5.399 msでした。CLIP-ReIDの新INT8・FP8は、同じ2クリップから設定選定用12 cropを除いた224件で、cosine P05が0.999602／0.995416、FP32とのID一致が両方224/224でした。下流の時系列・人物対応と速度はE2Eで確認します。[固定入力の全条件比較](nodes/tennis_scene/000077-run-i2303-vit-tensorrt.md)
- **TensorRT FP16の全クリップ評価**　固定court・ボール無効で `meiji_4cam/video_000` の69件を一括評価し、55件が生成されました。14件は人物対応またはコート向きで停止しました。再構築は12時間10分58秒、Peak VRAM 1.76 GiB、Peak RSS 6.16 GiBでした。GPU遊休時のCPU処理ではプレーヤーの三角測量が最多でした。採否は判断待ちです。 [全件評価と診断](nodes/tennis_scene/000083-run-i2363-trt-player-video.md)
- **人物モデルの共通速度基準**　固定6 clipで現行DINO、ViTPose-Hのflipあり・なし、HMR2特徴＋GVHMRのPT16・TRT16を再計測し、全22条件で計時が完了しました。parityは未判定です。検出・姿勢の比較図には疑似GTによる精度と実測速度を使います。PT32の暫定速度と初回parityは初回記録を参照してください。 [再計測の基準値](nodes/tennis_scene/000086-run-person-model-speed-low-precision.md) · [初回記録](nodes/tennis_scene/000085-run-person-model-speed-baseline.md)
- **2D姿勢の置換候補**　固定疑似BBoxではRTMPose-X TRT16が、ViTPose-H flipありTRT16に対しbatch 1のwall P50を28.7%、P95を10.1%、NVMLピークを60.2%削減し、平均OKSは0.95ポイント低下しました。実DINO BBoxと3D再構成で下流品質を確認する優先候補です。採用にはconfidenceの較正と重み・engineの利用条件の確認が残ります。 [固定疑似GTと実測の比較](nodes/tennis_scene/000091-run-meiji-topdown-pose-comparison.md)

## Court Detection

- **現行**: `multiscale_depth3` のcheckpointと、画像だけによる領域探索を組み合わせ、各カメラの先頭frameだけを推論します。[Meiji 3cameraの校正が通った](nodes/court_detection/000032-run-court-meiji-model-only-regions-20260922.md)ことが採用の根拠です。このcheckpointを学習したrunのノードはなく、独立したheld-out testでの精度も未確立です。
- **合成データのscene組合せ**（#1581）: [全6ペアと共通4scene評価](nodes/court_detection/000067-group-court-scene-pair-s42.md)が完了しました。固定5,500枚・33,160更新・seed42では、同群2ペアの実画像KP中央値平均57.36 pxに対し、異群4ペアは4.57 px（δ=+52.80 px）で、全5指標が同群側で悪い向きでした。P03（B00+B03）が実画像KP中央値で最良（2.97 px）、共通合成の4scene平均ではscene-4が最良です。単一seed・再開・会場とsurface等の交絡があり、一般的な因果効果や採用変更にはしません。旧[1〜4scene比較](nodes/court_detection/000055-group-court-scene-diversity-colab-s42.md)は当時の4条件の証拠として保持します。
- **backbone**（#983）: [凍結DINOv3 S/S+/B/Lの比較](nodes/court_detection/000044-group-i983-court-dinov3-shared1024-s42.md)では、出力ごとに最良のサイズが異なり、サイズによる一様な改善は見られませんでした（1 seed）。
- **後処理**（#2190）: revision 4で1frameの精度目標を動画共有へ移した。[旧処理復元](nodes/court_detection/000075-run-court-legacy-default-20261117.md)は全3545画像・8530領域の採否・Hが旧処理と完全一致。clip既定は`legacy`・RGB無効のまま、白線・品質部品と[run 69〜74](nodes/court_detection/000074-run-court-semantic-support-20261117.md)の負の結果を保持する。
- **動画共有**（#2191）: [人間が共通Hを承認](nodes/court_detection/000078-run-court-video-drift-lines-20261117.md)し、検査不能を警告にする明示方針 `unobservable_policy=warn` を既定にした。H・測定値・hashは維持し、全69clip×4cameraを利用可能（159/276件warning、失敗0）。確認範囲は3clip×4camera＋原寸crop。信頼できるmean10/max25px超過は失敗を維持する。自動診断は人工24/24一致だが、実データの隣接線誤抽出等が残り、clip途中のcamera移動を見逃す可能性がある。候補保持・線配置による対応検証は将来の改善。一括CLI接続は#1752が担当する。
- **次の比較候補**: 収録を分けた実画像testで、従来のhomography推定・PROSAC・共同推定のKP距離、geometry valid率、失敗率、処理時間、下流E2Eを比べます。scene組合せは追加seed、3sceneから1つずつ抜く配置、会場を固定したsurface対照が候補ですが、今回追加実行はしていません。画像数スケーリングは実施しない判断を維持し、B/Lの複数seedも残っています。

## Ball Detection

- **現行**: #935の混合FTのepoch 9です。2026-10-01のユーザー判断で既定にしました。根拠ノードは廃止したball_refinerトピックにあります（[epoch 9の検証](nodes/ball_refiner/000015-run-i935-evidence-mixed-e9-trainval-r17-20260930.md)、[既定切替](nodes/ball_refiner/000028-run-i935-confidence-r29-20261001.md)）。固定BゲートのGT比較は、p90が許容を超えて悪化したため不合格のままです（[run26](nodes/ball_refiner/000025-run-i935-source-b-gate-r26-20261001.md)）。精度の合格判定を経た採用ではありません。
- **採否の原則**: offline F1の最高値だけでは採用しません。[scratch版](nodes/ball_detection/000010-run-i618-convnext-v2-scratch.md)はF1が高い一方、実clipのcoverage低下とteleportを起こしたため採用しませんでした。[#934の混合FT](nodes/ball_detection/000022-run-i934-meiji-holdout-e0-r7.md)でも、Meiji holdoutのrecallは上がりましたが採用検出のp95が悪化し、欠損の低減と誤検出の抑制は両立していません。
- **長系列モデル**（親#1367）: 教師は入力Tの全frame、文脈はclip内・上限128と[決めました](nodes/ball_detection/000063-run-ball-all-frame-protocol-20261022.md)。配布storeの可搬化と、Colabでの逐次保存・新VMからの再開も確認済みです。[共通DPTからの逐次T64学習](nodes/ball_detection/000068-run-ball-sequential-length-20261025.md)では、query T64の平均/P95誤差（28.20/116.89 px）がDPT T64（36.56/183.20 px）より小さい一方、20 px以内の率は低くなりました（76.29%対86.58%）。追加の更新数と方式の差が交絡しており、seed42・validationだけの結果です。
- **事前学習＋事後学習**: [ConvNeXt V2＋DPTとquery事後学習](nodes/ball_detection/000060-run-i1049-convnext-query-posttrain-s42.md)は通常評価で親より改善しましたが、camera stress（特に1/4 FPSで41.11 px）で大きく悪化しました。頑健性の要件は満たしていません。
- **mHC queryの容量と構成**（#1675・#1899）: X3D/CSN＋4query/mHCの7容量条件は[12k学習](nodes/ball_detection/000088-group-i1675-capacity-12k-s42.md)でmean32.48〜35.38 pxとなり、最良のd512・8層でも旧query基準28.20 pxに届きません。[旧基準から1要素ずつ変更した4条件比較](nodes/ball_detection/000094-group-i1899-controlled-query-ablation-s42.md)も完了しました。4query＋mHCへの変更、事前学習込みのCNN置換はいずれも主指標を悪化させ、[前半からT64にした③](nodes/ball_detection/000093-run-i1899-convnext-1query-t64-t64-s42-12k.md)はmean28.53 px（基準比+0.33）で優位を確認できませんでした。③は途中でL4→RTX5060Tiへ移管したため、小差から系列長単独の効果や同等性は判定しません。全条件seed42・同じvalidationのみで、mHC単独・交互作用・seed分散と独立testは未確定です。12層・24層は保留です。
- **構成の効率**（#1859）: [L4の未学習計測](nodes/ball_detection/000079-run-ball-arch-efficiency-s42-20261111.md)では、CNN段の配置で推論時間が最大22%減り、MSDAを含むquery（T1/T2）は全層cross-attentionの約2倍の更新時間でした。本学習5条件はT1の[前半6k](nodes/ball_detection/000091-run-i1859-t1-front-t64-s42-6k.md)（mean 43.17 px）と後半5,500更新の保存で止まり、精度比較は未完です。
- **独立test**（#1476）: [評価契約](nodes/ball_detection/000080-run-ball-separate-test-contract-20261024.md)は記録しましたが、test配布は同名の3 folderに4 payloadが分かれたままでmanifestがなく、実モデルのtest評価は未実施です。評価・転送の実装はPR #1489をクローズしてbranchに残しています。
- **次の実験**: #1859の本学習の再開、test配布の復旧と独立testでの評価。

## 3D Ball Refiner（ball_refiner_3d）

- **方針**（2026-10-06）: 2Dで補完した点の誤差を三角測量に伝えず、観測の不足を3Dの欠損として保持して3D側で補完します。2D Refinerは同日に廃止しました。標準pipelineにはまだ入っていません。
- **現状**: [イベントヘッド付き回帰](nodes/ball_refiner_3d/000011-run-i1014-3d-eventhead-regression-eventonly-s42-20261006-v1.md)はtest RMSEが全体0.158 m・欠損0.428 mで、欠損の誤差は同じ入力の線形補間より約52%低くなりました。一方で正確な観測にも約6.8 cmの誤差を生み、欠損付近の速度振動が残ります。合成データ・単一seedの結果です。
- **物理head**（#1032）: [比較](nodes/ball_refiner_3d/000015-group-i1032-physics-heads-s42.md)では座標だけを出すモデルが最良で、場・区間初期状態のheadと教師を足すと、座標・イベントF1・物理指標がいずれも悪化しました。物理headは既定にしません。
- **次の実験**: loss係数、観測へのアンカー、区間分割の改善のどれを試すかはユーザーの判断待ちです。同じ入力・L1・GANなしで、イベント教師の有無も比べます。

## 人物（player_detection・person_tracking・player_association）

- **現行**: 2D検出はCOCO事前学習のDINO（personクラス、全画面、閾値.30）で全人物の候補を出し、選手かどうかはコート座標での滞在で判定します（#964のユーザー判断）。追跡はStrongSORT++＋pose/CLIPで、重複boxの統合は既定でoffです（[run 11](nodes/person_tracking/000013-run-i964-default-merge-r11-20260930.md)）。camera間の対応は、box下端の足元距離とCLIP-ReIDの外観をMILPで統合し、無ラベルclipで再較正した候補Aを既定にしました（[run 14](nodes/player_association/000005-run-i964-recalibration-fit-r14-20261001.md)、[run 16](nodes/player_association/000006-run-i964-unseen-r16-20261001.md)）。
- **根拠と限界**: 追跡方式は指標ごとに優位が入れ替わり、一律の勝者はありません（[run 10](nodes/person_tracking/000012-run-i964-tracker-hybrids-r10-20260930.md)）。予約した未見3clipの一回採点はpair F1 .720（3clip中2clipで決定）で、side欠測のclipも母数に残しています。旧追跡でのMeiji 4clip評価（[pair F1 0.997](nodes/player_association/000002-run-i933-association-meiji.md)）は、旧検出・旧追跡に限った結果です。AFLinkの重みは、利用条件が一次資料で未確認のまま使っています（[LICENSES.md](../LICENSES.md)）。
- **検出・統合姿勢の置換候補**　固定Meiji 4camの6 clipでRF-DETR-L TRT16はRecall@0.50が96.81%、計時P50が26.43 ms/画像でした。DINO TRT16は96.67%、94.79 ms/画像です。RF-DETR-LのRecall@0.75はDINOを下回り、実際のcropを使ったViTPose-Hと下流品質の確認が必要です。ECPoseは高速で、欠測を含む平均OKSは大きく低下しました。 [比較と採用時の制約](nodes/player_detection/000011-run-meiji-player-model-comparison.md)
- **次の実験**: clip_000の全pipelineでの品質確認と、ダブルスの実データでの検証。#937で学習した選手検出器は、明示的なcheckpoint指定のときだけ使います。

## コートside（court_side）

- **現行**　既定はball観測を使うclip単位の判定です。固定カメラと共有コートでは動画単位を選べます。ballのmin_marginは両モードとも0.15です。[合成ベンチマーク](nodes/court_side/000001-run-i932-synthetic-side-thresholds.md)
- **固定4カメラの実測**　meiji_4cam/video_000全69 clipでX3D/CSN+DPTの観測を統合すると、cam2・cam3がhalf-turn、margin 0.267。単独で一致51、margin不足18、矛盾0で、共有court sideは全69 clipで利用できました。[実測と全件表](nodes/court_side/000007-run-court-side-video-shared-20261118.md)
- **選手の足元による判定**　`court_side.evidence=player` で選択できます。meiji_4camのclip_000・027・037でcam2・cam3のhalf-turnが単独・共有とも一致し、共有marginは0.237でした。この3clipで照合距離3.0 mを選びました。[3clipの実測](nodes/court_side/000008-run-i2269-player-court-side-20261119.md)
- **限界と次の確認**　この動画での正解ラベルに対する誤判定率、別収録への一般化、全69 clipの人物対応以降は未検証です。旧ft-e13では別の51 clip中7 clipが確定しました（誤判定0、[過去の比較](nodes/court_side/000002-run-i932-meiji-clips-side-20260927.md)）。confidenceフィルタ廃止時の安全benchは元dataset欠測のため未完です。[安全benchの記録](nodes/court_side/000006-run-i935-unfiltered-production-safety-20261002.md)

## PLCS・BLCS・SLCS（研究用の推定モデル）

いずれも標準pipelineには入っていない研究用モデルです。PLCSとBLCSは単一対象のaxialモデルだけを残し、multi_object・tracking・axial reference・残差補正モデル・camera_view_v2専用データは廃止しました（[PLCS](../src/tasks/plcs/README.md)・[BLCS](../src/tasks/blcs/README.md)のREADME）。廃止した系統の比較結果は、アーカイブとノードに履歴として残っています。

- **PLCS**: 既存多視点axialモデル（`B0`）は位置と回転のPareto構造を持ちます（[loss・headの調整](nodes/plcs/000067-group-i545-loss-head-tuning.md)、[高周波jitter問題](nodes/plcs/000087-run-plcs-canonical-temporal-decomp-beta01-noaug.md)）。#2475/#2520 で単眼GVHMR型Early Fusion（カメラ内外パラメータ＋COCO17＋合成bbox、HMR2なし）と HTD-Refine Stage A（PVA-Heads）/ Stage B（`PLCSHTDRefiner`）を導入し、[段階アブレーションと実動画6クリップ横断評価](nodes/plcs/000132-group-i2520-monocular-gvhmr-htd.md)（[詳細レポート](reports/plcs-monocular-gvhmr-htd.md)）を完了しました。`M1`＋Stage B単視点（`M2-SV`）は合成3 splitでjerk -61〜-65%・PA-MPJPE 0.070〜0.100 m（`mixed`で4カメラ`B0`の0.092 mを上回る）、実動画`meiji_4cam`でjerk -87.6%・同一視点再投影中央値1.27 px・DLT比PA-MPJPE中央値11.1 cm・DLT欠損392フレーム100%復元をCPU 1.54 ms/f・HMR2/SMPL-X非依存で達成しました。一方、未見実カメラでの単眼絶対深度は光線方向オフセットが残り（vs DLT 3.70 m、多視点Stage B融合`M2-MV`で1.17 m・中央値1.04 mへ改善）、多視点絶対定位（`B0` 0.36〜0.56 m）には届きません。標準多カメラ既定（4カメラDLT+GVHMR）と`B0`は維持し、単眼PLCSは`tennis_scene`の明示プロファイルとして運用します（最終採否・`B0`整理は人間判断待ち）。次は、v1診断で見つかった既定loss重み（加速度整合による深度崩壊）と`torsion_angle`共線停止の修正、複数seed完走、実スタジアム視点分布拡張＋接地光線アンカーによる単眼深度改善です。
- **BLCS**: #1038の物理GT付きv3データで[再学習](nodes/blcs/000043-run-i1032-blcs-physics-v3-e200-s42.md)しました。[#1044の段階的損失比較](nodes/blcs/000068-group-i1044-progressive-losses-u4000.md)では、最小物理head等の構成（B）が主対照よりtest軌道RMSEを下げました（3seed平均 1.146→0.930 m）。ただし同じtest集合での少数seedの結果で、退行する層もあります。採用の閾値はユーザーの指定で設けず、結果を見て人が判断します。固定128frame窓の境界に誤差が集中することも分かっています。次は窓の結合、A/Bの長い同予算比較、B内部の損失の分離、独立した実3Dデータでの一般化です。
- **SLCS**: [実RGBの実験群](nodes/slcs/000009-group-slcs-real-rgb.md)で、ボールの低分散出力を脱してRGBの寄与を確認しましたが、入力欠損・位置の裾・時間的スパイクが残り、頑健性は未達で固定testも未評価です（[結果レポート](reports/slcs-real-rgb.md)）。v1 annotation layoutの廃止で、旧datasetでは再学習できません。Meijiの1clipはv2で[再生成](nodes/tennis_scene/000023-run-i931-v2-dataset-meiji-one-clip-20260927.md)し、残りはside・人物対応の確定後に再生成する予定です。

## 合成データ生成（synthetic_data_generation）

- **現行の方針**（2026-10-22のユーザー判断）: Courtの合成画像はすべてtrainに使い、実画像は動画単位で分けたval/testの評価専用にします。B00〜B03の3DGSシーンから層別counterのproduction recipeで生成し（[設計文書](reports/court-scale-trajectory-design.md)）、L4での分割生成とDriveからの復旧を[確認](nodes/synthetic_data_generation/000027-run-court-scale-colab-resume-l4-20261023.md)しました。
- **限界**: コート全体が写る視点が少なく、狭角ではKPの支持が減り、ぼけ・線の不一致も残ります（[描画品質の診断](nodes/synthetic_data_generation/000025-run-court-novel-view-quality-20261022.md)）。SfMの地面に高さの不整合があり、合成教師の幾何的な不確実性として扱います（[診断](nodes/synthetic_data_generation/000023-run-court-sfm-all-scenes-drift-20260920.md)）。
- **動的前景の合成**: 現行NHTのRGB/alpha/EDだけでは[情報不足](nodes/synthetic_data_generation/000030-run-player-depth-compositing-20261108.md)です。人間の「再学習せず近似」判断に従い、prefix特徴・透過率を使うfull-alpha closureを[選定](nodes/synthetic_data_generation/000031-run-player-prefix-approximation-20261108.md)しました。C5で公開APIと共通sample合成を実装します。局所色の正確さは保証せず、大きいGaussianの深度跨ぎも残ります。C3受入と第2フェーズ動画の人間確認は独立に必要です。
- **次の実験**: 写り方別の評価と、生成・学習の費用の確認（#1356）。

## 注釈（annotation・player_pose_annotation）

- **annotation**（#1088）: production campaignは現在、ball・framesだけをレビューから累積公開・追加学習・固定評価まで接続しています（[module README](../src/tasks/annotation/README.md)）。新規clipの[ball予測](nodes/annotation/000001-run-i1088-ball-prediction-pilot-20261015.md)は完走し、[player予測](nodes/annotation/000002-run-i1088-player-prediction-pilot-20261015.md)はDINOのCUDA拡張を読めずに停止しました。2サイクルの追加学習と、修正bboxから追跡・pose・累積公開への接続が残っています。
- **player_pose_annotation**（#988）: ボール検出の学習データ用に選手のposeを蓄積するcampaignです。[1,300 clipsの拡張campaign](nodes/player_pose_annotation/000003-run-i988-pose-after-review-20261005.md)は実行中として記録されたままです。次は公開・保留・失敗のcoverageとID切替の実例を監査し、下流の効果を独立した実験で確かめます。

## レビュー評価（review_evaluation）

既定はfixed（Astra・xhigh）を維持します。[固定7ケースの親判定](nodes/review_evaluation/000015-group-i1863-review-model-comparison-20261111.md)では、select投影が時間・使用量の20%以上改善を満たさず、selectを採用しません。両モデルの正解major見逃しは同数でしたが、Astraだけが正解表外の妥当な指摘を出しており、1反復ではSolを通常レビューに使う根拠が不足しています。反復を増やすかは人間の判断待ちです。正式レビューの実機受入と、[結果レポート](reports/review-model-comparison-20261111/README.md)に残した範囲外の別issue候補は未完了です。

## 結果を解釈するための規則

runは用途で区別します。標準pipelineが参照するcheckpoint、固定split・metricで施策を比べるbenchmark、特定の構造・loss・data契約の中だけで比べるfamily、経路や仮説の成立だけを確かめるdiagnosticです。

1. view数・dataset・target frame・単一/複数対象・metricが異なるrunを、直接順位付けしない。
2. 採用の判断では、単一metricの最高値より、実clipでの安定性・下流E2E・処理時間を優先する。
3. 単一seedの小さな差を、確立した効果とみなさない。
4. no-op、作業treeの取り違え、qualificationの失敗、holdoutでの棄却を、正の証拠に格上げしない。
5. 現行mainと正規化・loss・runtime・artifact schemaが異なるrunは、履歴の証拠と明記する。
6. 学習入力の運用設定の採用と、標準pipelineのcheckpointの更新は、別の判断として記録する。

## 横断的な未解決課題

| 領域 | 不足している証拠 |
|---|---|
| Court | 収録を分けた実画像testでの精度と下流E2E |
| Ball 2D | 独立testでの評価（#1476）、camera stressへの頑健性 |
| Ball 3D | 複数方式の同条件比較、別clip、bounce/hitの独立した正解 |
| side | 現行のボール検出器での決定率と誤判定 |
| 人物 | clip_000の全pipelineでの品質、ダブルスの実データ |
| SLCS | v2 datasetの再生成、固定testと複数seed |
