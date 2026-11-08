# Tennis Lab Knowledge Summary

更新日: 2026-11-05

研究トピックごとの**現在の判断**（採用中の構成とその根拠、未解決の課題、次の実験）をまとめた入口です。数値・条件・考察の正本は各[ノード](nodes/)、再現手順は[runs](runs/)にあります。2026-11-05までの日付順の記録は、[凍結アーカイブ](history/summary-2026-11-05.md)に原文のまま残しています。

- 標準pipelineのcheckpointと採用根拠: [deployments.yaml](registry/deployments.yaml)（CIがpipeline設定と照合する）
- トピックとsrcモジュールの対応、廃止したトピック: [tasks.yaml](registry/tasks.yaml)
- ノードの絞り込み: `kg_query.py`（[knowledge-control skill](../.agents/skills/knowledge-control/SKILL.md#plan-an-experiment)）

採用構成・評価契約・研究の優先順位が変わったときに、該当する節を書き換えます。runを1件追加するたびの追記はしません。結論を変えないrunは、ノードの `summary_impact: none` と本文で扱います。

## 標準pipeline（tennis_scene）

標準のシーン再構成は、各カメラのコートkeypoint・人物box/pose・ボールの2D観測からカメラを校正し、人物の関節とボールを三角測量します。PLCS・BLCS・SLCSの学習モデルは標準pipelineに入っていません。

- **動作確認**: 公開3カメラサンプル（Meiji clip_000）で、既定設定がSceneResult v2の保存まで完走しました（[標準pipeline](nodes/tennis_scene/000028-run-multiview-sample-standard-pipeline-20261021.md)、[クリーン導入](nodes/tennis_scene/000030-run-clean-standard-install-20261021.md)）。1clipの動作確認であり、独立した3D精度や全frameの目視合格は示していません。
- **ボール3Dの平滑化**: 三角測量の後、局所運動のconsensusで外れ値を除き、頑健な2次LOESSをかけます（既定で有効）。[導入時の比較](nodes/tennis_scene/000031-run-scene-ball-loess-20261026.md)では3D有効frameが821→859/1010に増え、隣接移動量のp95も下がりました。一方、元の2D観測への再投影中央値は3.554→4.810 pxに増えました。滑らかさと観測追従の交換関係であり、3D精度の向上とは扱いません。
- **ボール3Dの評価基盤**（2026-11-03）: clip_000の人間参照を使い、[固定入力のCPU replay比較](nodes/tennis_scene/000032-run-ball3d-evaluation-foundation-20261103.md)を作りました。[レビュー修正後](nodes/tennis_scene/000033-run-ball3d-foundation-review-20261103.md)、全方式に共通の768点でのLOVO中央値/p90は、既定が8.97/25.40 px、LOESSなしが9.70/28.02 pxです。共通点は全対象の52.53%に限られ、可用性を条件とする比較です。[誤差分解](nodes/tennis_scene/000034-run-ball3d-error-decomposition-20261103.md)では、top-8候補にargmax失敗の約半数を回収できる候補がある一方、既存の採用maskを保つ3D候補oracleの改善は小さく、校正誤差と同期ずれも分離できていません。
- **6方式の比較と採否**: [統合比較](nodes/tennis_scene/000041-run-ball3d-integrated-comparison-20261104.md)では後半LOVOとcoverageの一貫した改善は得られず、人間はbaseline維持を決定しました。後処理・三角測量側の追加改善はキャンセルされ、検出モデルの学習戦略の改善が必要と判断されました。比較は1 clip・近似校正・推定を含む2D参照に限られます。詳細は[凍結アーカイブ](history/summary-2026-11-05.md)を参照してください。

## Court Detection

- **現行**: `multiscale_depth3` のcheckpointと、画像だけによる領域探索を組み合わせ、各カメラの先頭frameだけを推論します。[Meiji 3cameraの校正が通った](nodes/court_detection/000032-run-court-meiji-model-only-regions-20260922.md)ことが採用の根拠です。このcheckpointを学習したrunのノードはなく、独立したheld-out testでの精度も未確立です。
- **合成データのscene数**（#1343、2026-10-30に完了）: [固定5500枚の1〜4scene比較](nodes/court_detection/000055-group-court-scene-diversity-colab-s42.md)では、4sceneが実画像valの集計5指標で最良でした（KP平均 53.95→23.61 px）。2sceneでは悪化しており、単調な改善ではありません。seed42の単一試行で追加順も固定のため、scene数だけの効果は確定していません。画像数スケーリングの本実験と追加seedは行わない判断です。
- **backbone**（#983）: [凍結DINOv3 S/S+/B/Lの比較](nodes/court_detection/000044-group-i983-court-dinov3-shared1024-s42.md)では、出力ごとに最良のサイズが異なり、サイズによる一様な改善は見られませんでした（1 seed）。
- **後処理**: KP・LINEの共同推定やPROSACは保存予測での診断までで、人手GTのある会場分離benchmarkはありません。Meijiの全frame処理ではCPUの幾何後処理が重く、並列化・batch化の余地があります（[時間分解](nodes/court_detection/000033-run-court-meiji-hybrid-cpu-profile-20260923.md)）。
- **次の実験**: 収録を分けた実画像testで、従来のhomography推定・PROSAC・共同推定のKP距離、geometry valid率、失敗率、処理時間、下流E2Eを比べます。シーン組合せの比較は[manifest作成](nodes/synthetic_data_generation/000029-run-court-scene-pair-manifests-v1.md)まで済み、学習と評価は親#1581で行います。B/Lの複数seedも残っています。

## Ball Detection

- **現行**: #935の混合FTのepoch 9です。2026-10-01のユーザー判断で既定にしました。根拠ノードは廃止したball_refinerトピックにあります（[epoch 9の検証](nodes/ball_refiner/000015-run-i935-evidence-mixed-e9-trainval-r17-20260930.md)、[既定切替](nodes/ball_refiner/000028-run-i935-confidence-r29-20261001.md)）。固定BゲートのGT比較は、p90が許容を超えて悪化したため不合格のままです（[run26](nodes/ball_refiner/000025-run-i935-source-b-gate-r26-20261001.md)）。精度の合格判定を経た採用ではありません。
- **採否の原則**: offline F1の最高値だけでは採用しません。[scratch版](nodes/ball_detection/000010-run-i618-convnext-v2-scratch.md)はF1が高い一方、実clipのcoverage低下とteleportを起こしたため採用しませんでした。[#934の混合FT](nodes/ball_detection/000022-run-i934-meiji-holdout-e0-r7.md)でも、Meiji holdoutのrecallは上がりましたが採用検出のp95が悪化し、欠損の低減と誤検出の抑制は両立していません。
- **長系列モデル**（親#1367）: 教師は入力Tの全frame、文脈はclip内・上限128と[決めました](nodes/ball_detection/000063-run-ball-all-frame-protocol-20261022.md)。配布storeの可搬化と、Colabでの逐次保存・新VMからの再開も確認済みです。[共通DPTからの逐次T64学習](nodes/ball_detection/000068-run-ball-sequential-length-20261025.md)では、query T64の平均/P95誤差（28.20/116.89 px）がDPT T64（36.56/183.20 px）より小さい一方、20 px以内の率は低くなりました（76.29%対86.58%）。追加の更新数と方式の差が交絡しており、seed42・validationだけの結果です。
- **事前学習＋事後学習**: [ConvNeXt V2＋DPTとquery事後学習](nodes/ball_detection/000060-run-i1049-convnext-query-posttrain-s42.md)は通常評価で親より改善しましたが、camera stress（特に1/4 FPSで41.11 px）で大きく悪化しました。頑健性の要件は満たしていません。
- **次の実験**: 更新数をそろえたT比較、query T32の残差とoverlayの確認、#1476・PR #1489を前提とした独立testでの評価。

## 3D Ball Refiner（ball_refiner_3d）

- **方針**（2026-10-06）: 2Dで補完した点の誤差を三角測量に伝えず、観測の不足を3Dの欠損として保持して3D側で補完します。2D Refinerは同日に廃止しました。標準pipelineにはまだ入っていません。
- **現状**: [イベントヘッド付き回帰](nodes/ball_refiner_3d/000011-run-i1014-3d-eventhead-regression-eventonly-s42-20261006-v1.md)はtest RMSEが全体0.158 m・欠損0.428 mで、欠損の誤差は同じ入力の線形補間より約52%低くなりました。一方で正確な観測にも約6.8 cmの誤差を生み、欠損付近の速度振動が残ります。合成データ・単一seedの結果です。
- **物理head**（#1032）: [比較](nodes/ball_refiner_3d/000015-group-i1032-physics-heads-s42.md)では座標だけを出すモデルが最良で、場・区間初期状態のheadと教師を足すと、座標・イベントF1・物理指標がいずれも悪化しました。物理headは既定にしません。
- **次の実験**: loss係数、観測へのアンカー、区間分割の改善のどれを試すかはユーザーの判断待ちです。同じ入力・L1・GANなしで、イベント教師の有無も比べます。

## 人物（player_detection・person_tracking・player_association）

- **現行**: 2D検出はCOCO事前学習のDINO（personクラス、全画面、閾値.30）で全人物の候補を出し、選手かどうかはコート座標での滞在で判定します（#964のユーザー判断）。追跡はStrongSORT++＋pose/CLIPで、重複boxの統合は既定でoffです（[run 11](nodes/person_tracking/000013-run-i964-default-merge-r11-20260930.md)）。camera間の対応は、box下端の足元距離とCLIP-ReIDの外観をMILPで統合し、無ラベルclipで再較正した候補Aを既定にしました（[run 14](nodes/player_association/000005-run-i964-recalibration-fit-r14-20261001.md)、[run 16](nodes/player_association/000006-run-i964-unseen-r16-20261001.md)）。
- **根拠と限界**: 追跡方式は指標ごとに優位が入れ替わり、一律の勝者はありません（[run 10](nodes/person_tracking/000012-run-i964-tracker-hybrids-r10-20260930.md)）。予約した未見3clipの一回採点はpair F1 .720（3clip中2clipで決定）で、side欠測のclipも母数に残しています。旧追跡でのMeiji 4clip評価（[pair F1 0.997](nodes/player_association/000002-run-i933-association-meiji.md)）は、旧検出・旧追跡に限った結果です。AFLinkの重みは、利用条件が一次資料で未確認のまま使っています（[LICENSES.md](../LICENSES.md)）。
- **次の実験**: clip_000の全pipelineでの品質確認と、ダブルスの実データでの検証。#937で学習した選手検出器は、明示的なcheckpoint指定のときだけ使います。

## コートside（court_side）

- **現行**: sideは学習モデルを使わず、ボールの2D観測だけを使う幾何的な仮説検定で決めます。閾値は[合成ベンチマーク](nodes/court_side/000001-run-i932-synthetic-side-thresholds.md)で決めました（min_margin 0.15）。誤判定を防いでいるのはmarginです。
- **限界**: 決定できる割合は、ボール検出の見落としと誤検出に左右されます。ft-e13の時点では、Meiji 51clipのうち検出器のボールで決まったのは7clipでした（誤判定0、[記録](nodes/court_side/000002-run-i932-meiji-clips-side-20260927.md)）。現行の検出器での全clip評価は記録されていません。2026-10-02に信頼度フィルタを外す判断をしましたが、安全benchの全件は元datasetの欠測で未実施です（[記録](nodes/court_side/000006-run-i935-unfiltered-production-safety-20261002.md)）。
- **次の実験**: 現行のボール検出器でのside決定率と誤判定の評価。

## PLCS・BLCS・SLCS（研究用の推定モデル）

いずれも標準pipelineには入っていない研究用モデルです。PLCSとBLCSは単一対象のaxialモデルだけを残し、multi_object・tracking・axial reference・残差補正モデル・camera_view_v2専用データは廃止しました（[PLCS](../src/tasks/plcs/README.md)・[BLCS](../src/tasks/blcs/README.md)のREADME）。廃止した系統の比較結果は、アーカイブとノードに履歴として残っています。

- **PLCS**: 位置と回転で最適な容量配分が異なり、単一の構成ですべての目的を同時に最適化できないPareto構造を確認しています（[loss・headの調整](nodes/plcs/000067-group-i545-loss-head-tuning.md)）。canonical poseはmotionを復元する代わりにjitterを過剰に生成します（[高周波成分の割合は予測0.391、GTは0.069](nodes/plcs/000087-run-plcs-canonical-temporal-decomp-beta01-noaug.md)）。reprojection lossも一方向の改善ではありません。次は、motionの相関を保ったままjitterを抑え、position・rotationと併用しても平均姿勢へ崩れない構成を探します。
- **BLCS**: #1038の物理GT付きv3データで[再学習](nodes/blcs/000043-run-i1032-blcs-physics-v3-e200-s42.md)しました。[#1044の段階的損失比較](nodes/blcs/000068-group-i1044-progressive-losses-u4000.md)では、最小物理head等の構成（B）が主対照よりtest軌道RMSEを下げました（3seed平均 1.146→0.930 m）。ただし同じtest集合での少数seedの結果で、退行する層もあります。採用の閾値はユーザーの指定で設けず、結果を見て人が判断します。固定128frame窓の境界に誤差が集中することも分かっています。次は窓の結合、A/Bの長い同予算比較、B内部の損失の分離、独立した実3Dデータでの一般化です。
- **SLCS**: [実RGBの実験群](nodes/slcs/000009-group-slcs-real-rgb.md)で、ボールの低分散出力を脱してRGBの寄与を確認しましたが、入力欠損・位置の裾・時間的スパイクが残り、頑健性は未達で固定testも未評価です（[結果レポート](reports/slcs-real-rgb.md)）。v1 annotation layoutの廃止で、旧datasetでは再学習できません。Meijiの1clipはv2で[再生成](nodes/tennis_scene/000023-run-i931-v2-dataset-meiji-one-clip-20260927.md)し、残りはside・人物対応の確定後に再生成する予定です。

## 合成データ生成（synthetic_data_generation）

- **現行の方針**（2026-10-22のユーザー判断）: Courtの合成画像はすべてtrainに使い、実画像は動画単位で分けたval/testの評価専用にします。B00〜B03の3DGSシーンから層別counterのproduction recipeで生成し（[設計文書](reports/court-scale-trajectory-design.md)）、L4での分割生成とDriveからの復旧を[確認](nodes/synthetic_data_generation/000027-run-court-scale-colab-resume-l4-20261023.md)しました。
- **限界**: コート全体が写る視点が少なく、狭角ではKPの支持が減り、ぼけ・線の不一致も残ります（[描画品質の診断](nodes/synthetic_data_generation/000025-run-court-novel-view-quality-20261022.md)）。SfMの地面に高さの不整合があり、合成教師の幾何的な不確実性として扱います（[診断](nodes/synthetic_data_generation/000023-run-court-sfm-all-scenes-drift-20260920.md)）。
- **次の実験**: 写り方別の評価と、生成・学習の費用の確認（#1356）。

## 注釈（annotation・player_pose_annotation）

- **annotation**（#1088）: production campaignは現在、ball・framesだけをレビューから累積公開・追加学習・固定評価まで接続しています（[module README](../src/tasks/annotation/README.md)）。新規clipの[ball予測](nodes/annotation/000001-run-i1088-ball-prediction-pilot-20261015.md)は完走し、[player予測](nodes/annotation/000002-run-i1088-player-prediction-pilot-20261015.md)はDINOのCUDA拡張を読めずに停止しました。2サイクルの追加学習と、修正bboxから追跡・pose・累積公開への接続が残っています。
- **player_pose_annotation**（#988）: ボール検出の学習データ用に選手のposeを蓄積するcampaignです。[1,300 clipsの拡張campaign](nodes/player_pose_annotation/000003-run-i988-pose-after-review-20261005.md)は実行中として記録されたままです。次は公開・保留・失敗のcoverageとID切替の実例を監査し、下流の効果を独立した実験で確かめます。

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
