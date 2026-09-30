# ユーザー提示用の提案 A–D（run22、すべて未実行）

この文書はorchestratorがユーザーへ転送する判断材料。設計・asset/default・較正・
#936の入力を変更する許可ではない。GPU投入、bundle export、較正fit、下流生成は今回0件。
結果の正本は[全比較表](comparison.md)、[回収監査](collection.json)、[動画選定](clip_selection.json)。
detector epoch9と[既存の同率規則](https://github.com/Motoki0705/tennis-lab/issues/935#issuecomment-5890186072)を維持する。

## A. 候補を基準にするheadを次のrefiner設計として採用する

**推奨案:** top-3候補＋±0.02uvの学習残差、1自由成分、計K=4、
12,000更新の予算を文脈なしrefinerの基準にする。bestは従来のMeiji選択側
observed/gap等重み位置NLLで選ぶ。今回の採用候補はepoch41（10,500更新）で、
最終epoch47を採る提案ではない。[READMEのproposed節](../../../src/tasks/ball_refiner/README.md#proposed-候補残差headを次の基準設計にする案)
へ設計案を記載した。候補がない成分の絶対平均分岐、free成分、full covariance、
全GMM＋amodal存在、NLLを保持する。検出点へのfallbackや点化は追加しない。

根拠は同一seed/cache/splitの3案比較。絶対headの長期化でも改善するが、
3kの候補残差だけで典型誤差はさらに小さくなり、12kで裾とgap NLLも改善する。
anchored_12kはdetectorより全source/camera/halfのp50/p90/p95が小さい。
一方、r18に対するp95はcalibration/cam0・cam1で悪化する。単一seed、
同じval収録への反復、head初期化も変えた比較という限界がある。
±0.02uvと3+1の比率が最適との主張はしない。

選択肢は「A1:このheadを基準設計として採用し、較正を次の課題にする（推奨）」と
「A2:追加seedによる再現確認までexperimentalを維持」。今回のコードではexperimentalを維持した。
A1の設計承認自体に計算費用はない。A2なら1seedあたり学習＋評価15–25分、
VRAM2–4GB、出力約450MBを見込む（今回の学習716秒・評価35秒・
training/evaluation約433MBが根拠）。追加seedは別grantと事前固定が必要。

## B. ball refiner componentを e9 + anchored_12k に切り替える

**推奨する採用候補**だが、ユーザー承認後の設定PRと元動画execute/load-only検証を条件とする。

| asset | 現行の専用recipeで検証済みの基準 SHA-256 | 提案候補 SHA-256 |
|---|---|---|
| detector | ft-e13 cd7927ad27e53ddd6aa77df28eca3c5e674552461ccda083a41e99e629857892 | e9 37f4c59aead00062829280ad591b3874886978891104a290f704a2c33c3c1b36 |
| refiner | r4 epoch9 ed659e142dd228949ec772bcfdf9db0f4c682081778853c15088f7202a848c76 | anchored_12k epoch41 985308b02d4b1b33c3bcfb40dbf2a846b900e730ee6d5dfc07c3a0ed561cd36c |

r18は比較pilotであり、default採用済みではない。専用recipeは現在bundle/checkpointを
明示する入口、標準sceneのball_detection既定はft-e13。単一のrefiner defaultが既に
全sceneへ適用されているとは説明しない。新bundleのモデル設定とe9の候補/前処理/
窓規則を固定し、実際にどのrecipeのasset参照を変更するかを設定PRに明示する。

承認後の再実行範囲:
1. anchored_12kのimmutable推論bundleをCPUでexportし、全hash・strict loadを検査。
2. Meiji val clip_010の3camera×270frameを元動画からdetector→refinerでexecute。
   別processのload-onlyで全GMM・存在・frame/PTS・採用窓・依存hashを照合する。
   今回のJPEG overlayはこの元動画経路の代わりにはならない。
3. 共有detector既定も変える場合、ball_detection依存のcourt_side、そこからの
   player_association等の依存artifact、ball三角測量/3D refiner、scene export・datasetを新identityで更新。
   court/person/poseモデルの再学習を要求する変更ではない。
4. #936の入力分布が変わるため、Dの較正bank→合成2D/3D条件→dev対照学習/評価を再実行。
   #936の現在のdevとcheckpointは旧入力の対照として保持する。
5. runtimeはsrc全体のcode fingerprintを持つので、コード変更を伴うcheckout更新では
   実装上全componentが無効化され得る。checkpoint/configだけの変更時は依存先へ限定される
   （[pipelineの正本](../../../src/tennis_scene/pipeline/README.md#成果物)）。
   全scene再実行の予算を2componentの費用で代用しない。

| 計算段階 | wall見積もり（待ち時間・実装を除く） | peak VRAM | 新規disk |
|---|---:|---:|---:|
| CPU bundle export/strict load | 1–3分 | 0 | 10MB以内 |
| 810frameの2component execute/load検査 | 5–10分 | 2–4GB | 150MB以内 |
| 設定PRのCPU CI | 10–20分 | 0 | ほぼなし |
| #936 bank/H dev96ラリー再生成 | bank 5–15分＋生成55–70分（4 CPU） | 0 | bank 10MB以内＋dev0.3GB以内 |
| #936 dev対照学習再実行 | 今回進行中のrun9実測回収後に確定 | 別grant、上限12GB | run9実測で確定 |

2componentの根拠は旧r7 cam0 detector9.14秒/refiner0.79秒＋起動・監査余裕。
H生成の根拠は[#936 run9の96ラリー実測52分・222MB](https://github.com/Motoki0705/tennis-lab/issues/936#issuecomment-5903897000)。
Hの固定予算でも新bankで数値的な失敗が増えないことを小probeで先に確認する。
標準scene全段はperson等の現行構成に依存し、#964完了前の確定見積もりはしない。
この費用表は追加実行の申請材料であり、本runのGPU grantは0のまま。

## C. 観測frameの裾の過信をどう扱うか

全Meiji valのobserved HDR50/90/95は0.515/0.836/0.883。
**checkpoint選択に使わないcalibration halfでは0.487/0.800/0.847**、
人工gapも0.470/0.841/0.880。全val gap 0.521/0.891/0.923だけを
「較正済み」として提示しない。chatの明示的不在NLLも高く、presenceは別課題。

| 案 | 内容・利点 | 制約と費用見積もり |
|---|---|---|
| C1: 未較正を明記して固定 | 今の予測を比較基準として保持する | 追加費用0。3Dへ過信を渡す課題は残る |
| C2: covariance scaleの後処理を検証（推奨する次の実験） | means/weight/presenceは固定し、保存GMMに1つの分散倍率をfit。必要なら入力候補の有無で明示した2群まで。GMM契約を維持し再学習不要 | CPU 10–30分、VRAM0、20MB以内（未実測）。50%の過拡大・面積/NLLの悪化を同時に検査し、不合格なら未較正値を維持 |
| C3: 広い自由成分のweight/scaleを含む較正、またはtailを重視する学習設計 | 単一scaleで中心と裾が両立しない場合の次案 | CPU fitなら30–60分、再学習なら別seed/損失を事前固定しGPU15–25分/2–4GB/約450MBから。今回未実装・未実行 |

C2の事前固定案は、6時刻clip群×3cameraのcalibration halfだけを使い、
leave-one-temporal-clip-group-outで倍率の安定性・位置NLL・HDR50/90/95・面積を出す。
最終fit後の同じframeのcoverageを未使用test性能とは呼ばない。選択halfで再選択しない。
倍率は標準偏差か共分散かをschemaで明示（Σ'=sΣならL'=sqrt(s)L）。
運用時の分岐はcandidate_valid等の入力から定義し、GTのobserved/occludedを参照しない。
連続frame相関・6群しかないこと・固定MCの誤差も報告する。まずCPU固定sampleで試験し、
最終採否では別MC seedの確認とclip群bootstrapを行う。現在の数字は一切補正しない。

## D. #936へ渡す2D較正資料と、合成noise modelの更新

#936が現在使う#959/r4の暫定bankは、全GMM残差・共分散・weight・presenceを
camera別/通常・人工gap別に連続blockで再標本化する。単にsigmaを10分の1にする方式ではない。

同じcalibration halfで、旧r4→anchored_12kのobserved中央値は46.17→3.53px（約13.1倍小さい）、
p95は568.05→377.6px（約1.50倍小さい）。人工gap中央値は91.60→35.34px（約2.59倍）。
「約10倍」は典型的な観測位置の説明に限る。裾、gap、存在、不確実性へ一律適用しない。
2D uv較正から作る合成3Dのm単位誤差も同倍率とは限らない。

ユーザーがA/B（および必要ならC）を決めた後、以下を**新しいimmutableなbundle**へ作る案:
- anchored best checkpoint/config/e9 cache/partitionと全入力hash、calibration 18camera-clip×2条件の
  全36 NPZを参照する2D診断manifest。各camera・gap長・source寸法・frame/PTS・観測/推定/不在/未知のmaskを明示。
- 位置誤差、位置NLL、HDR50/90/95coverage/面積、存在NLLと正負件数、clip群bootstrap。
  実遮蔽の推定点をobserved residual bankへ混ぜず、gapは人工証拠欠損と記す。
- K=4すべての成分についてmeans-target_uv、scale_tril_uv、mixture_logits、
  presence_logitsを保存し、camera_index/condition_index/source_artifact/source_frame/
  continuesで最大16frameの連続blockを追跡する。meansだけに縮約しない。
- #936のconsumerが要求するcalibration.json
  （schema ball_refiner_3d.degradation_calibration.v1、bank_sha256、components、明示status）と
  同じdirectoryのbank.npzを出す。report SHAも生成manifestに固定する。
- consumerのbuild_calibrationは現在#959形式
  ball_refiner_validation_diagnostics.v1のcalibration partitionを要求する。
  今回のball_refiner_cached_comparison.v1/comparison.jsonはそのままでは受理されない。
  保存済み36 NPZから監査付きadapterを作るか、既存diagnostic入口をCPUで実行する。
  schemaのラベルだけを付け替えて未計算の統計を埋めない。

元データは[anchored評価manifest](evaluation-anchored_12k/manifest.json)、
採点規則・全表は[comparison.json](comparison.json)、hashは[artifact一覧](artifact_hashes.json)。
BUNDLEは現在**未較正の比較証拠**であり、完成した#936入力bankではない。
将来のadapter/統計抽出はCPU5–15分・10MB以内、追加のHDR再計算/較正fitはCの予算。
#936の--calibration-reportで明示的に新reportを指定し、固定rally/frame/maskで新旧2D被覆を比較してから
新dev/fullを生成する。現在の96ラリー生成/学習は中止・上書きせず旧暫定入力の対照として残す。
新しいbankでのH dev生成はB表の55–70分/0.3GB目安、640件fullは別grantで再見積もり。

長い32/64frame gapへの外挿、camera間の独立再標本化、Meijiの確定負例不足は残る。
Cで較正を変更する場合はbankとすべての派生成果物のidentityを更新する。
#964の人物契約・COCO全画面0.30既定と独立した2D ballの提案であり、
#964/#936 branchや進行中worktreeには変更を加えていない。

