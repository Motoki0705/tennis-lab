# Run 9 addendum: offline linkingとnative KPRの比較（新結果の実行前）

2026-09-30。run 8の[主指標・照合・層別・失敗規則](../run-i964-tracker-matrix-r8-20260930/protocol.md)は変更しない。
同じ4 dev×3camera、COCO全画面 .30の40,531row、同じpose/CLIP/SOLIDERと
[照合済みKPR](../../nodes/person_tracking/000010-run-i964-kpr-native-features-r8-20260930.md)だけを使う。
未見予約clip、pipeline tracker/default、検出閾値、固定コート選別とv3は変更しない。
この文書をIssueへ投稿・commit・pushしてから新しい実データ追跡・採点を開始する。

## 追加する副指標（run 8結果を見た後の追加）

**group IDF1**: `select_linked_candidates` → `linked_timeline`が下流へ渡す
camera-local groupのbox/observed/IDを、run 8と同じ`tracking_units`/`score_units`で採点する。
handoff重複では既存規則の早いraw IDのboxを残す。group用にラベル照合を独立実行する。
raw-ID IDF1を置換せず、両方のIDTP/FP/FN、switch、fragment、選手保持、非選手残存を並べる。
旧6条件は保存済みrun 8出力をhash照合して利用し、主指標の一致を検査する。
camera×near/far、全8人物の50%以上保持、未照合予測、完走数、#933 pair F1/対応表も維持。

## 追加条件と順序

1. **Deep OC-SORT+pose/CLIP + Lab**: run 8のオンライン追跡結果へ旧`link_tracklets`を
   同じ`TrackletLinkPolicy()`で適用。元検出boxの胴Labを同じ関数で求める。
   gap60、overlap span3、containment .95、中心1 diagonal、size2/重複2.5、Lab20、端10観測。
   出力IDと重複解決だけを適用し、別の補間/平滑化は足さない。曖昧ならそのcameraは停止して空予測。
2. **StrongSORT++/CLIP（論文から再実装）**: 第3方式。下記の固定仕様。
3. (1)/(2)/run 8 Deep OC-SORT+pose/CLIPのうち、**raw-ID主指標、switch、fragment、名称順**
   （IDF1同点許容1e-6）で最良の方式を選び、その方式だけtracker encoderをKPRに替える。
   方式選択は事前に決めた分岐であり、KPRの結果を見て方式/閾値を再選択しない。
4. 検出rowが厳密に残る全候補で、camera間encoderにKPRを追加する。
   CLIP/SOLIDER/KPRの同じcrop sampling、固定幾何/v3を使い、ラベル再calibrationはしない。
   baselineの平滑化boxへ近い元検出のKPRを流用しないため、baselineは既存CLIP結果を維持する。

最終推薦は旧6条件中の候補4条件と追加(1)(2)(3)のraw-ID主指標を上記順で比較する。
group指標が支持しない場合はその不一致も報告し、主指標の順位を都合よく変更しない。
推薦は比較候補内のもの。pipeline既定採用は行わない。

## StrongSORT++の固定仕様・出自

根拠は[Du et al., arXiv v2 §III–V](https://arxiv.org/html/2202.13514v2)、
[公式設定](https://github.com/dyhBUPT/StrongSORT/blob/ee995076da5083e28d0da1f885297df62705ebd7/opts.py)。
公式repo commit `ee995076da5083e28d0da1f885297df62705ebd7`。公式コードはGPL-3.0。
論文の式/構造から推論のみを独自実装し、公式コードの移植とは称さない。
設定の数値・checkpointのkey/shape・合成入力の数値互換を確認する。法的なclean-room認定とは称さない。

- 全40,531rowを共用するため追加の検出score .6 gate/NMSは使わず、run 8のsource .30を維持。
  CLIPをBoT encoderの代わりに用い、固定cameraのECCはoff（run 8のCMC方針と同じ）。
- XYAH定速度Kalman、位置noise h/20・速度h/160、初期位置2倍/速度10倍、aspect noise .01/速度 .00001。
  NSA観測共分散 `(1-score)*R`、EMA .9、appearance/motion比 .98/.02、4D Mahalanobis gate 9.4877、
  matching cost上限 .45、vanilla Hungarian（cascadeなし）、残りはIoU距離≤.7、age≤1。
  max_age30、n_init3。実観測だけをemitし、confirmed後に最初の2frameを遡及補完しない。
- AFLink: 論文の2枝、30×3 `(frame,x,y)`、4層7×1 conv（32/64/128/256）、1×3 fusion、
  平均pool、512→128→2分類器。checkpointの時間/座標別BNを維持。
  前track末尾30/後track先頭30、ゼロpad後に両者共通の次元別min/maxで[-1,1]正規化。
  公式入口のtemporal `(0,30)` frame、spatial75px、非接続確率<.05、Hungarian連結。
  接続による重複/循環は停止。推論はCPU。
- AFLink重み: 公式READMEの[配布folder](https://drive.google.com/drive/folders/1Zk6TaSJPbpnqbz1w4kfhkKFCEzQbjfp_)の
  `AFLink_epoch20.pth`（file ID `1DFMUkL-dc-j8-fibcJIq-46Xoq_bFoO9`）、4,348,705 bytes、
  SHA-256 `b35cbeddd3acc48fece820bd640640e6bfb1f5fbf570aa79af26c6a38958daa4`。
  独立した重み利用条件は見つからない。再配布・MITの表示はせず今回のローカル研究比較に限定し、
  本番採用時の確認課題として【要判断】を残す。再実装が重みの条件も解決するとは扱わない。
- GSI: gap<20frameを線形補間し、XYWHを論文式14のRBF GPRで平滑化。
  tau10、lengthscale clip(`10*log(1000/L)`, .1,100)、kernel固定、noise alpha1e-10。
  補間boxは別の`interpolated` mask・元row=-1で保存する。
  **run 8の主指標/固定選別は実観測のみ**なので、GSI補間を実検出へ昇格させない。
  主表はAFLink後IDと元の実観測box。GSIの再構成は別配列/補助集計と動画で明示し、
  GSIが主指標のFN/fragmentを消したとは報告しない。厳密なMOTベンチマーク追試ではない。

## KPR native距離

[公式part距離](https://github.com/VlSomers/keypoint_promptable_reidentification/blob/e3e6ee2ffb74fd86a39518ce9a25ff91fbd973fa/torchreid/metrics/distance.py)の
**共通可視partのEuclidean距離の平均**を使う（6個のunit descriptor、mean combine）。
両者共通の可視partが無ければappearanceの証拠なしを明示し、別encoderへ戻さない。
trackerのsimilarity尺度へは明示的に`1-distance`（範囲[-1,1]）を渡す。
cosineとは呼ばず、part間のmaskを無視したflattenも行わない。trackerの既存重み/閾値は維持する。
時間更新はその方式のEMA係数を各partへ適用し、現在可視のpartだけ更新・再正規化する。
camera間では同じsample frameの可視partごとのunit平均を区間の代表とし、同じnative距離を使う。
association scoreは固定 `62.7*((1-distance)-.847)`、既存clip上限でclampする。
これはCLIP尺度の明示転用で、KPR固有の校正や最適性能とは解釈しない。
HL3 notice・既存重みhashを維持する。

## cam1 far診断と動画

run 8 Deep/CLIPのcam1 farで、raw主指標のID誤り/欠測が多い5秒窓上位3つを、重ならないよう
clip/frame順でtie breakして選ぶ。元row、競合検出、IoU gate、外観similarity/加重寄与、
pose有効joint/距離/寄与、first/OCRの割当を再生し記録する。
同じ窓でpose係数0とappearance係数0の反実仮想を診断専用に記録してよいが、候補や推薦に混ぜない。
小cropによる特徴不安定という仮説と、直接確認できたgate/割当/欠測を区別する。
3窓動画と、最良新条件対new+oldの既存規則による3camera上下レビュー動画を作る。

CPU単一process、torch≤4/OpenBLAS1/OpenCV1、pytest -n4。RAM available≥6GiB、追加disk≤5GB。
run 8のartifactは上書きしない。修正が必要なら旧結果と理由を保存し、閾値探索はしない。
