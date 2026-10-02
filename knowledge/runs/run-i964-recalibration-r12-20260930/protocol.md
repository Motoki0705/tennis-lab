# #964 run 12: 新既定trackによるcamera間対応の再較正

2026-09-30。**このprotocolをcommit/pushしIssueへ投稿してから、新しい特徴の生成・fitを開始する。**
仮説は「旧trackに由来する尺度が新trackへ移らない可能性」であり、pair F1低下の原因とはまだ断定しない。
前提は[run 11](../../nodes/person_tracking/000013-run-i964-default-merge-r11-20260930.md)と
[旧較正](../../nodes/player_association/000002-run-i933-association-meiji.md)。
結果に基づくclip交換、追加のしきい値探索、dev採点後の再fitをしない。

## 固定条件とデータ分離

- [ユーザー決定](https://github.com/Motoki0705/tennis-lab/issues/964#issuecomment-5910531924):
  COCO DINO全画面0.30、800/1333、StrongSORT++＋pose/CLIPのrun 10設定、merge off。
  固定コート座標選別・選別後cap6、v3の短い曖昧区間除外（1秒）とhandoff上限（.2秒）を維持する。
  detector/tracker/selectionのパラメータをfitしない。GSIは常にsyntheticで、較正の実観測に入れない。
- 候補母集団は旧 `i933-observe-v1-20260927/observe.json` の**clip key集合**。
  旧status・精度・画像内容は選定に使わない。4 devと予約manifestの全clipを除き、
  datasetに人物label/reviewファイルがあるclipも除く（存在だけを調べ、内容は読まない）。
  動画ごとにmetadataの `num_frames / fps` 昇順、同点はclip ID辞書順で**2本**選ぶ。
  3動画×2本=6本を使い、短縮実行・結果による交換・失敗clipの補充をしない。
  これは共有機の2時間枠に収め、動画単位の検証を可能にするための事前のサイズ制約。
- 4 devは `video_000/clip_000, video_000/clip_007, video_001/clip_001, video_002/clip_013`。
  予約未見は `video_000/clip_002, video_001/clip_003, video_002/clip_003`。
  予約manifestのhash/keyだけを確認し、映像・特徴・人物ラベルを開かない。
- 選定manifestには候補のmetadata/hash、全除外理由、選択理由、source区間を記録する。
  選択clipとdev/未見の元camera動画区間が重複しないこともmetadataで検証する。
  同じ収録・同じ選手であり、収録/人物の独立testではない。
- cameraのsideは旧#933と同じ注釈ball由来の固定decisionを用いる。
  これは人物ラベルではないが、完全自動pipelineの精度とは区別する。
  courtの校正は同じ検出checkpoint/configの既存artifactをchecksum検証して再利用できる。
  side未決定・court欠損は明記して停止し、成功clipだけで黙ってfitを続けない。

## 特徴のidentityと再利用

manifestに動画hash/timeline、detector hash・scope・resize・閾値・merge、
ViTPose hash/precision/crop設定、CLIP-ReIDモデル/hash/前処理、元rowを固定する。
特徴はこれらが一致する場合にだけ再利用し、追跡出力にはさらに
`TrackingConfig.identity()`、AFLink hash、選別設定・実装hashの一致を要求する。
旧ROI・旧tracker・旧cropの特徴を近いboxへの照合で代用しない。
devのrun 7 CLIP/pose archiveとrun 11 merge-off trackは再利用候補（採点はまだしない）。
無ラベル6clipの不足分だけを1件のqueue jobで生成する。
出力は新run専用で、旧結果・dataset annotationを上書きしない。

## 擬似対応とsampleの単位

既定track→固定選別→v3と同じ足元/switch分割を使う。source timelineを
frame 0起点の重複しない4秒窓に分ける。末尾も共有実観測が2秒以上なら使う。
窓中にswitch候補またはhandoffがあるtrackの組、無効足元、共有2秒未満を除外する。
cameraの異なる組について、同時観測の足元距離の中央値を計算する。

- positive: 中央値 <4m、各camera対で双方が最も近い相手であり、
  同じ窓の他の候補はすべて >10m（候補がなければ条件を満たす）。
- negative: 中央値 >10m。それ以外は曖昧としてfitから除き、数を保存する。
- 外観は既定の `CropSamplingConfig` に通った元rowのCLIPを、当該窓で平均してL2正規化。
  両者に有効sampleがある組だけcosineを使う。外観欠損はgeometry fitから除外しない。
  **外観scoreやassociateの予測IDで擬似ラベルを作らない。**
  固定選別にはCLIPが含まれるため完全な特徴独立ではないことを限界として記す。
- video→clip→camera対/足元のside（中央値yの符号）→窓の順で等重み。
  同じ層・窓の重複track pairは重みを分ける。外観fitは最後にpositive/negativeの総重みを各1/2へ揃える。
  連続frameや断片数を独立sample数として増幅しない。
  生pair数、重み、除外、camera×近遠の支持、欠測を全保存する。

## fitとしきい値選択（devラベル不使用）

1. geometry: positiveのpair中央値dに重みwを付け、
   `sigma_m = sqrt(sum(w*d*d)/(2*sum(w)))`。
   `geometry/affinity.py:rayleigh_scale` と同じRayleigh尺度で、frame距離の尺度と混同しない。
   <4mの擬似正例による切断/選択バイアスを明記する。
2. appearance: `logit = a*cosine+b` のclass-balanced weighted logistic lossに
   `1e-4*a*a/2` の固定L2罰則を加えて最小化。a>0を必要とし、
   slope=a、center=-b/a。初期値は旧62.7/.847、最適化許容差1e-9、最大1000 iteration。
   不収束・非有限・slope<=0は停止。旧値へ黙って戻さない。
3. **leave-one-video-out 3 fold**で上記尺度を2動画だけにfitし、残る1動画の
   full clipを既存associateへ通す。検証clip自身は尺度fitへ入れない。
   判定しきい値の候補は事前固定した3組だけ:

   |candidate|min_margin|max_runner_up_ratio|
   |---|---:|---:|
   |A（旧）|1.0|0.5|
   |B|2.0|0.4|
   |C|4.0|0.3|

   geometry score clip10、appearance score clip4、continuity2、presence .25、
   segment .25秒、switch窓.25秒/jump3m、領域、handoff .2秒、曖昧区間1秒は固定。
   score clipは数値安定化の固定上限でfit対象にしない。
4. foldごとのnegative pair-windowで、同じ非負IDを割り当てた共有実観測frame割合を測る。
   positiveも同じ定義で正接続割合を測る。上記の階層重みで集計する。
   undecidedはpositive recall=0、negative誤結合=0、棄却として数え、黙って母数から外さない。
   **各動画のnegative誤結合率<=1%かつ全体positive recall>=80%**の候補だけを可とする。
   可の中でpositive recall最大、同点（差<=1e-9）はA→B→Cを選ぶ。
   各foldの検証にpositive/negativeそれぞれ10 pair-window以上、
   各foldのfitに外観positive/negativeそれぞれ10 pair-window以上を要求する。
   支持不足・可の候補なしならfitは採用保留とし、dev採点せず原因を報告する。
5. 選んだしきい値を固定し、全6clipで尺度を1回fitする。全foldの尺度/loss/誤結合/棄却と
   全dataの値を残す。fold間でsigma最大/最小>2またはslope最大/最小>3なら不安定として採用保留。
   **最終YAML、fit証拠と入力hashをcommit/pushした後**にdevを開く。
   これは擬似対応の一致を当てる規則で、人間ラベルへの最適化や独立精度保証ではない。

## devの一回採点と判定

4 dev×3cameraについて、固定した同じtrack/CLIPに旧尺度/しきい値と最終YAMLを適用する
**1回の比較バッチ**だけを行う。旧保存値のコピーと再計算を区別する。
pair TP/FP/FN・F1、clip別decided、group accuracy、exclusion、switch/fragment、
label coverage、camera×near/far、理由付き停止を報告する。
較正前のrun 11 baselineとの不一致があれば入力identityの不具合として報告する。
dev数値が悪くても再fit・別候補選択はしない。採否の材料を全件残す。
予約未見は調整全体が凍結されるまで閉じたまま。

## 今回の資源と終了

GPU不足特徴: **1 queue job、resource=all、wall上限7200秒、peak見積り8–10GB**。
DINOとViTPose/CLIPは別段階で解放、pose batch4/appearance batch8、
torch allocator上限7GiB、NVML監視で当該jobが9.5GBに達したら停止する。
推定実行60–115分、disk推定1–2GB、run全体上限5GB。CPU最大4thread、pytest -n4、
RAM available>=6GiB。小batch再試行/2件目は今回登録せずorchestratorへ報告する。
enqueue前に実際の不足camera数・見積りをIssueへ投稿し、job ID記録後はWAITING_QUEUEで終了する。
fit/dev採点とclip_000全pipelineは次run（較正設定commit後、別GPU許可）へ渡す。
