# Run 8: 結果を見る前に固定する評価プロトコル

2026-09-30、#964。この文書をIssueへ投稿・commitしてからtrackerを実行する。
4 devは `video_000/clip_000`, `video_000/clip_007`, `video_001/clip_001`,
`video_002/clip_013`、各cam0/1/2、全10,491 camera-frame。未見予約clipは開かない。
ラベル・動画・校正・検出のidentityは[特徴plan](../run-i964-coco-person-features-r7-20260930/plan.json)の
sources.json参照とそのSHA-256を正とする。ラベルは各clipの
`annotations/player_association/labels.json`（#944レビュー、8選手）。

## 主指標と選択順

**固定コート選別後のcamera-local raw tracking IDについて、既知ラベルunit上のIDF1。**
人物unitは `(clip,camera,frame,person)`。同人物の重複参照boxを1unitにまとめる。
frameごとに予測boxとunit内全参照boxの最大IoUを取り、IoU≥0.5で1対1対応する。
対応数最大、次にIoU和最大のHungarian。入力は予測ID昇順・人物index昇順とし、完全同点は
固定版SciPyの決定的な出力に従う（score・encoder・ラベル人物名でtieを変えない）。
clip×camera内で全timelineを使いGT人物↔予測IDの対応数最大の割当を取りIDTPを計算。
IDFN=全参照player unit数−IDTP、IDFP=選択済み既知unit対応予測数−IDTP。
既知non-playerに対応した選択済み予測はIDFP。未対応・曖昧boxはGTが不完全なためIDFPに入れず別集計。
従って完全アノテーションMOTのIDF1とは呼ばない。全12 camera-clipのIDTP/FP/FNをpoolして
`2*IDTP/(2*IDTP+IDFP+IDFN)`を主指標にする。court連結groupのIDへ置換してraw断片化を隠さない。

推薦は**新検出を共用する外観+pose候補4条件**のIDF1最大。差≤1e-6ならID switch少、
次にfragment少、次にmethod名/encoder名辞書順。旧baselineを推薦対象にしない。
実行失敗は理由・対象を明示し、そのcameraの予測は空（全player unitがFN）として完走率を併記する。
部分的な成功出力を正規結果に昇格させない。全候補がbaselineを下回っても候補内の推薦と不足を併記し、
**pipeline defaultを変更しない**。

## 副指標・層別・動画

- ID switch: 同じGT選手の連続するmatched unitでraw IDが変わった回数（欠測を挟んでも比較）。
- Fragment: GTラベルが連続して存在する間の matched→unmatched→matched。参照自体の欠測は分断境界。
- 選手unit保持、8人物の50%以上保持、既知non-player残存、未対応選択box数、raw/selected ID数。
- camera別、near/far別とcamera×near/far別。同frameで2選手の最大面積参照boxのbottom-yが
  小さい方をfar、大きい方をnear。2選手が揃わない/同点はunknown。strataでも独立にID割当し、分母を併記。
- 既存#933 evaluatorのpair F1、group accuracy、coverage、人物→予測ID混同行列と対応表。
  camera内IDF1のswitchと#933のtrack上人物変化の検出率を混同しない。
- 最良候補とnew detection+old trackingで、選手unitごとの一致/欠落/ID誤りの相違が最多の
  5秒窓を各clipから1つ選ぶ。同数なら早い窓。3camera同時・baseline/候補上下の動画で
  ID、switch、fragment、既知人物を表示する。停止baselineは停止表示し代替出力を正規結果としない。

## Matrixと固定パラメータ

1. old detection + old tracking: #933 observe-v1保存artifact（COCO ROI、BoT-SORT motion/IoU、Lab再連結）。
2. new detection + old tracking: COCO full-frame .30、800/1333の同じ40,531rowを、旧BoT-SORT設定
   (`.25/.1/.25`, fuse_score=true, buffer30, match .8, sparseOptFlow, ReID off)と既存Lab規則へ渡す。
   旧の補間・平滑化boxとobserved maskを保持。raw capだけ外し、共通選別後に上限6。
3. BoT-SORT appearance+pose × {CLIP-ReID, SOLIDER}: 既存`BotSortPoseConfig`の数値を変更せず、
   raw max_tracksだけ無制限。IoU .05、cosine距離veto .25、appearance .35、pose .15、EMA .9、gap1秒。
4. Deep OC-SORT + pose × {CLIP-ReID, SOLIDER}: [公式](https://github.com/GerardMaggiolino/Deep-OC-SORT)の
   observation-centric Kalman再更新、方向速度、adaptive appearance weight/updateを使うadapter。
   IoU .3、delta_t3、inertia .2、appearance .75、alpha .95、adaptive weight .5、max_age30、min_hits3。
   scoreはsource .30のみ、CMC/grid/new-KF off（固定camera、共通embedding、公式best ablationの旧KF）。
   poseは既存の正規化距離/conf .3/4関節/scale .25を用い、対応scoreから .15×distanceを引く。
   poseはfirst/OCR両段に使う。これは独自pose追加で、論文の厳密な再現とは呼ばない。

2方式で要求の2–3を満たすためStrongSORT++は今回追加しない。random seed=0、CPU thread≤4、
OpenCV/OpenBLAS=1。tracker→検出rowを厳密に保存し、仮想/予測boxを新検出と称さない。
重み・全NPZ・入力manifest・実装・出力のhashと全設定を保存。決定的な追跡は再実行で配列一致を確認。
不具合修正が必要なら前結果を保持して理由をIssueへ記録。結果に基づく閾値探索はしない。

## 共通の選別・camera間対応

run 6/7の`LinkingConfig`と`select_linked_candidates`を変更せず全条件に適用する。
選別の外観は常にCLIP（encoder軸で選別閾値の意味を変えない）。sample_tracksの既定
（min height48、border4、overlap .1、最大16 samples）を用いる。
候補は同じ検出rowの保存CLIP特徴を参照する。旧trackingのKalman/平滑化boxは同一cropではないため、
そのboxで既存CPU encoder/cacheから特徴を計算し、近い検出の特徴を代用しない。
core |x|≤4.115 / |y|≤16.885 m、連結後distinct滞在25%、選択断片の全実観測保持、その後group上限6。
v3は1秒未満の曖昧区間除外、same-camera handoff≤.2秒を維持する。

camera間対応は全条件で同じassociate/config/sideを使い、CLIPを主比較とする。
さらに候補4条件それぞれでcamera間encoderをSOLIDERに替え、**tracker encoder×association encoder**を分離する。
SOLIDERでもslope62.7/center .847を固定し、再calibrationしない。これはCLIPで校正した尺度の転用であり、
SOLIDER固有の最適値を比較したとは解釈しない。undecidedは理由付きで報告し、IDを補完しない。
sideは既存注釈ball由来の固定判定で、end-to-end pipeline qualificationとは区別する。

## 既知の偏りと今回の境界

参照boxは旧COCO pipelineから、identityラベルは旧trackのレビューから作られている。
旧検出・旧追跡baselineにはbox形状とIDの両方で有利な偏りがある。旧経路の優位を方式採用の根拠にしない。
この4 singles clipは既に設計に使ったdevであり、未知状況・未検出人物・doublesへの一般化は未確認。
KPRは推論port/CPU parityを別途行い、成功後だけ同じrowの特徴jobを1件enqueueする。
KPR結果を今回のCLIP/SOLIDERの事前推薦へ混ぜない。最終採用と未見1回評価は後続run。
