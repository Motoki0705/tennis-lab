# PLCS固定track Re-ID

人物Re-IDとcourt sideは別モデル・別重み・別optimizer・別checkpointです。
BLCSのassociation機能とtasks/baseへのassociation共通化は廃止しました。
汎用のDataset/Lightning/attention基盤だけを既存共通実装から利用します。

## 入力

2D trackerはカメラ内の同一人物にシーン全体で同じIDを返し、退出・再登場でも維持します。
`FixedTrackRegistry`が各cameraのIDを固定slotへ配置します。slotは欠測・退出後も保持し、
再利用しません。P=4なら各camera累計4 trackまでです。5人目は明示的errorとなり、
時間crop/windowで対応表や人数をリセットしません。全camera共通の4人上限はありません。
生IDとslot番号はモデル特徴にしません。同じ数値のIDもcameraを跨いだ同一性を意味しません。

入力はhuman_kp `(B,V,T,P,17,2)`、human_vis `(B,V,T,P,17)`、
camera-local CourtKP14/visibility、padding_mask `(B,V,T)`です。
sideだけがreference_view_index `(B)`を追加で受け取ります。
実観測maskは補間bboxと区別します。時刻は約30fpsで同期し、入力区間全体を一度に推論します。
model入力上限を超えるclipを黙って圧縮・分割しません。

## Re-ID

各(camera,track)に共有初期値のqueryを1つ持たせ、次をstageごとに反復します。

1. `(B*V*P,T+1,D)`の全域時間attentionでqueryとその人物の観測だけを更新。
2. 時間情報を集約したqueryを`(B,V*P,D)`の全域attentionで更新。
3. 更新queryと時間tokenを次stageへ渡す。

人物方向P→1の圧縮、フレームごとのID分類、固定Kクラスはありません。
camera/slotの並び順をRoPE座標にせず、時間RoPEのみが時刻を表します。
最終queryを投影・L2正規化してtrack_embedding `(B,V,P,D)`とします。
RGBは使わず、特徴はpose・動き・court contextから抽出します。
全欠測trackをattention/matchingから除き、欠測値の数値を特徴に使いません。

同一scene内の異なるcameraの同一人物を正例、別人物を負例とするbalanced pair BCEで
cosineを学習します。温度・marginはloss設定で明示します。補助is_player headは
FP trackを区別し、教師-1同士を同一人物の正例にしません。
validation pair F1でcosine閾値を選び、checkpointのmatching_thresholdへ保存します。
同点の閾値は高い方を選びます。testには保存済み閾値を適用します。

matchingはcosine surplusの総和を最大化する整数最適化です。1人物groupあたり各camera高々1 track、
対応の推移律、group内の全pairが閾値を超えることを制約にします。相手のない有効人物も
単独groupとして保持し、全sceneのgroup数が4を超える場合を許します。
solver失敗を貪欲matchingへfallbackしません。global IDの数値自体に意味はありません。

## Side（暫定構成）

CourtSideModelは現行のview時間query＋mHC/view attentionを独立モデルへ分けた暫定実装です。
Re-ID出力・教師ID・推定sideを互いのモデル入力へ流しません。sideは参照cameraに対する
180度court half-turnで、参照viewはfalseです。人物のnear/far分類ではありません。
今回はRe-IDだけを学習し、結果を見てsideアーキテクチャを再検討します。
新side checkpointは未学習です。旧のside/ID同時推定checkpointは読み替えず拒否します。

## 合成データ・学習

`generate_reid_dataset`はACCAD full-source motionから1〜4人物/scene、1000scene、
seed42、scene単位800/100/100 splitを生成します。これは各camera累計4人契約を満たす
学習分布です。全sceneで4人を超える部分観測の入力も実装・テストでは扱います。
scene splitであり、未見人物/未見source motionのholdout精度ではありません。

cameraごとに独立なランダムlocal IDを付け、時系列内では固定します。
capacityはcrop前の全sceneのcamera別観測人物から検査します。
教師physical IDはtrack_person_id `(B,V,P)`にだけ残します。
noise・joint/track欠測はtrack同一性を壊しません。FPは未割当slotを一つ消費する継続trackです。
短いsourceも保持してpaddingします。固定人数のため同時人数の均等化は要求せず、
`run.require_uniform_occupancy=false`を明示します。full-source区間・同時人数上限は監査します。

```bash
.venv/bin/python -m src.tasks.plcs.scripts.generate_dataset --config-name generate_reid_dataset
# GPUは共有training queue経由で実行する。
.venv/bin/python -m src.tasks.plcs.scripts.train --config-name train_reid
```

初期recipeはD256、4stage、8head、FFN768、T512、batch4、勾配蓄積4、60epoch、seed42です。
最低val/lossのcheckpointを選び、testでpair precision/recall/F1、整合的matchingのprecision/recall/F1、
group正解率を計算します。教師・embedding・予測group・test split indexをpred_test.npzに保存します。
checkpointは各runのlogs/version_*/checkpoints、集計はassociation_metrics.jsonlとpredictionsに保存します。
旧association checkpointからの自動移行はありません。

推論はPlayerReIDPredictorとCourtSidePredictor、pipeline接続は
[person_association.py](../../tennis_scene/pipeline/components/person_association.py)です。
Re-ID結果はlocal track IDを使って元動画の全実観測frameへ戻します。
