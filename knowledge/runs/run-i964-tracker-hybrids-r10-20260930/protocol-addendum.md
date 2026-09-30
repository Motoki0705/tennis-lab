# Run 10 addendum: 指定された2つのhybrid（新結果の実行前）

2026-09-30、[ユーザー判断](https://github.com/Motoki0705/tennis-lab/issues/964#issuecomment-5904098464)。
この文書をIssueへ投稿・commit・pushしてから新条件の実データ追跡・採点を行う。
[run 8 protocol](../run-i964-tracker-matrix-r8-20260930/protocol.md)と
[run 9 addendum](../run-i964-tracker-linking-r9-20260930/protocol-addendum.md)を継承する。
変更点は以下の2条件だけ。パラメータ探索はしない。

## 追加条件

1. **StrongSORT++ + pose / CLIP** (`strongsort_pp_pose__clip_reid_market1501`):
   run 9の論文再実装StrongSORTに、Deep OC-SORT+poseと同じ`local_pose` / `pose_distance`を使う。
   poseを元検出box内で正規化し、双方confidence >= .3の共通関節が4以上のとき、
   confidenceの小さい側を重みにした平均Euclidean距離を .25 で割り、1にclipする。
   4関節未満はpose証拠なしとして加算0。最後に対応した実検出のposeを保持し、
   poseのEMA・Kalman予測・未観測pose補間は行わない。
   - confirmed trackの第1照合: `C = .98 * appearance_distance + .02 * Mahalanobis_distance + .15 * pose_distance`。
     従来のMahalanobis gate 9.4877を適用し、Hungarian後の最大cost .45は不変。
   - 未対応trackの残余IoU照合: `C = 1 - IoU + .15 * pose_distance`。
     対象（tentativeまたはage=1）と最大cost .7は不変。
   - poseコストは両段のHungarian前に加算する。外観とmotionの比を再正規化しない。
     上限をpose込みcostに適用するため、pose不一致で照合を拒否する場合もある。
   - その他の確認待ち、NSA、EMA、age、AFLink、GSIはrun 9のStrongSORT++と同じ。
     poseなし条件の既定は重み0を維持し、run 9のCLIP/KPR出力を回帰確認する。
2. **Deep OC-SORT+pose / CLIP + AFLink + GSI** (`deep_ocsort_pose_aflink_gsi__clip_reid_market1501`):
   run 9のDeep/CLIPと同じオンライン追跡（pose重み .15を含む）を再実行し、
   元row/ID/boxが保存済みオンライン結果と配列一致することを確認した後、
   run 9と同じAFLink→GSIを適用する。旧Lab連結は足さない。

両条件ともAFLinkはCPUで、前末尾30/後先頭30、共通min/max正規化、
`0 < gap < 30` frame、距離 <=75 px、非接続確率 <.05、Hungarian連結。
重みSHA-256は `b35cbeddd3acc48fece820bd640640e6bfb1f5fbf570aa79af26c6a38958daa4`。
GSIは `1 < gap < 20` の線形補間、XYWHの固定RBF GPR、tau10、
lengthscale `clip(10*log(1000/L), .1,100)`、alpha1e-10。
**主表のboxは元の実観測、IDはAFLink後**。GSI boxは別配列、補間mask、元row=-1で保存し、
固定選別・raw/group IDF1・#933 pair F1へ実観測として混入させない。
AFLink/GSIの効果といっても主指標の差はAFLink由来であり、GSIによるFN削減とは呼ばない。

## 入力・再現確認・評価

- run 9と同じ4 dev×3camera、10,491 camera-frame、COCO全画面 .30、40,531検出row、
  保存済みViTPose/CLIPを使う。検出・特徴は再推論せず、hashを確認する。
- 固定コート座標選別・CLIPによる断片group化・v3安全策・camera間CLIP対応は不変。
- 既存9条件のrun 9確定出力（`i964-linking-r9-20260930-v3`）をhash照合して保持し、
  保存追跡からraw/group照合・全層の指標・#933対応評価を再計算して数値の完全一致を確認する。
  旧baselineのboxは元検出boxへ置換しない。再利用と再実行の範囲は結果に分けて記す。
- 新2条件はオンライン追跡を二重実行して決定性を確認する。原StrongSORTのCLIP/KPRは
  pose重み0で再実行し、run 9のオンラインID/row/box一致を検証する。
- raw IDF1を主指標のままとし、group IDF1はrun 9で追加した副指標と明示する。
  全11条件で完走、raw/group IDF1、#933 pair F1（decided clipsとlabel box coverage）、
  switch/fragment、選手保持、非選手残存、camera×near/farを報告する。
  全camera停止は空予測/FN。未照合・unknown・人物50%保持も従来と同じ。

## 動画・判断材料

新2条件のraw IDF1が高い方（差<=1e-6ならswitch、fragment、名称順）を選び、
**Deep OC-SORT+pose/CLIPを上段、新条件を下段**にした3camera動画を1本作る。
run 8の規則で、人物unitのcorrect/miss/wrong_identity状態の差が最多の5秒窓を各clipから1つ、
同数なら最も早い窓を使う。raw/group IDとGSI syntheticを明示し、全frame読戻しとSHA-256を記録する。
raw/group/pair F1それぞれの見方で短い推薦を書く。既定はユーザー決定まで変更しない。
cam1-farの重複box対策は、期待効果、取り逃し/交差へのリスク、実装・CPU/GPU・disk見積りを
含む提案だけを出す。detector後処理/NMS/thresholdは変更しない。

## 境界

StrongSORT++は論文再実装を維持する。AFLink重みは
[run 9の暫定案B](https://github.com/Motoki0705/tennis-lab/issues/964#issuecomment-5903293645)どおり
このローカル比較だけに使用し、再配布・本番採用をしない。
旧track由来の部分参照ラベルの偏りとdev依存を引き継ぎ、未見性能とは呼ばない。
予約未見clip・pipeline tracker既定・#935/#936 branch/worktreeは触らない。
GPU0件、CPU最大4thread、OpenBLAS/OpenCV1、pytest -n4、RAM available >=6GiB、追加disk <=5GB。
