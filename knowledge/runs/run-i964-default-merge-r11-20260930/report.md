# Run 11 固定比較

pipeline追跡の既定はユーザー決定のStrongSORT++＋pose/CLIP。**重複統合はoffのまま**。同じ固定設定でoff/onを比較する。

統合ルール・元rowの保持・別人候補監査は`protocol-addendum.md`、camera別の件数は`merge_counts.csv`。

raw IDF1はrun 8の主指標。group IDF1はrun 8結果後に追加した副指標で、下流のcamera-local連結groupを測る。
全条件とも同じ固定コート選別。GSI補間は実観測へ昇格せず、主表は実観測のみ。
統合offはrun 10と全raw/group層・pair指標・ID/元box/元row/GSIが完全一致。productionの共通入口で両条件を再実行した。未見性能/完全GT MOTとは呼ばない。

|条件|完走|raw IDF1|group IDF1|#933 pair F1 / decided|対応label box coverage|raw switch/frag|group switch/frag|raw/group選手保持|非選手raw/group|人物50%|
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|strongsort_pp_pose_merge_off__CLIP|12/12|0.955395|0.969915|0.957119 / 4/4|25055/25147|4/21|1/21|19659/19659 of 20558|3/3|8/8|
|strongsort_pp_pose_merge_on__CLIP|12/12|0.955395|0.969915|0.957119 / 4/4|25054/25147|4/21|1/21|19659/19659 of 20558|3/3|8/8|

## camera × near/far（raw / group IDF1）

|条件|cam0 near|cam0 far|cam1 near|cam1 far|cam2 near|cam2 far|
|---|---:|---:|---:|---:|---:|---:|
|strongsort_pp_pose_merge_off__CLIP|0.9965 / 0.9965 (N=3270)|0.8787 / 0.9502 (N=3270)|0.9987 / 0.9987 (N=3430)|0.8964 / 0.9151 (N=3430)|0.9688 / 0.9688 (N=3367)|0.9835 / 0.9835 (N=3367)|
|strongsort_pp_pose_merge_on__CLIP|0.9965 / 0.9965 (N=3270)|0.8787 / 0.9502 (N=3270)|0.9987 / 0.9987 (N=3430)|0.8964 / 0.9151 (N=3430)|0.9688 / 0.9688 (N=3367)|0.9835 / 0.9835 (N=3367)|

unknownを含む全32層とIDTP/FP/FN・保持数は`comparison.csv`。停止/未照合数は`availability.csv`。

## camera間対応

pair F1はdecided clipのcountsをpool。4/4未満は条件付きの値で、coverageが異なる。
CLIPの固定calibrationを使用し、再較正はしていない。

|Tracker|camera間encoder|decided|pair F1|TP/FP/FN|group accuracy|label box coverage|
|---|---|---:|---:|---:|---:|---:|
|strongsort_pp_pose_merge_off__CLIP|CLIP|4/4|0.957119|18459/6/1648|0.763256|25055/25147|
|strongsort_pp_pose_merge_on__CLIP|CLIP|4/4|0.957119|18459/6/1648|0.763256|25054/25147|

人物→予測IDの全対応は`correspondence.csv`、未決定理由は`association.csv`。

旧box・旧track由来のラベルによるbaseline有利の偏り、選択後の部分参照指標、既知clipへの設計上の依存は残る。
StrongSORT++のGSI再構成とmaskは各tracking結果の`gsi`参照で監査できる。
