---
id: group-i647-blcs-line-cnn-token-ablation
type: group
title: BLCS CNN court token数アブレーション (#647)
issue: 647
members:
- run-i647-blcs-line-cnn-tok1-noaug-50ep-r2
- run-i647-blcs-line-cnn-tok4-noaug-50ep-r2
- run-i647-blcs-line-cnn-tok16-noaug-50ep-r2
parents:
- run-i647-blcs-line-noaug-50ep
tags:
- blcs
- court-line
- cnn
- line-map
- token-ablation
- no-augmentation
- negative-result
---

## まとめ

RANSAC線分のsort + flattenを廃止し、clean binary court-line mapを3段depthwise-separable CNNで直接圧縮した。共通条件はBLCS small axial、broadcast single-view、ball観測・court-line augmentationなし、50 epochであり、CNN feature mapのpool gridだけを`1 x 1`、`2 x 2`、`4 x 4`へ変更した。

| court token数 | test位置誤差 | endpoint誤差 | 位置1.2m以内 |
|---:|---:|---:|---:|
| 1 | 8.543m | 11.040m | 2.04% |
| 4 | 8.476m | **10.842m** | 1.84% |
| 16 | **8.323m** | 12.263m | 1.86% |

位置誤差はtoken数とともに`8.543 -> 8.476 -> 8.323m`と小幅に下がったが、16 tokenのendpoint誤差は`12.263m`へ悪化した。全条件のvalidation位置誤差は約`9m`で早期に頭打ちし、KP対照 [[run-i647-blcs-kp-small-100ep]] の位置`1.828m` / endpoint`3.046m`との差は埋まらなかった。したがって、RANSAC由来の順序不安定性は唯一の原因ではなく、CNN化と空間token数の増加だけではline mapから世界座標系のcourt geometryを学習できない。

次はtoken数の追加よりも、固定sampleへのoverfit試験で表現能力を切り分ける。記憶できる場合はcourt tokenごとの明示的なgrid位置encodingを比較し、記憶できない場合はline mapからhomography・camera poseなどの幾何量を明示推定する経路を検討する。
