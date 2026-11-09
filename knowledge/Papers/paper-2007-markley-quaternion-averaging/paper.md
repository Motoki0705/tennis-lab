---
id: paper-2007-markley-quaternion-averaging
type: paper
title: Quaternion Averaging
year: 2007
authors:
- F. Landis Markley
- Yang Cheng
- John L. Crassidis
- Yaakov Oshman
tasks:
- tennis_scene
source: https://ntrs.nasa.gov/citations/20070017872
---

## 研究の要点

unit quaternionの外積を重み付きで足した4×4行列から、最大固有vectorとして平均回転を得る。qと−qで結果が変わらず、成分平均がゼロになる問題を避ける。これはchordalな姿勢平均であり、SO(3)のgeodesic距離の平均と同一ではない。

## このプロジェクトとの関係

#1776の表示カメラ安定化で、有限のGaussian時間重みを付ける平均として採用する。元の論文を映像安定化アルゴリズムの実証とは扱わず、未来frameの使用・欠測区間・窓の端・元姿勢との偏差は本pipeline側で定義・検証する。原文PDFは保存せず、NASA一次資料への参照だけを保持する。

## 検証したい仮説・適用限界

細かな回転変動を抑えながら大きな首振りを残せるかを、角速度/加速度/jerk、周波数、姿勢偏差、先読みの量と動画で評価する。平均が一意でない広く散った回転は検出して拒否する。頭部推定の正確さや人間が感じる揺れの解消を、固有vectorの計算だけで保証しない。
