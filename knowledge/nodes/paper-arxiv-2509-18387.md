---
id: paper-arxiv-2509-18387
type: paper
title: "BlurBall: Joint Ball and Motion Blur Estimation for Table Tennis Ball Tracking"
curator: chatgpt-schedule
date: 2026-08-06
status: reviewed
external_ids:
  doi: null
  arxiv: "2509.18387"
  openreview: null
published_at: null
reviewed_at: 2026-08-06
evidence_level: fulltext-code-data
tasks: [ball_detection]
repo_paths:
  - src/tasks/ball_detection/data/types.py
  - src/tasks/ball_detection/data/dataset.py
  - src/utils/data/heatmaps.py
  - src/tasks/ball_detection/models/spatiotemporal_unet.py
  - src/tasks/ball_detection/training/lightning_module.py
sources:
  - kind: paper
    url: https://openaccess.thecvf.com/content/CVPR2026W/CVsports/html/Gossard_BlurBall_Joint_Ball_and_Motion_Blur_Estimation_for_Table_Tennis_CVPRW_2026_paper.html
  - kind: code
    url: https://github.com/cogsys-tuebingen/blurball
  - kind: dataset
    url: https://cloud.cs.uni-tuebingen.de/index.php/s/C3pJEPKWQAkono7
relations: []
tags: [literature, ball-detection, motion-blur, heatmap]
---

## 要約

高速なtable tennis映像でballが点ではなくstreakとして写ることを明示的に扱い、ball center、blur orientation、blur extentを共同推定する。64,119 framesへblur属性を付与し、streak端ではなく中央を位置labelとする。WASB/HRNet系multi-frame detectorへSqueeze-and-Excitationとblur-aware heatmapを導入する。

## 主要な主張と根拠

同一公開dataset上で、推奨threshold 0.7の1-step推論はF1 97.17を報告し、blurとpositionを併用したtrajectory評価ではposition-onlyより誤差を低下させた。実験が示すのはtable tennisの固定camera中心条件におけるblur-aware supervisionの有効性であり、tennis broadcast映像への一般化ではない。

## tennis-labへの適用可能性

現行datasetは点Gaussian targetを生成するため、まずmodelを変更せず、任意の`blur_angle`と`blur_half_length`をlabel contractへ追加し、線分または異方性Gaussian targetへ置換する最小比較が可能である。合格後にのみSE blockやblur属性headを検討することで、target変更とarchitecture変更の効果を分離できる。

## 制約・失敗条件

直線streak仮定はbounce、racket contact、非線形軌道、rolling shutterで破綻し得る。公開domainはtable tennisで、ball size、camera距離、背景、frame rateがtennisと異なる。重複frameを含む再encode映像に敏感であり、白色物体によるfalse positiveも残る。追加属性を持たない既存TrackNet labelにはannotationまたは推定処理が必要である。

## コード・データ・ライセンス

公式repositoryはtraining/evaluation、pretrained weights、dataset downloadを公開し、codeはMIT licenseである。dataset配布ページで独立した利用条件を確認できていないため、再配布や派生dataset公開の前に権利条件を確認する必要がある。
