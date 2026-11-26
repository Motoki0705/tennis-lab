補助3系列の結果を照合し、36条件の比較表、Pareto図2枚、代表6 clipの同一フレーム比較を更新しました。現行モデルを含む22条件に、固定入力での精度と速度がそろいました。[PR #2499](https://github.com/Motoki0705/tennis-lab/pull/2499)

- **検出置換**　RF-DETR-L TRT16は、同じCPU設定のDINO TRT16に対し、P50が26.43対94.79 ms/画像でした。Recall@0.50は96.81%対96.67%、@0.75は42.09%対47.07%です。検出置換の候補ですが、実BBoxを使ったViTPose-Hと下流品質は未確認です。
- **検出＋姿勢**　ECPose-L PT16は55.11 ms/画像、欠測込み平均OKS 0.2908でした。DINO＋ViTPose-H PT16は266.30 ms/画像、0.7631で、この疑似GTでは高速化に伴う品質差が大きく出ました。

EdgeCrafterとRT-DETRv4の計時はCPU 2スレッド、DINOとRF-DETR-Lは4スレッドです。RT-DETRv4 TRT16の2条件は、同一入力でRecallの急落が再現されたため、測定点を図に残してPareto計算から除外しました。EdgeCrafterのTensorRT精度低下の原因は未特定です。

既定FP32キャッシュの回帰を含む106件、pre-commit、knowledge検証が通過しました。`codex/parent-2492-issue-2487-primary` の `edeea50b43ea990fcae8dd2b0e1173948d36aa6d` をpush済みです。

PARENT_RESULT {"schema_version":1,"state":"waiting","wait":{"kind":"ci","pr":2499,"head":"edeea50b43ea990fcae8dd2b0e1173948d36aa6d","resume_on_success":false}}