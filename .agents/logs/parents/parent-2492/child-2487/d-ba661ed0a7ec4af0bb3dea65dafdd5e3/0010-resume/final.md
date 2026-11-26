helper 3系列の最終結果を取り込み、[44行の比較表](https://github.com/Motoki0705/tennis-lab/blob/9b55568985cf5d8dd1c443d1202aa90d6c319465/knowledge/runs/run-meiji-player-model-comparison/conditions.csv)とPareto図2枚を更新しました。速度・VRAMを掲載した18条件はPyTorch 4・OpenCV 1スレッドで、精度と速度の測定単位は画像batch 1です。batch 2の精度は未計測です。[PR #2499](https://github.com/Motoki0705/tennis-lab/pull/2499)

- **検出のPareto**　RF-DETR-L PT16／TRT16、ECDet-L／X O365 PT16。RF-DETR-L TRT16はP50 26.43 ms/画像、ECDet-X O365 PT16は50.75 ms/画像でRecall@0.50／0.75が97.97%[local path]。
- **検出＋姿勢のPareto**　DINO＋ViTPose-H PT16／TRT16、ECPose-L／X PT16。ECPose-L／Xの欠測込み平均OKSは0.2908／0.2588、現行PT16は0.7631でした。

TensorRT変換不具合の測定結果は表と図に残し、Pareto判定から除外しました。旧2スレッドで計時した条件は精度を掲載しています。計時レポートの差し替えと終端ログの欠落を拒否する回帰テストを含む133件、pre-commit、knowledge検証、保持ログ50件のSHA-256検査が通過しました。

採用判断には、候補の実BBoxを使ったViTPose-H、下流の追跡・3D品質、ライセンスの確認が残ります。TensorRT変換不具合の原因は未特定です。`codex/parent-2492-issue-2487-primary` に `9b55568985cf5d8dd1c443d1202aa90d6c319465` をpush済みです。

PARENT_RESULT {"schema_version":1,"state":"waiting","wait":{"kind":"ci","pr":2499,"head":"9b55568985cf5d8dd1c443d1202aa90d6c319465","resume_on_success":false}}