[#2499](https://github.com/Motoki0705/tennis-lab/pull/2499) に main `59faa32a2` と #2501 `8a25fe12b` を取り込み、単眼PLCS・FastHMR・ECPoseの経路と既存FP32キャッシュの互換性を検証しました。pushしたHEADは `c7c76e32a2bbd62721e8883007b1a1d8cc7bfb5d` です。PRのbaseは #2501 です。

- 固定6,844画像で、DINO PT16のRecall@0.50は96.69%、Recall@0.75は46.99%でした。DINO＋ViTPose-H PT16の検出欠測込み平均OKSは0.76313、PCK@0.10は77.99%でした。現行PT32精度はユーザー判断による対象外と記録しました。
- 修正版harnessでのDINO単体のバッチ1 P50はPT16が165.69 ms/画像、TRT16が94.79 ms/画像でした。parityは両方とも「PT32未計測」です。#2497の公開記録にDINO＋ViTPose-Hの一体計測がなく、この2条件の速度は未計測です。[暫定比較表](knowledge/runs/run-meiji-player-model-comparison/conditions.csv)
- 対象pytest 72件、pre-commit、knowledge検証が通過しました。#1739がECPose成果を明示的に拒否し、採用時に入力検査の拡張が必要なことをPRに記載しました。

候補の結果を受け取ってから比較表とPareto図を確定します。現行の一体速度は、修正版harnessでの追加計測を親に判断してほしいです。

PARENT_RESULT {"schema_version":1,"state":"needs_parent","wait":null}