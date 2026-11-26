[#2487 のdraft PR #2499](https://github.com/Motoki0705/tennis-lab/pull/2499) を作成し、CIを引き渡します。ブランチは `codex/parent-2492-issue-2487-primary`、commitは `bdbbb2127` です。draftは #2498 をbaseとし、作業ブランチの土台には #2497 も含みます。

代表6 clipの6,844画像で、保存済みTensorRT FP16出力を採点しました。DINOのRecallはIoU 0.50で **96.67%**、0.75で **47.07%**。DINO＋ViTPose-Hの検出欠測込み平均OKSは **0.7631**、PCK@0.10は **77.95%** でした。関連テスト24件、pre-commit、knowledge検査が通過しました。[採点結果と生出力のSHA-256](https://github.com/Motoki0705/tennis-lab/blob/bdbbb2127f8a842ae007ae42b0518e00f77d9de2/knowledge/runs/run-meiji-player-model-comparison/baseline-trt16.json)

PT32/PT16の精度、#2496の速度基準との照合、候補30条件の統合と可視化、`recipe=ecpose` の実装・回帰検証が残っています。候補adapterの契約は [#2487](https://github.com/Motoki0705/tennis-lab/issues/2487#issuecomment-7129890902) に共有済みです。

PARENT_RESULT {"schema_version":1,"state":"waiting","wait":{"kind":"ci","pr":2499,"head":"bdbbb2127f8a842ae007ae42b0518e00f77d9de2","resume_on_success":true,"next_step":"#2491の共通recipe実装とECPose adapterの状態を照合し、別経路の接続と既定recipeの回帰検証を進める。#2496の固定速度基準と補助担当の成果が揃った条件から同じPRに統合する。"}}