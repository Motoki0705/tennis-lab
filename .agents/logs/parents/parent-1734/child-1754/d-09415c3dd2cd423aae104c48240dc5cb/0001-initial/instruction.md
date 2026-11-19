## #1754 の実装: HMR2の画像特徴をGVHMRから分け、成果物として保存する

issue #1754 の仕様・受入・テスト設計に従って実装してください。今日の主経路は一括CLI（#1752）で、この子はそれを止めない最適化です。#1752・#1753とのconflictが最小になるように、変更を body_view_selection の後の HMR2特徴node・`gvhmr` の境界に閉じてください。
- 受入の中心は、segment・人物・frameの対応のunitテストと、分離前後で GVHMR・body_placement の出力が許容誤差内で一致すること。
- 実GPUでの比較は、queue経由で1clipに限る。
### 共通の前提（2026-11-17、親 #1734）
- 人間の目標: **今日中に、video全体（meiji_4cam / video_000 の全69clip）を一括処理できるようにする。** 子どうしは並行して動いている。
- このbranchは、#1751（PR #2182、branch `codex/scene-inference-runtime`、InferenceRuntime）の上に積み重ねている。最初に `origin/main` をmergeして最新にする（main には #2190 のコート変更が入っている）。PRのbaseは `codex/scene-inference-runtime` にする。#1751がmergeされたら、親がbaseを付け替える。
- GPU: 共有GPU lockは復旧した。GPUを使う処理は、必ずtraining queueを経由する（`TRAINING_QUEUE_DIR=[local path]`、`.agents/skills/training-queue/SKILL.md` に従う）。他の子の受入確認もqueueに入るので、GPUの処理は短く、固定した入力に限る。pytestは `CUDA_VISIBLE_DEVICES=''` で実行する。
- 静かなフォールバックを入れない。PRを作成し、CIの成功まで確認する。merge・issue closeは親が行う。PR本文は日本語で書き、`Closes #<子issue>（親 #1734）` を含める。
