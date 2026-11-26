親 #2627 は、テニスシーン3D再構成のアーキテクチャー（標準pipelineの手法＋実行システム）を日本語の技術報告として論文化する。親issue本文・合意と、担当子issueの本文を読む。#2630（閉）の調査コメントと親の統合コメント https://github.com/Motoki0705/tennis-lab/issues/2630#issuecomment-7189580548 に既存記録の数値・出典がある。

人間の判断（2026-11-26）: 各節の主張は人間が示したものを採り、必要な検証で数値の裏付けを作る。検証の結果が主張と異なる場合は、数値を変えずに記録し、最終報告で親へ明示する（親が人間に伝える）。

担当子issue: #2637

## 進め方と報告

- 担当子issueの「具体的な変更内容・対象範囲」と受入条件に従う。設計（条件・評価集合・指標・資源の見込み）を決めたら、子issueのコメントに書いてから実行する。
- 記録は knowledge-control skill（.agents/skills/knowledge-control/SKILL.md）に従い、記録専用PRにする（workers.mdの「ナレッジ記録PRの作成」、PR本文に `## 実行証拠`）。`Closes #<担当子issue>` を書く。
- GPUジョブは training queue（.agents/skills/training-queue/SKILL.md）に `add` し、queue workerは起動しない。メインcheckoutの `.training_queue` を共有する。1分以上かかるCPU処理は `parent.local_jobs start` で起動する。投入したら job ID を `PARENT_RESULT` の wait で宣言して終了し、結果を受けたターンで集計・記録する。
- 最終報告には、主張を支えるか（支持・部分的・不支持）と主要な数値、PR番号と完全なHEADを書く。
