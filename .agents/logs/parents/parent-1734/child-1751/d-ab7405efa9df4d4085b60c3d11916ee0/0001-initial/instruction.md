## 担当範囲
#1751（C3）を完了させる。推論runtimeがモデルを所有し、計画上そのモデルを使う最後のnodeの後で解放する。単一clipはN=1の計画として同じ仕組みを使う。

## 前提（2026-11-16時点）
- #1749（PR #2156）がmerge済み。`src/tennis_scene/pipeline/execution_plan.py` の `ExecutionPlan`（(clip_id, node) キーのgraph、nodeごとに1回だけのcache/load判定）と、`runner.py` の `PlanExecutor`（ready nodeの選択、clip完成の通知、`completion_results`）の上に作る。
- #1750（PR #2153）もmerge済み。DINOに `predict_batch` がある。ただし、batch化そのものはC8（#1755）の担当で、この子の対象外。
- C1（#1748）のrecorderはPR #2154（未merge・draft）にある。C3のruntimeがそのrecorderをそのまま使える形が望ましいが、#2154はまだ変わる可能性がある。load/unloadの回数・時間を記録できるhookを、recorderに依存しない形で用意する。#2154がmergeされたら接続する前提で、方法を報告する。
- ボール部品は、既存ConvNeXtとX3D/CSN＋DPT（#2151、PR #2155・未merge）を設定で選べるようになる予定。モデル共有キーは、モデル種別・重みの内容・device・dtype・前処理/推論設定から決める（親issueの方針B）。

## 資源と制約
- **2026-11-16の朝、別の親（#1990）が `data`・`outputs`・`ckpt` の配置を切り替えている。完了まで、data・outputs・ckptを使うGPUジョブ・推論は起動しない。** CPUのunit/integrationテスト（fake component・fake model）で実装と検証を進める。実モデルでの確認（GPU）が必要になったら、その時点で止めて、必要な確認内容を報告する。親が切替の完了を確認してから続きを依頼する。
- worktreeには `.venv` のsymlinkだけがある。`data`・`ckpt`・`outputs` はまだ張っていない。

## 報告
PR URL、runtimeのAPIとモデル共有キー、全cache hit・load専用で読み込まないことのテスト、例外・中断時の解放のテスト、GPU確認として残った項目。
