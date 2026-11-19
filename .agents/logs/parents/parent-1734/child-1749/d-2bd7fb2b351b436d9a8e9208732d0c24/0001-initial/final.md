[PR #2156](https://github.com/Motoki0705/tennis-lab/pull/2156)を作成しました。  
branch: `codex/scene-shared-plan` / commit: `0fdc24459`

| 共通化した責務 | 単一clip入口 | dataset入口 |
|---|---|---|
| 計画・cache/load・実行 | `ExecutionPlan` / `PlanExecutor`へ委譲 | 同じ1clip計画を利用 |
| 入力検証・改変検知 | mediaを検証 | manifest・mediaを検証 |
| scene最終化 | `finalize_clip` | 同じexportを参照 |
| 完成・失敗marker | run receiptを維持 | 共通publisherを呼出 |

- CPU回帰226件成功（追加29件）、pre-commit成功。
- 固定fake入力・code identityで31nodeの変更前後成果物が一致（生成時刻除外）。
- 対象HEADの[CI](https://github.com/Motoki0705/tennis-lab/actions/runs/43730532991)は`Python`を含む全check成功。

ローカルGPT一連E2Eは`os.pidfd_open`未提供で未完走です。変更前mainでも同じ失敗を確認しました。validatorは未指定のため試行0／完了0です。

親の統合レビュー・mergeと、merge後main CI確認が残ります。