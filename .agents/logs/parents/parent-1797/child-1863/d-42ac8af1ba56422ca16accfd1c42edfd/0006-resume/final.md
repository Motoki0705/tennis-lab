[PR #1907](https://github.com/Motoki0705/tennis-lab/pull/1907)へ2 commitをpushし、本文の契約・検証表を更新しました。

- **pr-1907-1:** 直接serveを含め、正式な判断捕捉の設定を明示拒否。
- **pr-1907-2:** 分類漏れ、重複、1指摘への複数goldを拒否。既存14 runの親判定は受理。
- **pr-1907-3:** 失敗finalは候補として保持し、品質集計から除外。AI起動前の失敗も件数へ計上。
- **pr-1907-4:** 公開入力のworktree参照2か所を無害化。元入力と公開版のdigestを保持し、verifyで変換を再現。PR全体105ファイルを検査。

テストは19件追加し、`pytest -n 0`で **38 passed、7.74秒**。各指摘の故障注入4件はすべて失敗を検出し、復元後に成功しました。ruff、mypy、pre-commit、Knowledge検証も成功。[検証記録](https://github.com/Motoki0705/tennis-lab/blob/81ab6eb41c64c6ad36dc9c763c2efadd1d37e78a/knowledge/reports/review-model-comparison-20261111/review-fix-validation.json)に保存しています。

再計算結果は不変です。両モデルとも正解6件検出・3件見逃し。select投影は使用量4.305%増、時間1.621%増で、fixed維持の結論も同じです。実LLM・実agentの追加起動は0回です。

**CIは成功していません。** [CI](https://github.com/Motoki0705/tennis-lab/actions/runs/43034986105)はCPU全10ジョブがOS依存準備のtimeoutで失敗し、pytest未到達。[Knowledge](https://github.com/Motoki0705/tennis-lab/actions/runs/43034986222)は本体成功、WebUIはDNS障害後に時間上限で取消でした。全失敗ログを確認済み。書込み制限に従い再実行は0回です。

branch: `codex/review-eval-acceptance`  
HEAD: `81ab6eb41c64c6ad36dc9c763c2efadd1d37e78a`

CI回復後の確認、親の再レビューとgateが残っています。