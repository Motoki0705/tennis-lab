[PR #1953](https://github.com/Motoki0705/tennis-lab/pull/1953)を作成しました。336時間での強制削除、停止連携、spin入口、04:30 JSTのtimerを実装済みです。

- branch: `codex/worktree-expiry-cleanup`
- HEAD: `e82ef1fc7`[local path]`codex/spin-ops-control`
- 広域回帰697件、最終確認87件、pre-commit、対象HEADのCI `Python`が成功。
- 親停止用の登録フックは独立commit `edf6dd637` に分離。
- 実repoのscan：115件、期限超過94件、birth time取得115件。実削除・ホスト有効化は未実施。

残りは親による統合レビュー、mainへの載せ替え、実ホスト検証・mergeです。validatorは指定なし（試行0／完了0）。