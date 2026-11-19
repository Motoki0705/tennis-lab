#2189 を実装し、PRを作成しました。[PR #2192](https://github.com/Motoki0705/tennis-lab/pull/2192)

branch `codex/worktree-cleanup-quarantine-residue`  
commit `4ad95c975`

- 関連216テスト、pre-commit、対象HEADのCI全14ジョブ（Python含む）が成功。
- 修正を外すと31ケースが失敗し、復元後の対象51ケースは成功。
- 本番stateコピーのdry-runを2回実行し、3件すべてが削除対象、参照0、エラー0。移行結果と本番stateのハッシュ一致も確認。

最古形式の照合には、保存済み管理birth timeと実体birth timeのナノ秒単位の一致を採用しました。11月16日04:03の別セッションのpruneと3件の登録消失との対応は、時系列からの推定です。

親による統合レビュー・merge後に、実ホスト3件の削除とstate確認が残っています。