配送fenceを[PR #2084](https://github.com/Motoki0705/tennis-lab/pull/2084)へ追加し、`Closes #2032`に更新しました。

- branch: `codex/board-project-io`
- HEAD: `c456901a9`
- 通常・強制・dispatcher・期限切れの担当変更が共通fenceへ参加。配送中のregistry解放と、受理不明時の非再送を維持
- 関連回帰: **1487 passed / 1 skipped**（Playwright未設定）
- ruff・mypy・pre-commit成功
- 対象HEADの[Python CI成功](https://github.com/Motoki0705/tennis-lab/actions/runs/43711102116/job/132809983779)

作業treeはcleanです。実サービス経路は未検証で、fakeを使用しました。残りは親側の正式レビュー1回・merge・merge後main CI確認です。